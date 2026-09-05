"""MCP gateway.

Agents do not talk to MCP servers. They talk to the gateway, which owns every connection
and applies the platform's controls to servers written by people who have never heard of
this platform.

What the gateway adds over a direct connection:

**Namespacing.** Tools are exposed as ``server.tool``, so two servers offering ``search``
coexist and an audit record says which one ran.

**Permission mapping.** An MCP server declares no permissions — the protocol has no concept
of one. The gateway assigns each imported tool a platform permission, defaulting to the
write-level permission for anything whose name suggests mutation. Defaulting to the
*restrictive* side is the important part: a server that adds a ``delete_everything`` tool
in a later release should be refused by default, not silently granted.

**Output containment.** Tool output from an untrusted server is content from outside the
trust boundary, and it is about to be placed into a model's context. It goes through the
guardrail pipeline first. This is the mitigation for the tool-poisoning class of attack,
where a benign-looking server returns a payload aimed at the model rather than at the user.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from eap.appops.mcp.client import MCPClient, MCPServerConfig, MCPTool
from eap.appops.tools.base import ToolResult
from eap.appops.tools.registry import ToolRegistry
from eap.identity.rbac import PERM_TOOL_EXECUTE, PERM_TOOL_EXECUTE_WRITE
from eap.platform.errors import NotFoundError
from eap.platform.telemetry import get_logger
from eap.secops.guardrails.base import Boundary
from eap.secops.guardrails.pipeline import GuardrailPipeline

log = get_logger(__name__)

# Verbs that indicate a tool changes state somewhere the platform cannot undo.
# fmt: off
MUTATING_HINTS = (
    "create", "update", "delete", "remove", "write", "send", "post", "put", "patch",
    "execute", "run", "deploy", "merge", "close", "publish", "upload", "insert", "drop",
    "revoke", "grant", "assign", "approve", "cancel", "restart", "scale", "terminate",
)
# fmt: on


def infers_mutation(tool_name: str, description: str = "") -> bool:
    """Classify a tool as mutating from its name, then its description.

    A heuristic, and it will occasionally be wrong in the safe direction — a read-only tool
    called ``run_query`` gets classified as mutating and requires the write permission. That
    is the correct way to be wrong.
    """
    lowered = tool_name.lower()
    if any(hint in lowered for hint in MUTATING_HINTS):
        return True
    return any(f" {hint}" in f" {description.lower()}" for hint in MUTATING_HINTS[:8])


@dataclass(slots=True)
class MCPBackedTool:
    """One MCP tool, wrapped so the platform's dispatcher can treat it like any other.

    Satisfies the ``Tool`` protocol structurally rather than by inheritance, so that the
    protocol stays a contract instead of becoming a base class with behaviour.
    """

    name: str
    description: str
    parameters: dict[str, Any]
    required_permission: str
    mutates: bool
    _client: MCPClient
    _remote_name: str
    _guardrails: GuardrailPipeline | None = None
    _trusted: bool = False

    async def run(self, arguments: dict[str, Any]) -> ToolResult:
        output = await self._client.call_tool(self._remote_name, arguments)

        if self._guardrails is not None and not self._trusted:
            decision = self._guardrails.evaluate(output, boundary=Boundary.TOOL_OUTPUT)
            if not decision.allowed:
                log.warning(
                    "mcp.tool_output_blocked",
                    tool=self.name,
                    blocked_by=decision.blocked_by,
                )
                return ToolResult(
                    output=(
                        "The tool returned content that the platform's guardrails refused to "
                        "pass on. Treat this tool call as failed and do not retry it."
                    ),
                    success=False,
                    metadata={"blocked_by": ",".join(decision.blocked_by)},
                )
            output = decision.text

        return ToolResult(output=output, success=True, metadata={"source": "mcp"})


class MCPGateway:
    """Owns every MCP connection and imports their tools into the platform registry."""

    def __init__(
        self,
        *,
        registry: ToolRegistry,
        guardrails: GuardrailPipeline | None = None,
    ) -> None:
        self._registry = registry
        self._guardrails = guardrails
        self._clients: dict[str, MCPClient] = {}
        self._imported: dict[str, list[str]] = {}

    @property
    def servers(self) -> tuple[str, ...]:
        return tuple(sorted(self._clients))

    def tools_from(self, server: str) -> tuple[str, ...]:
        return tuple(self._imported.get(server, ()))

    async def register_server(
        self,
        config: MCPServerConfig,
        *,
        allow_tools: frozenset[str] | None = None,
        deny_tools: frozenset[str] = frozenset(),
    ) -> list[str]:
        """Connect, enumerate, filter and import.

        ``allow_tools`` is the safer control of the two: an explicit allowlist means a server
        that grows new tools in a later release imports nothing new until someone decides it
        should. ``deny_tools`` is for the cases where enumerating everything wanted is
        impractical.
        """
        client = MCPClient(config)
        await client.connect()
        self._clients[config.name] = client

        discovered = await client.list_tools()
        imported: list[str] = []

        for tool in discovered:
            if allow_tools is not None and tool.name not in allow_tools:
                log.info(
                    "mcp.tool_skipped", server=config.name, tool=tool.name, reason="not_allowed"
                )
                continue
            if tool.name in deny_tools:
                log.info("mcp.tool_skipped", server=config.name, tool=tool.name, reason="denied")
                continue

            self._registry.register(self._wrap(tool, client, config))
            imported.append(tool.qualified_name)

        self._imported[config.name] = imported
        log.info(
            "mcp.server_registered",
            server=config.name,
            discovered=len(discovered),
            imported=len(imported),
            trusted=config.trusted,
        )
        return imported

    def _wrap(self, tool: MCPTool, client: MCPClient, config: MCPServerConfig) -> MCPBackedTool:
        mutates = infers_mutation(tool.name, tool.description)
        return MCPBackedTool(
            name=tool.qualified_name,
            description=tool.description or f"{tool.name} provided by MCP server {config.name}",
            parameters=tool.input_schema,
            required_permission=PERM_TOOL_EXECUTE_WRITE if mutates else PERM_TOOL_EXECUTE,
            mutates=mutates,
            _client=client,
            _remote_name=tool.name,
            _guardrails=self._guardrails,
            _trusted=config.trusted,
        )

    async def health(self) -> dict[str, bool]:
        return {name: client.connected for name, client in self._clients.items()}

    async def client_for(self, server: str) -> MCPClient:
        client = self._clients.get(server)
        if client is None:
            raise NotFoundError(f"no MCP server named '{server}'", known=sorted(self._clients))
        return client

    async def shutdown(self) -> None:
        for client in self._clients.values():
            await client.disconnect()
        self._clients.clear()
        self._imported.clear()
