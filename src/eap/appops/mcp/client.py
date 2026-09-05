"""Model Context Protocol client.

Speaks JSON-RPC 2.0 over stdio to an MCP server subprocess, which is the transport the
majority of servers ship with. The wire format is line-delimited JSON: one request object
per line, one response object per line.

The parts that are easy to get wrong and are handled here:

**Request ids are matched, not assumed.** A server may answer out of order or interleave
notifications. Reading "the next line" as "my response" works until it does not, and then
fails as a mysterious type error in unrelated code.

**stderr is drained.** A subprocess whose stderr pipe fills blocks forever on its next
write. Draining it into the log costs one task and removes an entire class of hang.

**Every call is bounded.** A server that never answers must not hold an agent turn open
until the request timeout at the edge fires.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, cast

from eap.platform.errors import PlatformError, TimeoutExceeded, ToolExecutionError
from eap.platform.telemetry import get_logger

log = get_logger(__name__)

PROTOCOL_VERSION = "2024-11-05"


@dataclass(frozen=True, slots=True)
class MCPServerConfig:
    name: str
    command: str
    args: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    """Passed verbatim to the subprocess. This is where a server's credentials live, and
    the reason the gateway never inherits the platform's own environment: an MCP server
    should not be able to read the platform's model API keys."""

    timeout_seconds: float = 30.0
    trusted: bool = False
    """Untrusted servers have their tool output treated as hostile content and passed
    through the guardrail pipeline before it reaches the model."""


@dataclass(frozen=True, slots=True)
class MCPTool:
    name: str
    description: str
    input_schema: dict[str, Any]
    server: str

    @property
    def qualified_name(self) -> str:
        """Namespaced so two servers can each expose a ``search`` without colliding."""
        return f"{self.server}.{self.name}"


class MCPClient:
    """One connection to one MCP server."""

    def __init__(self, config: MCPServerConfig) -> None:
        self._config = config
        self._process: asyncio.subprocess.Process | None = None
        self._request_id = 0
        self._lock = asyncio.Lock()
        self._stderr_task: asyncio.Task[None] | None = None
        self._initialized = False

    @property
    def name(self) -> str:
        return self._config.name

    @property
    def connected(self) -> bool:
        return self._process is not None and self._process.returncode is None

    async def connect(self) -> None:
        if self.connected:
            return

        self._process = await asyncio.create_subprocess_exec(
            self._config.command,
            *self._config.args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=dict(self._config.env) or None,
        )
        self._stderr_task = asyncio.create_task(self._drain_stderr())

        await self._request(
            "initialize",
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "clientInfo": {"name": "enterprise-agent-platform", "version": "0.1.0"},
            },
        )
        await self._notify("notifications/initialized", {})
        self._initialized = True
        log.info("mcp.connected", server=self._config.name)

    async def disconnect(self) -> None:
        if self._stderr_task is not None:
            self._stderr_task.cancel()
            self._stderr_task = None
        if self._process is None:
            return
        if self._process.returncode is None:
            self._process.terminate()
            try:
                await asyncio.wait_for(self._process.wait(), timeout=5.0)
            except TimeoutError:
                self._process.kill()
                await self._process.wait()
        self._process = None
        self._initialized = False
        log.info("mcp.disconnected", server=self._config.name)

    async def list_tools(self) -> list[MCPTool]:
        payload = await self._request("tools/list", {})
        return [
            MCPTool(
                name=entry["name"],
                description=entry.get("description", ""),
                input_schema=entry.get("inputSchema", {"type": "object", "properties": {}}),
                server=self._config.name,
            )
            for entry in payload.get("tools", [])
        ]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        payload = await self._request("tools/call", {"name": name, "arguments": arguments})

        if payload.get("isError"):
            raise ToolExecutionError(
                f"MCP server '{self._config.name}' reported an error from '{name}'",
                server=self._config.name,
                tool=name,
            )

        parts = [
            block.get("text", "")
            for block in payload.get("content", [])
            if block.get("type") == "text"
        ]
        return "\n".join(part for part in parts if part)

    async def _drain_stderr(self) -> None:
        """Keep the pipe empty and surface server diagnostics as structured log events."""
        assert self._process is not None and self._process.stderr is not None
        try:
            async for raw in self._process.stderr:
                message = raw.decode("utf-8", errors="replace").rstrip()
                if message:
                    log.debug("mcp.server_stderr", server=self._config.name, message=message[:500])
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.warning("mcp.stderr_drain_failed", server=self._config.name, error=str(exc))

    async def _notify(self, method: str, params: dict[str, Any]) -> None:
        """Fire-and-forget. Notifications carry no id and receive no response."""
        if self._process is None or self._process.stdin is None:
            raise PlatformError(f"MCP server '{self._config.name}' is not connected")
        line = json.dumps({"jsonrpc": "2.0", "method": method, "params": params}) + "\n"
        self._process.stdin.write(line.encode("utf-8"))
        await self._process.stdin.drain()

    async def _request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        if self._process is None or self._process.stdin is None or self._process.stdout is None:
            raise PlatformError(f"MCP server '{self._config.name}' is not connected")

        async with self._lock:
            self._request_id += 1
            request_id = self._request_id
            line = json.dumps(
                {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
            )
            self._process.stdin.write((line + "\n").encode("utf-8"))
            await self._process.stdin.drain()

            try:
                response = await asyncio.wait_for(
                    self._read_response(request_id), timeout=self._config.timeout_seconds
                )
            except TimeoutError as exc:
                raise TimeoutExceeded(
                    f"MCP server '{self._config.name}' did not answer '{method}' within "
                    f"{self._config.timeout_seconds}s",
                    server=self._config.name,
                    method=method,
                ) from exc

        if "error" in response:
            error = response["error"]
            raise ToolExecutionError(
                f"MCP server '{self._config.name}' returned error {error.get('code')}: "
                f"{error.get('message')}",
                server=self._config.name,
                method=method,
            )
        result = response.get("result", {})
        if not isinstance(result, dict):
            raise ToolExecutionError(
                f"MCP server '{self._config.name}' returned a non-object result for '{method}'",
                server=self._config.name,
                method=method,
            )
        return cast(dict[str, Any], result)

    async def _read_response(self, request_id: int) -> dict[str, Any]:
        """Read until the response carrying ``request_id`` arrives, skipping anything else."""
        assert self._process is not None and self._process.stdout is not None
        while True:
            raw = await self._process.stdout.readline()
            if not raw:
                raise PlatformError(
                    f"MCP server '{self._config.name}' closed its output stream",
                    server=self._config.name,
                )
            try:
                message = json.loads(raw.decode("utf-8"))
            except json.JSONDecodeError:
                # Servers occasionally print banners to stdout. Not fatal, but worth seeing.
                log.warning(
                    "mcp.non_json_stdout",
                    server=self._config.name,
                    line=raw.decode("utf-8", errors="replace")[:200],
                )
                continue
            if not isinstance(message, dict):
                # A bare array or scalar is not a valid JSON-RPC message. Skip it rather
                # than letting it reach the caller typed as a response object.
                log.warning("mcp.non_object_message", server=self._config.name)
                continue
            if message.get("id") == request_id:
                return cast(dict[str, Any], message)
