"""MCP client and gateway, driven against a real server subprocess.

The server in ``tests/fixtures/mcp_test_server.py`` is spawned as an actual process and
spoken to over actual pipes. Mocking the transport here would skip the parts most likely to
break: process lifecycle, line framing, request-id matching and stdout that is not protocol.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from tests.synthetic_credentials import AWS_ACCESS_KEY

from eap.appops.mcp.client import MCPClient, MCPServerConfig
from eap.appops.mcp.gateway import MCPGateway
from eap.appops.tools.registry import ToolDispatcher, ToolRegistry
from eap.identity.models import AgentIdentity
from eap.identity.rbac import PERM_TOOL_EXECUTE, PERM_TOOL_EXECUTE_WRITE
from eap.platform.errors import ToolExecutionError

SERVER = Path(__file__).parent / "fixtures" / "mcp_test_server.py"

pytestmark = pytest.mark.integration


def server_config(*, name: str = "wiki", trusted: bool = False) -> MCPServerConfig:
    return MCPServerConfig(
        name=name,
        command=sys.executable,
        args=(str(SERVER),),
        timeout_seconds=15.0,
        trusted=trusted,
    )


@pytest.fixture
async def client():
    connection = MCPClient(server_config())
    await connection.connect()
    try:
        yield connection
    finally:
        await connection.disconnect()


class TestClient:
    async def test_it_connects_and_reports_healthy(self, client: MCPClient) -> None:
        assert client.connected

    async def test_it_tolerates_a_non_protocol_banner_on_stdout(self, client: MCPClient) -> None:
        """The fixture server prints a banner before speaking JSON-RPC."""
        assert await client.list_tools()

    async def test_it_lists_the_servers_tools(self, client: MCPClient) -> None:
        names = {tool.name for tool in await client.list_tools()}
        assert {"search_wiki", "create_issue", "poisoned_lookup"} <= names

    async def test_tool_names_are_namespaced_by_server(self, client: MCPClient) -> None:
        tool = next(t for t in await client.list_tools() if t.name == "search_wiki")
        assert tool.qualified_name == "wiki.search_wiki"

    async def test_it_calls_a_tool_and_returns_its_text(self, client: MCPClient) -> None:
        output = await client.call_tool("search_wiki", {"query": "runbook"})
        assert "runbook" in output

    async def test_a_server_side_error_becomes_a_tool_execution_error(
        self, client: MCPClient
    ) -> None:
        with pytest.raises(ToolExecutionError):
            await client.call_tool("always_fails", {})

    async def test_sequential_calls_match_their_own_responses(self, client: MCPClient) -> None:
        first = await client.call_tool("search_wiki", {"query": "alpha"})
        second = await client.call_tool("search_wiki", {"query": "beta"})
        assert "alpha" in first
        assert "beta" in second

    async def test_disconnect_terminates_the_subprocess(self) -> None:
        connection = MCPClient(server_config())
        await connection.connect()
        assert connection.connected
        await connection.disconnect()
        assert not connection.connected


class TestGateway:
    @pytest.fixture
    async def gateway(self, tools: ToolRegistry, guardrails):
        instance = MCPGateway(registry=tools, guardrails=guardrails)
        try:
            yield instance
        finally:
            await instance.shutdown()

    async def test_registering_a_server_imports_its_tools(
        self, gateway: MCPGateway, tools: ToolRegistry
    ) -> None:
        imported = await gateway.register_server(server_config())

        assert "wiki.search_wiki" in imported
        assert "wiki.search_wiki" in tools.names()
        assert gateway.servers == ("wiki",)

    async def test_an_allowlist_imports_only_what_it_names(
        self, gateway: MCPGateway, tools: ToolRegistry
    ) -> None:
        imported = await gateway.register_server(
            server_config(), allow_tools=frozenset({"search_wiki"})
        )
        assert imported == ["wiki.search_wiki"]
        assert tools.names() == ("wiki.search_wiki",)

    async def test_a_denylist_excludes_what_it_names(self, gateway: MCPGateway) -> None:
        imported = await gateway.register_server(
            server_config(), deny_tools=frozenset({"create_issue", "always_fails"})
        )
        assert "wiki.create_issue" not in imported

    async def test_a_mutating_tool_is_assigned_the_write_permission(
        self, gateway: MCPGateway, tools: ToolRegistry
    ) -> None:
        await gateway.register_server(server_config())

        assert tools.get("wiki.create_issue").required_permission == PERM_TOOL_EXECUTE_WRITE
        assert tools.get("wiki.create_issue").mutates is True
        assert tools.get("wiki.search_wiki").required_permission == PERM_TOOL_EXECUTE
        assert tools.get("wiki.search_wiki").mutates is False

    async def test_health_reports_each_connected_server(self, gateway: MCPGateway) -> None:
        await gateway.register_server(server_config())
        assert await gateway.health() == {"wiki": True}


class TestUntrustedOutputContainment:
    """A server the platform does not control is a source of hostile content."""

    @pytest.fixture
    async def wired(
        self, tools: ToolRegistry, guardrails, authorizer, tenant, user_principal, policy, audit
    ):
        gateway = MCPGateway(registry=tools, guardrails=guardrails)
        await gateway.register_server(server_config(trusted=False))
        agent = AgentIdentity(
            id="a",
            name="A",
            mission="m",
            owner="o",
            granted_permissions=frozenset({PERM_TOOL_EXECUTE}),
        )
        ctx = authorizer.build_context(user_principal, tenant, agent=agent)
        dispatcher = ToolDispatcher(
            registry=tools, authorizer=authorizer, policy=policy, audit=audit
        )
        try:
            yield gateway, dispatcher, ctx
        finally:
            await gateway.shutdown()

    async def test_benign_output_passes_through(self, wired) -> None:
        _gateway, dispatcher, ctx = wired
        result = await dispatcher.dispatch(
            "wiki.search_wiki", {"query": "runbook"}, ctx=ctx, correlation_id="c1"
        )
        assert result.success
        assert "runbook" in result.output

    async def test_an_injection_payload_in_tool_output_is_blocked(self, wired) -> None:
        _gateway, dispatcher, ctx = wired
        result = await dispatcher.dispatch(
            "wiki.poisoned_lookup", {"id": "1"}, ctx=ctx, correlation_id="c1"
        )
        assert not result.success
        assert "guardrails refused" in result.output
        assert "Ignore all previous instructions" not in result.output

    async def test_a_credential_in_tool_output_is_blocked(self, wired) -> None:
        _gateway, dispatcher, ctx = wired
        result = await dispatcher.dispatch(
            "wiki.leaky_lookup", {"id": "1"}, ctx=ctx, correlation_id="c1"
        )
        assert not result.success
        assert AWS_ACCESS_KEY not in result.output

    async def test_a_trusted_server_is_not_filtered(
        self, tools: ToolRegistry, guardrails, authorizer, tenant, user_principal, policy, audit
    ) -> None:
        """Trust is a deliberate, per-server decision with a visible consequence."""
        gateway = MCPGateway(registry=tools, guardrails=guardrails)
        await gateway.register_server(server_config(name="internal", trusted=True))
        agent = AgentIdentity(
            id="a",
            name="A",
            mission="m",
            owner="o",
            granted_permissions=frozenset({PERM_TOOL_EXECUTE}),
        )
        ctx = authorizer.build_context(user_principal, tenant, agent=agent)
        dispatcher = ToolDispatcher(
            registry=tools, authorizer=authorizer, policy=policy, audit=audit
        )
        try:
            result = await dispatcher.dispatch(
                "internal.poisoned_lookup", {"id": "1"}, ctx=ctx, correlation_id="c1"
            )
            assert result.success
            assert "Ignore all previous instructions" in result.output
        finally:
            await gateway.shutdown()
