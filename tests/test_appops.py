"""Runtime plane: argument validation, guarded dispatch, MCP import rules, memory."""

from __future__ import annotations

from typing import Any

import pytest

from eap.appops.mcp.gateway import infers_mutation
from eap.appops.memory import MemoryManager, RunOutcome
from eap.appops.tools.base import FunctionTool, ToolResult, validate_arguments
from eap.appops.tools.registry import (
    ApprovalRequired,
    ToolDispatcher,
    ToolRegistry,
)
from eap.identity.models import AgentIdentity
from eap.identity.rbac import PERM_KNOWLEDGE_READ, PERM_TOOL_EXECUTE, PERM_TOOL_EXECUTE_WRITE
from eap.platform.errors import (
    AuthorizationError,
    NotFoundError,
    ToolExecutionError,
    ValidationError,
)

SEARCH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "minLength": 1, "maxLength": 200},
        "limit": {"type": "integer", "minimum": 1, "maximum": 50},
        "scope": {"type": "string", "enum": ["docs", "code"]},
    },
    "required": ["query"],
}


async def _search(arguments: dict[str, Any]) -> ToolResult:
    return ToolResult(output=f"results for {arguments['query']}")


async def _create_ticket(arguments: dict[str, Any]) -> ToolResult:
    return ToolResult(output=f"created ticket for {arguments.get('title', 'untitled')}")


async def _explode(_arguments: dict[str, Any]) -> ToolResult:
    raise RuntimeError("the downstream system rejected the call")


def read_tool() -> FunctionTool:
    return FunctionTool(
        name="search_docs",
        description="Search the documentation corpus.",
        parameters=SEARCH_SCHEMA,
        handler=_search,
        required_permission=PERM_TOOL_EXECUTE,
        mutates=False,
    )


def write_tool() -> FunctionTool:
    return FunctionTool(
        name="create_ticket",
        description="Create a ticket in the issue tracker.",
        parameters={
            "type": "object",
            "properties": {"title": {"type": "string"}},
            "required": ["title"],
        },
        handler=_create_ticket,
        required_permission=PERM_TOOL_EXECUTE_WRITE,
        mutates=True,
    )


class TestArgumentValidation:
    def test_valid_arguments_pass_through(self) -> None:
        assert validate_arguments(SEARCH_SCHEMA, {"query": "deploy", "limit": 5}) == {
            "query": "deploy",
            "limit": 5,
        }

    def test_missing_required_field_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="missing required"):
            validate_arguments(SEARCH_SCHEMA, {"limit": 5})

    def test_an_invented_field_is_refused_rather_than_dropped(self) -> None:
        with pytest.raises(ValidationError, match="does not accept"):
            validate_arguments(SEARCH_SCHEMA, {"query": "x", "sudo": True})

    def test_wrong_type_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="must be integer"):
            validate_arguments(SEARCH_SCHEMA, {"query": "x", "limit": "five"})

    def test_a_boolean_is_not_accepted_as_an_integer(self) -> None:
        with pytest.raises(ValidationError):
            validate_arguments(SEARCH_SCHEMA, {"query": "x", "limit": True})

    @pytest.mark.parametrize("limit", [0, 51])
    def test_numeric_bounds_are_enforced(self, limit: int) -> None:
        with pytest.raises(ValidationError):
            validate_arguments(SEARCH_SCHEMA, {"query": "x", "limit": limit})

    def test_enum_values_are_enforced(self) -> None:
        with pytest.raises(ValidationError, match="permitted values"):
            validate_arguments(SEARCH_SCHEMA, {"query": "x", "scope": "everything"})

    def test_string_length_bounds_are_enforced(self) -> None:
        with pytest.raises(ValidationError, match="maximum length"):
            validate_arguments(SEARCH_SCHEMA, {"query": "x" * 500})


class TestRegistry:
    def test_duplicate_registration_is_refused(self, tools: ToolRegistry) -> None:
        tools.register(read_tool())
        with pytest.raises(ValueError):
            tools.register(read_tool())

    def test_an_unknown_tool_raises_not_found(self, tools: ToolRegistry) -> None:
        with pytest.raises(NotFoundError):
            tools.get("no_such_tool")

    def test_the_manifest_hides_tools_the_caller_cannot_use(
        self, tools: ToolRegistry, security
    ) -> None:
        tools.register(read_tool())
        tools.register(write_tool())

        names = {schema.name for schema in tools.schemas_for(security)}
        # The agent fixture is granted tool:execute but not tool:execute:write.
        assert names == {"search_docs"}
        assert "create_ticket" not in tools.manifest_for(security)

    def test_the_manifest_hides_forbidden_tools(
        self, tools: ToolRegistry, authorizer, tenant, user_principal
    ) -> None:
        forbidding_agent = AgentIdentity(
            id="a",
            name="A",
            mission="m",
            owner="o",
            granted_permissions=frozenset({PERM_TOOL_EXECUTE, PERM_KNOWLEDGE_READ}),
            forbidden_tools=frozenset({"search_docs"}),
        )
        ctx = authorizer.build_context(user_principal, tenant, agent=forbidding_agent)
        tools.register(read_tool())
        assert tools.schemas_for(ctx) == ()

    def test_an_empty_manifest_says_so(self, tools: ToolRegistry, security) -> None:
        assert "no tools" in tools.manifest_for(security)


class TestDispatch:
    async def test_a_permitted_read_tool_runs(
        self, tools: ToolRegistry, dispatcher: ToolDispatcher, security
    ) -> None:
        tools.register(read_tool())
        result = await dispatcher.dispatch(
            "search_docs", {"query": "rollback"}, ctx=security, correlation_id="c1"
        )
        assert result.success
        assert "rollback" in result.output

    async def test_a_tool_outside_the_agent_grant_is_refused(
        self, tools: ToolRegistry, dispatcher: ToolDispatcher, security
    ) -> None:
        tools.register(write_tool())
        with pytest.raises(AuthorizationError) as exc:
            await dispatcher.dispatch(
                "create_ticket", {"title": "x"}, ctx=security, correlation_id="c1"
            )
        assert exc.value.details["reason"] == "agent_grant_narrower_than_caller"

    async def test_a_forbidden_tool_is_refused_even_with_the_permission(
        self, tools: ToolRegistry, dispatcher: ToolDispatcher, authorizer, tenant, user_principal
    ) -> None:
        agent = AgentIdentity(
            id="a",
            name="A",
            mission="m",
            owner="o",
            granted_permissions=frozenset({"*"}),
            forbidden_tools=frozenset({"search_docs"}),
        )
        ctx = authorizer.build_context(user_principal, tenant, agent=agent)
        tools.register(read_tool())

        with pytest.raises(AuthorizationError) as exc:
            await dispatcher.dispatch("search_docs", {"query": "x"}, ctx=ctx, correlation_id="c1")
        assert exc.value.details["reason"] == "agent_forbidden_tool"

    async def test_a_mutating_tool_requires_human_approval(
        self, tools: ToolRegistry, dispatcher: ToolDispatcher, authorizer, tenant, user_principal
    ) -> None:
        agent = AgentIdentity(
            id="a",
            name="A",
            mission="m",
            owner="o",
            granted_permissions=frozenset({PERM_TOOL_EXECUTE_WRITE}),
        )
        ctx = authorizer.build_context(user_principal, tenant, agent=agent)
        tools.register(write_tool())

        with pytest.raises(ApprovalRequired) as exc:
            await dispatcher.dispatch(
                "create_ticket", {"title": "outage"}, ctx=ctx, correlation_id="c1"
            )
        assert exc.value.tool == "create_ticket"
        assert exc.value.request_id

    async def test_an_approval_token_discharges_the_obligation(
        self, tools: ToolRegistry, dispatcher: ToolDispatcher, authorizer, tenant, user_principal
    ) -> None:
        agent = AgentIdentity(
            id="a",
            name="A",
            mission="m",
            owner="o",
            granted_permissions=frozenset({PERM_TOOL_EXECUTE_WRITE}),
        )
        ctx = authorizer.build_context(user_principal, tenant, agent=agent)
        tools.register(write_tool())

        result = await dispatcher.dispatch(
            "create_ticket",
            {"title": "outage"},
            ctx=ctx,
            correlation_id="c1",
            approval_token="approved-by-a-human",
        )
        assert result.success

    async def test_arguments_are_validated_before_the_tool_runs(
        self, tools: ToolRegistry, dispatcher: ToolDispatcher, security
    ) -> None:
        tools.register(read_tool())
        with pytest.raises(ValidationError):
            await dispatcher.dispatch(
                "search_docs", {"query": "x", "injected": "y"}, ctx=security, correlation_id="c1"
            )

    async def test_a_failing_tool_is_normalised_into_the_error_taxonomy(
        self, tools: ToolRegistry, dispatcher: ToolDispatcher, security
    ) -> None:
        tools.register(
            FunctionTool(
                name="broken",
                description="Always fails.",
                parameters={"type": "object", "properties": {}},
                handler=_explode,
            )
        )
        with pytest.raises(ToolExecutionError):
            await dispatcher.dispatch("broken", {}, ctx=security, correlation_id="c1")

    async def test_dispatch_writes_an_audit_record_without_argument_values(
        self, tools: ToolRegistry, dispatcher: ToolDispatcher, security, audit_sink
    ) -> None:
        tools.register(read_tool())
        await dispatcher.dispatch(
            "search_docs",
            {"query": "a-customer-name-that-must-not-be-logged"},
            ctx=security,
            correlation_id="c1",
        )
        records = [r for r in audit_sink.read_all() if str(r.action) == "tool.invoked"]

        assert len(records) == 1
        assert records[0].metadata["argument_keys"] == "query"
        assert "a-customer-name" not in str(records[0].metadata)

    async def test_a_denial_is_audited(
        self, tools: ToolRegistry, dispatcher: ToolDispatcher, security, audit_sink
    ) -> None:
        tools.register(write_tool())
        with pytest.raises(AuthorizationError):
            await dispatcher.dispatch(
                "create_ticket", {"title": "x"}, ctx=security, correlation_id="c1"
            )
        assert any(str(r.action) == "tool.denied" for r in audit_sink.read_all())


class TestMCPMutationInference:
    @pytest.mark.parametrize(
        "name",
        [
            "create_issue",
            "delete_branch",
            "update_record",
            "send_email",
            "run_query",
            "deploy_service",
            "revoke_token",
            "terminate_instance",
        ],
    )
    def test_mutating_verbs_are_detected(self, name: str) -> None:
        assert infers_mutation(name)

    @pytest.mark.parametrize("name", ["search", "list_issues", "get_file", "read_config"])
    def test_read_only_names_are_not_flagged(self, name: str) -> None:
        assert not infers_mutation(name)

    def test_a_description_can_reveal_mutation_the_name_hides(self) -> None:
        assert infers_mutation("issue_manager", "Create and update issues in the tracker.")

    def test_an_unknown_verb_defaults_to_read_only_but_needs_the_execute_permission(self) -> None:
        # Classification only decides which permission is required; both still require one.
        assert not infers_mutation("frobnicate")


class TestMemory:
    def test_session_transcript_keeps_the_opening_turn(self, memory: MemoryManager) -> None:
        for index in range(30):
            memory.record_turn(
                tenant_id="acme",
                principal_id="alice",
                session_id="s1",
                role="user",
                content=f"turn {index}",
            )
        transcript = memory.session_transcript(
            tenant_id="acme", principal_id="alice", session_id="s1"
        )
        assert transcript[0].content == "turn 0"
        assert transcript[-1].content == "turn 29"
        assert len(transcript) <= 12

    def test_sessions_are_isolated_from_each_other(self, memory: MemoryManager) -> None:
        memory.record_turn(
            tenant_id="acme", principal_id="alice", session_id="s1", role="user", content="a"
        )
        memory.record_turn(
            tenant_id="acme", principal_id="alice", session_id="s2", role="user", content="b"
        )
        s1 = memory.session_transcript(tenant_id="acme", principal_id="alice", session_id="s1")
        assert [entry.content for entry in s1] == ["a"]

    def test_memory_does_not_leak_across_principals(self, memory: MemoryManager) -> None:
        memory.record_turn(
            tenant_id="acme", principal_id="alice", session_id="s", role="user", content="secret"
        )
        bob = memory.session_transcript(tenant_id="acme", principal_id="bob", session_id="s")
        assert bob == []

    def test_memory_does_not_leak_across_tenants(self, memory: MemoryManager) -> None:
        memory.record_turn(
            tenant_id="acme", principal_id="alice", session_id="s", role="user", content="secret"
        )
        other = memory.session_transcript(tenant_id="globex", principal_id="alice", session_id="s")
        assert other == []

    def test_prior_failures_are_recalled(self, memory: MemoryManager) -> None:
        memory.record_outcome(
            tenant_id="acme",
            principal_id="alice",
            outcome=RunOutcome(
                run_id="r1",
                question="q",
                succeeded=False,
                summary="tool timed out",
                error="TimeoutExceeded",
            ),
        )
        memory.record_outcome(
            tenant_id="acme",
            principal_id="alice",
            outcome=RunOutcome(run_id="r2", question="q", succeeded=True, summary="fine"),
        )
        failures = memory.prior_failures(tenant_id="acme", principal_id="alice")
        assert len(failures) == 1
        assert failures[0].metadata["run_id"] == "r1"

    def test_long_term_facts_are_ordered_by_importance(self, memory: MemoryManager) -> None:
        memory.remember_fact(
            tenant_id="acme", principal_id="alice", content="minor", importance=0.2
        )
        memory.remember_fact(
            tenant_id="acme", principal_id="alice", content="critical", importance=0.95
        )
        facts = memory.known_facts(tenant_id="acme", principal_id="alice")
        assert facts[0].content == "critical"

    def test_importance_is_clamped(self, memory: MemoryManager) -> None:
        memory.remember_fact(tenant_id="acme", principal_id="alice", content="x", importance=5.0)
        assert memory.known_facts(tenant_id="acme", principal_id="alice")[0].importance == 1.0
