"""Tool registry and the guarded dispatch path.

Everything a tool call must survive, in order:

1. The tool exists.
2. The agent's own design permits it (``forbidden_tools``).
3. The caller ∩ agent permission intersection covers ``required_permission``.
4. Governance policy allows it, and any obligations it attaches are discharged.
5. The arguments validate against the declared schema.
6. Only then does anything execute.

The ordering is not incidental. Permission is checked before argument validation so that a
denied caller cannot use validation error messages to probe a tool's shape, and policy is
consulted before execution so that an obligation such as human approval can still block.
"""

from __future__ import annotations

from typing import Any

from eap.appops.tools.base import Tool, ToolResult, validate_arguments
from eap.identity.models import SecurityContext
from eap.identity.rbac import Authorizer
from eap.llmops.providers.base import ToolSchema
from eap.platform.errors import (
    AuthorizationError,
    NotFoundError,
    PolicyViolation,
    ToolExecutionError,
)
from eap.platform.telemetry import get_logger
from eap.secops.audit import AuditAction, AuditLog, Outcome
from eap.secops.policy import DataClassification, PolicyEngine, PolicyRequest

log = get_logger(__name__)


class ApprovalRequired(Exception):
    """Raised when policy attaches a human-approval obligation that is not yet discharged.

    An exception rather than a return value because the orchestrator must not be able to
    continue by accident. Ignoring a returned "needs approval" flag is one missing branch;
    ignoring a raised exception takes deliberate effort.
    """

    def __init__(self, *, tool: str, reason: str, request_id: str) -> None:
        super().__init__(f"tool '{tool}' requires human approval: {reason}")
        self.tool = tool
        self.reason = reason
        self.request_id = request_id


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"a tool named '{tool.name}' is already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        tool = self._tools.get(name)
        if tool is None:
            raise NotFoundError(f"no tool named '{name}'", available=sorted(self._tools))
        return tool

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._tools))

    def schemas_for(self, ctx: SecurityContext) -> tuple[ToolSchema, ...]:
        """Only advertise tools this caller may actually use.

        Showing a model a tool it will be refused wastes a turn and teaches it to try
        things that fail. The manifest is filtered to what will succeed.
        """
        visible: list[ToolSchema] = []
        for tool in self._tools.values():
            if ctx.agent is not None and not ctx.agent.may_use_tool(tool.name):
                continue
            if not ctx.has(tool.required_permission):
                continue
            visible.append(
                ToolSchema(name=tool.name, description=tool.description, parameters=tool.parameters)
            )
        return tuple(sorted(visible, key=lambda schema: schema.name))

    def manifest_for(self, ctx: SecurityContext) -> str:
        """Render the visible tools for a planning prompt."""
        schemas = self.schemas_for(ctx)
        if not schemas:
            return "(no tools are available to this caller)"
        lines = []
        for schema in schemas:
            properties = schema.parameters.get("properties", {})
            arguments = ", ".join(sorted(properties)) or "no arguments"
            lines.append(f"- {schema.name}({arguments}): {schema.description}")
        return "\n".join(lines)


class ToolDispatcher:
    """The only sanctioned way to execute a tool."""

    def __init__(
        self,
        *,
        registry: ToolRegistry,
        authorizer: Authorizer,
        policy: PolicyEngine,
        audit: AuditLog | None = None,
    ) -> None:
        self._registry = registry
        self._authorizer = authorizer
        self._policy = policy
        self._audit = audit

    async def dispatch(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        ctx: SecurityContext,
        correlation_id: str,
        classification: DataClassification = DataClassification.INTERNAL,
        approval_token: str | None = None,
    ) -> ToolResult:
        tool = self._registry.get(name)

        if ctx.agent is not None and not ctx.agent.may_use_tool(name):
            self._deny(ctx, name, correlation_id, "tool is on the agent's forbidden list")
            raise AuthorizationError(
                f"agent '{ctx.agent.id}' is forbidden from using '{name}'",
                permission=tool.required_permission,
                reason="agent_forbidden_tool",
            )

        try:
            self._authorizer.require(ctx, tool.required_permission)
        except AuthorizationError as exc:
            self._deny(ctx, name, correlation_id, exc.message)
            raise

        decision = self._policy.evaluate(
            PolicyRequest(
                action="tool.execute",
                security=ctx,
                resource=name,
                tool=name,
                classification=classification,
                attributes={"tool_mutates": tool.mutates},
            )
        )
        if not decision.allowed:
            self._deny(ctx, name, correlation_id, decision.reason)
            raise PolicyViolation(decision.reason, rule=decision.rule_id, tool=name)

        needs_approval = "require_human_approval" in decision.obligations or (
            ctx.agent is not None and ctx.agent.needs_approval_for(name)
        )
        if needs_approval and not approval_token:
            request_id = f"approval_{correlation_id}_{name}"
            if self._audit is not None:
                self._audit.record(
                    AuditAction.APPROVAL_REQUESTED,
                    outcome=Outcome.DENIED,
                    tenant_id=ctx.tenant.id,
                    actor=str(ctx.principal),
                    correlation_id=correlation_id,
                    resource=name,
                    reason="human approval obligation not discharged",
                    request_id=request_id,
                )
            raise ApprovalRequired(tool=name, reason=decision.reason, request_id=request_id)

        validated = validate_arguments(tool.parameters, arguments)

        try:
            result = await tool.run(validated)
        except Exception as exc:
            if self._audit is not None:
                self._audit.record(
                    AuditAction.TOOL_INVOKED,
                    outcome=Outcome.ERROR,
                    tenant_id=ctx.tenant.id,
                    actor=str(ctx.principal),
                    correlation_id=correlation_id,
                    resource=name,
                    reason=type(exc).__name__,
                    mutates=str(tool.mutates),
                )
            raise ToolExecutionError(f"tool '{name}' failed: {exc}", tool=name) from exc

        if self._audit is not None:
            self._audit.record(
                AuditAction.TOOL_INVOKED,
                outcome=Outcome.ALLOWED if result.success else Outcome.ERROR,
                tenant_id=ctx.tenant.id,
                actor=str(ctx.principal),
                correlation_id=correlation_id,
                resource=name,
                mutates=str(tool.mutates),
                approved=str(bool(approval_token)),
                # Arguments are recorded by key only. Values routinely carry customer data,
                # and the audit log has a longer retention than most data agreements allow.
                argument_keys=",".join(sorted(validated)),
            )

        log.info(
            "tool.executed",
            tool=name,
            success=result.success,
            mutates=tool.mutates,
            tenant_id=ctx.tenant.id,
        )
        return result

    def _deny(self, ctx: SecurityContext, tool: str, correlation_id: str, reason: str) -> None:
        log.warning("tool.denied", tool=tool, reason=reason, tenant_id=ctx.tenant.id)
        if self._audit is not None:
            self._audit.record(
                AuditAction.TOOL_DENIED,
                outcome=Outcome.DENIED,
                tenant_id=ctx.tenant.id,
                actor=str(ctx.principal),
                correlation_id=correlation_id,
                resource=tool,
                reason=reason,
            )
