"""Role resolution and authorization.

Roles are defined once, resolved into a flat permission set at authentication time, and
never consulted again during the turn. Resolving on every check would make permission
evaluation depend on mutable global state halfway through a request.
"""

from __future__ import annotations

from eap.identity.models import (
    AgentIdentity,
    Principal,
    Role,
    SecurityContext,
    Tenant,
)
from eap.platform.errors import AuthorizationError, TenantIsolationError

# Permission vocabulary. Every guarded action in the platform names one of these.
PERM_AGENT_INVOKE = "agent:invoke"
PERM_AGENT_ADMIN = "agent:admin"
PERM_KNOWLEDGE_READ = "knowledge:read"
PERM_KNOWLEDGE_WRITE = "knowledge:write"
PERM_TOOL_EXECUTE = "tool:execute"
PERM_TOOL_EXECUTE_WRITE = "tool:execute:write"
PERM_MODEL_INVOKE = "model:invoke"
PERM_MODEL_ROUTE_OVERRIDE = "model:route:override"
PERM_AUDIT_READ = "audit:read"
PERM_POLICY_ADMIN = "policy:admin"
PERM_TENANT_ADMIN = "tenant:admin"

BUILTIN_ROLES: dict[str, Role] = {
    "agent.reader": Role(
        name="agent.reader",
        permissions=frozenset({PERM_KNOWLEDGE_READ}),
        description="Query the knowledge base. Cannot invoke an agent.",
    ),
    "agent.user": Role(
        name="agent.user",
        permissions=frozenset({PERM_AGENT_INVOKE, PERM_MODEL_INVOKE, PERM_TOOL_EXECUTE}),
        inherits=("agent.reader",),
        description="Run agents with read-only tools. The default role for a human caller.",
    ),
    "agent.operator": Role(
        name="agent.operator",
        permissions=frozenset({PERM_TOOL_EXECUTE_WRITE, PERM_KNOWLEDGE_WRITE}),
        inherits=("agent.user",),
        description="Run agents with tools that mutate external systems.",
    ),
    "knowledge.curator": Role(
        name="knowledge.curator",
        permissions=frozenset({PERM_KNOWLEDGE_READ, PERM_KNOWLEDGE_WRITE}),
        description="Ingest and retire corpora. Holds no model or tool authority.",
    ),
    "security.auditor": Role(
        name="security.auditor",
        permissions=frozenset({PERM_AUDIT_READ, PERM_KNOWLEDGE_READ}),
        description="Read the audit chain. Deliberately cannot invoke anything.",
    ),
    "platform.admin": Role(
        name="platform.admin",
        permissions=frozenset(
            {
                PERM_AGENT_ADMIN,
                PERM_POLICY_ADMIN,
                PERM_TENANT_ADMIN,
                PERM_AUDIT_READ,
                PERM_MODEL_ROUTE_OVERRIDE,
            }
        ),
        inherits=("agent.operator",),
        description="Full control of one tenant.",
    ),
}


class RoleRegistry:
    """Resolves role names into a flat permission set, following inheritance."""

    def __init__(self, roles: dict[str, Role] | None = None) -> None:
        self._roles = dict(roles or BUILTIN_ROLES)

    def register(self, role: Role) -> None:
        self._roles[role.name] = role

    def get(self, name: str) -> Role | None:
        return self._roles.get(name)

    def resolve(self, role_names: frozenset[str] | set[str]) -> frozenset[str]:
        """Flatten roles to permissions.

        Cycles in ``inherits`` are tolerated rather than raised: a misconfigured role graph
        should degrade to the permissions it can reach, not take the gateway down.
        Unknown role names are ignored for the same reason — a role that was deleted from
        the registry but still appears in a live token grants nothing.
        """
        permissions: set[str] = set()
        seen: set[str] = set()
        pending = list(role_names)
        while pending:
            name = pending.pop()
            if name in seen:
                continue
            seen.add(name)
            role = self._roles.get(name)
            if role is None:
                continue
            permissions |= role.permissions
            pending.extend(role.inherits)
        return frozenset(permissions)


class Authorizer:
    """The single place a permission decision is made."""

    def __init__(self, registry: RoleRegistry | None = None) -> None:
        self._registry = registry or RoleRegistry()

    def build_context(
        self,
        principal: Principal,
        tenant: Tenant,
        *,
        agent: AgentIdentity | None = None,
    ) -> SecurityContext:
        if principal.tenant_id != tenant.id:
            raise TenantIsolationError(
                "principal does not belong to the requested tenant",
                principal_tenant=principal.tenant_id,
                requested_tenant=tenant.id,
            )
        if not tenant.active:
            raise AuthorizationError("tenant is suspended", tenant_id=tenant.id)

        permissions = self._registry.resolve(principal.roles)
        if principal.scopes:
            # A token may carry scopes narrower than the principal's roles — for instance a
            # delegated token minted for one job. Narrow, never widen.
            permissions = frozenset(p for p in permissions if p in principal.scopes)
        return SecurityContext(
            principal=principal, tenant=tenant, permissions=permissions, agent=agent
        )

    def require(self, ctx: SecurityContext, permission: str) -> None:
        """Assert a permission or raise. The exception carries what was missing and why."""
        if ctx.has(permission):
            return
        if ctx.agent is not None and _held_by_caller_but_not_agent(ctx, permission):
            raise AuthorizationError(
                f"agent '{ctx.agent.id}' is not granted '{permission}' even though the "
                "caller holds it",
                permission=permission,
                agent_id=ctx.agent.id,
                reason="agent_grant_narrower_than_caller",
            )
        raise AuthorizationError(
            f"principal is not granted '{permission}'",
            permission=permission,
            principal=str(ctx.principal),
        )

    def require_tenant(self, ctx: SecurityContext, resource_tenant_id: str) -> None:
        """Assert that a resource being touched belongs to the caller's tenant."""
        if resource_tenant_id != ctx.tenant.id:
            raise TenantIsolationError(
                "cross-tenant access attempt",
                caller_tenant=ctx.tenant.id,
                resource_tenant=resource_tenant_id,
            )


def _held_by_caller_but_not_agent(ctx: SecurityContext, permission: str) -> bool:
    from eap.identity.models import _matches

    return _matches(permission, ctx.permissions)
