"""Who is acting.

An agent turn has two identities, and conflating them is the root of most agent security
incidents. The **principal** is the human or service that asked for the work. The **agent
identity** is what the runtime is allowed to do while carrying out that request. The
effective permission set is the *intersection* of the two, so an agent can never be used
to widen its caller's reach, and a highly-privileged caller cannot accidentally hand an
agent more authority than the agent was designed to hold.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum


class PrincipalType(StrEnum):
    USER = "user"
    SERVICE = "service"
    AGENT = "agent"


@dataclass(frozen=True, slots=True)
class Tenant:
    """An isolation boundary. Data, budgets, policies and audit are scoped to one."""

    id: str
    name: str
    data_residency: str = "us"
    active: bool = True


@dataclass(frozen=True, slots=True)
class Principal:
    """An authenticated caller, derived from a verified token. Never built from user input."""

    subject: str
    tenant_id: str
    principal_type: PrincipalType
    roles: frozenset[str] = frozenset()
    scopes: frozenset[str] = frozenset()
    email: str | None = None
    display_name: str | None = None
    issued_at: datetime | None = None
    expires_at: datetime | None = None

    def __str__(self) -> str:
        return f"{self.principal_type}:{self.subject}@{self.tenant_id}"


@dataclass(frozen=True, slots=True)
class AgentIdentity:
    """What an agent is, and — more importantly — what it is not.

    ``non_goals`` and ``forbidden_tools`` are not documentation. They are enforced: the
    orchestrator refuses to dispatch a tool that appears in ``forbidden_tools`` even when
    the calling principal holds the permission for it, because the constraint belongs to
    the agent's design rather than to the caller's authority.
    """

    id: str
    name: str
    mission: str
    owner: str
    non_goals: tuple[str, ...] = ()
    granted_permissions: frozenset[str] = frozenset()
    forbidden_tools: frozenset[str] = frozenset()
    max_tool_calls_per_turn: int = 8
    requires_human_approval_for: frozenset[str] = frozenset()

    def may_use_tool(self, tool_name: str) -> bool:
        return tool_name not in self.forbidden_tools

    def needs_approval_for(self, tool_name: str) -> bool:
        return tool_name in self.requires_human_approval_for


@dataclass(frozen=True, slots=True)
class Role:
    """A named bundle of permissions, optionally inheriting from other roles."""

    name: str
    permissions: frozenset[str] = frozenset()
    inherits: tuple[str, ...] = ()
    description: str = ""


@dataclass(frozen=True, slots=True)
class SecurityContext:
    """The resolved authority for one unit of work.

    Built once per turn by the gateway, then passed down. Components ask this object
    whether an action is allowed; they never re-derive permissions from roles themselves.
    """

    principal: Principal
    tenant: Tenant
    permissions: frozenset[str]
    agent: AgentIdentity | None = None
    attributes: dict[str, str] = field(default_factory=dict)

    @property
    def effective_permissions(self) -> frozenset[str]:
        """Caller authority narrowed by the agent's own grant. Intersection, never union.

        Plain set intersection is wrong here, because the two sides may express the same
        authority at different granularity: a caller holding ``knowledge:read`` and an agent
        granted ``knowledge:*`` intersect to nothing literally, while the correct answer is
        ``knowledge:read``. So each side is kept only where the *other* side covers it under
        wildcard matching, which is the intersection of what the two grants actually mean.
        """
        if self.agent is None:
            return self.permissions
        granted = self.agent.granted_permissions
        from_caller = {p for p in self.permissions if _matches(p, granted)}
        from_agent = {p for p in granted if _matches(p, self.permissions)}
        return frozenset(from_caller | from_agent)

    def has(self, permission: str) -> bool:
        return _matches(permission, self.effective_permissions)

    def has_any(self, permissions: Iterable[str]) -> bool:
        return any(self.has(p) for p in permissions)


def _matches(required: str, granted: frozenset[str]) -> bool:
    """Permission matching with a single trailing wildcard segment.

    ``knowledge:*`` grants ``knowledge:read``. ``*`` grants everything. Wildcards only
    expand rightwards, so ``*:admin`` is not a thing: a permission language that can match
    in the middle is one nobody can audit by reading it.
    """
    if "*" in granted or required in granted:
        return True
    parts = required.split(":")
    for depth in range(len(parts) - 1, 0, -1):
        if ":".join(parts[:depth]) + ":*" in granted:
            return True
    return False
