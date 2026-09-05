"""Agent memory.

RAG answers "what does the organisation know". Memory answers "what does this agent need to
remember". They are different systems with different lifetimes, and merging them produces a
knowledge base slowly poisoned by conversational noise.

Three kinds, because they expire differently:

* **Session** — the current conversation. Bounded by turn count, dropped when the session
  ends. Oldest turns are evicted first, but the opening turn is kept: it usually carries the
  task definition that everything after it refers to.
* **Episodic** — outcomes of past runs. What was attempted, whether it worked. This is what
  lets an agent stop repeating a tool call that failed the same way three times yesterday.
* **Long-term** — durable facts and preferences, written deliberately rather than
  accumulated. Requires an explicit write, because memory that fills itself is memory that
  fills with garbage.

Every read and write is scoped by tenant *and* principal. Memory is the easiest place in an
agent platform to leak one user's context into another's session, since it sits outside the
retrieval path where people concentrate their isolation review.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any, Protocol

from eap.platform.clock import SYSTEM_CLOCK, Clock
from eap.platform.errors import TenantIsolationError


class MemoryKind(StrEnum):
    SESSION = "session"
    EPISODIC = "episodic"
    LONG_TERM = "long_term"


@dataclass(frozen=True, slots=True)
class MemoryEntry:
    kind: MemoryKind
    tenant_id: str
    principal_id: str
    session_id: str | None
    content: str
    created_at: datetime
    role: str = "user"
    importance: float = 0.5
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def scope_key(self) -> str:
        return f"{self.tenant_id}:{self.principal_id}"


@dataclass(frozen=True, slots=True)
class RunOutcome:
    """One episodic record: what the agent tried and how it went."""

    run_id: str
    question: str
    succeeded: bool
    summary: str
    tools_used: tuple[str, ...] = ()
    error: str | None = None


class MemoryStore(Protocol):
    def remember(self, entry: MemoryEntry) -> None: ...

    def recall(
        self, *, tenant_id: str, principal_id: str, kind: MemoryKind, limit: int = 20
    ) -> list[MemoryEntry]: ...


class InMemoryStore:
    """Process-local memory, partitioned by tenant and principal."""

    def __init__(self, *, session_turns: int = 40, episodic_limit: int = 200) -> None:
        self._sessions: dict[str, deque[MemoryEntry]] = defaultdict(
            lambda: deque(maxlen=session_turns)
        )
        self._episodic: dict[str, deque[MemoryEntry]] = defaultdict(
            lambda: deque(maxlen=episodic_limit)
        )
        self._long_term: dict[str, list[MemoryEntry]] = defaultdict(list)

    def remember(self, entry: MemoryEntry) -> None:
        if entry.kind is MemoryKind.SESSION:
            key = f"{entry.scope_key}:{entry.session_id or 'default'}"
            self._sessions[key].append(entry)
        elif entry.kind is MemoryKind.EPISODIC:
            self._episodic[entry.scope_key].append(entry)
        else:
            self._long_term[entry.scope_key].append(entry)

    def recall(
        self,
        *,
        tenant_id: str,
        principal_id: str,
        kind: MemoryKind,
        limit: int = 20,
        session_id: str | None = None,
    ) -> list[MemoryEntry]:
        scope = f"{tenant_id}:{principal_id}"
        if kind is MemoryKind.SESSION:
            entries = list(self._sessions.get(f"{scope}:{session_id or 'default'}", ()))
        elif kind is MemoryKind.EPISODIC:
            entries = list(self._episodic.get(scope, ()))
        else:
            entries = sorted(
                self._long_term.get(scope, ()), key=lambda e: e.importance, reverse=True
            )
        return entries[-limit:] if kind is not MemoryKind.LONG_TERM else entries[:limit]

    def forget_session(self, *, tenant_id: str, principal_id: str, session_id: str) -> int:
        key = f"{tenant_id}:{principal_id}:{session_id}"
        removed = len(self._sessions.get(key, ()))
        self._sessions.pop(key, None)
        return removed


class MemoryManager:
    """The API the orchestrator uses. Enforces scoping on every call."""

    def __init__(
        self,
        store: InMemoryStore | None = None,
        *,
        clock: Clock = SYSTEM_CLOCK,
        session_window: int = 12,
    ) -> None:
        self._store = store or InMemoryStore()
        self._clock = clock
        self._window = session_window

    def record_turn(
        self,
        *,
        tenant_id: str,
        principal_id: str,
        session_id: str,
        role: str,
        content: str,
    ) -> None:
        self._store.remember(
            MemoryEntry(
                kind=MemoryKind.SESSION,
                tenant_id=tenant_id,
                principal_id=principal_id,
                session_id=session_id,
                content=content,
                role=role,
                created_at=self._clock.now(),
            )
        )

    def record_outcome(self, *, tenant_id: str, principal_id: str, outcome: RunOutcome) -> None:
        self._store.remember(
            MemoryEntry(
                kind=MemoryKind.EPISODIC,
                tenant_id=tenant_id,
                principal_id=principal_id,
                session_id=None,
                content=outcome.summary,
                role="system",
                created_at=self._clock.now(),
                importance=0.8 if not outcome.succeeded else 0.4,
                metadata={
                    "run_id": outcome.run_id,
                    "question": outcome.question,
                    "succeeded": outcome.succeeded,
                    "tools_used": list(outcome.tools_used),
                    "error": outcome.error,
                },
            )
        )

    def remember_fact(
        self,
        *,
        tenant_id: str,
        principal_id: str,
        content: str,
        importance: float = 0.7,
        **metadata: Any,
    ) -> None:
        self._store.remember(
            MemoryEntry(
                kind=MemoryKind.LONG_TERM,
                tenant_id=tenant_id,
                principal_id=principal_id,
                session_id=None,
                content=content,
                role="system",
                created_at=self._clock.now(),
                importance=max(0.0, min(1.0, importance)),
                metadata=metadata,
            )
        )

    def session_transcript(
        self, *, tenant_id: str, principal_id: str, session_id: str
    ) -> list[MemoryEntry]:
        """The recent window, with the opening turn kept.

        Dropping turn one is a common and quietly damaging default: it is usually where the
        task was stated, and every later turn is a pronoun referring back to it.
        """
        entries = self._store.recall(
            tenant_id=tenant_id,
            principal_id=principal_id,
            kind=MemoryKind.SESSION,
            limit=1000,
            session_id=session_id,
        )
        if len(entries) <= self._window:
            return entries
        return [entries[0], *entries[-(self._window - 1) :]]

    def prior_failures(
        self, *, tenant_id: str, principal_id: str, limit: int = 5
    ) -> list[MemoryEntry]:
        entries = self._store.recall(
            tenant_id=tenant_id,
            principal_id=principal_id,
            kind=MemoryKind.EPISODIC,
            limit=50,
        )
        return [entry for entry in entries if not entry.metadata.get("succeeded", True)][-limit:]

    def known_facts(
        self, *, tenant_id: str, principal_id: str, limit: int = 10
    ) -> list[MemoryEntry]:
        return self._store.recall(
            tenant_id=tenant_id,
            principal_id=principal_id,
            kind=MemoryKind.LONG_TERM,
            limit=limit,
        )

    def assert_scope(self, entry: MemoryEntry, *, tenant_id: str) -> None:
        if entry.tenant_id != tenant_id:
            raise TenantIsolationError(
                "memory entry belongs to a different tenant",
                entry_tenant=entry.tenant_id,
                caller_tenant=tenant_id,
            )
