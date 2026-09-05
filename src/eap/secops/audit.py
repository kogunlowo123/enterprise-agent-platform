"""Tamper-evident audit log.

An audit trail that can be edited after the fact is a liability rather than a control. Each
record stores the SHA-256 of its own canonical content chained with the hash of the record
before it, so any modification, deletion or reordering breaks verification from that point
onward. This is the same construction as a git commit chain, and it gives the same
property: you cannot rewrite history without it being obvious.

The chain proves *integrity*, not *availability* — an attacker with write access to the
sink can still truncate it. Production deployments append to storage with an object-lock or
WORM retention policy, and export the head hash to a separate system on a schedule so that
truncation is detectable too.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

from eap.platform.clock import SYSTEM_CLOCK, Clock

GENESIS_HASH = "0" * 64


class AuditAction(StrEnum):
    """The vocabulary of auditable events. Closed on purpose: an open string field turns
    into fifty spellings of the same event within a quarter."""

    AUTH_SUCCEEDED = "auth.succeeded"
    AUTH_FAILED = "auth.failed"
    AUTHZ_DENIED = "authz.denied"
    AGENT_RUN_STARTED = "agent.run.started"
    AGENT_RUN_COMPLETED = "agent.run.completed"
    AGENT_RUN_FAILED = "agent.run.failed"
    GUARDRAIL_TRIPPED = "guardrail.tripped"
    POLICY_DENIED = "policy.denied"
    TOOL_INVOKED = "tool.invoked"
    TOOL_DENIED = "tool.denied"
    APPROVAL_REQUESTED = "approval.requested"
    APPROVAL_GRANTED = "approval.granted"
    MODEL_INVOKED = "model.invoked"
    KNOWLEDGE_INGESTED = "knowledge.ingested"
    KNOWLEDGE_RETRIEVED = "knowledge.retrieved"
    BUDGET_EXCEEDED = "budget.exceeded"


class Outcome(StrEnum):
    ALLOWED = "allowed"
    DENIED = "denied"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class AuditRecord:
    sequence: int
    timestamp: datetime
    action: AuditAction
    outcome: Outcome
    tenant_id: str
    actor: str
    correlation_id: str
    resource: str | None = None
    reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    previous_hash: str = GENESIS_HASH
    record_hash: str = ""

    def canonical_payload(self) -> str:
        """The exact bytes that are hashed.

        Sorted keys and a fixed separator, so that two processes hashing the same logical
        record always agree. ``record_hash`` is excluded — it is the output.
        """
        payload = {
            "sequence": self.sequence,
            "timestamp": self.timestamp.isoformat(),
            "action": str(self.action),
            "outcome": str(self.outcome),
            "tenant_id": self.tenant_id,
            "actor": self.actor,
            "correlation_id": self.correlation_id,
            "resource": self.resource,
            "reason": self.reason,
            "metadata": self.metadata,
            "previous_hash": self.previous_hash,
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)

    def compute_hash(self) -> str:
        return hashlib.sha256(self.canonical_payload().encode("utf-8")).hexdigest()

    def to_json(self) -> str:
        record = asdict(self)
        record["timestamp"] = self.timestamp.isoformat()
        record["action"] = str(self.action)
        record["outcome"] = str(self.outcome)
        return json.dumps(record, sort_keys=True, separators=(",", ":"), default=str)


class AuditSink(Protocol):
    """Where records land. Append-only by contract: there is no update or delete."""

    def append(self, record: AuditRecord) -> None: ...

    def read_all(self) -> Iterator[AuditRecord]: ...


class InMemoryAuditSink:
    """For tests and local development. ``Settings`` refuses it in staging and prod."""

    def __init__(self) -> None:
        self._records: list[AuditRecord] = []

    def append(self, record: AuditRecord) -> None:
        self._records.append(record)

    def read_all(self) -> Iterator[AuditRecord]:
        yield from self._records

    def __len__(self) -> int:
        return len(self._records)


class FileAuditSink:
    """Newline-delimited JSON on disk, opened in append mode per write.

    Reopening per record is slower than holding the handle, and that is the point: a
    long-lived handle can be truncated out from under the process and buffered records
    lost. Durability beats throughput for this particular file.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: AuditRecord) -> None:
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(record.to_json() + "\n")

    def read_all(self) -> Iterator[AuditRecord]:
        if not self._path.exists():
            return
        with self._path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    yield _record_from_dict(json.loads(line))


@dataclass(frozen=True, slots=True)
class ChainVerification:
    valid: bool
    records_checked: int
    head_hash: str
    broken_at: int | None = None
    reason: str | None = None


class AuditLog:
    """Writes records into a hash chain and verifies the chain on demand."""

    def __init__(self, sink: AuditSink, *, clock: Clock = SYSTEM_CLOCK) -> None:
        self._sink = sink
        self._clock = clock
        self._sequence = 0
        self._head = GENESIS_HASH
        self._restore_head()

    def _restore_head(self) -> None:
        """Resume an existing chain rather than forking a new one on restart."""
        for record in self._sink.read_all():
            self._sequence = record.sequence
            self._head = record.record_hash

    @property
    def head_hash(self) -> str:
        return self._head

    def record(
        self,
        action: AuditAction,
        *,
        outcome: Outcome,
        tenant_id: str,
        actor: str,
        correlation_id: str,
        resource: str | None = None,
        reason: str | None = None,
        **metadata: Any,
    ) -> AuditRecord:
        self._sequence += 1
        draft = AuditRecord(
            sequence=self._sequence,
            timestamp=self._clock.now(),
            action=action,
            outcome=outcome,
            tenant_id=tenant_id,
            actor=actor,
            correlation_id=correlation_id,
            resource=resource,
            reason=reason,
            metadata=metadata,
            previous_hash=self._head,
        )
        sealed = AuditRecord(**{**asdict(draft), "record_hash": draft.compute_hash()})
        self._sink.append(sealed)
        self._head = sealed.record_hash
        return sealed

    def verify(self) -> ChainVerification:
        """Recompute every hash and confirm each record points at its predecessor."""
        previous = GENESIS_HASH
        expected_sequence = 0
        count = 0

        for record in self._sink.read_all():
            count += 1
            expected_sequence += 1

            if record.sequence != expected_sequence:
                return ChainVerification(
                    valid=False,
                    records_checked=count,
                    head_hash=previous,
                    broken_at=record.sequence,
                    reason=f"sequence gap: expected {expected_sequence}, found {record.sequence}",
                )
            if record.previous_hash != previous:
                return ChainVerification(
                    valid=False,
                    records_checked=count,
                    head_hash=previous,
                    broken_at=record.sequence,
                    reason="record does not chain to its predecessor",
                )
            if record.compute_hash() != record.record_hash:
                return ChainVerification(
                    valid=False,
                    records_checked=count,
                    head_hash=previous,
                    broken_at=record.sequence,
                    reason="record content does not match its hash",
                )
            previous = record.record_hash

        return ChainVerification(valid=True, records_checked=count, head_hash=previous)


def _record_from_dict(data: dict[str, Any]) -> AuditRecord:
    return AuditRecord(
        sequence=int(data["sequence"]),
        timestamp=datetime.fromisoformat(data["timestamp"]),
        action=AuditAction(data["action"]),
        outcome=Outcome(data["outcome"]),
        tenant_id=data["tenant_id"],
        actor=data["actor"],
        correlation_id=data["correlation_id"],
        resource=data.get("resource"),
        reason=data.get("reason"),
        metadata=data.get("metadata") or {},
        previous_hash=data["previous_hash"],
        record_hash=data["record_hash"],
    )


def build_audit_sink(kind: str, *, file_path: str) -> AuditSink:
    if kind == "file":
        return FileAuditSink(file_path)
    return InMemoryAuditSink()
