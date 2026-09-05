"""Guardrail contracts.

A guardrail inspects text at a boundary and returns findings. It never raises: the decision
to block belongs to the pipeline, which knows the configured policy, and to the caller,
which knows whether it is running in shadow mode. A detector that raises cannot be run in
shadow mode, and a control nobody dares enable is not a control.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
from typing import Protocol


class Severity(IntEnum):
    """Ordered so that findings sort and compare numerically."""

    INFO = 10
    LOW = 20
    MEDIUM = 30
    HIGH = 40
    CRITICAL = 50


class Boundary(StrEnum):
    """Where the text was intercepted. Determines which detectors are relevant."""

    USER_INPUT = "user_input"
    RETRIEVED_CONTEXT = "retrieved_context"
    TOOL_OUTPUT = "tool_output"
    MODEL_OUTPUT = "model_output"


@dataclass(frozen=True, slots=True)
class Finding:
    detector: str
    category: str
    severity: Severity
    confidence: float
    message: str
    span: tuple[int, int] | None = None
    evidence: str | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be within [0, 1]")


@dataclass(frozen=True, slots=True)
class GuardrailResult:
    text: str
    """The text after any redaction. Identical to the input when nothing was rewritten."""

    findings: tuple[Finding, ...] = ()
    modified: bool = False
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def max_severity(self) -> Severity | None:
        return max((f.severity for f in self.findings), default=None)

    @property
    def max_confidence(self) -> float:
        return max((f.confidence for f in self.findings), default=0.0)

    def at_or_above(self, severity: Severity) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.severity >= severity)


class Guardrail(Protocol):
    name: str

    def inspect(self, text: str, *, boundary: Boundary) -> GuardrailResult: ...
