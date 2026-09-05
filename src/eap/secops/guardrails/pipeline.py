"""Guardrail composition.

Runs a chain of detectors over one piece of text and turns the aggregate into a single
decision. Detectors are pure and order-independent for detection; order only matters for
rewriting, so redaction runs before scoring and the injection detector sees the redacted
text. That ordering is deliberate: a credential should never reach the model even if the
turn is ultimately allowed.
"""

from __future__ import annotations

from dataclasses import dataclass

from eap.platform.errors import GuardrailTripped
from eap.secops.guardrails.base import Boundary, Finding, Guardrail, GuardrailResult, Severity
from eap.secops.guardrails.injection import InjectionDetector
from eap.secops.guardrails.pii import CREDENTIAL_CATEGORIES, SensitiveDataDetector


@dataclass(frozen=True, slots=True)
class GuardrailDecision:
    allowed: bool
    text: str
    findings: tuple[Finding, ...]
    blocked_by: tuple[str, ...] = ()
    reason: str | None = None

    @property
    def modified(self) -> bool:
        return bool(self.findings) and any(f.category not in ("ip_address",) for f in self.findings)

    def raise_if_blocked(self) -> None:
        if not self.allowed:
            raise GuardrailTripped(
                self.reason or "content blocked by guardrail",
                rule=",".join(self.blocked_by) or "guardrail",
                categories=sorted({f.category for f in self.findings}),
            )


class GuardrailPipeline:
    """The enforcement point. Detectors decide *what*; this decides *whether*."""

    def __init__(
        self,
        *,
        guardrails: tuple[Guardrail, ...] | None = None,
        block_on_injection: bool = True,
        injection_threshold: float = 0.5,
        block_severity: Severity = Severity.CRITICAL,
        redact: bool = True,
    ) -> None:
        self._guardrails = guardrails or (
            SensitiveDataDetector(redact=redact),
            InjectionDetector(),
        )
        self._block_on_injection = block_on_injection
        self._injection_threshold = injection_threshold
        self._block_severity = block_severity

    def evaluate(self, text: str, *, boundary: Boundary) -> GuardrailDecision:
        findings: list[Finding] = []
        current = text
        injection_score = 0.0

        for guardrail in self._guardrails:
            result: GuardrailResult = guardrail.inspect(current, boundary=boundary)
            findings.extend(result.findings)
            current = result.text
            if "score" in result.metadata:
                injection_score = max(injection_score, float(result.metadata["score"]))

        blocked_by: list[str] = []
        reasons: list[str] = []

        credentials = [f for f in findings if f.category in CREDENTIAL_CATEGORIES]
        if credentials:
            blocked_by.append("credential_exposure")
            reasons.append(
                f"{len(credentials)} credential-shaped value(s) found; these must be rotated"
            )

        severe = [f for f in findings if f.severity >= self._block_severity]
        if severe and not credentials:
            blocked_by.append("severity_threshold")
            reasons.append(f"{len(severe)} finding(s) at or above {self._block_severity.name}")

        if self._block_on_injection and injection_score >= self._injection_threshold:
            blocked_by.append("prompt_injection")
            reasons.append(
                f"injection score {injection_score:.2f} met the {self._injection_threshold:.2f} "
                f"threshold at boundary {boundary}"
            )

        return GuardrailDecision(
            allowed=not blocked_by,
            text=current,
            findings=tuple(findings),
            blocked_by=tuple(blocked_by),
            reason="; ".join(reasons) or None,
        )

    def evaluate_input(self, text: str) -> GuardrailDecision:
        return self.evaluate(text, boundary=Boundary.USER_INPUT)

    def evaluate_retrieved(self, text: str) -> GuardrailDecision:
        return self.evaluate(text, boundary=Boundary.RETRIEVED_CONTEXT)

    def evaluate_output(self, text: str) -> GuardrailDecision:
        return self.evaluate(text, boundary=Boundary.MODEL_OUTPUT)
