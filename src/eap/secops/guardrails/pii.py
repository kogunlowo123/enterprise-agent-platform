"""Sensitive data detection and redaction.

Covers two classes of value with different consequences.

**Personal data** (email, phone, national identifiers, payment cards) is redacted rather
than blocked. Blocking a support conversation because it contains a customer's email would
make the platform unusable; replacing the value with a stable placeholder keeps the turn
working while keeping the value out of the provider's logs.

**Credentials** (API keys, private keys, tokens) are treated as critical. A live key that
reaches a model provider must be rotated regardless of what the provider does with it, so
these findings are surfaced at a severity that stops the turn under default policy.

Detectors that can be checked arithmetically are: payment cards run through Luhn, so the
sixteen-digit order number in a shipping confirmation does not get flagged as a card.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from eap.secops.guardrails.base import Boundary, Finding, GuardrailResult, Severity


@dataclass(frozen=True, slots=True)
class Rule:
    name: str
    regex: re.Pattern[str]
    severity: Severity
    confidence: float
    placeholder: str
    validator: Callable[[str], bool] | None = None
    redact: bool = True


def luhn_valid(value: str) -> bool:
    """Mod-10 checksum. Filters the large majority of numeric strings that are not cards."""
    digits = [int(c) for c in value if c.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for index, digit in enumerate(reversed(digits)):
        if index % 2 == 1:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


def _entropy_at_least(threshold: float) -> Callable[[str], bool]:
    """Reject low-entropy matches such as ``token = "changeme"``."""

    def check(value: str) -> bool:
        import math
        from collections import Counter

        candidate = value.split("=")[-1].strip().strip("\"'")
        if len(candidate) < 16:
            return False
        counts = Counter(candidate)
        entropy = -sum(
            (n / len(candidate)) * math.log2(n / len(candidate)) for n in counts.values()
        )
        return entropy >= threshold

    return check


RULES: tuple[Rule, ...] = (
    Rule(
        "email",
        re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]{2,}\b"),
        Severity.LOW,
        0.95,
        "[EMAIL]",
    ),
    Rule(
        "phone_e164",
        re.compile(
            r"(?<![\w.])\+?[1-9]\d{0,2}[\s.-]?\(?\d{2,4}\)?[\s.-]?\d{3,4}[\s.-]?\d{3,4}(?![\w.])"
        ),
        Severity.LOW,
        0.55,
        "[PHONE]",
    ),
    Rule(
        "us_ssn",
        re.compile(r"\b(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b"),
        Severity.HIGH,
        0.9,
        "[SSN]",
    ),
    Rule(
        "payment_card",
        re.compile(r"\b(?:\d[ -]?){13,19}\b"),
        Severity.HIGH,
        0.9,
        "[PAYMENT_CARD]",
        validator=luhn_valid,
    ),
    Rule(
        "ip_address",
        re.compile(
            r"\b(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\b"
        ),
        Severity.INFO,
        0.6,
        "[IP]",
        redact=False,
    ),
    Rule(
        "aws_access_key",
        re.compile(r"\b(?:AKIA|ASIA|AROA|AIDA)[0-9A-Z]{16}\b"),
        Severity.CRITICAL,
        0.98,
        "[AWS_ACCESS_KEY]",
    ),
    Rule(
        "github_token",
        re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr|github_pat)_[A-Za-z0-9_]{22,}\b"),
        Severity.CRITICAL,
        0.98,
        "[GITHUB_TOKEN]",
    ),
    Rule(
        "openai_key",
        re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}\b"),
        Severity.CRITICAL,
        0.95,
        "[OPENAI_KEY]",
    ),
    Rule(
        "anthropic_key",
        re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}\b"),
        Severity.CRITICAL,
        0.98,
        "[ANTHROPIC_KEY]",
    ),
    Rule(
        "slack_token",
        re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"),
        Severity.CRITICAL,
        0.95,
        "[SLACK_TOKEN]",
    ),
    Rule(
        "private_key",
        re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |PGP )?PRIVATE KEY-----"),
        Severity.CRITICAL,
        0.99,
        "[PRIVATE_KEY]",
    ),
    Rule(
        "jwt",
        re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"),
        Severity.HIGH,
        0.9,
        "[JWT]",
    ),
    Rule(
        "generic_secret_assignment",
        re.compile(
            r"(?i)\b(?:api[_-]?key|secret|password|passwd|token|credential)\b\s*[:=]\s*"
            r"[\"']?([A-Za-z0-9_\-./+]{16,})[\"']?"
        ),
        Severity.HIGH,
        0.7,
        "[SECRET]",
        validator=_entropy_at_least(3.0),
    ),
)

CREDENTIAL_CATEGORIES = frozenset(
    {
        "aws_access_key",
        "github_token",
        "openai_key",
        "anthropic_key",
        "slack_token",
        "private_key",
        "jwt",
        "generic_secret_assignment",
    }
)


class SensitiveDataDetector:
    """Finds personal data and credentials, and optionally redacts them in place."""

    name = "sensitive_data"

    def __init__(self, *, rules: tuple[Rule, ...] = RULES, redact: bool = True) -> None:
        self._rules = rules
        self._redact = redact

    def inspect(self, text: str, *, boundary: Boundary = Boundary.USER_INPUT) -> GuardrailResult:
        if not text:
            return GuardrailResult(text=text)

        findings: list[Finding] = []
        # Collect every replacement first, then apply them right-to-left, so that earlier
        # spans stay valid while the string is being rewritten.
        replacements: list[tuple[int, int, str]] = []

        for rule in self._rules:
            for match in rule.regex.finditer(text):
                value = match.group(0)
                if rule.validator is not None and not rule.validator(value):
                    continue
                findings.append(
                    Finding(
                        detector=self.name,
                        category=rule.name,
                        severity=rule.severity,
                        confidence=rule.confidence,
                        message=f"{rule.name} detected at {boundary}",
                        span=match.span(),
                        evidence=_mask(value),
                    )
                )
                if self._redact and rule.redact:
                    replacements.append((match.start(), match.end(), rule.placeholder))

        redacted = text
        if replacements:
            # Overlapping matches (a JWT also matching generic_secret_assignment) would
            # corrupt the output if applied twice. Keep the first claim on each span.
            replacements.sort(key=lambda item: (item[0], -item[1]))
            applied: list[tuple[int, int, str]] = []
            cursor = -1
            for start, end, placeholder in replacements:
                if start >= cursor:
                    applied.append((start, end, placeholder))
                    cursor = end
            for start, end, placeholder in reversed(applied):
                redacted = redacted[:start] + placeholder + redacted[end:]

        return GuardrailResult(
            text=redacted,
            findings=tuple(findings),
            modified=redacted != text,
            metadata={"redactions": str(len(replacements)), "boundary": str(boundary)},
        )

    def contains_credentials(self, text: str) -> bool:
        result = self.inspect(text)
        return any(f.category in CREDENTIAL_CATEGORIES for f in result.findings)


def _mask(value: str) -> str:
    """Evidence for the audit log that identifies the hit without reproducing the secret."""
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:3]}{'*' * (len(value) - 6)}{value[-3:]}"
