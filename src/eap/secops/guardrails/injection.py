"""Prompt injection detection.

Scores text against pattern families drawn from OWASP LLM01. Scoring is additive with
diminishing returns, so a document that trips one weak pattern is not treated the same as
one that trips four strong ones.

Two properties matter more than raw accuracy:

**Boundary awareness.** ``"ignore previous instructions"`` typed by a user into a chat box
is usually them being sloppy about their own conversation. The same string arriving inside
a retrieved wiki page or a tool response is indirect injection — content that no human
chose to send. Retrieved and tool boundaries therefore carry a higher multiplier, because
that is where the real attacks live.

**Honesty about what this is.** Pattern matching catches unsophisticated and opportunistic
injection. It does not catch a determined attacker who paraphrases, encodes or translates.
It is one layer: the ones that carry the actual weight are least-privilege tool grants,
the caller ∩ agent permission intersection, human approval on write tools, and egress
control. Those hold when this detector misses, which it will.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from eap.secops.guardrails.base import Boundary, Finding, GuardrailResult, Severity


@dataclass(frozen=True, slots=True)
class Pattern:
    name: str
    regex: re.Pattern[str]
    weight: float
    severity: Severity
    description: str


def _compile(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.IGNORECASE | re.MULTILINE)


PATTERNS: tuple[Pattern, ...] = (
    Pattern(
        "instruction_override",
        _compile(
            r"\b(ignore|disregard|forget|override|discard)\b[\w\s,]{0,30}?"
            r"\b(previous|prior|above|earlier|all|any|initial|original)\b"
            r"[\w\s,]{0,30}?\b(instruction|prompt|rule|direction|command|guideline)s?\b"
        ),
        # On its own this phrasing is close to conclusive: it has no benign use in a
        # question, so one match is enough to cross the default enforcement threshold.
        weight=0.55,
        severity=Severity.HIGH,
        description="attempts to void instructions already in the context",
    ),
    Pattern(
        "role_reassignment",
        _compile(
            r"\b(you\s+are\s+now|from\s+now\s+on\s+you|act\s+as|pretend\s+to\s+be|"
            r"roleplay\s+as|behave\s+as|simulate\s+being)\b"
        ),
        weight=0.30,
        severity=Severity.MEDIUM,
        description="attempts to redefine the agent's identity or mission",
    ),
    Pattern(
        "system_prompt_extraction",
        _compile(
            r"\b(reveal|show|print|repeat|output|display|dump|disclose|verbatim|tell|give)\b"
            r"[\w\s,'\"]{0,30}?\b(system\s+prompt|initial\s+instruction|your\s+instruction|"
            r"prompt\s+above|context\s+window|developer\s+message)s?\b"
        ),
        weight=0.50,
        severity=Severity.HIGH,
        description="attempts to exfiltrate the system prompt or developer message",
    ),
    Pattern(
        "guardrail_bypass",
        _compile(
            r"\b(developer\s+mode|god\s+mode|jailbreak|DAN\s+mode|unrestricted\s+mode|"
            r"without\s+(any\s+)?(restriction|filter|limitation|censorship)|"
            r"bypass\s+(the\s+)?(safety|guardrail|filter|restriction))\b"
        ),
        weight=0.55,
        severity=Severity.HIGH,
        description="named jailbreak techniques and explicit bypass requests",
    ),
    Pattern(
        "fake_system_turn",
        _compile(
            r"(^|\n)\s*(\[|<|##\s*)?(system|assistant|developer)\s*(\]|>|:)"
            r"|<\|(im_start|im_end|system|endoftext)\|>"
        ),
        weight=0.50,
        severity=Severity.HIGH,
        description="forged conversation turn markers or model control tokens",
    ),
    Pattern(
        "tool_coercion",
        _compile(
            r"\b(call|invoke|execute|run|use)\b[\w\s]{0,20}?\b(tool|function|command|api)\b"
            r"[\w\s]{0,40}?\b(without|skip|bypass|no\s+need)\b"
            r"|\bdo\s+not\s+(ask|request|seek)\b[\w\s]{0,20}\b(permission|approval|confirm)"
        ),
        weight=0.50,
        severity=Severity.CRITICAL,
        description="attempts to drive tool use while skipping approval",
    ),
    Pattern(
        "exfiltration_channel",
        _compile(
            r"\b(send|post|upload|transmit|forward|leak|email|exfiltrate)\b"
            r"[\w\s,]{0,40}?\b(to|at)\b\s*(https?://|www\.|[\w.-]+@[\w.-]+\.\w+)"
            r"|!\[[^\]]*\]\(https?://[^)]*\{"
        ),
        weight=0.50,
        severity=Severity.CRITICAL,
        description="instructs the agent to send data to an attacker-controlled endpoint",
    ),
    Pattern(
        "authority_spoofing",
        _compile(
            r"\b(this\s+is\s+(an?\s+)?(official|authorised|authorized|urgent|admin|"
            r"security)\s+(message|request|override|instruction)"
            r"|as\s+(your|the)\s+(administrator|developer|owner|creator))\b"
        ),
        # Tuned so this alone clears the default threshold from a retrieved document or a
        # tool response (0.35 x 1.6) but not from a user typing it (0.35). A person
        # claiming authority in a chat box is a claim; a stored document making the same
        # claim to the model is an attack, because no human chose to send it.
        weight=0.35,
        severity=Severity.MEDIUM,
        description="claims institutional authority to justify an instruction",
    ),
    Pattern(
        "hidden_channel",
        _compile(r"(?s)<!--.{0,400}?(ignore|instruct|system|prompt|tool).{0,400}?-->"),
        weight=0.35,
        severity=Severity.HIGH,
        description="instructions concealed in markup that renders invisibly",
    ),
    Pattern(
        "encoded_payload",
        _compile(
            r"\b(base64|rot13|hex|url)\s*(decode|encoded?)\b[\w\s:]{0,20}"
            r"|\bdecode\s+(this|the\s+following)\b"
        ),
        weight=0.25,
        severity=Severity.MEDIUM,
        description="asks the model to decode a payload, hiding it from pattern matching",
    ),
)

# Retrieved documents and tool responses are content the user never authored. An override
# attempt arriving through those channels is indirect injection and weighs more.
BOUNDARY_MULTIPLIER: dict[Boundary, float] = {
    Boundary.USER_INPUT: 1.0,
    Boundary.RETRIEVED_CONTEXT: 1.6,
    Boundary.TOOL_OUTPUT: 1.6,
    Boundary.MODEL_OUTPUT: 1.2,
}

# Written as escapes rather than literals. A source file containing actual bidirectional
# control characters is unreviewable — you cannot see what you are approving — and static
# analysers flag it as Trojan Source (CWE-838), correctly. Writing the detector for
# invisible characters using invisible characters would be a poor joke at reviewers'
# expense.
_ZERO_WIDTH = re.compile(
    "["
    "\u200b-\u200f"  # zero-width space, ZWNJ, ZWJ, LRM, RLM
    "\u202a-\u202e"  # bidirectional embedding, override and pop
    "\u2060-\u2064"  # word joiner and the invisible operators
    "\ufeff"  # zero-width no-break space (BOM when leading)
    "]"
)


class InjectionDetector:
    """Pattern-and-heuristic detector for direct and indirect prompt injection."""

    name = "prompt_injection"

    def __init__(self, *, patterns: tuple[Pattern, ...] = PATTERNS) -> None:
        self._patterns = patterns

    def inspect(self, text: str, *, boundary: Boundary = Boundary.USER_INPUT) -> GuardrailResult:
        if not text.strip():
            return GuardrailResult(text=text)

        normalised, obfuscation = self._normalise(text)
        findings: list[Finding] = []
        weights: list[float] = []
        multiplier = BOUNDARY_MULTIPLIER[boundary]

        for pattern in self._patterns:
            match = pattern.regex.search(normalised)
            if match is None:
                continue
            confidence = min(1.0, pattern.weight * multiplier)
            weights.append(pattern.weight)
            findings.append(
                Finding(
                    detector=self.name,
                    category=pattern.name,
                    severity=pattern.severity,
                    confidence=round(confidence, 3),
                    message=f"{pattern.description} (boundary={boundary})",
                    span=match.span(),
                    evidence=_excerpt(normalised, match.span()),
                )
            )

        if obfuscation:
            weights.append(0.30)
            findings.append(
                Finding(
                    detector=self.name,
                    category="unicode_obfuscation",
                    severity=Severity.HIGH,
                    confidence=round(min(1.0, 0.30 * multiplier), 3),
                    message=(
                        "text contains zero-width or bidirectional control characters, which "
                        "hide content from human review while the model still reads it"
                    ),
                )
            )

        score = _aggregate(weights, multiplier)
        return GuardrailResult(
            text=text,
            findings=tuple(findings),
            metadata={"score": f"{score:.3f}", "boundary": str(boundary)},
        )

    def score(self, text: str, *, boundary: Boundary = Boundary.USER_INPUT) -> float:
        result = self.inspect(text, boundary=boundary)
        return float(result.metadata.get("score", "0"))

    @staticmethod
    def _normalise(text: str) -> tuple[str, bool]:
        """Strip the tricks that defeat naive matching, and report having seen them.

        NFKC folds full-width and mathematical alphanumerics onto ASCII, so ``ｉｇｎｏｒｅ``
        and ``𝗂𝗀𝗇𝗈𝗋𝖾`` both reach the patterns as ``ignore``. Zero-width characters are
        removed because ``i\u200bgnore`` reads as ``ignore`` to a tokenizer but not to a
        regex — and their presence is itself evidence.
        """
        stripped = _ZERO_WIDTH.sub("", text)
        had_obfuscation = stripped != text
        return unicodedata.normalize("NFKC", stripped), had_obfuscation


def _aggregate(weights: list[float], multiplier: float) -> float:
    """Combine pattern weights with diminishing returns.

    Probabilistic OR rather than a sum: four independent 0.4 signals should read as
    "almost certainly", not as 1.6 clipped to 1.0, and one signal should never saturate.
    """
    if not weights:
        return 0.0
    inverse = 1.0
    for weight in sorted(weights, reverse=True):
        inverse *= 1.0 - min(0.95, weight)
    return round(min(1.0, (1.0 - inverse) * multiplier), 3)


def _excerpt(text: str, span: tuple[int, int], *, padding: int = 24) -> str:
    start = max(0, span[0] - padding)
    end = min(len(text), span[1] + padding)
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(text) else ""
    return f"{prefix}{text[start:end]}{suffix}"
