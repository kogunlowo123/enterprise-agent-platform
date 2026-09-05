"""Governance policy engine.

RBAC answers "may this caller do this kind of thing". Policy answers "given everything we
know about this specific request, should it happen anyway". They are separate because the
questions have different shapes: permissions are static grants, policies are contextual
rules that read the tenant, the model, the data classification and the time of day.

Rules are ordered and the first matching DENY wins. Deny-overrides rather than
allow-overrides, because a governance engine where adding a rule can *widen* access is one
nobody can reason about.

The rule set is data, not code: it is expressed as :class:`PolicyRule` objects that a
platform admin can load from configuration, and every decision returns the id of the rule
that produced it so an auditor can trace a refusal back to a specific line.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from eap.identity.models import SecurityContext
from eap.platform.errors import PolicyViolation


class Effect(StrEnum):
    ALLOW = "allow"
    DENY = "deny"


class DataClassification(StrEnum):
    """How sensitive the material in this request is. Drives residency and model choice."""

    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


_ORDER = {
    DataClassification.PUBLIC: 0,
    DataClassification.INTERNAL: 1,
    DataClassification.CONFIDENTIAL: 2,
    DataClassification.RESTRICTED: 3,
}


@dataclass(frozen=True, slots=True)
class PolicyRequest:
    """Everything a rule may inspect. Rules receive this and nothing else."""

    action: str
    security: SecurityContext
    resource: str | None = None
    classification: DataClassification = DataClassification.INTERNAL
    model: str | None = None
    provider: str | None = None
    tool: str | None = None
    estimated_cost_usd: float = 0.0
    attributes: dict[str, Any] = field(default_factory=dict)

    @property
    def tenant_id(self) -> str:
        return self.security.tenant.id

    @property
    def residency(self) -> str:
        return self.security.tenant.data_residency


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    effect: Effect
    rule_id: str
    reason: str
    obligations: tuple[str, ...] = ()
    """Conditions the caller must satisfy for an ALLOW to stand — for instance
    ``require_human_approval`` or ``redact_pii``. An allow with an unmet obligation is a
    deny; the orchestrator is responsible for discharging them."""

    @property
    def allowed(self) -> bool:
        return self.effect is Effect.ALLOW


@dataclass(frozen=True, slots=True)
class PolicyRule:
    id: str
    description: str
    effect: Effect
    applies_to: Callable[[PolicyRequest], bool]
    obligations: tuple[str, ...] = ()

    def matches(self, request: PolicyRequest) -> bool:
        return self.applies_to(request)


class PolicyEngine:
    """Evaluates rules in order. Deny wins; the first matching allow ends evaluation."""

    def __init__(self, rules: Sequence[PolicyRule] | None = None) -> None:
        self._rules: list[PolicyRule] = list(rules if rules is not None else default_rules())

    def add(self, rule: PolicyRule) -> None:
        self._rules.append(rule)

    @property
    def rules(self) -> tuple[PolicyRule, ...]:
        return tuple(self._rules)

    def evaluate(self, request: PolicyRequest) -> PolicyDecision:
        obligations: list[str] = []

        for rule in self._rules:
            if not rule.matches(request):
                continue
            if rule.effect is Effect.DENY:
                return PolicyDecision(effect=Effect.DENY, rule_id=rule.id, reason=rule.description)
            obligations.extend(rule.obligations)

        return PolicyDecision(
            effect=Effect.ALLOW,
            rule_id="default_allow",
            reason="no rule denied this request",
            obligations=tuple(dict.fromkeys(obligations)),
        )

    def enforce(self, request: PolicyRequest) -> PolicyDecision:
        decision = self.evaluate(request)
        if not decision.allowed:
            raise PolicyViolation(decision.reason, rule=decision.rule_id, action=request.action)
        return decision


def at_least(classification: DataClassification) -> Callable[[PolicyRequest], bool]:
    def check(request: PolicyRequest) -> bool:
        return _ORDER[request.classification] >= _ORDER[classification]

    return check


def default_rules() -> tuple[PolicyRule, ...]:
    """A defensible starting set. Deployments extend it; they should not need to weaken it."""
    return (
        PolicyRule(
            id="residency.eu_data_stays_in_eu",
            description=(
                "Tenants with EU residency may not send confidential or higher material to a "
                "provider endpoint outside the EU"
            ),
            effect=Effect.DENY,
            applies_to=lambda r: (
                r.residency == "eu"
                and at_least(DataClassification.CONFIDENTIAL)(r)
                and bool(r.attributes.get("provider_region", "us") != "eu")
            ),
        ),
        PolicyRule(
            id="model.restricted_data_requires_approved_model",
            description=(
                "Restricted material may only be processed by a model on the tenant's approved list"
            ),
            effect=Effect.DENY,
            applies_to=lambda r: (
                r.classification is DataClassification.RESTRICTED
                and r.model is not None
                and r.model not in set(r.attributes.get("approved_models", ()))
            ),
        ),
        PolicyRule(
            id="tool.write_tools_need_human_approval",
            description="Tools that mutate an external system require a human in the loop",
            effect=Effect.ALLOW,
            applies_to=lambda r: bool(r.tool) and bool(r.attributes.get("tool_mutates", False)),
            obligations=("require_human_approval",),
        ),
        PolicyRule(
            id="data.confidential_prompts_are_redacted",
            description="Confidential or higher material is redacted before leaving the boundary",
            effect=Effect.ALLOW,
            applies_to=at_least(DataClassification.CONFIDENTIAL),
            obligations=("redact_pii", "disable_prompt_capture"),
        ),
        PolicyRule(
            id="cost.single_call_ceiling",
            description="A single model call may not exceed 5 USD without an override permission",
            effect=Effect.DENY,
            applies_to=lambda r: (
                r.estimated_cost_usd > 5.0 and not r.security.has("model:route:override")
            ),
        ),
        PolicyRule(
            id="agent.agents_may_not_delegate_upward",
            description=(
                "A principal that is itself an agent may not invoke tools that mutate state; "
                "chained agent delegation must terminate at a human-authorised caller"
            ),
            effect=Effect.DENY,
            applies_to=lambda r: (
                r.security.principal.principal_type == "agent"
                and bool(r.attributes.get("tool_mutates", False))
            ),
        ),
    )
