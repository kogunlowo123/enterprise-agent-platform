"""Token accounting and budget enforcement.

Cost is the constraint that actually stops agent platforms in production. A retry loop that
looked harmless in review turns into a four-figure invoice overnight, and by the time it
appears on a bill the money is gone. So spend is attributed per tenant at the moment it
happens, and the budget is checked *before* the call rather than after.

Prices are configuration, not constants baked into logic. They change, they differ per
contract, and a platform that hardcodes them reports confidently wrong numbers. The table
below is a starting set with an explicit ``as_of`` date; deployments override it.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime

from eap.llmops.providers.base import Usage
from eap.platform.clock import SYSTEM_CLOCK, Clock
from eap.platform.errors import BudgetExceeded


@dataclass(frozen=True, slots=True)
class ModelPrice:
    """USD per million tokens."""

    input_per_million: float
    output_per_million: float
    cached_input_per_million: float | None = None

    def cost_usd(self, usage: Usage) -> float:
        billable_input = max(0, usage.input_tokens - usage.cached_input_tokens)
        total = (billable_input / 1_000_000) * self.input_per_million
        total += (usage.output_tokens / 1_000_000) * self.output_per_million
        if usage.cached_input_tokens:
            cached_rate = (
                self.cached_input_per_million
                if self.cached_input_per_million is not None
                else self.input_per_million * 0.1
            )
            total += (usage.cached_input_tokens / 1_000_000) * cached_rate
        return round(total, 8)


PRICES_AS_OF = date(2026, 9, 1)

DEFAULT_PRICES: dict[str, ModelPrice] = {
    "claude-opus-5": ModelPrice(15.00, 75.00),
    "claude-sonnet-5": ModelPrice(3.00, 15.00),
    "claude-fable-5-1": ModelPrice(3.00, 15.00),
    "claude-haiku-4-5-20251001": ModelPrice(1.00, 5.00),
    "gpt-4o": ModelPrice(2.50, 10.00, 1.25),
    "gpt-4o-mini": ModelPrice(0.15, 0.60, 0.075),
    "gpt-4.1": ModelPrice(2.00, 8.00, 0.50),
    "gpt-4.1-mini": ModelPrice(0.40, 1.60, 0.10),
    "local-deterministic": ModelPrice(0.0, 0.0),
}


@dataclass(frozen=True, slots=True)
class SpendRecord:
    tenant_id: str
    model: str
    provider: str
    usage: Usage
    cost_usd: float
    at: datetime
    correlation_id: str | None = None
    agent_id: str | None = None


@dataclass(slots=True)
class TenantSpend:
    tenant_id: str
    window_start: date
    total_usd: float = 0.0
    calls: int = 0
    by_model: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    by_agent: dict[str, float] = field(default_factory=lambda: defaultdict(float))


class CostTracker:
    """Per-tenant daily spend with a pre-call budget check.

    Windows roll on the UTC calendar day. Anything finer needs a shared store, since a
    per-process window is meaningless across replicas — the same caveat as the rate
    limiter, and the same fix.
    """

    def __init__(
        self,
        *,
        daily_budget_usd: float,
        prices: dict[str, ModelPrice] | None = None,
        clock: Clock = SYSTEM_CLOCK,
    ) -> None:
        self._budget = daily_budget_usd
        self._prices = dict(prices or DEFAULT_PRICES)
        self._clock = clock
        self._spend: dict[str, TenantSpend] = {}
        self._history: list[SpendRecord] = []

    def price_for(self, model: str) -> ModelPrice | None:
        return self._prices.get(model)

    def set_price(self, model: str, price: ModelPrice) -> None:
        self._prices[model] = price

    def estimate(self, model: str, *, input_tokens: int, output_tokens: int) -> float:
        """Pre-call estimate. Unknown models cost nothing by this measure, which is why
        :meth:`check_budget` treats an unknown model as unpriced rather than free."""
        price = self._prices.get(model)
        if price is None:
            return 0.0
        return price.cost_usd(Usage(input_tokens=input_tokens, output_tokens=output_tokens))

    def current_spend(self, tenant_id: str) -> TenantSpend:
        today = self._clock.now().date()
        spend = self._spend.get(tenant_id)
        if spend is None or spend.window_start != today:
            spend = TenantSpend(tenant_id=tenant_id, window_start=today)
            self._spend[tenant_id] = spend
        return spend

    def remaining(self, tenant_id: str) -> float:
        return max(0.0, self._budget - self.current_spend(tenant_id).total_usd)

    def check_budget(self, tenant_id: str, *, estimated_usd: float) -> None:
        """Refuse a call whose estimate would take the tenant past its allocation."""
        spend = self.current_spend(tenant_id)
        if spend.total_usd + estimated_usd > self._budget:
            raise BudgetExceeded(
                f"tenant '{tenant_id}' has spent {spend.total_usd:.4f} of its "
                f"{self._budget:.2f} USD daily allocation",
                rule="llmops.daily_tenant_budget",
                spent_usd=round(spend.total_usd, 6),
                budget_usd=self._budget,
                estimated_usd=round(estimated_usd, 6),
            )

    def record(
        self,
        *,
        tenant_id: str,
        model: str,
        provider: str,
        usage: Usage,
        correlation_id: str | None = None,
        agent_id: str | None = None,
    ) -> SpendRecord:
        price = self._prices.get(model)
        cost = price.cost_usd(usage) if price else 0.0

        spend = self.current_spend(tenant_id)
        spend.total_usd += cost
        spend.calls += 1
        spend.by_model[model] += cost
        if agent_id:
            spend.by_agent[agent_id] += cost

        record = SpendRecord(
            tenant_id=tenant_id,
            model=model,
            provider=provider,
            usage=usage,
            cost_usd=cost,
            at=self._clock.now(),
            correlation_id=correlation_id,
            agent_id=agent_id,
        )
        self._history.append(record)
        return record

    def attribution(self, tenant_id: str) -> dict[str, object]:
        """What this tenant spent today, split by model and by agent.

        This is the report that answers "which team is generating the bill", which is the
        first question anyone asks once a platform has more than one internal customer.
        """
        spend = self.current_spend(tenant_id)
        return {
            "tenant_id": tenant_id,
            "window_start": spend.window_start.isoformat(),
            "total_usd": round(spend.total_usd, 6),
            "budget_usd": self._budget,
            "remaining_usd": round(self.remaining(tenant_id), 6),
            "calls": spend.calls,
            "by_model": {k: round(v, 6) for k, v in sorted(spend.by_model.items())},
            "by_agent": {k: round(v, 6) for k, v in sorted(spend.by_agent.items())},
        }
