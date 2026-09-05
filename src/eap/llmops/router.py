"""Model routing.

Applications ask for a *route* — ``fast``, ``balanced``, ``deep`` — not a model. That
indirection is the point: swapping which model serves ``balanced``, or failing over to a
second vendor during an outage, then becomes a configuration change instead of a deploy
across every consuming service.

Each route is an ordered chain of candidates. The router walks it, guarded by a circuit
breaker per provider and a retry policy per attempt, and stops at the first success. A
candidate is skipped without being attempted when its provider's circuit is open or when
its estimated cost would breach the tenant's budget — cheap checks first, so a doomed call
is never dispatched.

Budget is checked before the call and recorded after it, against the *actual* usage the
provider reported. Estimating and then trusting the estimate is how platforms discover a
20% accounting drift a month later.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from eap.llmops.cost import CostTracker
from eap.llmops.providers.base import CompletionRequest, CompletionResponse, LLMProvider
from eap.netops.resilience import CircuitBreaker, RetryPolicy, call_with_resilience
from eap.platform.errors import (
    BudgetExceeded,
    ConfigurationError,
    NoProviderAvailable,
    PlatformError,
)
from eap.platform.telemetry import get_logger

log = get_logger(__name__)

BreakerFactory = Callable[[str], CircuitBreaker]
"""Builds the breaker guarding one provider. Injected so tests can supply a manual clock."""


@dataclass(frozen=True, slots=True)
class Candidate:
    provider: str
    model: str
    max_output_tokens: int | None = None
    reason: str = ""


@dataclass(frozen=True, slots=True)
class Route:
    """A named chain. First entry is the intent; the rest are fallbacks in order."""

    name: str
    candidates: tuple[Candidate, ...]
    description: str = ""

    def __post_init__(self) -> None:
        if not self.candidates:
            raise ConfigurationError(f"route '{self.name}' has no candidates")


@dataclass(frozen=True, slots=True)
class RoutingDecision:
    response: CompletionResponse
    route: str
    candidate: Candidate
    attempts: tuple[str, ...]
    cost_usd: float
    fell_back: bool

    @property
    def text(self) -> str:
        return self.response.text


def default_routes() -> dict[str, Route]:
    """Routes spanning two vendors, so a single provider outage is survivable.

    Fallbacks cross vendors deliberately. A chain of three Anthropic models does nothing
    when Anthropic is the thing that is down.
    """
    return {
        "fast": Route(
            name="fast",
            description="Classification, extraction, routing decisions. Latency over depth.",
            candidates=(
                Candidate("anthropic", "claude-haiku-4-5-20251001", reason="primary"),
                Candidate("openai", "gpt-4o-mini", reason="cross-vendor fallback"),
                Candidate("local", "local-deterministic", reason="last resort"),
            ),
        ),
        "balanced": Route(
            name="balanced",
            description="The default. Most agent turns land here.",
            candidates=(
                Candidate("anthropic", "claude-sonnet-5", reason="primary"),
                Candidate("openai", "gpt-4.1", reason="cross-vendor fallback"),
                Candidate("anthropic", "claude-haiku-4-5-20251001", reason="degraded capability"),
            ),
        ),
        "deep": Route(
            name="deep",
            description="Multi-step reasoning, ambiguous policy, high-consequence output.",
            candidates=(
                Candidate("anthropic", "claude-opus-5", reason="primary"),
                Candidate("anthropic", "claude-sonnet-5", reason="degraded capability"),
                Candidate("openai", "gpt-4.1", reason="cross-vendor fallback"),
            ),
        ),
    }


class ModelRouter:
    """Selects a provider and model per request, with fallback and budget enforcement."""

    def __init__(
        self,
        *,
        providers: dict[str, LLMProvider],
        routes: dict[str, Route] | None = None,
        cost: CostTracker | None = None,
        retry: RetryPolicy | None = None,
        timeout_seconds: float = 45.0,
        breaker_factory: BreakerFactory | None = None,
    ) -> None:
        if not providers:
            raise ConfigurationError("the router needs at least one provider")
        self._providers = providers
        self._routes = routes or default_routes()
        self._cost = cost
        self._retry = retry or RetryPolicy()
        self._timeout = timeout_seconds
        self._breakers: dict[str, CircuitBreaker] = {}
        self._breaker_factory = breaker_factory or (
            lambda name: CircuitBreaker(name=f"provider:{name}")
        )

    @property
    def routes(self) -> dict[str, Route]:
        return dict(self._routes)

    def register_route(self, route: Route) -> None:
        self._routes[route.name] = route

    def breaker_for(self, provider: str) -> CircuitBreaker:
        if provider not in self._breakers:
            self._breakers[provider] = self._breaker_factory(provider)
        return self._breakers[provider]

    def candidates_for(
        self, route_name: str, *, pinned_model: str | None = None
    ) -> tuple[Candidate, ...]:
        """Resolve a route to its chain, or a pinned model to a single candidate."""
        if pinned_model:
            for provider_name, provider in self._providers.items():
                if pinned_model in provider.supported_models:
                    return (Candidate(provider_name, pinned_model, reason="caller pinned"),)
            raise ConfigurationError(f"no registered provider serves model '{pinned_model}'")

        route = self._routes.get(route_name)
        if route is None:
            raise ConfigurationError(
                f"unknown route '{route_name}'; known routes: {sorted(self._routes)}"
            )
        return route.candidates

    async def complete(
        self,
        request: CompletionRequest,
        *,
        tenant_id: str,
        route: str = "balanced",
        correlation_id: str | None = None,
        agent_id: str | None = None,
    ) -> RoutingDecision:
        candidates = self.candidates_for(route, pinned_model=request.model)
        attempts: list[str] = []
        budget_error: BudgetExceeded | None = None
        last_error: PlatformError | None = None

        for position, candidate in enumerate(candidates):
            provider = self._providers.get(candidate.provider)
            if provider is None:
                continue

            label = f"{candidate.provider}/{candidate.model}"
            breaker = self.breaker_for(candidate.provider)
            if not breaker.allows_request():
                attempts.append(f"{label}:circuit_open")
                continue

            if self._cost is not None:
                estimated = self._cost.estimate(
                    candidate.model,
                    input_tokens=_estimate_input_tokens(request),
                    output_tokens=candidate.max_output_tokens or request.max_output_tokens,
                )
                try:
                    self._cost.check_budget(tenant_id, estimated_usd=estimated)
                except BudgetExceeded as exc:
                    # Keep walking: a cheaper fallback may still fit inside the allocation.
                    budget_error = exc
                    attempts.append(f"{label}:over_budget")
                    continue

            attempt_request = CompletionRequest(
                messages=request.messages,
                model=candidate.model,
                max_output_tokens=candidate.max_output_tokens or request.max_output_tokens,
                temperature=request.temperature,
                tools=request.tools,
                stop_sequences=request.stop_sequences,
                extra=request.extra,
            )

            # Bound to locals rather than closed over: the loop rebinds ``provider`` and
            # ``attempt_request`` on the next iteration, and a retry that fires after that
            # would otherwise call the wrong provider with the wrong request.
            def attempt(
                p: LLMProvider = provider, r: CompletionRequest = attempt_request
            ) -> Awaitable[CompletionResponse]:
                return p.complete(r)

            try:
                response = await call_with_resilience(
                    attempt,
                    breaker=breaker,
                    retry=self._retry,
                    timeout_seconds=self._timeout,
                )
            except PlatformError as exc:
                last_error = exc
                attempts.append(f"{label}:{exc.code}")
                log.warning(
                    "router.candidate_failed",
                    route=route,
                    candidate=label,
                    position=position,
                    error=exc.code,
                )
                continue

            attempts.append(f"{label}:ok")
            cost_usd = 0.0
            if self._cost is not None:
                record = self._cost.record(
                    tenant_id=tenant_id,
                    model=response.model,
                    provider=response.provider,
                    usage=response.usage,
                    correlation_id=correlation_id,
                    agent_id=agent_id,
                )
                cost_usd = record.cost_usd

            log.info(
                "router.completed",
                route=route,
                candidate=label,
                fell_back=position > 0,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                cost_usd=cost_usd,
                latency_ms=response.latency_ms,
            )

            return RoutingDecision(
                response=response,
                route=route,
                candidate=candidate,
                attempts=tuple(attempts),
                cost_usd=cost_usd,
                fell_back=position > 0,
            )

        # Every candidate over budget and none failed technically: the budget is the story.
        if budget_error is not None and last_error is None:
            raise budget_error

        raise NoProviderAvailable(
            f"every candidate on route '{route}' was exhausted", attempted=attempts
        )


def _estimate_input_tokens(request: CompletionRequest) -> int:
    return max(1, sum(len(m.content) for m in request.messages) // 4)
