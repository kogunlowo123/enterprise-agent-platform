"""Network plane: token bucket arithmetic, circuit state machine, retry, egress."""

from __future__ import annotations

import asyncio
import random

import pytest

from eap.netops.egress import EgressGuard
from eap.netops.ratelimit import InMemoryBucketStore, LayeredRateLimiter, RateLimiter
from eap.netops.resilience import (
    CircuitBreaker,
    CircuitState,
    RetryPolicy,
    call_with_resilience,
    is_retryable,
)
from eap.platform.clock import ManualClock
from eap.platform.errors import (
    CircuitOpen,
    PolicyViolation,
    ProviderError,
    RateLimited,
    TimeoutExceeded,
    ValidationError,
)


class TestTokenBucket:
    def test_burst_is_available_immediately(self, clock: ManualClock) -> None:
        limiter = RateLimiter(rate_per_minute=60, burst=5, clock=clock)
        for _ in range(5):
            assert limiter.check("k").allowed
        assert not limiter.check("k").allowed

    def test_tokens_refill_at_the_configured_rate(self, clock: ManualClock) -> None:
        limiter = RateLimiter(rate_per_minute=60, burst=5, clock=clock)
        for _ in range(5):
            limiter.check("k")
        assert not limiter.check("k").allowed

        clock.advance(1.0)  # 60/min == 1 token per second
        assert limiter.check("k").allowed

    def test_refill_is_capped_at_burst(self, clock: ManualClock) -> None:
        limiter = RateLimiter(rate_per_minute=60, burst=3, clock=clock)
        clock.advance(3600)
        for _ in range(3):
            assert limiter.check("k").allowed
        assert not limiter.check("k").allowed

    def test_a_refused_request_does_not_consume_tokens(self, clock: ManualClock) -> None:
        """A client retrying hard must still recover on schedule."""
        limiter = RateLimiter(rate_per_minute=60, burst=2, clock=clock)
        limiter.check("k")
        limiter.check("k")
        for _ in range(50):
            assert not limiter.check("k").allowed

        clock.advance(1.0)
        assert limiter.check("k").allowed

    def test_retry_after_reflects_the_actual_deficit(self, clock: ManualClock) -> None:
        limiter = RateLimiter(rate_per_minute=60, burst=1, clock=clock)
        limiter.check("k")
        decision = limiter.check("k", cost=2.0)
        assert not decision.allowed
        assert decision.retry_after_seconds == pytest.approx(2.0, abs=0.01)

    def test_keys_are_independent(self, clock: ManualClock) -> None:
        limiter = RateLimiter(rate_per_minute=60, burst=1, clock=clock)
        assert limiter.check("a").allowed
        assert limiter.check("b").allowed
        assert not limiter.check("a").allowed

    def test_enforce_raises_with_retry_after(self, clock: ManualClock) -> None:
        limiter = RateLimiter(rate_per_minute=60, burst=1, clock=clock)
        limiter.enforce("k")
        with pytest.raises(RateLimited) as exc:
            limiter.enforce("k")
        assert exc.value.retry_after_seconds > 0

    def test_the_store_evicts_long_idle_buckets(self, clock: ManualClock) -> None:
        store = InMemoryBucketStore(max_keys=10)
        limiter = RateLimiter(rate_per_minute=600, burst=10, clock=clock, store=store)
        for index in range(10):
            limiter.check(f"key-{index}")
        clock.advance(7200)
        for index in range(10, 15):
            limiter.check(f"key-{index}")
        assert len(store) < 15

    def test_invalid_configuration_is_refused_at_construction(self, clock: ManualClock) -> None:
        with pytest.raises(ValueError):
            RateLimiter(rate_per_minute=0, burst=5, clock=clock)


class TestLayeredLimits:
    def test_tenant_exhaustion_blocks_a_principal_with_budget(self, clock: ManualClock) -> None:
        layered = LayeredRateLimiter(
            principal=RateLimiter(rate_per_minute=6000, burst=100, clock=clock),
            tenant=RateLimiter(rate_per_minute=60, burst=2, clock=clock),
        )
        layered.enforce(principal_id="alice", tenant_id="acme")
        layered.enforce(principal_id="alice", tenant_id="acme")
        with pytest.raises(RateLimited) as exc:
            layered.enforce(principal_id="alice", tenant_id="acme")
        assert exc.value.details["scope"] == "tenant"

    def test_one_principal_cannot_starve_another(self, clock: ManualClock) -> None:
        layered = LayeredRateLimiter(
            principal=RateLimiter(rate_per_minute=60, burst=2, clock=clock),
            tenant=RateLimiter(rate_per_minute=6000, burst=100, clock=clock),
        )
        layered.enforce(principal_id="noisy", tenant_id="acme")
        layered.enforce(principal_id="noisy", tenant_id="acme")
        with pytest.raises(RateLimited):
            layered.enforce(principal_id="noisy", tenant_id="acme")
        layered.enforce(principal_id="quiet", tenant_id="acme")


class TestCircuitBreaker:
    def test_opens_after_the_failure_threshold(self, clock: ManualClock) -> None:
        breaker = CircuitBreaker(name="p", failure_threshold=3, clock=clock)
        for _ in range(2):
            breaker.record_failure()
        assert breaker.state is CircuitState.CLOSED
        breaker.record_failure()
        assert breaker.state is CircuitState.OPEN
        assert not breaker.allows_request()

    def test_moves_to_half_open_after_the_reset_window(self, clock: ManualClock) -> None:
        breaker = CircuitBreaker(
            name="p", failure_threshold=1, reset_timeout_seconds=30.0, clock=clock
        )
        breaker.record_failure()
        clock.advance(29.0)
        assert breaker.state is CircuitState.OPEN
        clock.advance(2.0)
        assert breaker.state is CircuitState.HALF_OPEN

    def test_half_open_admits_exactly_one_probe(self, clock: ManualClock) -> None:
        breaker = CircuitBreaker(
            name="p", failure_threshold=1, reset_timeout_seconds=1.0, clock=clock
        )
        breaker.record_failure()
        clock.advance(2.0)
        assert breaker.allows_request()
        assert not breaker.allows_request()

    def test_a_failed_probe_reopens_the_circuit(self, clock: ManualClock) -> None:
        breaker = CircuitBreaker(
            name="p", failure_threshold=5, reset_timeout_seconds=1.0, clock=clock
        )
        for _ in range(5):
            breaker.record_failure()
        clock.advance(2.0)
        assert breaker.allows_request()
        breaker.record_failure()
        assert breaker.state is CircuitState.OPEN

    def test_a_successful_probe_closes_the_circuit(self, clock: ManualClock) -> None:
        breaker = CircuitBreaker(
            name="p", failure_threshold=1, reset_timeout_seconds=1.0, clock=clock
        )
        breaker.record_failure()
        clock.advance(2.0)
        breaker.allows_request()
        breaker.record_success()
        assert breaker.state is CircuitState.CLOSED
        assert breaker.allows_request()

    def test_require_closed_raises_when_open(self, clock: ManualClock) -> None:
        breaker = CircuitBreaker(name="p", failure_threshold=1, clock=clock)
        breaker.record_failure()
        with pytest.raises(CircuitOpen):
            breaker.require_closed()


class TestRetry:
    def test_full_jitter_stays_within_the_exponential_ceiling(self) -> None:
        policy = RetryPolicy(base_delay_seconds=1.0, max_delay_seconds=100.0)
        rng = random.Random(7)
        for attempt in range(1, 6):
            ceiling = min(100.0, 1.0 * 2 ** (attempt - 1))
            for _ in range(50):
                assert 0.0 <= policy.delay_for(attempt, rng=rng) <= ceiling

    @pytest.mark.parametrize(
        ("error", "expected"),
        [
            (ProviderError("rate limited", provider="x", retryable=True), True),
            (ProviderError("bad request", provider="x", retryable=False), False),
            (TimeoutExceeded("slow"), True),
            (ValidationError("malformed"), False),
            (ConnectionError("reset"), True),
        ],
    )
    def test_error_classification(self, error: BaseException, expected: bool) -> None:
        assert is_retryable(error) is expected

    async def test_a_transient_failure_is_retried_then_succeeds(self) -> None:
        attempts = 0

        async def flaky() -> str:
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise ProviderError("temporary", provider="x", retryable=True)
            return "recovered"

        async def no_sleep(_seconds: float) -> None:
            return None

        result = await call_with_resilience(
            flaky, retry=RetryPolicy(max_attempts=4), sleep=no_sleep
        )
        assert result == "recovered"
        assert attempts == 3

    async def test_a_terminal_failure_is_not_retried(self) -> None:
        attempts = 0

        async def broken() -> str:
            nonlocal attempts
            attempts += 1
            raise ProviderError("malformed request", provider="x", retryable=False)

        with pytest.raises(ProviderError):
            await call_with_resilience(broken, retry=RetryPolicy(max_attempts=5))
        assert attempts == 1

    async def test_a_slow_call_raises_timeout_exceeded(self) -> None:
        async def slow() -> str:
            await asyncio.sleep(5)
            return "never"

        with pytest.raises(TimeoutExceeded):
            await call_with_resilience(
                slow, retry=RetryPolicy(max_attempts=1), timeout_seconds=0.01
            )

    async def test_repeated_failures_trip_the_breaker(self, clock: ManualClock) -> None:
        breaker = CircuitBreaker(name="p", failure_threshold=2, clock=clock)

        async def always_fails() -> str:
            raise ProviderError("down", provider="p", retryable=True)

        async def no_sleep(_seconds: float) -> None:
            return None

        with pytest.raises(ProviderError):
            await call_with_resilience(
                always_fails, breaker=breaker, retry=RetryPolicy(max_attempts=2), sleep=no_sleep
            )
        assert breaker.state is CircuitState.OPEN

        with pytest.raises(CircuitOpen):
            await call_with_resilience(always_fails, breaker=breaker)


class TestEgress:
    @pytest.fixture
    def guard(self) -> EgressGuard:
        return EgressGuard(("api.anthropic.com", "api.github.com"), resolve=False)

    def test_allowlisted_host_passes(self, guard: EgressGuard) -> None:
        assert guard.check("https://api.anthropic.com/v1/messages").allowed

    def test_subdomains_of_an_allowlisted_host_pass(self, guard: EgressGuard) -> None:
        assert guard.check("https://uploads.api.github.com/x").allowed

    def test_unknown_host_is_refused(self, guard: EgressGuard) -> None:
        decision = guard.check("https://collector.evil.example/steal")
        assert not decision.allowed
        assert "allowlist" in (decision.reason or "")

    def test_plain_http_is_refused_even_for_an_allowlisted_host(self, guard: EgressGuard) -> None:
        assert not guard.check("http://api.github.com/x").allowed

    def test_a_lookalike_host_is_refused(self, guard: EgressGuard) -> None:
        assert not guard.check("https://api.github.com.evil.example/x").allowed

    def test_enforce_raises_a_policy_violation(self, guard: EgressGuard) -> None:
        with pytest.raises(PolicyViolation):
            guard.enforce("https://not-allowed.example/x")

    def test_cloud_metadata_address_is_in_a_blocked_range(self) -> None:
        import ipaddress

        from eap.netops.egress import BLOCKED_NETWORKS

        metadata = ipaddress.ip_address("169.254.169.254")
        assert any(metadata in network for network in BLOCKED_NETWORKS)

    def test_loopback_and_private_ranges_are_blocked(self) -> None:
        import ipaddress

        from eap.netops.egress import BLOCKED_NETWORKS

        for address in ("127.0.0.1", "10.0.0.5", "192.168.1.1", "172.16.0.1"):
            parsed = ipaddress.ip_address(address)
            assert any(parsed in network for network in BLOCKED_NETWORKS)
