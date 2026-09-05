"""Retry, circuit breaking and timeouts for calls that leave the process.

Model providers fail in three distinguishable ways and each wants a different response:

* **Transient** (429, 503, connection reset) — retry with backoff and full jitter.
* **Persistent** (the provider is down) — stop retrying and shed load, or every replica
  spends its whole timeout budget queueing against a dead dependency. That is the circuit
  breaker's job.
* **Terminal** (400, 401, content policy) — do not retry. A malformed request stays
  malformed, and retrying an auth failure just burns the rate limit.

Backoff uses *full* jitter (``random(0, base * 2^n)``) rather than a fixed exponential.
Fixed backoff synchronises every client that failed at the same instant into retrying at
the same instant, which is how a recovering service gets knocked over a second time.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import TypeVar

from eap.platform.clock import SYSTEM_CLOCK, Clock
from eap.platform.errors import CircuitOpen, PlatformError, ProviderError, TimeoutExceeded
from eap.platform.telemetry import get_logger

T = TypeVar("T")
log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    max_attempts: int = 3
    base_delay_seconds: float = 0.25
    max_delay_seconds: float = 8.0

    def delay_for(self, attempt: int, *, rng: random.Random | None = None) -> float:
        """Full jitter. ``attempt`` is 1-based; the delay follows attempt 1's failure."""
        ceiling = min(self.max_delay_seconds, self.base_delay_seconds * (2 ** (attempt - 1)))
        source = rng or random
        return source.uniform(0.0, ceiling)


def is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, ProviderError):
        return exc.retryable
    if isinstance(exc, (TimeoutExceeded, asyncio.TimeoutError)):
        return True
    if isinstance(exc, PlatformError):
        return exc.status_code in (429, 502, 503, 504)
    return isinstance(exc, (ConnectionError, OSError))


class CircuitState(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Three-state breaker guarding one dependency.

    CLOSED passes traffic and counts consecutive failures. At the threshold it OPENs and
    fails fast for ``reset_timeout``. It then moves to HALF_OPEN and admits a single probe:
    success closes it, failure reopens it. Admitting exactly one probe — rather than
    reopening the floodgates — is what stops a recovering dependency being re-flattened.
    """

    def __init__(
        self,
        *,
        name: str,
        failure_threshold: int = 5,
        reset_timeout_seconds: float = 30.0,
        clock: Clock = SYSTEM_CLOCK,
    ) -> None:
        self._name = name
        self._threshold = failure_threshold
        self._reset_timeout = reset_timeout_seconds
        self._clock = clock
        self._state = CircuitState.CLOSED
        self._failures = 0
        self._opened_at = 0.0
        self._probe_in_flight = False

    @property
    def state(self) -> CircuitState:
        if self._state is CircuitState.OPEN:
            if (self._clock.monotonic() - self._opened_at) >= self._reset_timeout:
                self._state = CircuitState.HALF_OPEN
                self._probe_in_flight = False
        return self._state

    @property
    def name(self) -> str:
        return self._name

    def allows_request(self) -> bool:
        state = self.state
        if state is CircuitState.CLOSED:
            return True
        if state is CircuitState.OPEN:
            return False
        if self._probe_in_flight:
            return False
        self._probe_in_flight = True
        return True

    def record_success(self) -> None:
        self._failures = 0
        self._probe_in_flight = False
        if self._state is not CircuitState.CLOSED:
            log.info("circuit.closed", circuit=self._name)
        self._state = CircuitState.CLOSED

    def record_failure(self) -> None:
        self._failures += 1
        self._probe_in_flight = False
        if self._state is CircuitState.HALF_OPEN or self._failures >= self._threshold:
            if self._state is not CircuitState.OPEN:
                log.warning("circuit.opened", circuit=self._name, failures=self._failures)
            self._state = CircuitState.OPEN
            self._opened_at = self._clock.monotonic()

    def require_closed(self) -> None:
        if not self.allows_request():
            raise CircuitOpen(
                f"circuit '{self._name}' is open; shedding load while it recovers",
                circuit=self._name,
                retry_after_seconds=round(
                    max(0.0, self._reset_timeout - (self._clock.monotonic() - self._opened_at)), 2
                ),
            )


async def call_with_resilience(
    operation: Callable[[], Awaitable[T]],
    *,
    breaker: CircuitBreaker | None = None,
    retry: RetryPolicy | None = None,
    timeout_seconds: float | None = None,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    rng: random.Random | None = None,
) -> T:
    """Run ``operation`` under a timeout, a retry policy and a circuit breaker.

    The three compose in that order: the timeout bounds one attempt, the retry policy
    bounds the set of attempts, and the breaker decides whether to attempt at all.
    """
    policy = retry or RetryPolicy()
    last_error: BaseException | None = None

    for attempt in range(1, policy.max_attempts + 1):
        if breaker is not None:
            breaker.require_closed()
        try:
            if timeout_seconds is None:
                result = await operation()
            else:
                try:
                    result = await asyncio.wait_for(operation(), timeout=timeout_seconds)
                except TimeoutError as exc:
                    raise TimeoutExceeded(
                        f"operation exceeded {timeout_seconds}s", timeout_seconds=timeout_seconds
                    ) from exc
        except Exception as exc:
            last_error = exc
            if breaker is not None:
                breaker.record_failure()
            if not is_retryable(exc) or attempt == policy.max_attempts:
                raise
            delay = policy.delay_for(attempt, rng=rng)
            log.warning(
                "call.retrying",
                attempt=attempt,
                max_attempts=policy.max_attempts,
                delay_seconds=round(delay, 3),
                error=type(exc).__name__,
            )
            await sleep(delay)
        else:
            if breaker is not None:
                breaker.record_success()
            return result

    assert last_error is not None
    raise last_error
