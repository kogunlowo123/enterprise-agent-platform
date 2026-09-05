"""Rate limiting.

Token bucket, because it is the only common algorithm that expresses both a sustained rate
and a burst allowance in two numbers a capacity planner can reason about. A fixed window
lets a caller send two full windows' worth of traffic across a boundary instant; a sliding
log costs memory proportional to traffic. The bucket costs two floats per key.

Buckets are keyed per principal *and* per tenant, and both must have capacity. One noisy
service account should not be able to consume its whole tenant's allocation, and one busy
tenant should not affect another.

State is in-process. That is correct for a single instance and wrong behind a load
balancer, where N replicas each grant the full limit. :class:`RateLimiter` therefore takes
a backing store, and a Redis-backed store is the production swap — the algorithm does not
change, only where the two floats live.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from eap.platform.clock import SYSTEM_CLOCK, Clock
from eap.platform.errors import RateLimited


@dataclass(slots=True)
class Bucket:
    tokens: float
    updated_at: float


@dataclass(frozen=True, slots=True)
class LimitDecision:
    allowed: bool
    key: str
    remaining: float
    retry_after_seconds: float = 0.0


class BucketStore(Protocol):
    def get(self, key: str) -> Bucket | None: ...

    def put(self, key: str, bucket: Bucket) -> None: ...


class InMemoryBucketStore:
    """Process-local buckets with opportunistic eviction.

    Full buckets are indistinguishable from absent ones, so any bucket that has refilled
    completely can be dropped. Without this, the map grows once per distinct principal and
    never shrinks — a slow leak that only shows up after a month in production.
    """

    def __init__(self, *, max_keys: int = 100_000) -> None:
        self._buckets: dict[str, Bucket] = {}
        self._max_keys = max_keys

    def get(self, key: str) -> Bucket | None:
        return self._buckets.get(key)

    def put(self, key: str, bucket: Bucket) -> None:
        self._buckets[key] = bucket
        if len(self._buckets) > self._max_keys:
            self._evict_full(bucket.updated_at)

    def _evict_full(self, now: float) -> None:
        stale = [k for k, b in self._buckets.items() if now - b.updated_at > 3600]
        for key in stale[: len(stale) // 2 or 1]:
            self._buckets.pop(key, None)

    def __len__(self) -> int:
        return len(self._buckets)


class RateLimiter:
    """Token bucket over a pluggable store."""

    def __init__(
        self,
        *,
        rate_per_minute: int,
        burst: int,
        clock: Clock = SYSTEM_CLOCK,
        store: BucketStore | None = None,
    ) -> None:
        if rate_per_minute <= 0 or burst <= 0:
            raise ValueError("rate_per_minute and burst must both be positive")
        self._refill_per_second = rate_per_minute / 60.0
        self._capacity = float(burst)
        self._clock = clock
        self._store = store or InMemoryBucketStore()

    def check(self, key: str, *, cost: float = 1.0) -> LimitDecision:
        """Consume ``cost`` tokens if available. Costly operations can charge more than one."""
        now = self._clock.monotonic()
        bucket = self._store.get(key) or Bucket(tokens=self._capacity, updated_at=now)

        elapsed = max(0.0, now - bucket.updated_at)
        tokens = min(self._capacity, bucket.tokens + elapsed * self._refill_per_second)

        if tokens >= cost:
            self._store.put(key, Bucket(tokens=tokens - cost, updated_at=now))
            return LimitDecision(allowed=True, key=key, remaining=round(tokens - cost, 3))

        # Do not consume on refusal: a client that retries hard would otherwise hold its own
        # bucket permanently empty and never recover.
        self._store.put(key, Bucket(tokens=tokens, updated_at=now))
        deficit = cost - tokens
        return LimitDecision(
            allowed=False,
            key=key,
            remaining=round(tokens, 3),
            retry_after_seconds=round(deficit / self._refill_per_second, 3),
        )

    def enforce(self, key: str, *, cost: float = 1.0) -> LimitDecision:
        decision = self.check(key, cost=cost)
        if not decision.allowed:
            raise RateLimited(
                f"rate limit exceeded for {key}",
                retry_after_seconds=decision.retry_after_seconds,
                key=key,
            )
        return decision


class LayeredRateLimiter:
    """Applies a per-principal and a per-tenant limit together.

    Both are checked before either is consumed, so a request rejected by the tenant limit
    does not silently burn the principal's allowance.
    """

    def __init__(self, *, principal: RateLimiter, tenant: RateLimiter) -> None:
        self._principal = principal
        self._tenant = tenant

    def enforce(self, *, principal_id: str, tenant_id: str, cost: float = 1.0) -> LimitDecision:
        tenant_check = self._tenant.check(f"tenant:{tenant_id}", cost=0.0)
        principal_check = self._principal.check(f"principal:{principal_id}", cost=0.0)
        for decision, scope in ((tenant_check, "tenant"), (principal_check, "principal")):
            if decision.remaining < cost:
                retry = round((cost - decision.remaining) / max(1e-9, cost), 3)
                raise RateLimited(
                    f"{scope} rate limit exceeded",
                    retry_after_seconds=max(retry, decision.retry_after_seconds, 1.0),
                    scope=scope,
                    key=decision.key,
                )
        self._tenant.enforce(f"tenant:{tenant_id}", cost=cost)
        return self._principal.enforce(f"principal:{principal_id}", cost=cost)
