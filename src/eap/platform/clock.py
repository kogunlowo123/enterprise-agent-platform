"""Injectable time.

Rate limiters, circuit breakers, budget windows and token expiry all depend on the clock.
If they read ``time.monotonic()`` directly, their tests can only be written with sleeps,
which makes the suite slow and flaky. Every component that needs time takes a ``Clock``,
and tests hand it a ``ManualClock`` they advance by hand.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Protocol, runtime_checkable


@runtime_checkable
class Clock(Protocol):
    def monotonic(self) -> float:
        """Seconds from an arbitrary origin. Never goes backwards. Use for durations."""

    def now(self) -> datetime:
        """Current wall-clock time, timezone-aware UTC. Use for timestamps on records."""


class SystemClock:
    """The real clock. The only implementation that reaches production."""

    __slots__ = ()

    def monotonic(self) -> float:
        return time.monotonic()

    def now(self) -> datetime:
        return datetime.now(UTC)


class ManualClock:
    """A clock the caller drives. Test fixture, not a stub: the arithmetic is real."""

    __slots__ = ("_monotonic", "_wall")

    def __init__(self, *, start: datetime | None = None) -> None:
        self._wall = start or datetime(2026, 1, 1, tzinfo=UTC)
        self._monotonic = 0.0

    def monotonic(self) -> float:
        return self._monotonic

    def now(self) -> datetime:
        return self._wall

    def advance(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError("a monotonic clock cannot move backwards")
        self._monotonic += seconds
        self._wall = datetime.fromtimestamp(self._wall.timestamp() + seconds, tz=UTC)


SYSTEM_CLOCK = SystemClock()
