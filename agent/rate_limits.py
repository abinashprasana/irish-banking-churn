"""Process-local safety limits for the shared Groq free-tier quota."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, MutableMapping
from datetime import datetime, timezone
import threading
import time


PROVIDER_REQUESTS_PER_MINUTE = 30
PROVIDER_REQUESTS_PER_DAY = 1_000
DAILY_SAFETY_MARGIN = 50
DAILY_REQUEST_CAP = PROVIDER_REQUESTS_PER_DAY - DAILY_SAFETY_MARGIN
SESSION_RUN_CAP = 5

DAILY_LIMIT_MESSAGE = "daily demo limit reached, please try again tomorrow"
MINUTE_LIMIT_MESSAGE = "demo request limit reached, please wait a minute and try again"
SESSION_LIMIT_MESSAGE = "session demo limit reached (5 runs), please try again later"


class RateLimitSafetyError(RuntimeError):
    """Raised before a request would exceed a local provider safety limit."""


class SessionRunLimitError(RateLimitSafetyError):
    """Raised before a visitor would exceed the per-session run cap."""


class InMemoryRequestQuota:
    """Thread-safe request counter that resets each UTC day.

    This intentionally mirrors the provider's request limits rather than agent
    runs: every physical SDK request, including a retry, consumes one slot.
    """

    def __init__(
        self,
        *,
        requests_per_minute: int = PROVIDER_REQUESTS_PER_MINUTE,
        daily_request_cap: int = DAILY_REQUEST_CAP,
        now: Callable[[], datetime] | None = None,
        monotonic: Callable[[], float] | None = None,
    ) -> None:
        if requests_per_minute < 1 or daily_request_cap < 1:
            raise ValueError("quota limits must be positive")
        self.requests_per_minute = requests_per_minute
        self.daily_request_cap = daily_request_cap
        self._now = now or (lambda: datetime.now(timezone.utc))
        self._monotonic = monotonic or time.monotonic
        self._lock = threading.Lock()
        self._utc_day = self._normalised_now().date()
        self._daily_requests = 0
        self._minute_requests: deque[float] = deque()

    def _normalised_now(self) -> datetime:
        value = self._now()
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    def _roll_day_if_needed(self) -> None:
        current_day = self._normalised_now().date()
        if current_day != self._utc_day:
            self._utc_day = current_day
            self._daily_requests = 0
            self._minute_requests.clear()

    def reserve_request(self) -> None:
        """Atomically reserve one physical provider request."""

        with self._lock:
            self._roll_day_if_needed()
            if self._daily_requests >= self.daily_request_cap:
                raise RateLimitSafetyError(DAILY_LIMIT_MESSAGE)

            now_monotonic = self._monotonic()
            while (
                self._minute_requests
                and now_monotonic - self._minute_requests[0] >= 60.0
            ):
                self._minute_requests.popleft()
            if len(self._minute_requests) >= self.requests_per_minute:
                raise RateLimitSafetyError(MINUTE_LIMIT_MESSAGE)

            self._minute_requests.append(now_monotonic)
            self._daily_requests += 1

    def ensure_run_available(self) -> None:
        """Block a new agent run once the daily safety threshold is reached."""

        with self._lock:
            self._roll_day_if_needed()
            if self._daily_requests >= self.daily_request_cap:
                raise RateLimitSafetyError(DAILY_LIMIT_MESSAGE)

    def snapshot(self) -> dict[str, int | str]:
        """Return non-secret counter state for diagnostics and UI copy."""

        with self._lock:
            self._roll_day_if_needed()
            return {
                "utc_day": self._utc_day.isoformat(),
                "daily_requests": self._daily_requests,
                "daily_requests_remaining": max(
                    0, self.daily_request_cap - self._daily_requests
                ),
                "minute_requests": len(self._minute_requests),
            }


GLOBAL_REQUEST_QUOTA = InMemoryRequestQuota()


def reserve_session_run(
    session_state: MutableMapping[str, object],
    *,
    key: str = "retention_live_run_count",
) -> int:
    """Reserve one live agent run and return the number remaining."""

    raw_count = session_state.get(key, 0)
    if isinstance(raw_count, bool) or not isinstance(raw_count, int):
        raise TypeError(f"{key} must be an integer")
    if raw_count >= SESSION_RUN_CAP:
        raise SessionRunLimitError(SESSION_LIMIT_MESSAGE)
    next_count = raw_count + 1
    session_state[key] = next_count
    return SESSION_RUN_CAP - next_count
