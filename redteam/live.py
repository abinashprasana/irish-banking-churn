"""Live mode helpers: request pacing, a run budget, and token usage recording.

Everything here wraps the existing live client from ``agent.loop.create_live_client``.
The API key is read by ``agent.loop.resolve_groq_api_key`` from the environment
only. Nothing in this module prints, stores or logs it.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import threading
import time
from typing import Any

from agent.rate_limits import (
    DAILY_LIMIT_MESSAGE,
    DAILY_REQUEST_CAP,
    GEMINI_DAILY_REQUEST_CAP,
    GEMINI_REQUESTS_PER_MINUTE,
    PROVIDER_REQUESTS_PER_MINUTE,
    InMemoryRequestQuota,
    RateLimitSafetyError,
)


QUOTA_STATE_PATH = Path(__file__).resolve().parents[1] / ".redteam_quota.json"
PROVIDER_LIMITS = {
    "groq": (PROVIDER_REQUESTS_PER_MINUTE, DAILY_REQUEST_CAP),
    "gemini": (GEMINI_REQUESTS_PER_MINUTE, GEMINI_DAILY_REQUEST_CAP),
}
# Keep this many tokens of headroom in the provider's per minute token window.
# Groq's free tier allows 8,000 tokens per minute and one agent turn resends
# 3,500 to 4,500 prompt tokens, so wait for the window once fewer than 5,000 remain.
TOKEN_HEADROOM = 5_000
# A 429 for a per minute limit is retried after the provider's retry-after wait.
MINUTE_LIMIT_RETRIES = 4
MAX_RETRY_WAIT_SECONDS = 90.0


class BudgetExhausted(RuntimeError):
    """Raised before a request would exceed the --max-requests budget."""


def _parse_reset(value: str | None) -> float:
    """Parse Groq reset strings such as '7.66s', '1m2.5s' or '120ms'."""

    if not value:
        return 0.0
    total = 0.0
    for amount, unit in re.findall(r"([0-9.]+)(ms|h|m|s)", value):
        total += float(amount) * {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}[unit]
    return total


@dataclass
class UsageLedger:
    """Per request usage taken from the API response usage fields and headers."""

    requests: list[dict[str, Any]] = field(default_factory=list)
    remaining_tokens: int | None = None
    token_reset_seconds: float = 0.0
    observed_at: float = 0.0

    def record(self, response: Any, latency: float, headers: Any | None) -> None:
        # Groq returns SDK objects and Gemini returns plain dictionaries.
        def field(value: Any, name: str) -> Any:
            return value.get(name) if isinstance(value, dict) else getattr(value, name, None)

        usage = field(response, "usage")
        self.requests.append(
            {
                "model": field(response, "model"),
                "prompt_tokens": field(usage, "prompt_tokens"),
                "completion_tokens": field(usage, "completion_tokens"),
                "total_tokens": field(usage, "total_tokens"),
                "latency_seconds": round(latency, 3),
            }
        )
        if headers is not None:
            remaining = headers.get("x-ratelimit-remaining-tokens")
            if remaining is not None and str(remaining).isdigit():
                self.remaining_tokens = int(remaining)
                self.token_reset_seconds = _parse_reset(headers.get("x-ratelimit-reset-tokens"))
                self.observed_at = time.monotonic()

    def token_wait(self) -> float:
        if self.remaining_tokens is None or self.remaining_tokens >= TOKEN_HEADROOM:
            return 0.0
        elapsed = time.monotonic() - self.observed_at
        return max(0.0, self.token_reset_seconds - elapsed)

    def totals(self) -> dict[str, Any]:
        def total(key: str) -> int:
            return sum(item[key] or 0 for item in self.requests)

        return {
            "successful_responses": len(self.requests),
            "prompt_tokens": total("prompt_tokens"),
            "completion_tokens": total("completion_tokens"),
            "total_tokens": total("total_tokens"),
            "models": sorted({item["model"] for item in self.requests if item["model"]}),
        }


class PacedQuota(InMemoryRequestQuota):
    """Waits at the provider's per minute limit instead of failing, keeps its daily
    cap across runs in a local file, and stops at the run's --max-requests budget."""

    def __init__(
        self,
        max_requests: int,
        ledger: UsageLedger,
        *,
        state_path: Path | None = None,
        sleep: Any = time.sleep,
        provider: str = "groq",
        scope: str | None = None,
    ) -> None:
        per_minute, per_day = PROVIDER_LIMITS[provider]
        super().__init__(requests_per_minute=per_minute, daily_request_cap=per_day)
        if state_path is None:
            # Provider quotas are per model, so each model keeps its own daily count.
            suffix = f"{provider}_{scope}" if scope else provider
            suffix = re.sub(r"[^A-Za-z0-9._-]", "_", suffix)
            state_path = QUOTA_STATE_PATH.with_name(f".redteam_quota_{suffix}.json")
        if max_requests < 1:
            raise ValueError("max_requests must be positive")
        self.max_requests = max_requests
        self.used = 0
        self._ledger = ledger
        self._state_path = state_path
        self._sleep = sleep
        self._pace_lock = threading.Lock()

    def _load_day(self) -> dict[str, Any]:
        today = datetime.now(timezone.utc).date().isoformat()
        try:
            state = json.loads(self._state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            state = {}
        if state.get("utc_day") != today:
            state = {"utc_day": today, "requests": 0}
        return state

    def reserve_request(self) -> None:
        with self._pace_lock:
            state = self._load_day()
            if state["requests"] >= self.daily_request_cap:
                raise RateLimitSafetyError(DAILY_LIMIT_MESSAGE)
            if self.used >= self.max_requests:
                raise BudgetExhausted(f"--max-requests budget of {self.max_requests} reached")
            while True:
                now = self._monotonic()
                while self._minute_requests and now - self._minute_requests[0] >= 60.0:
                    self._minute_requests.popleft()
                if len(self._minute_requests) < self.requests_per_minute:
                    break
                self._sleep(60.0 - (now - self._minute_requests[0]) + 0.05)
            wait = self._ledger.token_wait()
            if wait:
                self._sleep(wait)
            self._minute_requests.append(self._monotonic())
            self.used += 1
            state["requests"] += 1
            self._state_path.write_text(json.dumps(state), encoding="utf-8")


def _minute_limit_wait(exc: Exception) -> float | None:
    """Seconds to wait when a 429 is a per minute limit; None for any other error."""

    if getattr(exc, "status_code", None) != 429 or "per minute" not in str(exc):
        return None
    response = getattr(exc, "response", None)
    header = getattr(response, "headers", {}).get("retry-after") if response is not None else None
    match = re.search(r"try again in ([0-9.]+)s", str(exc))
    for value in (header, match.group(1) if match else None):
        try:
            if value is not None:
                return min(float(value) + 1.0, MAX_RETRY_WAIT_SECONDS)
        except ValueError:
            continue
    return 10.0


class _RecordingCompletions:
    """Stands in for the SDK completions object so usage and rate headers are kept."""

    def __init__(
        self,
        completions: Any,
        ledger: UsageLedger,
        guarded: Any | None = None,
        sleep: Any = time.sleep,
    ) -> None:
        self._completions = completions
        self._ledger = ledger
        self._guarded = guarded
        self._sleep = sleep

    def _call(self, **kwargs: Any) -> Any:
        started = time.monotonic()
        raw_api = getattr(self._completions, "with_raw_response", None)
        if raw_api is not None:
            raw = raw_api.create(**kwargs)
            response, headers = raw.parse(), raw.headers
        else:
            response, headers = self._completions.create(**kwargs), None
        self._ledger.record(response, time.monotonic() - started, headers)
        return response

    def create(self, **kwargs: Any) -> Any:
        for attempt in range(MINUTE_LIMIT_RETRIES + 1):
            try:
                return self._call(**kwargs)
            except Exception as exc:
                wait = _minute_limit_wait(exc)
                if wait is None or attempt == MINUTE_LIMIT_RETRIES:
                    raise
                self._sleep(wait)
                # Each retry is a real request, so it goes through the same budget.
                if self._guarded is not None:
                    self._guarded._quota_guard.reserve_request()
                    self._guarded.request_count += 1
        raise RuntimeError("unreachable")


def attach_usage_recorder(live_client: Any, ledger: UsageLedger) -> Any:
    """Wrap the SDK completions inside a GroqLiveClient without touching agent code."""

    guarded = live_client.chat.completions
    guarded._completions = _RecordingCompletions(guarded._completions, ledger, guarded)
    return live_client


def contains_key_material(serialised: str) -> bool:
    return "gsk_" in serialised
