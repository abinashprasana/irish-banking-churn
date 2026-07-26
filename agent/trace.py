"""Structured, JSON-serialisable reasoning trace events."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


TRACE_TYPES = frozenset(
    {"model_thought", "tool_call", "tool_result", "gate_check", "final_output"}
)


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True, slots=True)
class TraceEvent:
    step: int
    type: str
    content: Any
    timestamp: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "type": self.type,
            "content": self.content,
            "timestamp": self.timestamp,
        }


class TraceRecorder:
    def __init__(self, clock: Callable[[], str] = utc_timestamp) -> None:
        self._clock = clock
        self._events: list[TraceEvent] = []

    def log(self, event_type: str, content: Any) -> TraceEvent:
        if event_type not in TRACE_TYPES:
            raise ValueError(f"unsupported trace event type: {event_type}")
        event = TraceEvent(
            step=len(self._events) + 1,
            type=event_type,
            content=content,
            timestamp=self._clock(),
        )
        self._events.append(event)
        return event

    def as_list(self) -> list[dict[str, Any]]:
        return [event.as_dict() for event in self._events]

