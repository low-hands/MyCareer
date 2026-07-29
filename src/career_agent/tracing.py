from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol
from uuid import uuid4


@dataclass(slots=True)
class TraceEvent:
    name: str
    at: datetime = field(default_factory=lambda: datetime.now(UTC))
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RunTrace:
    trace_id: str
    thread_id: str
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime | None = None
    events: list[TraceEvent] = field(default_factory=list)
    error: str | None = None


class TraceSink(Protocol):
    def start(self, thread_id: str) -> RunTrace: ...

    def event(self, trace: RunTrace, name: str, **attributes: Any) -> None: ...

    def finish(self, trace: RunTrace, error: Exception | None = None) -> None: ...


class InMemoryTraceSink:
    """Small default sink; production adapters can export the same events."""

    def __init__(self) -> None:
        self.traces: dict[str, RunTrace] = {}

    def start(self, thread_id: str) -> RunTrace:
        trace = RunTrace(trace_id=uuid4().hex, thread_id=thread_id)
        self.traces[trace.trace_id] = trace
        return trace

    def event(self, trace: RunTrace, name: str, **attributes: Any) -> None:
        trace.events.append(TraceEvent(name=name, attributes=attributes))

    def finish(self, trace: RunTrace, error: Exception | None = None) -> None:
        trace.finished_at = datetime.now(UTC)
        trace.error = str(error) if error else None
