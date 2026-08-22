from __future__ import annotations

from datetime import datetime, timezone
from threading import Lock
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field


EventType = Literal[
    "run_started",
    "run_resumed",
    "run_completed",
    "run_interrupted",
    "node_started",
    "node_completed",
    "node_failed",
    "node_interrupted",
    "model_attempt",
    "model_succeeded",
    "model_failed",
]


class RunEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    sequence: int = Field(ge=1)
    event_type: EventType
    stage: str
    attempt: int | None = Field(default=None, ge=1)
    occurred_at: datetime
    duration_ms: int | None = Field(default=None, ge=0)
    outcome: Literal["started", "succeeded", "failed", "interrupted"]
    details: dict[str, Any] = Field(default_factory=dict)
    error_code: str | None = None
    error_detail: str | None = None
    recoverable: bool | None = None


class RunTrace(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    events: tuple[RunEvent, ...] = ()


class TraceRecorder(Protocol):
    def record(
        self,
        run_id: str,
        event_type: EventType,
        stage: str,
        *,
        attempt: int | None = None,
        duration_ms: int | None = None,
        outcome: Literal["started", "succeeded", "failed", "interrupted"] = "started",
        details: dict[str, Any] | None = None,
        error_code: str | None = None,
        error_detail: str | None = None,
        recoverable: bool | None = None,
    ) -> RunEvent: ...

    def snapshot(self, run_id: str) -> RunTrace: ...


class InMemoryTraceRecorder:
    def __init__(self) -> None:
        self._events: dict[str, list[RunEvent]] = {}
        self._lock = Lock()

    def record(
        self,
        run_id: str,
        event_type: EventType,
        stage: str,
        *,
        attempt: int | None = None,
        duration_ms: int | None = None,
        outcome: Literal["started", "succeeded", "failed", "interrupted"] = "started",
        details: dict[str, Any] | None = None,
        error_code: str | None = None,
        error_detail: str | None = None,
        recoverable: bool | None = None,
    ) -> RunEvent:
        with self._lock:
            events = self._events.setdefault(run_id, [])
            event = RunEvent(
                run_id=run_id,
                sequence=len(events) + 1,
                event_type=event_type,
                stage=stage,
                attempt=attempt,
                occurred_at=datetime.now(timezone.utc),
                duration_ms=duration_ms,
                outcome=outcome,
                details=details or {},
                error_code=error_code,
                error_detail=error_detail,
                recoverable=recoverable,
            )
            events.append(event)
            return event

    def snapshot(self, run_id: str) -> RunTrace:
        with self._lock:
            return RunTrace(run_id=run_id, events=tuple(self._events.get(run_id, ())))

    def restore(self, trace: RunTrace) -> None:
        with self._lock:
            self._events[trace.run_id] = list(trace.events)


class NoopTraceRecorder:
    def record(
        self,
        run_id: str,
        event_type: EventType,
        stage: str,
        *,
        attempt: int | None = None,
        duration_ms: int | None = None,
        outcome: Literal["started", "succeeded", "failed", "interrupted"] = "started",
        details: dict[str, Any] | None = None,
        error_code: str | None = None,
        error_detail: str | None = None,
        recoverable: bool | None = None,
    ) -> RunEvent:
        return RunEvent(
            run_id=run_id,
            sequence=1,
            event_type=event_type,
            stage=stage,
            attempt=attempt,
            occurred_at=datetime.now(timezone.utc),
            duration_ms=duration_ms,
            outcome=outcome,
            details=details or {},
            error_code=error_code,
            error_detail=error_detail,
            recoverable=recoverable,
        )

    def snapshot(self, run_id: str) -> RunTrace:
        return RunTrace(run_id=run_id)
