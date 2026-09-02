from __future__ import annotations

from datetime import datetime, timezone
from threading import Lock
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from career_agent.security.redaction import redact, redact_text


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
    "turn_completed",
    "turn_failed",
    "capability_failed",
    "presentation_degraded",
]

ModelCallCategory = Literal[
    "orchestrator_decision",
    "capability_agent",
    "planner",
    "evaluator",
    "writer",
    "legacy_router",
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
    model_call_category: ModelCallCategory | None = None


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
        model_call_category: ModelCallCategory | None = None,
    ) -> RunEvent: ...

    def snapshot(self, run_id: str) -> RunTrace: ...


def safe_trace_fields(
    *,
    details: dict[str, Any] | None,
    error_detail: str | None,
) -> tuple[dict[str, Any], str | None]:
    """Apply the mandatory storage boundary for every trace implementation."""

    safe_details = redact(details or {})
    safe_error = (
        redact_text(error_detail)[:2000] if error_detail is not None else None
    )
    return safe_details, safe_error


def validate_model_call_category(
    event_type: EventType,
    model_call_category: ModelCallCategory | None,
) -> None:
    """Enforce classification at the new-write boundary, not while reading v1."""

    is_model_event = event_type in {
        "model_attempt",
        "model_succeeded",
        "model_failed",
    }
    if is_model_event != (model_call_category is not None):
        raise ValueError(
            "model events require model_call_category and non-model events "
            "must not carry it"
        )


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
        model_call_category: ModelCallCategory | None = None,
    ) -> RunEvent:
        validate_model_call_category(event_type, model_call_category)
        safe_details, safe_error = safe_trace_fields(
            details=details,
            error_detail=error_detail,
        )
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
                details=safe_details,
                error_code=error_code,
                error_detail=safe_error,
                recoverable=recoverable,
                model_call_category=model_call_category,
            )
            events.append(event)
            return event

    def snapshot(self, run_id: str) -> RunTrace:
        with self._lock:
            return RunTrace(run_id=run_id, events=tuple(self._events.get(run_id, ())))

    def restore(self, trace: RunTrace) -> None:
        with self._lock:
            self._events[trace.run_id] = list(trace.events)

    def forget(self, run_id: str) -> None:
        """Release one run's events.

        This recorder accumulates every event for the process lifetime, so a
        caller that bounds its own per-run caches has to be able to release these
        too. Safe only where the events are already on a persisted record;
        ``restore`` puts them back.
        """
        with self._lock:
            self._events.pop(run_id, None)


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
        model_call_category: ModelCallCategory | None = None,
    ) -> RunEvent:
        validate_model_call_category(event_type, model_call_category)
        safe_details, safe_error = safe_trace_fields(
            details=details,
            error_detail=error_detail,
        )
        return RunEvent(
            run_id=run_id,
            sequence=1,
            event_type=event_type,
            stage=stage,
            attempt=attempt,
            occurred_at=datetime.now(timezone.utc),
            duration_ms=duration_ms,
            outcome=outcome,
            details=safe_details,
            error_code=error_code,
            error_detail=safe_error,
            recoverable=recoverable,
            model_call_category=model_call_category,
        )

    def snapshot(self, run_id: str) -> RunTrace:
        return RunTrace(run_id=run_id)


def model_call_counts(trace: RunTrace) -> dict[ModelCallCategory, int]:
    """Count recorded call attempts once, independently from tool delegation.

    Attempts are the stable denominator: success and failure are outcomes of
    the same call and must not double-count it. The mapping is deliberately
    sparse: an absent category means this trace contains no observation for it,
    not that the corresponding component made zero calls. In particular, most
    capability workers do not yet receive a TraceRecorder, so returning
    ``capability_agent: 0`` would turn missing instrumentation into a false
    operational claim.
    """

    counts: dict[ModelCallCategory, int] = {}
    for event in trace.events:
        if event.event_type == "model_attempt" and event.model_call_category is not None:
            counts[event.model_call_category] = (
                counts.get(event.model_call_category, 0) + 1
            )
    return counts
