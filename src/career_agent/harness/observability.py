from __future__ import annotations

from contextvars import ContextVar
from datetime import datetime, timezone
from functools import wraps
import hashlib
from threading import Lock
from time import perf_counter
from typing import Any, Callable, Literal, Protocol

from langchain_core.callbacks import BaseCallbackHandler
from pydantic import BaseModel, ConfigDict, Field

from career_agent.harness.capability_steps import notify_capability_step
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
    # A turn that never started: the conversation already had one running.
    # Recorded because the gate that rejects it is process-local, and whether
    # that is a real limitation depends on how often contention actually
    # happens — which nothing currently measures.
    "turn_rejected",
    "capability_failed",
    "presentation_degraded",
    "context_compacted",
    # The complete-request estimate a load took, as numbers. Paired with the
    # run's first model_succeeded it gives estimator against provider count,
    # which compaction events alone recorded only on turns that compacted.
    "context_estimated",
    # A summarizer call that failed, with the conversation's consecutive count
    # and whether compaction is now suspended. The worker traces nothing itself.
    "context_compaction_failed",
    "memory_tombstone_observed",
    "memory_context_observed",
    "memory_proposal_expired",
    "working_notes_oversize",
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


# Bound by the outer turn and inherited by synchronous capability/workflow
# calls.  Keeping it in the harness package lets isolated workers emit model
# telemetry without importing MainAgentRuntime (which would invert the
# dependency and create a cycle).
ACTIVE_TRACE_CONTEXT: ContextVar[tuple[TraceRecorder, str] | None] = ContextVar(
    "active_agent_trace_context",
    default=None,
)


def conversation_trace_key(user_id: str, conversation_id: str) -> str:
    """Pseudonymous join key for events from one owned conversation."""

    return hashlib.sha256(
        f"{user_id}\0{conversation_id}".encode("utf-8")
    ).hexdigest()


def record_active_trace(
    event_type: EventType,
    stage: str,
    *,
    outcome: Literal["started", "succeeded", "failed", "interrupted"],
    duration_ms: int | None = None,
    error_code: str | None = None,
    error_detail: str | None = None,
    recoverable: bool | None = None,
    details: dict[str, Any] | None = None,
    model_call_category: ModelCallCategory | None = None,
) -> None:
    """Best-effort event write for code running inside the active turn."""

    context = ACTIVE_TRACE_CONTEXT.get()
    if context is None:
        return
    recorder, run_id = context
    try:
        recorder.record(
            run_id,
            event_type,
            stage,
            outcome=outcome,
            duration_ms=duration_ms,
            error_code=error_code,
            error_detail=error_detail,
            recoverable=recoverable,
            details=details,
            model_call_category=model_call_category,
        )
    except Exception:
        # Observability cannot become authority over a business operation.
        return


TraceStage = str | Callable[..., str]


def traced_model_call(
    stage: TraceStage,
    *,
    when: Callable[..., bool] | None = None,
):
    """Trace one worker method without persisting any of its input or output.

    ``stage`` may be a fixed internal label or a callable receiving the same
    arguments as the decorated method.  The event carries only that label and
    the worker class name: never prompts, resume/JD text, interview answers, or
    model output.
    """

    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            if when is not None:
                try:
                    should_trace = when(*args, **kwargs)
                except Exception:
                    # A diagnostic predicate cannot become a new failure mode.
                    should_trace = False
                if not should_trace:
                    return function(*args, **kwargs)
            try:
                resolved_stage = (
                    stage(*args, **kwargs) if callable(stage) else stage
                )
            except Exception:
                # Preserve the model call even if a future signature change
                # leaves the more specific stage resolver stale.
                resolved_stage = function.__name__
            safe_stage = str(resolved_stage)
            notify_capability_step(safe_stage, kind="model")
            if ACTIVE_TRACE_CONTEXT.get() is None:
                return function(*args, **kwargs)
            details = {
                "worker": type(args[0]).__name__ if args else function.__qualname__
            }
            started = perf_counter()
            record_active_trace(
                "model_attempt",
                safe_stage,
                outcome="started",
                details=details,
                model_call_category="capability_agent",
            )
            try:
                result = function(*args, **kwargs)
            except Exception as error:
                retryable = getattr(error, "retryable", None)
                record_active_trace(
                    "model_failed",
                    safe_stage,
                    outcome="failed",
                    duration_ms=int((perf_counter() - started) * 1000),
                    error_code=getattr(error, "code", type(error).__name__),
                    # Worker details can contain provider validation fragments
                    # derived from a resume, JD, email, or interview answer.
                    # The stable code is enough for aggregation; keep the
                    # detail structural instead of trusting every producer to
                    # redact its exception payload correctly.
                    error_detail=type(error).__name__,
                    recoverable=retryable if isinstance(retryable, bool) else None,
                    details=details,
                    model_call_category="capability_agent",
                )
                raise
            record_active_trace(
                "model_succeeded",
                safe_stage,
                outcome="succeeded",
                duration_ms=int((perf_counter() - started) * 1000),
                details=details,
                model_call_category="capability_agent",
            )
            return result

        return wrapped

    return decorate


class CapabilityModelTraceCallback(BaseCallbackHandler):
    """Trace each LangChain chat-model request made inside a Deep Agent.

    A Deep Agent invocation can contain several model/tool/model cycles, so a
    decorator around ``agent.invoke`` would undercount them.  This callback is
    attached to the actual ChatOpenAI instance and records one pair per model
    request, while deliberately ignoring prompts, messages, generations and
    token content supplied by LangChain.
    """

    def __init__(self, *, stage: str, worker: str) -> None:
        self._stage = stage
        self._worker = worker
        self._started: dict[object, float] = {}
        self._requests = 0
        self._lock = Lock()

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        *,
        run_id: object,
        **kwargs: Any,
    ) -> None:
        del serialized, messages, kwargs
        with self._lock:
            if run_id in self._started:
                return
            self._started[run_id] = perf_counter()
            self._requests += 1
            requests = self._requests
        notify_capability_step(self._stage, kind="model", index=requests)
        if ACTIVE_TRACE_CONTEXT.get() is None:
            return
        record_active_trace(
            "model_attempt",
            self._stage,
            outcome="started",
            details={"worker": self._worker},
            model_call_category="capability_agent",
        )

    def on_llm_end(self, response: Any, *, run_id: object, **kwargs: Any) -> None:
        del response, kwargs
        with self._lock:
            started = self._started.pop(run_id, None)
        if started is None:
            return
        record_active_trace(
            "model_succeeded",
            self._stage,
            outcome="succeeded",
            duration_ms=int((perf_counter() - started) * 1000),
            details={"worker": self._worker},
            model_call_category="capability_agent",
        )

    def on_llm_error(
        self, error: BaseException, *, run_id: object, **kwargs: Any
    ) -> None:
        del kwargs
        with self._lock:
            started = self._started.pop(run_id, None)
        if started is None:
            return
        notify_capability_step(self._stage, kind="retry")
        if ACTIVE_TRACE_CONTEXT.get() is None:
            return
        retryable = getattr(error, "retryable", None)
        record_active_trace(
            "model_failed",
            self._stage,
            outcome="failed",
            duration_ms=int((perf_counter() - started) * 1000),
            error_code=getattr(error, "code", type(error).__name__),
            error_detail=type(error).__name__,
            recoverable=retryable if isinstance(retryable, bool) else None,
            details={"worker": self._worker},
            model_call_category="capability_agent",
        )


class CapabilityToolStepCallback(BaseCallbackHandler):
    """Announce each LangChain tool a Deep Agent runs, by tool name only.

    Passed through ``invoke(config=...)`` so it inherits to the agent's tool
    nodes; provider-hosted tools such as Responses API web search never run
    as LangChain tools and so never reach it.
    """

    def __init__(self, *, stage: str) -> None:
        self._stage = stage

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        **kwargs: Any,
    ) -> None:
        del input_str
        name = kwargs.get("name") or serialized.get("name")
        if not isinstance(name, str) or not name:
            return
        notify_capability_step(f"{self._stage}.{name}", kind="tool")


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
