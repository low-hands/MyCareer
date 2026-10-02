"""Request-scoped lifecycle around the Main Agent execution graph.

The coordinator owns ingress concerns that are deliberately outside graph
state: request idempotency, stream binding, trace/action invocation context,
turn receipts, and the final commit/delivery envelope.  The graph host still
owns agent semantics; this module only brackets one invocation of them.
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Generic, Protocol, TypeVar
from uuid import uuid4

from career_agent.harness.observability import ACTIVE_TRACE_CONTEXT, TraceRecorder
from career_agent.harness.streaming import (
    ContentDeltaEvent,
    ProgressEvent,
    PublicStreamEvent,
    StreamEventSink,
    TurnCompletedEvent,
    TurnFailedEvent,
    TurnInputResource,
    TurnStartedEvent,
)
from career_agent.storage.turn_receipts import (
    REPLAYED_EVENT_TYPES,
    SQLiteTurnReceiptStore,
    TurnReceipt,
)


STREAM_SINK: ContextVar[StreamEventSink | None] = ContextVar(
    "main_agent_stream_sink",
    default=None,
)
TRACE_CONTEXT = ACTIVE_TRACE_CONTEXT
ACTION_INVOCATION: ContextVar[tuple[str, str | None] | None] = ContextVar(
    "main_agent_action_invocation",
    default=None,
)

TurnResultT = TypeVar("TurnResultT")


class TurnInProgressError(RuntimeError):
    """The request's first attempt has not settled, so it cannot be replayed yet."""

    def __init__(self, request_id: str) -> None:
        super().__init__(f"request {request_id} is still running")
        self.request_id = request_id


@dataclass(frozen=True)
class ReplayedTurn:
    """A repeated request answered from its receipt; nothing executed."""

    turn_id: str
    request_id: str
    events: tuple[PublicStreamEvent, ...]

    @property
    def assistant_message(self) -> str:
        return "".join(
            event.delta
            for event in self.events
            if isinstance(event, ContentDeltaEvent)
        )


class TurnLifecycleHost(Protocol[TurnResultT]):
    """Agent-specific operations bracketed by :class:`TurnCoordinator`."""

    def _emit(self, event: PublicStreamEvent) -> None: ...

    def _run_and_commit_turn(
        self,
        *,
        user_id: str,
        conversation_id: str,
        user_message: str,
        interaction_response: Any | None,
        before_commit: Callable[[TurnResultT], None],
        input_resources: tuple[TurnInputResource, ...],
    ) -> TurnResultT: ...

    def _deliver_reply(
        self, *, result: TurnResultT, conversation_id: str
    ) -> None: ...

    def _record_turn(
        self, *, turn_id: str, conversation_id: str, result: TurnResultT
    ) -> None: ...

    def _deliver_stream_events(
        self, *, result: TurnResultT, turn_id: str, conversation_id: str
    ) -> None: ...

    def _invalidate_episode_reconciliation(self, user_id: str) -> None: ...

    def _record_turn_failed(
        self,
        *,
        turn_id: str,
        conversation_id: str,
        error: Exception,
        reply_delivered: bool,
    ) -> None: ...

    def _emit_turn_failure(
        self,
        *,
        turn_id: str,
        error: Exception,
        reply_delivered: bool,
    ) -> None: ...


class TurnCoordinator(Generic[TurnResultT]):
    """Own one request lifecycle without owning Agent or domain semantics."""

    def __init__(
        self,
        *,
        host: TurnLifecycleHost[TurnResultT],
        receipt_store: SQLiteTurnReceiptStore | None,
        trace_recorder: TraceRecorder | None,
    ) -> None:
        self._host = host
        self._receipt_store = receipt_store
        self._trace_recorder = trace_recorder

    def run(
        self,
        *,
        user_id: str,
        conversation_id: str,
        user_message: str,
        request_id: str | None,
        interaction_response: Any | None,
        event_sink: StreamEventSink | None,
        input_resources: tuple[TurnInputResource, ...],
    ) -> TurnResultT | ReplayedTurn:
        turn_id = uuid4().hex
        if request_id is not None:
            request_id = request_id.strip()
            if not request_id or len(request_id) > 200:
                raise ValueError("request_id must contain 1 to 200 characters")

        receipt_owner: tuple[SQLiteTurnReceiptStore, str] | None = None
        if request_id is not None and self._receipt_store is not None:
            existing = self._receipt_store.begin(
                user_id=user_id,
                conversation_id=conversation_id,
                request_id=request_id,
                turn_id=turn_id,
            )
            if existing is not None:
                return self._replay(existing, event_sink=event_sink)
            receipt_owner = (self._receipt_store, request_id)

        answered: list[PublicStreamEvent] = []
        if receipt_owner is not None:
            event_sink = self._answer_recording_sink(event_sink, answered)
        sink_token = STREAM_SINK.set(event_sink)
        action_token = ACTION_INVOCATION.set((turn_id, request_id))
        trace_token = TRACE_CONTEXT.set(
            (self._trace_recorder, turn_id)
            if self._trace_recorder is not None
            else None
        )
        self._host._emit(TurnStartedEvent(turn_id=turn_id))
        self._host._emit(
            ProgressEvent(
                stage="loading_context",
                message="正在读取对话和职业上下文……",
            )
        )
        reply_delivered = False

        def deliver_reply(result: TurnResultT) -> None:
            nonlocal reply_delivered
            self._host._deliver_reply(
                result=result,
                conversation_id=conversation_id,
            )
            reply_delivered = True

        try:
            result = self._host._run_and_commit_turn(
                user_id=user_id,
                conversation_id=conversation_id,
                user_message=user_message,
                interaction_response=interaction_response,
                before_commit=deliver_reply,
                input_resources=input_resources,
            )
            self._host._record_turn(
                turn_id=turn_id,
                conversation_id=conversation_id,
                result=result,
            )
            self._host._deliver_stream_events(
                result=result,
                turn_id=turn_id,
                conversation_id=conversation_id,
            )
            if receipt_owner is not None:
                self._settle_receipt(
                    receipt_owner,
                    user_id=user_id,
                    conversation_id=conversation_id,
                    turn_id=turn_id,
                    answered=tuple(answered),
                )
            return result
        except Exception as error:
            self._host._invalidate_episode_reconciliation(user_id)
            self._host._record_turn_failed(
                turn_id=turn_id,
                conversation_id=conversation_id,
                error=error,
                reply_delivered=reply_delivered,
            )
            if receipt_owner is not None:
                self._settle_receipt(
                    receipt_owner,
                    user_id=user_id,
                    conversation_id=conversation_id,
                    turn_id=turn_id,
                    answered=None,
                )
            self._host._emit_turn_failure(
                turn_id=turn_id,
                error=error,
                reply_delivered=reply_delivered,
            )
            raise
        finally:
            STREAM_SINK.reset(sink_token)
            TRACE_CONTEXT.reset(trace_token)
            ACTION_INVOCATION.reset(action_token)

    @staticmethod
    def _answer_recording_sink(
        event_sink: StreamEventSink | None,
        answered: list[PublicStreamEvent],
    ) -> StreamEventSink:
        def sink(event: PublicStreamEvent) -> None:
            if isinstance(event, REPLAYED_EVENT_TYPES):
                answered.append(event)
            if event_sink is not None:
                event_sink(event)

        return sink

    @staticmethod
    def _settle_receipt(
        owner: tuple[SQLiteTurnReceiptStore, str],
        *,
        user_id: str,
        conversation_id: str,
        turn_id: str,
        answered: tuple[PublicStreamEvent, ...] | None,
        body_expires_at: datetime | None = None,
    ) -> None:
        store, request_id = owner
        try:
            if answered is None:
                store.fail(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    request_id=request_id,
                    turn_id=turn_id,
                )
            else:
                store.commit(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    request_id=request_id,
                    turn_id=turn_id,
                    events=answered,
                    body_expires_at=body_expires_at,
                )
        except Exception:
            # A receipt is only a retry convenience. It cannot undo a durable
            # business turn if its own best-effort write fails.
            return

    def _replay(
        self,
        receipt: TurnReceipt,
        *,
        event_sink: StreamEventSink | None,
    ) -> ReplayedTurn:
        sink_token = STREAM_SINK.set(event_sink)
        try:
            if receipt.status == "RUNNING":
                self._host._emit(
                    TurnFailedEvent(
                        turn_id=receipt.turn_id,
                        code="TURN_IN_PROGRESS",
                        message="这条请求仍在处理中；稍后重新读取对话即可看到结果。",
                    )
                )
                raise TurnInProgressError(receipt.request_id)
            self._host._emit(TurnStartedEvent(turn_id=receipt.turn_id))
            events = receipt.events
            if receipt.content_status != "available":
                events = (
                    ContentDeltaEvent(
                        delta=(
                            "内容已删除。"
                            if receipt.content_status == "deleted"
                            else "回执正文已过期，请查看历史对话。"
                        ),
                        delivery="synthetic",
                    ),
                    TurnCompletedEvent(turn_id=receipt.turn_id),
                )
            for event in events:
                self._host._emit(event)
        finally:
            STREAM_SINK.reset(sink_token)
        return ReplayedTurn(
            turn_id=receipt.turn_id,
            request_id=receipt.request_id,
            events=events,
        )
