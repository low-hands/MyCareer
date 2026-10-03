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
from typing import Callable, Generic, Literal, Protocol, TypeVar
from uuid import uuid4

from career_agent.harness.observability import ACTIVE_TRACE_CONTEXT, TraceRecorder
from career_agent.agent.context_builder import TurnContextBuilder
from career_agent.agent.context_manager import ContextManager
from career_agent.agent.main_agent_contracts import (
    ConversationResourceReference,
    ConversationTaskState,
    MainAgentContext,
    ToolObservation,
)
from career_agent.services.episode_consolidation import drafts_from_tool_results
from career_agent.harness.streaming import (
    ContentDeltaEvent,
    ProgressEvent,
    PublicStreamEvent,
    StreamEventSink,
    TurnCompletedEvent,
    TurnFailedEvent,
    TurnInputResource,
    InteractionResponse,
    TurnStartedEvent,
)
from career_agent.storage.turn_receipts import (
    REPLAYED_EVENT_TYPES,
    SQLiteTurnReceiptStore,
    TurnReceipt,
)
from career_agent.storage.context import DeliveredBodyDraft


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


class CommittableTurnResult(Protocol):
    context: MainAgentContext
    assistant_message: str
    tool_result: ToolObservation | None
    tool_results: tuple[ToolObservation, ...]
    model_message: str
    career_memory_scope_keys: tuple[str, ...]


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

    def _reconcile_episodes(self, user_id: str) -> None: ...

    def _owns_next_turn(self, task: ConversationTaskState) -> bool: ...

    def _prepare_questionnaire_continuation(
        self,
        *,
        user_id: str,
        conversation_id: str,
        response: InteractionResponse,
        task: ConversationTaskState,
    ) -> MainAgentContext: ...

    def _run_loaded_context(
        self,
        context: MainAgentContext,
        *,
        bare_confirmation_target: Literal[
            "career_fact", "job_intent", "free_text_preference"
        ]
        | None = None,
    ) -> TurnResultT: ...

    def _run_interaction_response(
        self,
        *,
        context: MainAgentContext,
        conversation_id: str,
        response: InteractionResponse,
    ) -> TurnResultT: ...

    def _run_owned_workflow_turn(
        self, *, context: MainAgentContext, user_message: str
    ) -> TurnResultT: ...

    def _commit_interrupted_turn(
        self, *, context: MainAgentContext, error: Exception
    ) -> None: ...

    def _conversation_content(
        self,
        result: ToolObservation | None,
        *,
        screen: str,
        composed: bool,
    ) -> str: ...

    def _durable_screen(self, result: TurnResultT) -> str: ...

    def _turn_resource_refs(
        self, results: tuple[ToolObservation, ...]
    ) -> tuple[ConversationResourceReference, ...]: ...

    def _delivered_bodies(
        self, results: tuple[ToolObservation, ...]
    ) -> tuple[DeliveredBodyDraft, ...]: ...

    def _active_turn_id(self) -> str | None: ...

    def _attach_destructive_confirmation(self, result: TurnResultT) -> None: ...

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
        context_manager: ContextManager,
        context_builder: TurnContextBuilder,
        receipt_store: SQLiteTurnReceiptStore | None,
        trace_recorder: TraceRecorder | None,
    ) -> None:
        self._host = host
        self._context_manager = context_manager
        self._context_builder = context_builder
        self._receipt_store = receipt_store
        self._trace_recorder = trace_recorder

    def run(
        self,
        *,
        user_id: str,
        conversation_id: str,
        user_message: str,
        request_id: str | None,
        interaction_response: InteractionResponse | None,
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
            result = self._run_and_commit_turn(
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

    def _before_commit(
        self,
        result: TurnResultT,
        hook: Callable[[TurnResultT], None] | None,
    ) -> None:
        if hook is not None:
            hook(result)
        self._host._emit(
            ProgressEvent(stage="saving", message="正在保存本轮状态……")
        )

    def _run_and_commit_turn(
        self,
        *,
        user_id: str,
        conversation_id: str,
        user_message: str,
        interaction_response: InteractionResponse | None,
        before_commit: Callable[[TurnResultT], None] | None,
        input_resources: tuple[TurnInputResource, ...],
    ) -> TurnResultT:
        self._host._reconcile_episodes(user_id)
        prepared = self._context_builder.prepare(
            user_id=user_id,
            conversation_id=conversation_id,
            input_resources=input_resources,
        )
        routing_task = prepared.routing_task
        if (
            interaction_response is not None
            and interaction_response.scope == "questionnaire"
        ):
            return self._run_questionnaire_turn(
                user_id=user_id,
                conversation_id=conversation_id,
                response=interaction_response,
                task=routing_task,
                before_commit=before_commit,
            )
        if interaction_response is not None:
            context = self._context_builder.load_turn(
                prepared,
                user_id=user_id,
                conversation_id=conversation_id,
                user_message=user_message,
                route_profile=False,
            )
            try:
                result = self._host._run_interaction_response(
                    context=context,
                    conversation_id=conversation_id,
                    response=interaction_response,
                )
            except Exception as error:
                self._host._commit_interrupted_turn(context=context, error=error)
                raise
            self._before_commit(result, before_commit)
            result_tools = result.tool_results or (
                (result.tool_result,) if result.tool_result else ()
            )
            self._context_manager.commit_turn(
                context=context,
                task=result.context.task,
                assistant_message=self._host._conversation_content(
                    result.tool_result,
                    screen=result.assistant_message,
                    composed=False,
                ),
                assistant_bodies=self._host._delivered_bodies(result_tools),
                assistant_resource_refs=self._host._turn_resource_refs(()),
                episode_drafts=drafts_from_tool_results(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    tool_results=result_tools,
                ),
                memory_scope_keys=result.career_memory_scope_keys,
                turn_id=self._host._active_turn_id(),
            )
            return result

        if self._host._owns_next_turn(routing_task):
            context = self._context_builder.load_workflow_turn(
                prepared,
                user_id=user_id,
                conversation_id=conversation_id,
            )
            try:
                result = self._host._run_owned_workflow_turn(
                    context=context,
                    user_message=user_message,
                )
            except Exception as error:
                self._host._commit_interrupted_turn(context=context, error=error)
                raise
            self._before_commit(result, before_commit)
            if self._host._owns_next_turn(result.context.task):
                self._context_manager.commit_workflow_turn(
                    context=context,
                    task=result.context.task,
                )
            else:
                self._context_manager.commit_workflow_exit(
                    context=context,
                    task=result.context.task,
                    assistant_message=self._host._conversation_content(
                        result.tool_result,
                        screen=self._host._durable_screen(result),
                        composed=bool(result.model_message),
                    ),
                    assistant_resource_refs=self._host._turn_resource_refs(
                        result.tool_results
                    ),
                    assistant_bodies=self._host._delivered_bodies(
                        result.tool_results
                    ),
                    turn_id=self._host._active_turn_id(),
                )
            return result

        context = self._context_builder.load_turn(
            prepared,
            user_id=user_id,
            conversation_id=conversation_id,
            user_message=user_message,
            route_profile=True,
        )
        try:
            result = self._host._run_loaded_context(
                context,
                bare_confirmation_target=prepared.bare_confirmation_target,
            )
        except Exception as error:
            self._host._commit_interrupted_turn(context=context, error=error)
            raise
        self._before_commit(result, before_commit)
        result_tools = result.tool_results or (
            (result.tool_result,) if result.tool_result else ()
        )
        if self._host._owns_next_turn(result.context.task):
            held = self._context_manager.commit_workflow_entry(
                context=context,
                task=result.context.task,
                episode_drafts=drafts_from_tool_results(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    tool_results=result_tools,
                ),
            )
            result.context = result.context.model_copy(update={"task": held})
        else:
            self._context_manager.commit_turn(
                context=context,
                task=result.context.task,
                assistant_message=self._host._conversation_content(
                    result.tool_result,
                    screen=self._host._durable_screen(result),
                    composed=bool(result.model_message),
                ),
                assistant_resource_refs=self._host._turn_resource_refs(
                    result.tool_results
                ),
                assistant_bodies=self._host._delivered_bodies(
                    result.tool_results
                ),
                episode_drafts=drafts_from_tool_results(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    tool_results=result_tools,
                ),
                memory_scope_keys=result.career_memory_scope_keys,
                turn_id=self._host._active_turn_id(),
            )
        self._host._attach_destructive_confirmation(result)
        return result

    def _run_questionnaire_turn(
        self,
        *,
        user_id: str,
        conversation_id: str,
        response: InteractionResponse,
        task: ConversationTaskState,
        before_commit: Callable[[TurnResultT], None] | None,
    ) -> TurnResultT:
        context = self._host._prepare_questionnaire_continuation(
            user_id=user_id,
            conversation_id=conversation_id,
            response=response,
            task=task,
        )
        try:
            result = self._host._run_loaded_context(context)
        except Exception as error:
            self._host._commit_interrupted_turn(context=context, error=error)
            raise
        self._before_commit(result, before_commit)
        result_tools = result.tool_results or (
            (result.tool_result,) if result.tool_result else ()
        )
        self._context_manager.commit_turn(
            context=context,
            task=result.context.task,
            assistant_message=self._host._conversation_content(
                result.tool_result,
                screen=self._host._durable_screen(result),
                composed=bool(result.model_message),
            ),
            assistant_resource_refs=self._host._turn_resource_refs(
                result.tool_results
            ),
            assistant_bodies=self._host._delivered_bodies(result.tool_results),
            episode_drafts=drafts_from_tool_results(
                user_id=user_id,
                conversation_id=conversation_id,
                tool_results=result_tools,
            ),
            memory_scope_keys=result.career_memory_scope_keys,
            turn_id=self._host._active_turn_id(),
        )
        return result

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
