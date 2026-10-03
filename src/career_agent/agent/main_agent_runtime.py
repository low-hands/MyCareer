from __future__ import annotations

from collections.abc import Callable
from datetime import datetime

from threading import Event
from typing import Any, ClassVar, Literal

from career_agent.agent.decision_engine import DecisionEngine
from career_agent.agent.presentation_engine import PresentationEngine
from career_agent.agent.result_presenter import ResultPresenter
from career_agent.agent.presentation.interaction_renderer import InteractionRenderer
from career_agent.agent.presentation.presenter import TurnPresenter
from career_agent.agent.runtime_observability import RuntimeObservability
from career_agent.agent.context_manager import ContextManager
from career_agent.agent.context_builder import keyword_tool_profile
from career_agent.agent.career_context import CareerContextProjector
from career_agent.agent.main_agent_contracts import (
    AgentDecision,
    ConversationResourceReference,
    ConversationTaskState,
    DECISION_OBSERVATION_BODY_LIMIT,
    DecisionMaker,
    DecisionObservation,
    MainAgentContext,
    MAX_DECISION_OBSERVATIONS,
    ToolObservation,
    ToolProfile,
)
from career_agent.agent.summary_text import clamp
from career_agent.harness.observability import (
    EventType,
    ModelCallCategory,
    TraceRecorder,
)
from career_agent.agent.delivery_policy import condenses_message
from career_agent.storage.capability_confirmations import (
    SQLiteCapabilityConfirmationStore,
)
from career_agent.agent.main_agent_reducers import reduce_task_state
from career_agent.agent.main_agent_tools import MainAgentToolOutput, MainAgentToolRegistry
from career_agent.agent.main_state import MainAgentState, PendingAction
from career_agent.agent.runtime_composition import install_main_runtime_components
from career_agent.agent.turn_models import (
    InteractionReceipt,
    MainAgentTurnResult,
    ModelDecision,
    RuntimeAction,
    RuntimePolicyAction,
)
from career_agent.agent.middleware.argument_projection import (
    project_atomic_arguments,
    project_runtime_owned_arguments,
    project_workflow_arguments,
)
from career_agent.services.episode_reconciliation import EpisodeReconciler
from career_agent.harness.graph_routing import GraphRoutingPolicy
from career_agent.harness.turn_router import TurnRouter
from career_agent.harness.streaming import (
    InteractionRequiredEvent,
    InteractionResponse,
    PublicStreamEvent,
    StreamEventSink,
    TurnInputResource,
)
from career_agent.storage.action_executions import (
    SQLiteActionExecutionStore,
)
from career_agent.storage.context import DeliveredBodyDraft
from career_agent.storage.turn_receipts import (
    SQLiteTurnReceiptStore,
)
from career_agent.agent.turn_coordinator import (
    ACTION_INVOCATION as _ACTION_INVOCATION,
    STREAM_SINK as _STREAM_SINK,
    TRACE_CONTEXT as _TRACE_CONTEXT,
    ReplayedTurn,
    TurnInProgressError,
)

ACTION_EXECUTION_POLICY_EPOCH = 1

DEFAULT_MAX_READ_CALLS = 6
DEFAULT_MAX_WRITE_CALLS = 1
DEFAULT_MAX_EXTERNAL_WRITE_CALLS = 1
DEFAULT_MAX_PROJECTION_REFUSALS = 2
DEFAULT_MAX_AUTHORIZATION_REFUSALS = 1
DEFAULT_MAX_FAILURE_RETRIES = 2


class MainAgentRuntime:
    _INTERACTION_RENDERER_STATES = InteractionRenderer.RENDERER_STATES

    @classmethod
    def _has_interaction_renderer(cls, state: str) -> bool:
        return state in cls._INTERACTION_RENDERER_STATES

    def __init__(self, *, context_manager: ContextManager, decision_maker: DecisionMaker, tools: MainAgentToolRegistry, career_context_projector: CareerContextProjector | None = None, max_read_calls: int = DEFAULT_MAX_READ_CALLS, max_write_calls: int = DEFAULT_MAX_WRITE_CALLS, max_external_write_calls: int = DEFAULT_MAX_EXTERNAL_WRITE_CALLS, max_projection_refusals: int = DEFAULT_MAX_PROJECTION_REFUSALS, max_authorization_refusals: int = DEFAULT_MAX_AUTHORIZATION_REFUSALS, max_failure_retries: int = DEFAULT_MAX_FAILURE_RETRIES, owned_resources: tuple[Any, ...] = (), trace_recorder: TraceRecorder | None = None, action_execution_store: SQLiteActionExecutionStore | None = None, capability_confirmation_store: SQLiteCapabilityConfirmationStore | None = None, turn_receipt_store: SQLiteTurnReceiptStore | None = None, action_policy_epoch: int = ACTION_EXECUTION_POLICY_EPOCH, episode_reconciler: EpisodeReconciler | None = None) -> None:
        if max_read_calls < 1:
            raise ValueError("max_read_calls must be at least one")
        if max_write_calls < 1:
            raise ValueError("max_write_calls must be at least one")
        if max_external_write_calls < 1:
            raise ValueError("max_external_write_calls must be at least one")
        if max_projection_refusals < 1:
            raise ValueError("max_projection_refusals must be at least one")
        if max_authorization_refusals < 1:
            raise ValueError("max_authorization_refusals must be at least one")
        if max_failure_retries < 0:
            raise ValueError("max_failure_retries cannot be negative")
        if action_policy_epoch < 1:
            raise ValueError("action_policy_epoch must be positive")
        if (
            max_read_calls
            + max_write_calls
            + max_external_write_calls
            + max_projection_refusals
            + max_authorization_refusals
            > MAX_DECISION_OBSERVATIONS
        ):
            raise ValueError(
                "read, write, and refusal budgets must fit the observation window"
            )
        self._context_manager = context_manager
        # Exposed for the CLI's post-turn maintenance notice, which is an
        # operator concern and deliberately never reaches the decision model.
        self.context_manager = context_manager
        context_manager.on_compaction(self._announce_compaction)
        self._decision_maker = decision_maker
        self._tools = tools
        self._max_read_calls = max_read_calls
        self._max_write_calls = max_write_calls
        self._max_external_write_calls = max_external_write_calls
        self._max_projection_refusals = max_projection_refusals
        self._max_authorization_refusals = max_authorization_refusals
        self._max_failure_retries = max_failure_retries
        self._trace_recorder = trace_recorder
        self._capability_confirmation_store = capability_confirmation_store
        self._action_policy_epoch = action_policy_epoch
        self._owned_resources = owned_resources
        self._closed = False
        install_main_runtime_components(
            self,
            context_manager=context_manager,
            decision_maker=decision_maker,
            tools=tools,
            career_context_projector=career_context_projector,
            max_read_calls=max_read_calls,
            max_write_calls=max_write_calls,
            max_external_write_calls=max_external_write_calls,
            max_projection_refusals=max_projection_refusals,
            max_authorization_refusals=max_authorization_refusals,
            max_failure_retries=max_failure_retries,
            trace_recorder=trace_recorder,
            action_execution_store=action_execution_store,
            capability_confirmation_store=capability_confirmation_store,
            turn_receipt_store=turn_receipt_store,
            action_policy_epoch=action_policy_epoch,
            episode_reconciler=episode_reconciler,
        )

    @staticmethod
    def _route_entry(state: MainAgentState) -> Literal["hydrate", "authorize"]:
        return GraphRoutingPolicy.route_entry(state)

    def close(self) -> None:
        if self._closed:
            return
        for resource in reversed(self._owned_resources):
            close = getattr(resource, "close", None)
            if close is not None:
                close()
        self._closed = True

    @staticmethod
    def _emit(event: PublicStreamEvent) -> None:
        RuntimeObservability.emit(event)

    @classmethod
    def _emit_capability_started(cls, name: str) -> None:
        RuntimeObservability.emit_capability_started(name)

    @classmethod
    def _emit_capability_completed(cls, name: str, state: str) -> None:
        RuntimeObservability.emit_capability_completed(name, state)

    def run_turn(
        self,
        *,
        user_id: str,
        conversation_id: str,
        user_message: str,
        request_id: str | None = None,
        interaction_response: InteractionResponse | None = None,
        event_sink: StreamEventSink | None = None,
        input_resources: tuple[TurnInputResource, ...] = (),
    ) -> MainAgentTurnResult | ReplayedTurn:
        """Run one committed turn and optionally publish presentation-only events.

        The sink is held outside graph state and checkpoints. A broken observer
        never gets authority to fail or mutate the business turn.

        With a receipt store, ``request_id`` identifies the turn as a whole: a
        request whose key already committed is answered from its receipt and
        returns a ``ReplayedTurn`` without executing anything.
        """
        return self._turn_coordinator.run(
            user_id=user_id,
            conversation_id=conversation_id,
            user_message=user_message,
            request_id=request_id,
            interaction_response=interaction_response,
            event_sink=event_sink,
            input_resources=input_resources,
        )

    def _emit_turn_failure(
        self,
        *,
        turn_id: str,
        error: Exception,
        reply_delivered: bool,
    ) -> None:
        RuntimeObservability.emit_turn_failure(
            turn_id=turn_id,
            error=error,
            reply_delivered=reply_delivered,
        )

    @staticmethod
    def _emit_trace(
        event_type: Literal["capability_failed", "presentation_degraded"],
        stage: str,
        *,
        error_code: str | None = None,
        error_detail: str | None = None,
        details: dict[str, Any] | None = None,
        recoverable: bool | None = None,
    ) -> None:
        RuntimeObservability.emit_trace(
            event_type,
            stage,
            error_code=error_code,
            error_detail=error_detail,
            details=details,
            recoverable=recoverable,
        )

    @staticmethod
    def _record_trace_event(
        event_type: EventType,
        stage: str,
        *,
        outcome: Literal["started", "succeeded", "failed", "interrupted"],
        duration_ms: int | None = None,
        error_code: str | None = None,
        error_detail: str | None = None,
        details: dict[str, Any] | None = None,
        recoverable: bool | None = None,
        model_call_category: ModelCallCategory | None = None,
    ) -> None:
        RuntimeObservability.record_trace_event(
            event_type,
            stage,
            outcome=outcome,
            duration_ms=duration_ms,
            error_code=error_code,
            error_detail=error_detail,
            details=details,
            recoverable=recoverable,
            model_call_category=model_call_category,
        )

    def _record_turn(
        self,
        *,
        turn_id: str,
        conversation_id: str,
        result: MainAgentTurnResult,
    ) -> None:
        self._runtime_observability.record_turn(
            turn_id=turn_id,
            conversation_id=conversation_id,
            result=result,
        )

    def record_rejected_turn(self, *, user_id: str, conversation_id: str) -> None:
        self._runtime_observability.record_rejected_turn(
            user_id=user_id,
            conversation_id=conversation_id,
        )

    def record_capture_continuation(
        self,
        *,
        user_id: str,
        conversation_id: str | None,
        capture_event_id: str | None,
        phase: Literal["intent", "settled"],
        status: str,
        turn_id: str | None = None,
        error_code: str | None = None,
    ) -> None:
        self._runtime_observability.record_capture_continuation(
            user_id=user_id,
            conversation_id=conversation_id,
            capture_event_id=capture_event_id,
            phase=phase,
            status=status,
            turn_id=turn_id,
            error_code=error_code,
        )

    def _record_turn_failed(
        self,
        *,
        turn_id: str,
        conversation_id: str,
        error: Exception,
        reply_delivered: bool = False,
    ) -> None:
        self._runtime_observability.record_turn_failed(
            turn_id=turn_id,
            conversation_id=conversation_id,
            error=error,
            reply_delivered=reply_delivered,
        )

    def _commit_interrupted_turn(
        self, *, context: MainAgentContext, error: Exception
    ) -> None:
        self._reconciliation.commit_interrupted_turn(
            context=context,
            error=error,
        )

    def _reconcile_episodes(self, user_id: str) -> None:
        self._reconciliation.reconcile_episodes(user_id)

    def _invalidate_episode_reconciliation(self, user_id: str) -> None:
        self._reconciliation.invalidate_episode_reconciliation(user_id)

    def _attach_destructive_confirmation(self, result: MainAgentTurnResult) -> None:
        self._confirmation_coordinator.attach_destructive_confirmation(result)

    def _prepare_questionnaire_continuation(
        self,
        *,
        user_id: str,
        conversation_id: str,
        response: InteractionResponse,
        task: ConversationTaskState,
    ) -> MainAgentContext:
        return self._interaction_coordinator.prepare_questionnaire_continuation(
            user_id=user_id,
            conversation_id=conversation_id,
            response=response,
            task=task,
        )

    @staticmethod
    def _active_turn_id() -> str | None:
        invocation = _ACTION_INVOCATION.get()
        return invocation[0] if invocation is not None else None

    def _deliver_stream_events(
        self,
        *,
        result: MainAgentTurnResult,
        turn_id: str,
        conversation_id: str,
    ) -> None:
        self._stream_adapter.deliver_events(
            result=result,
            turn_id=turn_id,
            conversation_id=conversation_id,
        )

    def _deliver_reply(
        self,
        *,
        result: MainAgentTurnResult,
        conversation_id: str,
    ) -> None:
        self._stream_adapter.deliver_reply(
            result=result,
            conversation_id=conversation_id,
        )

    @staticmethod
    def _interaction_event(
        *,
        result: MainAgentTurnResult,
        conversation_id: str,
    ) -> InteractionRequiredEvent | None:
        renderer = InteractionRenderer(
            active_turn_id=MainAgentRuntime._active_turn_id,
            assistant_message=MainAgentRuntime._assistant_message,
            has_interaction_renderer=MainAgentRuntime._has_interaction_renderer,
        )
        return renderer.event(
            result=result,
            conversation_id=conversation_id,
        )

    def _run_interaction_response(
        self,
        *,
        context: MainAgentContext,
        conversation_id: str,
        response: InteractionResponse,
    ) -> MainAgentTurnResult:
        return self._confirmation_coordinator.run_owner_confirmation(
            context=context, conversation_id=conversation_id, response=response
        )

    def accepts_background_turn(self, *, user_id: str, conversation_id: str) -> bool:
        return self._turn_router.accepts_background_turn(
            user_id=user_id,
            conversation_id=conversation_id,
        )

    @staticmethod
    def _owns_next_turn(task: ConversationTaskState) -> bool:
        return TurnRouter.owns_next_turn(task)

    def _run_loaded_context(
        self,
        context: MainAgentContext,
        *,
        bare_confirmation_target: Literal[
            "career_fact",
            "job_intent",
            "free_text_preference",
        ] | None = None,
    ) -> MainAgentTurnResult:
        return self._turn_router.run_loaded_context(
            context,
            bare_confirmation_target=bare_confirmation_target,
        )

    def _decision_tool_schemas(
        self, profile: ToolProfile, task: ConversationTaskState | None = None
    ) -> tuple[dict[str, Any], ...]:
        return self._decision_engine.tool_schemas(profile, task)

    def _offers_tool(
        self,
        name: str,
        profile: ToolProfile = "core",
        task: ConversationTaskState | None = None,
    ) -> bool:
        return TurnRouter.offers_tool(name, profile, task)

    DECISION_HEARTBEAT_SECONDS: ClassVar[float] = 15.0
    def _announce_compaction(self, phase: str) -> None:
        RuntimeObservability.announce_compaction(phase)

    def _heartbeat(
        self,
        sink: StreamEventSink | None,
        *,
        stage: str,
        describe: Callable[[int], str],
        step_fields: Callable[[], dict[str, str]] | None = None,
    ) -> Event:
        return RuntimeObservability.heartbeat(
            sink,
            interval=self.DECISION_HEARTBEAT_SECONDS,
            stage=stage,
            describe=describe,
            step_fields=step_fields,
        )

    def _decision_heartbeat(self, sink: StreamEventSink | None) -> Event:
        return self._heartbeat(
            sink,
            stage="deciding",
            describe=lambda waited: f"仍在等待模型判断（已等待 {waited} 秒）……",
        )

    def _run_capability(
        self,
        pending: PendingAction,
        run: Callable[[], MainAgentToolOutput],
    ) -> MainAgentToolOutput:
        return RuntimeObservability.run_capability(
            pending,
            run,
            heartbeat_interval=self.DECISION_HEARTBEAT_SECONDS,
        )

    def _decide(self, state: MainAgentState) -> MainAgentState:
        return self._decision_engine.decide(state)

    def _run_owned_workflow_turn(
        self, *, context: MainAgentContext, user_message: str
    ) -> MainAgentTurnResult:
        return self._turn_router.run_owned_workflow_turn(
            context=context,
            user_message=user_message,
        )

    def _hydrate_career_context(self, state: MainAgentState) -> MainAgentState:
        return self._context_hydrator.hydrate(state)

    @staticmethod
    def _tool_call_fingerprint(decision: AgentDecision) -> str:
        return DecisionEngine.tool_call_fingerprint(decision)

    @staticmethod
    def _route_decision(
        state: MainAgentState,
    ) -> Literal["authorize", "present", "interrupt"]:
        return GraphRoutingPolicy.route_decision(state)

    def _authorize(self, state: MainAgentState) -> MainAgentState:
        return self._authorization_engine.authorize(state)

    @staticmethod
    def _after_authorize(
        state: MainAgentState,
    ) -> Literal["act", "observe", "present", "interrupt"]:
        return GraphRoutingPolicy.after_authorize(state)

    def _act(self, state: MainAgentState) -> MainAgentState:
        return self._capability_executor.act(state)

    @staticmethod
    def _project_runtime_workflow_arguments(
        state: MainAgentState, name: str
    ) -> dict[str, Any]:
        return project_runtime_owned_arguments(state, name)

    @staticmethod
    def _project_workflow_arguments(
        context: MainAgentContext,
        name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        return project_workflow_arguments(context, name, arguments)

    def _observe(self, state: MainAgentState) -> MainAgentState:
        return self._observation_reducer.reduce(state)

    @staticmethod
    def _after_observe(
        state: MainAgentState,
    ) -> Literal["hydrate", "decide", "present", "interrupt"]:
        return GraphRoutingPolicy.after_observe(state)

    @staticmethod
    def _present(state: MainAgentState) -> MainAgentState:
        return PresentationEngine.present(
            state,
            renderer=MainAgentRuntime._presentation_adapter(),
        )

    @staticmethod
    def _presentation_adapter() -> TurnPresenter:
        return TurnPresenter(render_result=MainAgentRuntime._assistant_message)

    @staticmethod
    def _screen_message(result: MainAgentToolOutput) -> str:
        return MainAgentRuntime._presentation_adapter()._screen_message(result)

    @staticmethod
    def _turn_resource_refs(
        results: tuple[MainAgentToolOutput, ...],
        input_refs: tuple[ConversationResourceReference, ...] = (),
    ) -> tuple[ConversationResourceReference, ...]:
        return TurnPresenter.turn_resource_refs(results, input_refs)

    @staticmethod
    def _turn_is_card_backed(results: tuple[MainAgentToolOutput, ...]) -> bool:
        return MainAgentRuntime._presentation_adapter()._turn_is_card_backed(results)

    @staticmethod
    def _durable_screen(result: MainAgentTurnResult) -> str:
        return TurnPresenter.durable_screen(result)

    @staticmethod
    def _undelivered_bodies(results: tuple[MainAgentToolOutput, ...]) -> str:
        return MainAgentRuntime._presentation_adapter()._undelivered_bodies(results)

    @staticmethod
    def _delivered_bodies(
        results: tuple[MainAgentToolOutput, ...],
    ) -> tuple[DeliveredBodyDraft, ...]:
        return MainAgentRuntime._presentation_adapter().delivered_bodies(results)

    def _interrupt(self, state: MainAgentState) -> MainAgentState:
        return self._interaction_renderer.interrupt(state)

    @staticmethod
    def _tool_observation(
        name: str,
        result: MainAgentToolOutput,
        arguments: dict[str, Any] | None = None,
    ) -> DecisionObservation:
        receipt = clamp(result.message) or "工具已返回，但没有提供结果摘要。"
        body = None
        if condenses_message(result.state):
            rendered = MainAgentRuntime._assistant_message(result)
            body = clamp(rendered, limit=DECISION_OBSERVATION_BODY_LIMIT) or None
        if isinstance(result, ToolObservation):
            return DecisionObservation(
                tool_name=result.tool_name,
                state=result.state,
                message=receipt,
                body=body,
                facts=dict(result.facts),
                next_action=result.next_action,
                arguments=dict(arguments or {}),
                # The handle survives the body. Clearing keeps the reference for
                # the same reason tool-result clearing keeps the tool_use record.
                resource_ref=result.resource_ref,
                resource_refs=result.resource_refs,
            )
        return DecisionObservation(
            tool_name=name,
            state=result.state,
            message=receipt,
            body=body,
            facts=dict(result.facts),
            next_action=result.next_action,
            arguments=dict(arguments or {}),
            resource_ref=result.resource_ref,
            resource_refs=result.resource_refs,
        )

    @staticmethod
    def _conversation_content(
        result: MainAgentToolOutput | None,
        *,
        screen: str,
        composed: bool,
    ) -> str:
        return TurnPresenter.conversation_content(
            result,
            screen=screen,
            composed=composed,
        )

    @staticmethod
    def _assistant_message(result: MainAgentToolOutput) -> str:
        return ResultPresenter.present(
            result, report_degraded=MainAgentRuntime._emit_trace
        )

    @staticmethod
    def _project_atomic_tool_arguments(
        context: MainAgentContext,
        name: str,
        arguments: dict[str, object],
    ) -> dict[str, object]:
        return project_atomic_arguments(
            context,
            name,
            arguments,
            source_turn_id=MainAgentRuntime._active_turn_id(),
        )

    @staticmethod
    def _update_mock_interview_task(
        context: MainAgentContext, result: ToolObservation
    ) -> MainAgentContext:
        task = context.task
        session_id = result.payload.get("session_id")
        if result.state in {
            "mock_interview_answer_required",
            "mock_interview_running",
        }:
            if not isinstance(session_id, str) or not session_id:
                raise ValueError("Mock interview result has no session_id")
            task = task.enter_workflow(
                "mock_interview",
                run_id=session_id,
                phase=result.state,
                candidates=(),
            )
        elif result.state in {
            "mock_interview_completed",
            "mock_interview_cancelled",
            "mock_interview_restart_failed",
            "no_mock_interview_to_restart",
        }:
            if task.active_workflow == "mock_interview":
                task = task.leave_workflow()
        elif result.state == "failed" and task.active_workflow == "mock_interview":
            # A persisted answer can be retried. Keep ownership instead of
            # stranding the graph after a transient Worker failure.
            task = task.enter_workflow(
                "mock_interview",
                run_id=task.run_id or str(session_id),
                phase="failed",
                candidates=(),
            )
        elif result.state in {
            "mock_interview_checkpoint_missing",
            "mock_interview_graph_incompatible",
        } and task.active_workflow == "mock_interview":
            task = task.enter_workflow(
                "mock_interview",
                run_id=task.run_id or str(session_id),
                phase=result.state,
                candidates=(),
            )
        return context.model_copy(update={"task": task})

    @staticmethod
    def _update_atomic_task(
        context: MainAgentContext,
        result: ToolObservation,
        *,
        now: datetime | None = None,
    ) -> MainAgentContext:
        return context.model_copy(
            update={"task": reduce_task_state(context.task, result, now=now)}
        )
