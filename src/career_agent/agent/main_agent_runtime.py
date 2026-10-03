from __future__ import annotations

from collections.abc import Callable
from datetime import datetime

import json
from threading import Event
from typing import Any, ClassVar, Literal

from career_agent.agent.authorization_engine import AuthorizationEngine
from career_agent.agent.execution.capability_executor import CapabilityExecutor
from career_agent.agent.decision_engine import DecisionEngine
from career_agent.agent.observation_reducer import ObservationReducer
from career_agent.agent.presentation_engine import PresentationEngine
from career_agent.agent.result_presenter import ResultPresenter
from career_agent.agent.presentation.interaction_renderer import InteractionRenderer
from career_agent.agent.presentation.presenter import TurnPresenter
from career_agent.agent.presentation.stream_adapter import StreamAdapter
from career_agent.agent.execution.reconciliation import ReconciliationCoordinator
from career_agent.agent.runtime_observability import RuntimeObservability
from career_agent.agent.interaction_coordinator import (
    InteractionCoordinator,
    QuestionnaireContinuationError,
)
from career_agent.agent.context_manager import ContextManager
from career_agent.agent.context_builder import (
    TurnContextBuilder,
    keyword_tool_profile,
)
from career_agent.agent.career_context import CareerContextProjector
from career_agent.harness.capability_steps import (
    CapabilityStep,
)
from career_agent.agent.main_agent_contracts import (
    AgentDecision,
    AttachedResumeContext,
    ConversationResourceReference,
    ConversationTaskState,
    DECISION_OBSERVATION_BODY_LIMIT,
    DecisionMaker,
    DecisionObservation,
    MainAgentContext,
    MAX_DECISION_OBSERVATIONS,
    SavedJobCandidateContextItem,
    TOOL_PROFILE_NAMES,
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
from career_agent.agent.tool_effects import (
    ToolEffect,
)
from career_agent.agent.middleware.contracts import AuthorizationRefusalKind
from career_agent.storage.capability_confirmations import (
    SQLiteCapabilityConfirmationStore,
)
from career_agent.agent.main_agent_reducers import reduce_task_state
from career_agent.agent.main_agent_tools import MainAgentToolOutput, MainAgentToolRegistry
from career_agent.agent.main_state import LoopControl, MainAgentState, PendingAction
from career_agent.agent.main_graph import build_main_graph
from career_agent.agent.turn_models import (
    InteractionReceipt,
    MainAgentTurnResult,
    ModelDecision,
    OriginKind,
    Originator,
    RuntimeAction,
    RuntimePolicyAction,
    TurnOrigin,
)
from career_agent.agent.middleware.argument_projection import (
    project_atomic_arguments,
    project_runtime_owned_arguments,
    project_workflow_arguments,
)
from career_agent.agent.job_analysis_contracts import JobAnalysisResult
from career_agent.agent.resume_job_match_contracts import ResumeJobMatchResult
from career_agent.agent.resume_tailoring_contracts import ResumeTailoringResult
from career_agent.domain.job_comparison import JobComparison
from career_agent.domain.applications.models import ApplicationStatus
from career_agent.agent.mock_interview_contracts import (
    MockInterviewGraphResult,
)
from career_agent.domain.interview_preparation import InterviewPreparationResult
from career_agent.domain.job_research import JobResearchDraft
from career_agent.services.episode_reconciliation import EpisodeReconciler
from career_agent.harness.agent_loop import AgentLoop
from career_agent.harness.confirmation_coordinator import (
    CapabilityConfirmationCoordinator,
)
from career_agent.harness.graph_routing import GraphRoutingPolicy
from career_agent.harness.turn_router import TurnRouter
from career_agent.harness.streaming import (
    InteractionOption,
    InteractionRequiredEvent,
    InteractionResponse,
    JobResourceReadyEvent,
    PublicStreamEvent,
    ReportReadyEvent,
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
    TurnCoordinator,
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
        self._career_context_projector = career_context_projector
        self._max_read_calls = max_read_calls
        self._max_write_calls = max_write_calls
        self._max_external_write_calls = max_external_write_calls
        self._max_projection_refusals = max_projection_refusals
        self._max_authorization_refusals = max_authorization_refusals
        self._max_failure_retries = max_failure_retries
        self._trace_recorder = trace_recorder
        self._runtime_observability = RuntimeObservability(
            trace_recorder=trace_recorder,
        )
        self._capability_confirmation_store = capability_confirmation_store
        # Transitional compatibility for startup recovery and focused tests.
        # Turn lifecycle operations are owned by ``_turn_coordinator``; callers
        # that enumerate orphaned receipts still need direct store access until
        # recovery is moved to the ingress layer in the next extraction.
        self._turn_receipt_store = turn_receipt_store
        self._action_policy_epoch = action_policy_epoch
        self._reconciliation = ReconciliationCoordinator(
            context_manager=context_manager,
            action_execution_store=action_execution_store,
            episode_reconciler=episode_reconciler,
        )
        self._context_builder = TurnContextBuilder(
            context_manager=context_manager,
            tools=tools,
            owns_next_turn=self._owns_next_turn,
        )
        self._turn_coordinator = TurnCoordinator(
            host=self,
            context_manager=context_manager,
            context_builder=self._context_builder,
            receipt_store=turn_receipt_store,
            trace_recorder=trace_recorder,
        )
        self._authorization_engine = AuthorizationEngine(
            host=self,
            tools=tools,
            confirmation_store=capability_confirmation_store,
            max_read_calls=max_read_calls,
            max_write_calls=max_write_calls,
            max_external_write_calls=max_external_write_calls,
            max_projection_refusals=max_projection_refusals,
            max_authorization_refusals=max_authorization_refusals,
            max_failure_retries=max_failure_retries,
        )
        self._capability_executor = CapabilityExecutor(
            host=self,
            tools=tools,
            action_execution_store=action_execution_store,
            action_policy_epoch=action_policy_epoch,
        )
        self._observation_reducer = ObservationReducer(
            host=self,
            context_manager=context_manager,
        )
        self._decision_engine = DecisionEngine(
            host=self,
            context_manager=context_manager,
            decision_maker_provider=lambda: self._decision_maker,
            tools=tools,
            career_memory_enabled=career_context_projector is not None,
        )
        self._interaction_coordinator = InteractionCoordinator(
            context_manager=context_manager,
            tools=tools,
            confirmation_store=capability_confirmation_store,
        )
        self._interaction_renderer = InteractionRenderer(
            active_turn_id=self._active_turn_id,
            assistant_message=self._assistant_message,
            has_interaction_renderer=self._has_interaction_renderer,
        )
        self._stream_adapter = StreamAdapter(host=self, emit=self._emit)
        self._owned_resources = owned_resources
        self._closed = False
        request_token_usage = getattr(decision_maker, "request_token_usage", None)
        if callable(request_token_usage):

            def estimate_complete_request(
                context: MainAgentContext,
            ) -> tuple[int, int]:
                if self._career_context_projector is not None:
                    context = context.model_copy(
                        update={
                            "career_memory": self._career_context_projector.project(
                                user_id=context.profile.user_id,
                                query=context.user_message,
                            )
                        }
                    )
                return request_token_usage(
                    context, self._decision_tool_schemas(
                        context.task.tool_profile, context.task
                    )
                )

            static_request_token_usage = getattr(
                decision_maker, "static_request_token_usage", None
            )
            if not callable(static_request_token_usage):
                raise ValueError(
                    "decision makers that report request token usage must also "
                    "report static request token usage"
                )
            # The message and recent-window caps must not move with the
            # profile, so they are derived from the largest profile's request.
            static_tokens, max_input_tokens = max(
                (
                    static_request_token_usage(self._decision_tool_schemas(profile))
                    for profile in TOOL_PROFILE_NAMES
                ),
                key=lambda usage: usage[0],
            )
            self._context_manager.configure_request_token_estimator(
                estimate_complete_request,
                static_input_tokens=static_tokens,
                max_input_tokens=max_input_tokens,
            )

        self._graph = build_main_graph(self)
        self._agent_loop = AgentLoop(
            graph=self._graph,
            memory_scope_keys=TurnRouter.free_text_preference_scope_keys,
            # Keep artifact support lazy: several narrow registry doubles only
            # implement the workflow surface they exercise, just as before.
            deliver_resume_artifact=lambda **kwargs: (
                self._tools.deliver_resume_artifact(**kwargs)
            ),
        )
        self._turn_router = TurnRouter(
            context_manager=context_manager,
            confirmation_store=capability_confirmation_store,
            agent_loop=self._agent_loop,
        )
        self._confirmation_coordinator = CapabilityConfirmationCoordinator(
            confirmation_store=capability_confirmation_store,
            interaction_coordinator=self._interaction_coordinator,
            agent_loop=self._agent_loop,
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

    @staticmethod
    def _public_capability(name: str) -> str:
        return RuntimeObservability.public_capability(name)

    _CAPABILITY_LABELS: ClassVar[dict[str, str]] = (
        RuntimeObservability.CAPABILITY_LABELS
    )

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

    def _observability_service(self) -> RuntimeObservability:
        service = getattr(self, "_runtime_observability", None)
        if service is None:
            service = RuntimeObservability(
                trace_recorder=getattr(self, "_trace_recorder", None),
            )
            self._runtime_observability = service
        return service

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
        self._observability_service().record_turn(
            turn_id=turn_id,
            conversation_id=conversation_id,
            result=result,
        )

    def record_rejected_turn(self, *, user_id: str, conversation_id: str) -> None:
        self._observability_service().record_rejected_turn(
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
        self._observability_service().record_capture_continuation(
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
        self._observability_service().record_turn_failed(
            turn_id=turn_id,
            conversation_id=conversation_id,
            error=error,
            reply_delivered=reply_delivered,
        )

    def _reconciliation_service(self) -> ReconciliationCoordinator:
        service = getattr(self, "_reconciliation", None)
        if service is None:
            # Compatibility for focused callers that construct the runtime
            # without __init__ and inject the former fields directly.
            service = ReconciliationCoordinator(
                context_manager=getattr(self, "_context_manager", None),
                action_execution_store=getattr(
                    self, "_action_execution_store", None
                ),
                episode_reconciler=getattr(self, "_episode_reconciler", None),
                reconciled_users=getattr(self, "_reconciled_users", None),
                reconcile_guard=getattr(
                    self, "_episode_reconcile_guard", None
                ),
                user_locks=getattr(self, "_episode_reconcile_locks", None),
            )
            self._reconciliation = service
        return service

    def _commit_interrupted_turn(
        self, *, context: MainAgentContext, error: Exception
    ) -> None:
        self._reconciliation_service().commit_interrupted_turn(
            context=context,
            error=error,
        )

    def _reconcile_episodes(self, user_id: str) -> None:
        self._reconciliation_service().reconcile_episodes(user_id)

    def _invalidate_episode_reconciliation(self, user_id: str) -> None:
        self._reconciliation_service().invalidate_episode_reconciliation(user_id)

    def _episode_reconcile_user_lock(self, user_id: str):
        return self._reconciliation_service().user_lock(user_id)

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
    def _attach_input_resources(
        context: MainAgentContext,
        attached_resumes: tuple[AttachedResumeContext, ...],
        attached_jobs: tuple[SavedJobCandidateContextItem, ...] = (),
        active_application: tuple[
            str | None, ApplicationStatus | None
        ] = (None, None),
    ) -> MainAgentContext:
        return TurnContextBuilder.attach_input_resources(
            context,
            attached_resumes,
            attached_jobs,
            active_application,
        )

    def _refresh_saved_job_focus(self, context: MainAgentContext) -> MainAgentContext:
        return self._context_builder.refresh_saved_job_focus(context)

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
        self._stream_delivery_adapter().deliver_events(
            result=result,
            turn_id=turn_id,
            conversation_id=conversation_id,
        )

    def _stream_delivery_adapter(self) -> StreamAdapter:
        adapter = getattr(self, "_stream_adapter", None)
        if adapter is None:
            adapter = StreamAdapter(host=self, emit=self._emit)
            self._stream_adapter = adapter
        return adapter

    @staticmethod
    def _resource_ready_event(
        reference: ConversationResourceReference,
    ) -> ReportReadyEvent | JobResourceReadyEvent:
        return StreamAdapter.resource_ready_event(reference)

    def _deliver_reply(
        self,
        *,
        result: MainAgentTurnResult,
        conversation_id: str,
    ) -> None:
        self._stream_delivery_adapter().deliver_reply(
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

    def _run_owner_confirmation(
        self,
        *,
        context: MainAgentContext,
        conversation_id: str,
        response: InteractionResponse,
    ) -> MainAgentTurnResult:
        return self._confirmation_coordinator.run_owner_confirmation(
            context=context,
            conversation_id=conversation_id,
            response=response,
        )

    @staticmethod
    def _settled_confirmation_turn(
        context: MainAgentContext,
        message: str,
        *,
        state: str,
        action: str = "confirm",
    ) -> MainAgentTurnResult:
        return CapabilityConfirmationCoordinator.settled_turn(
            context,
            message,
            state=state,
            action=action,
        )

    @staticmethod
    def _mock_interview_resume_choice_event(
        event_id: str, prompt: str, task: ConversationTaskState
    ) -> InteractionRequiredEvent:
        return InteractionRenderer._mock_interview_resume_choice_event(
            event_id, prompt, task
        )

    @staticmethod
    def _selection_options(
        tool_result: MainAgentToolOutput | None,
        task: ConversationTaskState,
    ) -> tuple[InteractionOption, ...]:
        return InteractionRenderer._selection_options(tool_result, task)

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

    def _explicit_span_prelude(self, context: MainAgentContext) -> MainAgentState:
        return self._turn_router.explicit_span_prelude(context)

    def _registered_schemas(self) -> tuple[dict[str, Any], ...]:
        return self._decision_engine.registered_schemas()

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

    def _run_free_text_preference_confirmation(
        self,
        context: MainAgentContext,
    ) -> MainAgentTurnResult:
        return self._turn_router.run_runtime_policy_tool(
            context,
            policy="free_text_preference_confirmation",
            tool_name="propose_free_text_preference_confirmation",
            arguments={"selection_index": 1},
        )

    def _run_runtime_policy_tool(
        self,
        context: MainAgentContext,
        *,
        policy: Literal[
            "free_text_preference_confirmation",
            "free_text_preference_activation",
            "career_fact_confirmation",
            "job_intent_confirmation",
        ],
        tool_name: str,
        arguments: dict[str, Any],
    ) -> MainAgentTurnResult:
        return self._turn_router.run_runtime_policy_tool(
            context,
            policy=policy,
            tool_name=tool_name,
            arguments=arguments,
        )

    DECISION_HEARTBEAT_SECONDS: ClassVar[float] = 15.0
    _COMPACTION_MESSAGES = RuntimeObservability.COMPACTION_MESSAGES
    _CAPABILITY_STEP_MESSAGES = RuntimeObservability.CAPABILITY_STEP_MESSAGES
    _CAPABILITY_TOOL_MESSAGES = RuntimeObservability.CAPABILITY_TOOL_MESSAGES

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

    @classmethod
    def _capability_step_label(cls, step: CapabilityStep) -> str | None:
        return RuntimeObservability.capability_step_label(step)

    @staticmethod
    def _capability_step_message(label: str, step: CapabilityStep) -> str:
        return RuntimeObservability.capability_step_message(label, step)

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

    def _decision_note_only_tokens(
        self, context: MainAgentContext, decision: AgentDecision
    ) -> tuple[str, ...]:
        return self._decision_engine.note_only_tokens(context, decision)

    def _run_owned_workflow_turn(
        self, *, context: MainAgentContext, user_message: str
    ) -> MainAgentTurnResult:
        return self._turn_router.run_owned_workflow_turn(
            context=context,
            user_message=user_message,
        )

    def _hydrate_career_context(self, state: MainAgentState) -> MainAgentState:
        context = state["context"]
        free_text_scope_keys = self._free_text_preference_scope_keys(context)
        # A prelude read that brought the turn here has been observed; the
        # model's first decision must not inherit its policy ownership.
        pending: PendingAction = {}
        if self._career_context_projector is None:
            return {
                "pending": pending,
                "career_memory_scope_keys": free_text_scope_keys,
            }
        memory = self._career_context_projector.project(
            user_id=context.profile.user_id,
            query=context.user_message,
        )
        return {
            "pending": pending,
            "context": context.model_copy(update={"career_memory": memory}),
            "career_memory_scope_keys": tuple(
                dict.fromkeys(
                    (
                        *(binding.entry_id for binding in memory.telemetry_bindings),
                        *free_text_scope_keys,
                    )
                )
            ),
        }

    @staticmethod
    def _free_text_preference_scope_keys(
        context: MainAgentContext,
    ) -> tuple[str, ...]:
        return TurnRouter.free_text_preference_scope_keys(context)

    @staticmethod
    def _tool_call_fingerprint(decision: AgentDecision) -> str:
        return DecisionEngine.tool_call_fingerprint(decision)

    @staticmethod
    def _route_decision(
        state: MainAgentState,
    ) -> Literal["authorize", "present", "interrupt"]:
        return GraphRoutingPolicy.route_decision(state)

    @staticmethod
    def _control(state: MainAgentState) -> LoopControl:
        return state.get("control", {})

    @staticmethod
    def _last_result(state: MainAgentState) -> MainAgentToolOutput | None:
        return AgentLoop.last_result(state)

    def _authorize(self, state: MainAgentState) -> MainAgentState:
        return self._authorization_engine.authorize(state)

    def _budget_bucket(
        self, control: LoopControl, *, name: str, effect: ToolEffect
    ) -> tuple[str, int, int]:
        # Compatibility shim for callers that inspect budget classification.
        return self._authorization_engine.budget_bucket(
            control, name=name, effect=effect
        )

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

    @staticmethod
    def _reraise_security_refusal(error: ValueError) -> None:
        """Keep the least-privilege boundary hard, unlike a soft refusal.

        A projection error says one of two things: the object the model named is
        not there (a wrong but ordinary choice, softened below), or the model is
        reaching for identifiers or argument shapes it must never be able to
        touch. The second must still kill the turn before it commits anything:
        softening it would turn the guard into a suggestion.
        """
        message = str(error)
        if (
            "cannot accept internal identifier" in message
            or "Extra inputs are not permitted" in message
            or message.startswith("Unknown ")
        ):
            raise error

    @staticmethod
    def _rejection_observation(name: str, error: ValueError) -> ToolObservation:
        """The soft form of a projection refusal, safe to present.

        A model-selected tool whose preconditions fail at projection used to
        raise through the whole turn, killing it with a canned failure. That
        gave the model no way to recover and the user no say. Now the refusal
        returns as an ordinary result and the model decides what to do with it —
        re-select, list what is available, or ask the user for the one thing
        only the user has.

        The state is unconditional. An earlier version chose between
        ``needs_user`` and ``invalid_input`` by looking up the capability in a
        reroute table; that decision now belongs to the model, and the counter
        in ``_authorize`` bounds how many times it may take it.
        """
        return ToolObservation(
            tool_name=name,
            state="invalid_input",
            message=f"这步暂时做不到：{error}。",
            # The one thing the state cannot say: this is not a failure to retry
            # but a selection that did not hold. Deleting ``REROUTE_FIELDS`` gave
            # the model this decision; leaving the hint empty would have given it
            # the decision without the knowledge the table used to carry.
            next_action=(
                "这不是失败，是选择或参数不成立。原样重试没有意义："
                "换一个已经在上下文里的对象，或者向用户要一个只有他才有的信息。"
            ),
        )

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
    def _validated(model, payload: object):
        return ResultPresenter.validated(
            model, payload, report_degraded=MainAgentRuntime._emit_trace
        )

    @staticmethod
    def _mock_interview_result(
        result: MainAgentToolOutput,
    ) -> MockInterviewGraphResult | None:
        return ResultPresenter.mock_interview_result(
            result, report_degraded=MainAgentRuntime._emit_trace
        )

    @staticmethod
    def _has_backed_card(result: MainAgentToolOutput) -> bool:
        return TurnPresenter.has_backed_card(result)

    @staticmethod
    def _resume_job_match_result(
        result: MainAgentToolOutput,
    ) -> ResumeJobMatchResult | None:
        return ResultPresenter.resume_job_match_result(
            result, report_degraded=MainAgentRuntime._emit_trace
        )

    @staticmethod
    def _job_analysis_result(
        result: MainAgentToolOutput,
    ) -> JobAnalysisResult | None:
        return ResultPresenter.job_analysis_result(
            result, report_degraded=MainAgentRuntime._emit_trace
        )

    @staticmethod
    def _resume_tailoring_result(
        result: MainAgentToolOutput,
    ) -> ResumeTailoringResult | None:
        return ResultPresenter.resume_tailoring_result(
            result, report_degraded=MainAgentRuntime._emit_trace
        )

    @staticmethod
    def _job_comparison(result: ToolObservation) -> JobComparison | None:
        return ResultPresenter.job_comparison(result)

    @staticmethod
    def _interview_preparation_result(
        result: ToolObservation,
    ) -> InterviewPreparationResult | None:
        return ResultPresenter.interview_preparation_result(result)

    @staticmethod
    def _job_research_draft(result: ToolObservation) -> JobResearchDraft | None:
        return ResultPresenter.job_research_draft(result)

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
