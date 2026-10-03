"""Composition root for the main-agent runtime.

This module owns concrete component construction and dependency wiring.  The
runtime is touched only while it is being assembled; constructed components
retain their explicit narrow dependencies rather than a runtime host.
"""

from __future__ import annotations

from typing import Any

from career_agent.agent.runtime.authorization_engine import AuthorizationEngine
from career_agent.agent.context.career import CareerContextProjector
from career_agent.agent.context.turn_builder import TurnContextBuilder
from career_agent.agent.context.manager import ContextManager
from career_agent.agent.runtime.decision_engine import DecisionEngine
from career_agent.agent.execution.capability_executor import CapabilityExecutor
from career_agent.agent.execution.reconciliation import ReconciliationCoordinator
from career_agent.agent.runtime.interaction_coordinator import InteractionCoordinator
from career_agent.agent.contracts.main_agent import (
    DecisionMaker,
    MainAgentContext,
    TOOL_PROFILE_NAMES,
)
from career_agent.agent.capabilities.registry import MainAgentToolRegistry
from career_agent.agent.runtime.graph import MainGraphNodes, build_main_graph
from career_agent.agent.runtime.observation_reducer import ObservationReducer
from career_agent.agent.presentation.interaction_renderer import InteractionRenderer
from career_agent.agent.presentation.stream_adapter import StreamAdapter
from career_agent.agent.runtime.observability import RuntimeObservability
from career_agent.agent.runtime.turn_coordinator import (
    TurnCoordinator,
    TurnLifecycleOperations,
)
from career_agent.harness.agent_loop import AgentLoop
from career_agent.harness.confirmation_coordinator import (
    CapabilityConfirmationCoordinator,
)
from career_agent.harness.context_hydrator import ContextHydrator
from career_agent.harness.observability import TraceRecorder
from career_agent.harness.turn_router import TurnRouter
from career_agent.services.episode_reconciliation import EpisodeReconciler
from career_agent.storage.action_executions import SQLiteActionExecutionStore
from career_agent.storage.capability_confirmations import (
    SQLiteCapabilityConfirmationStore,
)
from career_agent.storage.turn_receipts import SQLiteTurnReceiptStore


def install_main_runtime_components(
    runtime: Any,
    *,
    context_manager: ContextManager,
    decision_maker: DecisionMaker,
    tools: MainAgentToolRegistry,
    career_context_projector: CareerContextProjector | None,
    max_read_calls: int,
    max_write_calls: int,
    max_external_write_calls: int,
    max_projection_refusals: int,
    max_authorization_refusals: int,
    max_failure_retries: int,
    trace_recorder: TraceRecorder | None,
    action_execution_store: SQLiteActionExecutionStore | None,
    capability_confirmation_store: SQLiteCapabilityConfirmationStore | None,
    turn_receipt_store: SQLiteTurnReceiptStore | None,
    action_policy_epoch: int,
    episode_reconciler: EpisodeReconciler | None,
) -> None:
    """Build and install every collaborating component in dependency order."""

    runtime._runtime_observability = RuntimeObservability(
        trace_recorder=trace_recorder,
    )
    runtime._reconciliation = ReconciliationCoordinator(
        context_manager=context_manager,
        action_execution_store=action_execution_store,
        episode_reconciler=episode_reconciler,
    )
    runtime._context_builder = TurnContextBuilder(
        context_manager=context_manager,
        tools=tools,
        owns_next_turn=runtime._owns_next_turn,
    )
    runtime._turn_coordinator = TurnCoordinator(
        operations=TurnLifecycleOperations(
            emit=runtime._emit,
            reconcile_episodes=runtime._reconcile_episodes,
            invalidate_episode_reconciliation=(
                runtime._invalidate_episode_reconciliation
            ),
            owns_next_turn=runtime._owns_next_turn,
            prepare_questionnaire_continuation=(
                runtime._prepare_questionnaire_continuation
            ),
            run_loaded_context=runtime._run_loaded_context,
            run_interaction_response=runtime._run_interaction_response,
            run_owned_workflow_turn=runtime._run_owned_workflow_turn,
            commit_interrupted_turn=runtime._commit_interrupted_turn,
            conversation_content=runtime._conversation_content,
            durable_screen=runtime._durable_screen,
            turn_resource_refs=runtime._turn_resource_refs,
            delivered_bodies=runtime._delivered_bodies,
            active_turn_id=runtime._active_turn_id,
            attach_destructive_confirmation=(
                runtime._attach_destructive_confirmation
            ),
            deliver_reply=runtime._deliver_reply,
            record_turn=runtime._record_turn,
            deliver_stream_events=runtime._deliver_stream_events,
            record_turn_failed=runtime._record_turn_failed,
            emit_turn_failure=runtime._emit_turn_failure,
        ),
        context_manager=context_manager,
        context_builder=runtime._context_builder,
        receipt_store=turn_receipt_store,
        trace_recorder=trace_recorder,
    )
    runtime._authorization_engine = AuthorizationEngine(
        tools=tools,
        confirmation_store=capability_confirmation_store,
        offers_tool=runtime._offers_tool,
        project_runtime_workflow_arguments=(
            runtime._project_runtime_workflow_arguments
        ),
        project_atomic_tool_arguments=runtime._project_atomic_tool_arguments,
        project_workflow_arguments=runtime._project_workflow_arguments,
        record_trace_event=runtime._record_trace_event,
        max_read_calls=max_read_calls,
        max_write_calls=max_write_calls,
        max_external_write_calls=max_external_write_calls,
        max_projection_refusals=max_projection_refusals,
        max_authorization_refusals=max_authorization_refusals,
        max_failure_retries=max_failure_retries,
    )
    runtime._capability_executor = CapabilityExecutor(
        tools=tools,
        action_execution_store=action_execution_store,
        action_policy_epoch=action_policy_epoch,
        emit_capability_started=runtime._emit_capability_started,
        emit_capability_completed=runtime._emit_capability_completed,
        run_capability=runtime._run_capability,
    )
    runtime._observation_reducer = ObservationReducer(
        context_manager=context_manager,
        emit_trace=runtime._emit_trace,
        update_mock_interview_task=runtime._update_mock_interview_task,
        update_atomic_task=runtime._update_atomic_task,
        tool_call_fingerprint=runtime._tool_call_fingerprint,
        tool_observation=runtime._tool_observation,
    )
    runtime._decision_engine = DecisionEngine(
        emit=runtime._emit,
        decision_heartbeat=runtime._decision_heartbeat,
        record_trace_event=runtime._record_trace_event,
        project_atomic_tool_arguments=runtime._project_atomic_tool_arguments,
        project_workflow_arguments=runtime._project_workflow_arguments,
        context_manager=context_manager,
        decision_maker_provider=lambda: runtime._decision_maker,
        tools=tools,
        career_memory_enabled=career_context_projector is not None,
    )
    runtime._interaction_coordinator = InteractionCoordinator(
        context_manager=context_manager,
        tools=tools,
        confirmation_store=capability_confirmation_store,
    )
    runtime._turn_presenter = runtime._presentation_adapter()
    runtime._interaction_renderer = InteractionRenderer(
        active_turn_id=runtime._active_turn_id,
        assistant_message=runtime._turn_presenter._assistant_message,
        has_interaction_renderer=runtime._has_interaction_renderer,
    )
    runtime._stream_adapter = StreamAdapter(
        interaction_renderer=runtime._interaction_renderer,
        presenter=runtime._turn_presenter,
        emit=runtime._emit,
    )
    runtime._context_hydrator = ContextHydrator(
        career_context_projector=career_context_projector,
    )

    _configure_request_token_estimator(
        runtime,
        context_manager=context_manager,
        decision_maker=decision_maker,
    )

    runtime._graph = build_main_graph(
        MainGraphNodes(
            hydrate=runtime._hydrate_career_context,
            decide=runtime._decide,
            authorize=runtime._authorize,
            act=runtime._act,
            observe=runtime._observe,
            present=runtime._present,
            interrupt=runtime._interrupt,
            route_entry=runtime._route_entry,
            route_decision=runtime._route_decision,
            after_authorize=runtime._after_authorize,
            after_observe=runtime._after_observe,
        )
    )
    runtime._agent_loop = AgentLoop(
        graph=runtime._graph,
        memory_scope_keys=ContextHydrator.free_text_preference_scope_keys,
        # Keep artifact support lazy for narrow registry doubles.
        deliver_resume_artifact=lambda **kwargs: (
            runtime._tools.deliver_resume_artifact(**kwargs)
        ),
    )
    runtime._turn_router = TurnRouter(
        context_manager=context_manager,
        confirmation_store=capability_confirmation_store,
        agent_loop=runtime._agent_loop,
    )
    runtime._confirmation_coordinator = CapabilityConfirmationCoordinator(
        confirmation_store=capability_confirmation_store,
        interaction_coordinator=runtime._interaction_coordinator,
        agent_loop=runtime._agent_loop,
    )


def _configure_request_token_estimator(
    runtime: Any,
    *,
    context_manager: ContextManager,
    decision_maker: DecisionMaker,
) -> None:
    request_token_usage = getattr(decision_maker, "request_token_usage", None)
    if not callable(request_token_usage):
        return

    def estimate_complete_request(context: MainAgentContext) -> tuple[int, int]:
        hydrated = runtime._context_hydrator.project_context(context)
        return request_token_usage(
            hydrated,
            runtime._decision_tool_schemas(
                hydrated.task.tool_profile,
                hydrated.task,
            ),
        )

    static_request_token_usage = getattr(
        decision_maker, "static_request_token_usage", None
    )
    if not callable(static_request_token_usage):
        raise ValueError(
            "decision makers that report request token usage must also "
            "report static request token usage"
        )
    static_tokens, max_input_tokens = max(
        (
            static_request_token_usage(
                runtime._decision_tool_schemas(profile)
            )
            for profile in TOOL_PROFILE_NAMES
        ),
        key=lambda usage: usage[0],
    )
    context_manager.configure_request_token_estimator(
        estimate_complete_request,
        static_input_tokens=static_tokens,
        max_input_tokens=max_input_tokens,
    )
