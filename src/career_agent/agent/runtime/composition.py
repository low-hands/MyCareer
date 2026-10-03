"""Composition root for the main-agent runtime.

This module owns concrete component construction and dependency wiring without
depending on the ``MainAgentRuntime`` facade.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph

from career_agent.agent.runtime.authorization_engine import AuthorizationEngine
from career_agent.agent.runtime.checkpoint_lifecycle import CheckpointLifecycle
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
from career_agent.agent.runtime.observation_reducer import (
    ObservationReducer,
    tool_observation,
)
from career_agent.agent.runtime.ports import DecisionMakerSlot, RuntimePorts
from career_agent.agent.runtime.suspension import suspend_for_interaction
from career_agent.agent.presentation.engine import PresentationEngine
from career_agent.agent.presentation.interaction_renderer import InteractionRenderer
from career_agent.agent.presentation.presenter import TurnPresenter
from career_agent.agent.presentation.factory import build_turn_presenter
from career_agent.agent.presentation.stream_adapter import StreamAdapter
from career_agent.agent.middleware.argument_projection import (
    project_runtime_owned_arguments,
    project_workflow_arguments,
)
from career_agent.agent.runtime.observability import RuntimeObservability
from career_agent.agent.runtime.turn_coordinator import (
    TurnCoordinator,
    TurnLifecycleOperations,
    active_turn_id,
)
from career_agent.harness.agent_loop import AgentLoop, main_graph_thread_id
from career_agent.harness.confirmation_coordinator import (
    CapabilityConfirmationCoordinator,
)
from career_agent.harness.context_hydrator import ContextHydrator
from career_agent.harness.graph_routing import GraphRoutingPolicy
from career_agent.harness.observability import TraceRecorder
from career_agent.harness.turn_router import TurnRouter
from career_agent.services.episode_reconciliation import EpisodeReconciler
from career_agent.storage.action_executions import SQLiteActionExecutionStore
from career_agent.storage.capability_confirmations import (
    SQLiteCapabilityConfirmationStore,
)
from career_agent.storage.turn_receipts import SQLiteTurnReceiptStore

@dataclass(frozen=True)
class RuntimeComponents:
    """Fully assembled collaborators owned by ``MainAgentRuntime``."""

    runtime_observability: RuntimeObservability
    reconciliation: ReconciliationCoordinator
    context_builder: TurnContextBuilder
    turn_coordinator: TurnCoordinator
    authorization_engine: AuthorizationEngine
    capability_executor: CapabilityExecutor
    observation_reducer: ObservationReducer
    decision_engine: DecisionEngine
    interaction_coordinator: InteractionCoordinator
    turn_presenter: TurnPresenter
    interaction_renderer: InteractionRenderer
    stream_adapter: StreamAdapter
    context_hydrator: ContextHydrator
    graph: CompiledStateGraph
    agent_loop: AgentLoop
    turn_router: TurnRouter
    confirmation_coordinator: CapabilityConfirmationCoordinator


def build_main_runtime_components(
    *,
    context_manager: ContextManager,
    decision_maker: DecisionMaker,
    decision_maker_slot: DecisionMakerSlot,
    tools: MainAgentToolRegistry,
    ports: RuntimePorts,
    decision_heartbeat_seconds: float,
    checkpointer: Any | None,
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
) -> RuntimeComponents:
    """Build every collaborator from explicit dependencies."""

    runtime_observability = RuntimeObservability(
        trace_recorder=trace_recorder,
    )
    reconciliation = ReconciliationCoordinator(
        context_manager=context_manager,
        action_execution_store=action_execution_store,
        episode_reconciler=episode_reconciler,
    )
    context_builder = TurnContextBuilder(
        context_manager=context_manager,
        tools=tools,
        owns_next_turn=TurnRouter.owns_next_turn,
    )
    authorization_engine = AuthorizationEngine(
        tools=tools,
        confirmation_store=capability_confirmation_store,
        offers_tool=TurnRouter.offers_tool,
        project_runtime_workflow_arguments=project_runtime_owned_arguments,
        project_atomic_tool_arguments=ports.project_atomic_tool_arguments,
        project_workflow_arguments=project_workflow_arguments,
        record_trace_event=RuntimeObservability.record_trace_event,
        max_read_calls=max_read_calls,
        max_write_calls=max_write_calls,
        max_external_write_calls=max_external_write_calls,
        max_projection_refusals=max_projection_refusals,
        max_authorization_refusals=max_authorization_refusals,
        max_failure_retries=max_failure_retries,
    )
    capability_executor = CapabilityExecutor(
        tools=tools,
        action_execution_store=action_execution_store,
        action_policy_epoch=action_policy_epoch,
        emit_capability_started=ports.emit_capability_started,
        emit_capability_completed=ports.emit_capability_completed,
        run_capability=lambda pending, run: RuntimeObservability.run_capability(
            pending,
            run,
            heartbeat_interval=decision_heartbeat_seconds,
        ),
    )
    observation_reducer = ObservationReducer(
        context_manager=context_manager,
        emit_trace=RuntimeObservability.emit_trace,
        update_mock_interview_task=ports.update_mock_interview_task,
        update_atomic_task=ports.update_atomic_task,
        tool_call_fingerprint=DecisionEngine.tool_call_fingerprint,
        tool_observation=tool_observation,
    )
    decision_engine = DecisionEngine(
        emit=RuntimeObservability.emit,
        decision_heartbeat=lambda sink: RuntimeObservability.heartbeat(
            sink,
            interval=decision_heartbeat_seconds,
            stage="deciding",
            describe=lambda waited: (
                f"仍在等待模型判断（已等待 {waited} 秒）……"
            ),
        ),
        record_trace_event=RuntimeObservability.record_trace_event,
        project_atomic_tool_arguments=ports.project_atomic_tool_arguments,
        project_workflow_arguments=project_workflow_arguments,
        context_manager=context_manager,
        decision_maker_provider=decision_maker_slot.get,
        tools=tools,
        career_memory_enabled=career_context_projector is not None,
    )
    interaction_coordinator = InteractionCoordinator(
        context_manager=context_manager,
        tools=tools,
        confirmation_store=capability_confirmation_store,
    )
    turn_presenter = build_turn_presenter(
        report_degraded=RuntimeObservability.emit_trace,
    )
    interaction_renderer = InteractionRenderer(
        active_turn_id=active_turn_id,
        assistant_message=turn_presenter._assistant_message,
        has_interaction_renderer=ports.has_interaction_renderer,
    )
    stream_adapter = StreamAdapter(
        interaction_renderer=interaction_renderer,
        presenter=turn_presenter,
        emit=RuntimeObservability.emit,
    )
    context_hydrator = ContextHydrator(
        career_context_projector=career_context_projector,
    )

    _configure_request_token_estimator(
        context_manager=context_manager,
        decision_maker=decision_maker,
        context_hydrator=context_hydrator,
        decision_engine=decision_engine,
    )

    effective_checkpointer = checkpointer or InMemorySaver()
    graph = build_main_graph(
        MainGraphNodes(
            hydrate=context_hydrator.hydrate,
            decide=decision_engine.decide,
            authorize=authorization_engine.authorize,
            act=capability_executor.act,
            observe=observation_reducer.reduce,
            present=lambda state: PresentationEngine.present(
                state,
                renderer=turn_presenter,
            ),
            interrupt=interaction_renderer.interrupt,
            suspend=suspend_for_interaction,
            route_entry=GraphRoutingPolicy.route_entry,
            route_decision=GraphRoutingPolicy.route_decision,
            after_authorize=GraphRoutingPolicy.after_authorize,
            after_observe=GraphRoutingPolicy.after_observe,
        ),
        checkpointer=effective_checkpointer,
    )
    agent_loop = AgentLoop(
        graph=graph,
        memory_scope_keys=ContextHydrator.free_text_preference_scope_keys,
        # Keep artifact support lazy for narrow registry doubles.
        deliver_resume_artifact=lambda **kwargs: (
            tools.deliver_resume_artifact(**kwargs)
        ),
    )
    checkpoint_lifecycle = CheckpointLifecycle(
        checkpointer=effective_checkpointer,
        thread_id=main_graph_thread_id,
    )
    turn_router = TurnRouter(
        context_manager=context_manager,
        confirmation_store=capability_confirmation_store,
        agent_loop=agent_loop,
    )
    confirmation_coordinator = CapabilityConfirmationCoordinator(
        confirmation_store=capability_confirmation_store,
        interaction_coordinator=interaction_coordinator,
        agent_loop=agent_loop,
    )
    turn_coordinator = TurnCoordinator(
        operations=TurnLifecycleOperations(
            emit=RuntimeObservability.emit,
            reconcile_episodes=reconciliation.reconcile_episodes,
            invalidate_episode_reconciliation=(
                reconciliation.invalidate_episode_reconciliation
            ),
            owns_next_turn=TurnRouter.owns_next_turn,
            prepare_questionnaire_continuation=(
                interaction_coordinator.prepare_questionnaire_continuation
            ),
            run_loaded_context=turn_router.run_loaded_context,
            resume_questionnaire=agent_loop.resume_model,
            settle_checkpoint=checkpoint_lifecycle.settle,
            run_interaction_response=(
                confirmation_coordinator.run_owner_confirmation
            ),
            run_owned_workflow_turn=turn_router.run_owned_workflow_turn,
            commit_interrupted_turn=reconciliation.commit_interrupted_turn,
            conversation_content=TurnPresenter.conversation_content,
            durable_screen=TurnPresenter.durable_screen,
            turn_resource_refs=TurnPresenter.turn_resource_refs,
            delivered_bodies=turn_presenter.delivered_bodies,
            active_turn_id=active_turn_id,
            attach_destructive_confirmation=(
                confirmation_coordinator.attach_destructive_confirmation
            ),
            deliver_reply=stream_adapter.deliver_reply,
            record_turn=runtime_observability.record_turn,
            deliver_stream_events=stream_adapter.deliver_events,
            record_turn_failed=runtime_observability.record_turn_failed,
            emit_turn_failure=RuntimeObservability.emit_turn_failure,
        ),
        context_manager=context_manager,
        context_builder=context_builder,
        receipt_store=turn_receipt_store,
        trace_recorder=trace_recorder,
    )
    return RuntimeComponents(
        runtime_observability=runtime_observability,
        reconciliation=reconciliation,
        context_builder=context_builder,
        turn_coordinator=turn_coordinator,
        authorization_engine=authorization_engine,
        capability_executor=capability_executor,
        observation_reducer=observation_reducer,
        decision_engine=decision_engine,
        interaction_coordinator=interaction_coordinator,
        turn_presenter=turn_presenter,
        interaction_renderer=interaction_renderer,
        stream_adapter=stream_adapter,
        context_hydrator=context_hydrator,
        graph=graph,
        agent_loop=agent_loop,
        turn_router=turn_router,
        confirmation_coordinator=confirmation_coordinator,
    )


def _configure_request_token_estimator(
    *,
    context_manager: ContextManager,
    decision_maker: DecisionMaker,
    context_hydrator: ContextHydrator,
    decision_engine: DecisionEngine,
) -> None:
    request_token_usage = getattr(decision_maker, "request_token_usage", None)
    if not callable(request_token_usage):
        return

    def estimate_complete_request(context: MainAgentContext) -> tuple[int, int]:
        hydrated = context_hydrator.project_context(context)
        return request_token_usage(
            hydrated,
            decision_engine.tool_schemas(
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
                decision_engine.tool_schemas(profile)
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
