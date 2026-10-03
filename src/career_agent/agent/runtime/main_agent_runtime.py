from __future__ import annotations

from typing import Any, Literal

from career_agent.agent.runtime.observability import RuntimeObservability
from career_agent.agent.runtime.ports import DecisionMakerSlot, RuntimePorts
from career_agent.agent.context.manager import ContextManager
from career_agent.agent.context.turn_builder import keyword_tool_profile
from career_agent.agent.context.career import CareerContextProjector
from career_agent.agent.contracts.main_agent import (
    ConversationTaskState,
    DecisionMaker,
    MainAgentContext,
    MAX_DECISION_OBSERVATIONS,
)
from career_agent.harness.observability import TraceRecorder
from career_agent.storage.capability_confirmations import (
    SQLiteCapabilityConfirmationStore,
)
from career_agent.agent.capabilities.registry import MainAgentToolRegistry
from career_agent.agent.runtime.state import MainAgentState
from career_agent.agent.runtime.composition import build_main_runtime_components
from career_agent.agent.contracts.turn import (
    InteractionReceipt,
    MainAgentTurnResult,
    ModelDecision,
    RuntimeAction,
    RuntimePolicyAction,
)
from career_agent.services.episode_reconciliation import EpisodeReconciler
from career_agent.harness.streaming import InteractionResponse, StreamEventSink, TurnInputResource
from career_agent.storage.action_executions import (
    SQLiteActionExecutionStore,
)
from career_agent.storage.turn_receipts import (
    SQLiteTurnReceiptStore,
)
from career_agent.agent.runtime.turn_coordinator import (
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
DEFAULT_DECISION_HEARTBEAT_SECONDS = 15.0


class MainAgentRuntime:
    def __init__(self, *, context_manager: ContextManager, decision_maker: DecisionMaker, tools: MainAgentToolRegistry, career_context_projector: CareerContextProjector | None = None, runtime_ports: RuntimePorts | None = None, decision_heartbeat_seconds: float = DEFAULT_DECISION_HEARTBEAT_SECONDS, max_read_calls: int = DEFAULT_MAX_READ_CALLS, max_write_calls: int = DEFAULT_MAX_WRITE_CALLS, max_external_write_calls: int = DEFAULT_MAX_EXTERNAL_WRITE_CALLS, max_projection_refusals: int = DEFAULT_MAX_PROJECTION_REFUSALS, max_authorization_refusals: int = DEFAULT_MAX_AUTHORIZATION_REFUSALS, max_failure_retries: int = DEFAULT_MAX_FAILURE_RETRIES, owned_resources: tuple[Any, ...] = (), trace_recorder: TraceRecorder | None = None, action_execution_store: SQLiteActionExecutionStore | None = None, capability_confirmation_store: SQLiteCapabilityConfirmationStore | None = None, turn_receipt_store: SQLiteTurnReceiptStore | None = None, action_policy_epoch: int = ACTION_EXECUTION_POLICY_EPOCH, episode_reconciler: EpisodeReconciler | None = None) -> None:
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
        if decision_heartbeat_seconds <= 0:
            raise ValueError("decision_heartbeat_seconds must be positive")
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
        context_manager.on_compaction(RuntimeObservability.announce_compaction)
        self._decision_maker_slot = DecisionMakerSlot(decision_maker)
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
        self._runtime_ports = runtime_ports or RuntimePorts()
        components = build_main_runtime_components(
            context_manager=context_manager,
            decision_maker=decision_maker,
            decision_maker_slot=self._decision_maker_slot,
            tools=tools,
            ports=self._runtime_ports,
            decision_heartbeat_seconds=decision_heartbeat_seconds,
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
        self._components = components
        self._runtime_observability = components.runtime_observability
        self._turn_coordinator = components.turn_coordinator
        self._turn_router = components.turn_router

    def close(self) -> None:
        if self._closed:
            return
        for resource in reversed(self._owned_resources):
            close = getattr(resource, "close", None)
            if close is not None:
                close()
        self._closed = True

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

    def accepts_background_turn(self, *, user_id: str, conversation_id: str) -> bool:
        return self._turn_router.accepts_background_turn(
            user_id=user_id,
            conversation_id=conversation_id,
        )

    def replace_decision_maker(self, decision_maker: DecisionMaker) -> None:
        """Replace the decision provider through its explicit runtime slot."""

        self._decision_maker_slot.replace(decision_maker)
