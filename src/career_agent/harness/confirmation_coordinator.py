from __future__ import annotations

from career_agent.agent.interaction_coordinator import InteractionCoordinator
from career_agent.agent.main_agent_contracts import (
    CONFIRMATION_SPECS,
    AgentDecision,
    MainAgentContext,
    ToolCall,
    ToolObservation,
    confirmation_arguments_snapshot,
)
from career_agent.agent.main_state import MainAgentState
from career_agent.agent.turn_models import InteractionReceipt, MainAgentTurnResult
from career_agent.harness.agent_loop import AgentLoop
from career_agent.harness.streaming import InteractionResponse
from career_agent.storage.capability_confirmations import (
    SQLiteCapabilityConfirmationStore,
)


class CapabilityConfirmationCoordinator:
    """Seal destructive proposals and resume exactly the action an owner approved."""

    def __init__(
        self,
        *,
        confirmation_store: SQLiteCapabilityConfirmationStore | None,
        interaction_coordinator: InteractionCoordinator,
        agent_loop: AgentLoop,
    ) -> None:
        self._confirmation_store = confirmation_store
        self._interaction_coordinator = interaction_coordinator
        self._agent_loop = agent_loop

    def attach_destructive_confirmation(self, result: MainAgentTurnResult) -> None:
        """Seal a destructive proposal after its task state becomes durable."""

        store = self._confirmation_store
        if store is None:
            return
        proposal_by_state = {
            spec.proposed_state: name
            for name, spec in CONFIRMATION_SPECS.items()
            if spec.requires_seal
        }
        observation = next(
            (
                item
                for item in reversed(result.tool_results)
                if item.state in proposal_by_state
            ),
            None,
        )
        if observation is None and result.tool_result is not None:
            observation = result.tool_result
        confirm_tool = (
            proposal_by_state.get(observation.state)
            if observation is not None
            else None
        )
        if confirm_tool is None:
            return
        spec = CONFIRMATION_SPECS[confirm_tool]
        proposal = getattr(result.context.task, spec.slot)
        if proposal is None:
            return
        context = result.context
        display_summary = (
            proposal.constraint
            if confirm_tool == "confirm_constraint_retirement"
            else observation.message[:500]
        )
        sealed = store.seal(
            user_id=context.profile.user_id,
            conversation_id=context.conversation_id,
            capability=confirm_tool,
            display_summary=display_summary,
            arguments=confirmation_arguments_snapshot(
                context.task,
                confirm_tool,
                user_id=context.profile.user_id,
                conversation_id=context.conversation_id,
            ),
            policy_revision=context.preferences.behavior_policy.revision,
        )
        if sealed.status != "PENDING":
            return
        marked = observation.model_copy(
            update={
                "payload": {
                    **observation.payload,
                    "confirmation_id": sealed.confirmation_id,
                    "confirmation_summary": sealed.display_summary,
                }
            }
        )
        result.tool_result = marked
        result.tool_results = tuple(
            marked if item is observation else item for item in result.tool_results
        )

    def run_owner_confirmation(
        self,
        *,
        context: MainAgentContext,
        conversation_id: str,
        response: InteractionResponse,
    ) -> MainAgentTurnResult:
        def invoke_confirmed(sealed) -> MainAgentState:
            return self._agent_loop.invoke(
                context,
                decision=AgentDecision(
                    action="tool_call",
                    tool_call=ToolCall(name=sealed.capability, arguments={}),
                ),
                pending={
                    "name": sealed.capability,
                    "arguments": sealed.arguments,
                    "owner_confirmed": True,
                    "confirmation_id": sealed.confirmation_id,
                },
            )

        resolution = self._interaction_coordinator.resolve_confirmation(
            context=context,
            conversation_id=conversation_id,
            response=response,
            invoke_confirmed=invoke_confirmed,
        )
        if resolution.graph_state is None:
            if resolution.result is None:
                raise RuntimeError("confirmation resolution has no result")
            return self.settled_turn(
                context,
                resolution.result.message,
                state=resolution.result.state,
                action=resolution.action,
            )
        return self._agent_loop.result_from_state(
            resolution.graph_state,
            origin=InteractionReceipt(
                scope="capability_confirmation",
                action=resolution.action,
            ),
        )

    @staticmethod
    def settled_turn(
        context: MainAgentContext,
        message: str,
        *,
        state: str,
        action: str = "confirm",
    ) -> MainAgentTurnResult:
        result = ToolObservation(
            tool_name="capability_confirmation",
            state=state,
            message=message,
        )
        return MainAgentTurnResult(
            origin=InteractionReceipt(
                scope="capability_confirmation",
                action=action,
            ),
            context=context,
            assistant_message=message,
            tool_result=result,
            tool_results=(result,),
        )
