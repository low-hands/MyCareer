from __future__ import annotations

from typing import Any, Literal

from career_agent.agent.context.manager import ContextManager
from career_agent.agent.resources.conversation_span import explicit_sequence_span
from career_agent.agent.contracts.context import MainAgentContext
from career_agent.agent.contracts.decisions import (
    AgentDecision,
    ToolCall,
)
from career_agent.agent.contracts.task_state import ConversationTaskState
from career_agent.agent.runtime.state import MainAgentState
from career_agent.agent.capabilities.selection_strategy import SearchStrategy
from career_agent.agent.contracts.turn import (
    MainAgentTurnResult,
    RuntimeAction,
    RuntimePolicyAction,
)
from career_agent.harness.agent_loop import AgentLoop
from career_agent.services.free_text_preferences import is_explicit_confirmation
from career_agent.storage.capability_confirmations import (
    SQLiteCapabilityConfirmationStore,
)


PolicyName = Literal[
    "free_text_preference_confirmation",
    "free_text_preference_activation",
    "career_fact_confirmation",
    "job_intent_confirmation",
]


class TurnRouter:
    """Choose a turn ingress before the fixed LangGraph topology runs."""

    def __init__(
        self,
        *,
        context_manager: ContextManager,
        confirmation_store: SQLiteCapabilityConfirmationStore | None,
        agent_loop: AgentLoop,
        selection_strategy: SearchStrategy | None = None,
    ) -> None:
        self._context_manager = context_manager
        self._confirmation_store = confirmation_store
        self._agent_loop = agent_loop
        self._selection_strategy = selection_strategy or SearchStrategy()

    def accepts_background_turn(self, *, user_id: str, conversation_id: str) -> bool:
        task = self._context_manager.get_task(
            user_id=user_id,
            conversation_id=conversation_id,
        )
        if self.owns_next_turn(task):
            return False
        store = self._confirmation_store
        return store is None or not store.pending_for_conversation(
            user_id=user_id,
            conversation_id=conversation_id,
            policy_revision=None,
        )

    @staticmethod
    def owns_next_turn(task: ConversationTaskState) -> bool:
        """Whether a resumable mock interview consumes the next user message."""

        return task.active_workflow == "mock_interview" and task.phase not in {
            "mock_interview_checkpoint_missing",
            "mock_interview_graph_incompatible",
        }

    def run_loaded_context(
        self,
        context: MainAgentContext,
        *,
        bare_confirmation_target: Literal[
            "career_fact",
            "job_intent",
            "free_text_preference",
        ]
        | None = None,
    ) -> MainAgentTurnResult:
        if (
            bare_confirmation_target == "career_fact"
            and context.task.pending_career_fact is not None
            and is_explicit_confirmation(context.user_message)
        ):
            return self.run_runtime_policy_tool(
                context,
                policy="career_fact_confirmation",
                tool_name="confirm_career_fact",
                arguments={},
            )
        if (
            bare_confirmation_target == "job_intent"
            and context.task.pending_job_intent_update is not None
            and is_explicit_confirmation(context.user_message)
        ):
            return self.run_runtime_policy_tool(
                context,
                policy="job_intent_confirmation",
                tool_name="confirm_job_intent",
                arguments={},
            )
        if (
            bare_confirmation_target == "free_text_preference"
            and context.task.pending_free_text_preference is not None
            and not context.task.pending_free_text_preference.needs_scope_clarification
            and is_explicit_confirmation(context.user_message)
        ):
            return self.run_runtime_policy_tool(
                context,
                policy="free_text_preference_activation",
                tool_name="confirm_free_text_preference",
                arguments={},
            )
        if (
            context.task.pending_free_text_preference is None
            and any(
                item.status == "quarantined"
                for item in context.free_text_preferences
            )
        ):
            return self.run_runtime_policy_tool(
                context,
                policy="free_text_preference_confirmation",
                tool_name="propose_free_text_preference_confirmation",
                arguments={"selection_index": 1},
            )
        return self._agent_loop.run_model(
            context,
            prelude=self.explicit_span_prelude(context),
        )

    def explicit_span_prelude(self, context: MainAgentContext) -> MainAgentState:
        if context.through_sequence < 1 or context.recent_from_sequence is None:
            return {}
        span = explicit_sequence_span(context.user_message)
        if span is None or not self._selection_strategy.offers_tool(
            context, "read_conversation_span",
        ):
            return {}
        arguments = {
            "from_sequence": span.from_sequence,
            "through_sequence": span.through_sequence,
        }
        return {
            "decision": AgentDecision(
                action="tool_call",
                tool_call=ToolCall(
                    name="read_conversation_span",
                    arguments=arguments,
                ),
            ),
            "pending": {
                "name": "read_conversation_span",
                "policy_owned": True,
                "policy_prelude": True,
                "arguments": arguments,
            },
        }

    def run_runtime_policy_tool(
        self,
        context: MainAgentContext,
        *,
        policy: PolicyName,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> MainAgentTurnResult:
        decision = AgentDecision(
            action="tool_call",
            tool_call=ToolCall(name=tool_name, arguments=arguments),
        )
        turn = self._agent_loop.run_decided(
            context,
            decision=decision,
            pending={
                "name": tool_name,
                "policy_owned": True,
                "arguments": arguments,
            },
            origin=RuntimePolicyAction(policy=policy),
        )
        if turn.tool_result is None:
            raise RuntimeError(f"{policy} policy produced no result")
        return turn

    def run_owned_workflow_turn(
        self,
        *,
        context: MainAgentContext,
        user_message: str,
    ) -> MainAgentTurnResult:
        session_id = context.task.run_id
        if session_id is None:
            raise ValueError("Active mock interview has no resumable session")
        entry = (
            "retry_mock_interview"
            if context.task.phase == "failed"
            else "handle_mock_interview_input"
        )
        owned_workflow = context.task.active_workflow
        if owned_workflow == "none":
            raise ValueError("owned workflow turn requires an active workflow")
        decision = AgentDecision(
            action="tool_call",
            tool_call=ToolCall(name=entry, arguments={}),
        )
        runtime_arguments = (
            {} if entry == "retry_mock_interview" else {"message": user_message}
        )
        return self._agent_loop.run_decided(
            context,
            decision=decision,
            pending={
                "name": entry,
                "runtime_owned": True,
                "arguments": runtime_arguments,
            },
            origin=RuntimeAction(workflow=owned_workflow),
        )
