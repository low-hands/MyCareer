from types import SimpleNamespace
from datetime import datetime, timezone

import pytest

from agent_test_support import CatalogSchemaRegistry
from career_agent.agent.capabilities.catalog import CAPABILITIES
from career_agent.agent.contracts.context import MainAgentContext
from career_agent.agent.contracts.decisions import AgentDecision, ToolCall
from career_agent.agent.contracts.observations import ToolObservation
from career_agent.agent.contracts.profile import CareerProfileContext, MemoryTombstoneProposal, ConstraintRetirementProposal
from career_agent.agent.contracts.task_state import ConversationTaskState
from career_agent.agent.context.manager import ContextManager
from career_agent.agent.middleware.approval import ApprovalMiddleware
from career_agent.agent.presentation.factory import interaction_event
from career_agent.agent.runtime.main_agent_runtime import MainAgentRuntime
from career_agent.harness.streaming import InteractionResponse
from career_agent.storage.capability_confirmations import SQLiteCapabilityConfirmationStore
from career_agent.storage.context import CareerContextStore


ALWAYS = {n for n, d in CAPABILITIES.items() if d.approval_policy == "always"}


@pytest.mark.parametrize("name", sorted(ALWAYS | {"create_interview"}))
def test_confirmation_prompt_distinguishes_external_destructive_and_owner_rules(tmp_path, name):
    task = ConversationTaskState()
    arguments = {"user_id": "u1", "conversation_id": "c1"}
    if name == "confirm_memory_tombstone":
        proposal = MemoryTombstoneProposal(target_kind="career_evidence", detail_ref="detail_" + "a" * 24, reason="删除误记")
        task = ConversationTaskState().with_pending_proposal("pending_memory_tombstone", proposal, proposed_at=datetime.now(timezone.utc))
        arguments["proposal"] = proposal
    elif name == "confirm_constraint_retirement":
        proposal = ConstraintRetirementProposal(target_kind="conversation_constraint", constraint="旧约束", reason="不再适用")
        task = ConversationTaskState().with_pending_proposal("pending_constraint_retirement", proposal, proposed_at=datetime.now(timezone.utc))
        arguments["proposal"] = proposal
    context = MainAgentContext(conversation_id="c1", profile=CareerProfileContext(user_id="u1"), user_message="执行", task=task)
    class Tools:
        def invoke_atomic_tool(self, *args):
            return ToolObservation(tool_name="get_calendar_proposal", state="calendar_proposal_ready", message="预览", payload={"operation": "create"})
    store = SQLiteCapabilityConfirmationStore(tmp_path / "confirmation.sqlite3")
    result = ApprovalMiddleware(tools=Tools(), confirmation_store=store).seal({"context": context}, name=name, arguments=arguments)
    observation = result["pending"]["result"]
    assert observation.state == "capability_confirmation_required"
    if CAPABILITIES[name].external_write:
        assert "外部写入" in observation.message
        assert "你设置了" not in observation.message
    elif CAPABILITIES[name].destructive:
        assert "系统强制要求确认" in observation.message
        assert "你设置了" not in observation.message
    else:
        assert "你设置了此操作需要确认" in observation.message
    unavailable = ApprovalMiddleware(tools=Tools(), confirmation_store=None).seal({"context": context}, name=name, arguments=arguments)
    assert ("你设置了" in unavailable.reason) == (name == "create_interview")


@pytest.mark.parametrize("name", ["resolve_email_event", "restart_mock_interview"])
def test_new_always_tools_stop_before_execution_and_execute_only_after_owner_click(tmp_path, name):
    class Registry(CatalogSchemaRegistry):
        def __init__(self):
            super().__init__()
            self.calls = []
        def capability_kind(self, name):
            return CAPABILITIES[name].execution_kind
        def invoke_atomic_tool(self, name, arguments):
            self.calls.append((name, arguments))
            return ToolObservation(tool_name=name, state="email_event_resolved", message="邮件事件已忽略。", execution_outcome="committed")
        def invoke_workflow(self, name, arguments):
            self.calls.append((name, arguments))
            return ToolObservation(tool_name=name, state="no_mock_interview_to_restart", message="没有卡住的练习。", execution_outcome="not_committed")
    arguments = {"selection_index": 1, "approve": False} if name == "resolve_email_event" else {}
    task = ConversationTaskState(
        email_event_candidates=({"email_event_id": "email-1", "event_type": "unclear", "status": "pending_confirmation", "summary": "测试邮件"},),
    ) if name == "resolve_email_event" else ConversationTaskState(
        workflow={"kind": "mock_interview", "run_id": "run-1", "phase": "mock_interview_checkpoint_missing"},
    )
    task = task.add_loaded_capabilities((name,))
    store = CareerContextStore(tmp_path / "context.sqlite3")
    store.upsert_task(user_id="u1", conversation_id="c1", task=task)
    manager = ContextManager(store)
    manager.upsert_profile(CareerProfileContext(user_id="u1"))
    decisions = [AgentDecision(action="tool_call", tool_call=ToolCall(name=name, arguments=arguments))]
    def decide(context, tool_names):
        assert decisions, "confirmation must not reconsult the model"
        return decisions.pop(0)
    registry = Registry()
    runtime = MainAgentRuntime(context_manager=manager, decision_maker=SimpleNamespace(decide=decide), tools=registry,
                               capability_confirmation_store=SQLiteCapabilityConfirmationStore(tmp_path / "context.sqlite3"))
    stopped = runtime.run_turn(user_id="u1", conversation_id="c1", user_message="执行这个操作")
    assert registry.calls == []
    assert stopped.tool_result.state == "capability_confirmation_required"
    assert "系统强制要求确认" in stopped.tool_result.message
    gate = interaction_event(result=stopped, conversation_id="c1")
    runtime.run_turn(user_id="u1", conversation_id="c1", user_message="确认",
                     interaction_response=InteractionResponse(interaction_id=gate.interaction_id, scope="capability_confirmation", action="confirm"))
    assert [n for n, _ in registry.calls] == [name]
    assert not decisions
