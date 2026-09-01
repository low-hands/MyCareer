from career_agent.agent.context_manager import ContextManager
import pytest

from career_agent.agent.main_agent_contracts import (
    AgentDecision,
    CandidateContextItem,
    CareerProfileContext,
    ConversationTaskState,
    ToolCall,
)
from career_agent.agent.main_agent_runtime import MainAgentRuntime
from career_agent.agent.main_agent_tools import MainAgentToolRegistry
from career_agent.services.email_tracking import EmailSyncResult
from career_agent.storage.context import CareerContextStore


class EmailService:
    def __init__(self):
        self.calls = []

    def sync(self, **kwargs):
        self.calls.append(kwargs)
        return EmailSyncResult(
            accounts_synced=2,
            messages_seen=3,
            candidate_messages=1,
            events_created=(),
        )

    def list_events(self, **kwargs):
        return ()


class Decisions:
    def __init__(self):
        self.values = [
            AgentDecision(
                action="tool_call",
                tool_call=ToolCall(name="sync_application_emails", arguments={}),
            ),
            AgentDecision(action="final", message="邮箱已检查。"),
        ]

    def decide(self, context, tool_specs):
        names = {spec["function"]["name"] for spec in tool_specs}
        assert {
            "sync_application_emails",
            "list_email_events",
        }.issubset(names)
        return self.values.pop(0)


def test_email_sync_is_registered_as_workflow_and_updates_task_state(tmp_path) -> None:
    manager = ContextManager(CareerContextStore(tmp_path / "context.sqlite3"))
    manager.upsert_profile(CareerProfileContext(user_id="u1"))
    email_service = EmailService()
    tools = MainAgentToolRegistry(
        email_tracking_service=email_service,
    )
    runtime = MainAgentRuntime(
        context_manager=manager,
        decision_maker=Decisions(),
        tools=tools,
    )

    result = runtime.run_turn(
        user_id="u1", conversation_id="c1", user_message="检查一下邮箱进展"
    )

    assert tools.capability_kind("sync_application_emails") == "workflow"
    assert tools.capability_kind("list_email_events") == "atomic_tool"
    assert email_service.calls == [{"user_id": "u1", "account_id": None}]
    # Email sync records its own phase and leaves the workflow slot free, so a
    # suspended job-discovery run stays resumable across the sync.
    assert result.context.task.email_sync_phase == "email_sync_complete"
    assert result.context.task.active_workflow == "none"
    assert result.context.task.phase is None


def test_email_sync_does_not_evict_a_suspended_job_discovery_run(tmp_path) -> None:
    manager = ContextManager(CareerContextStore(tmp_path / "context.sqlite3"))
    manager.upsert_profile(CareerProfileContext(user_id="u1"))
    suspended = ConversationTaskState().enter_workflow(
        "job_discovery",
        run_id="run-1",
        phase="selection_required",
        candidates=(
            CandidateContextItem(
                result_ref="boss:1", title="AI Engineer", company_name="Acme"
            ),
        ),
    )
    manager.commit_turn(
        context=manager.load_for_turn(
            user_id="u1", conversation_id="c1", user_message="找工作"
        ),
        task=suspended,
        assistant_message="请选择一个岗位。",
    )
    runtime = MainAgentRuntime(
        context_manager=manager,
        decision_maker=Decisions(),
        tools=MainAgentToolRegistry(email_tracking_service=EmailService()),
    )

    result = runtime.run_turn(
        user_id="u1", conversation_id="c1", user_message="先看下邮箱"
    )

    # The pending selection survives the detour, so the next turn can still
    # resume run-1 instead of being told to start over.
    task = result.context.task
    assert (task.active_workflow, task.run_id, task.phase) == (
        "job_discovery",
        "run-1",
        "selection_required",
    )
    assert task.email_sync_phase == "email_sync_complete"


def test_workflow_slot_and_run_id_cannot_drift_apart() -> None:
    with pytest.raises(ValueError):
        ConversationTaskState(active_workflow="job_discovery")
    with pytest.raises(ValueError):
        ConversationTaskState(run_id="run-1")

    left = ConversationTaskState().enter_workflow(
        "job_discovery", run_id="run-1", phase="selection_required"
    ).leave_workflow()
    assert (left.active_workflow, left.run_id, left.phase) == ("none", None, None)
