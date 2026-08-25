from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from career_agent.agent.context_manager import ContextManager
from career_agent.agent.main_agent_contracts import AgentDecision, CareerProfileContext, ToolCall
from career_agent.agent.main_agent_runtime import MainAgentRuntime
from career_agent.agent.main_agent_tools import MainAgentToolRegistry
from career_agent.domain.calendar import (
    CalendarChangeProposal,
    CalendarEventLink,
    CalendarEventPayload,
)
from career_agent.storage.context import CareerContextStore


class Gateway:
    def advance(self, **kwargs):
        raise AssertionError("job discovery should not run")


class Calendar:
    def __init__(self):
        now = datetime(2026, 8, 26, tzinfo=timezone.utc)
        self.proposal = CalendarChangeProposal(
            id="proposal-1", user_id="u1", calendar_account_id="account-1",
            interview_round_id="interview-1", operation="create",
            external_event_id="ca12345",
            payload=CalendarEventPayload(
                title="面试 · Acme · AI Engineer", description="岗位面试",
                start_at=now + timedelta(days=1),
                end_at=now + timedelta(days=1, hours=1),
                timezone="Asia/Shanghai",
            ),
            payload_hash="a" * 64, status="pending", created_at=now,
            expires_at=now + timedelta(minutes=15),
        )
        self.execute_calls = []

    def prepare_interview_sync(self, **kwargs):
        return self.proposal

    def execute_proposal(self, **kwargs):
        self.execute_calls.append(kwargs)
        executed = self.proposal.model_copy(
            update={"status": "executed", "executed_at": self.proposal.created_at}
        )
        link = CalendarEventLink(
            id="link-1", user_id="u1", calendar_account_id="account-1",
            interview_round_id="interview-1", external_event_id="ca12345",
            status="active", last_payload_hash="a" * 64,
            created_at=self.proposal.created_at, updated_at=self.proposal.created_at,
        )
        return SimpleNamespace(proposal=executed, link=link)


class Decisions:
    def __init__(self, values):
        self.values = list(values)

    def decide(self, context, tool_specs):
        return self.values.pop(0)


def test_calendar_preview_blocks_same_turn_write_and_confirmation_executes_next_turn(
    tmp_path,
) -> None:
    manager = ContextManager(CareerContextStore(tmp_path / "context.sqlite3"))
    manager.upsert_profile(CareerProfileContext(user_id="u1"))
    calendar = Calendar()
    tools = MainAgentToolRegistry(Gateway(), calendar_service=calendar)
    first_runtime = MainAgentRuntime(
        context_manager=manager,
        decision_maker=Decisions(
            [
                AgentDecision(
                    action="tool_call",
                    tool_call=ToolCall(
                        name="prepare_interview_calendar_sync",
                        arguments={"interview_round_id": "interview-1"},
                    ),
                ),
                AgentDecision(
                    action="tool_call",
                    tool_call=ToolCall(name="execute_calendar_proposal", arguments={}),
                ),
            ]
        ),
        tools=tools,
    )

    preview = first_runtime.run_turn(
        user_id="u1", conversation_id="c1", user_message="把这场面试加到日历"
    )

    assert calendar.execute_calls == []
    assert preview.context.task.active_calendar_proposal_id == "proposal-1"
    assert "面试 · Acme · AI Engineer" in preview.assistant_message
    assert "只有你明确确认后" in preview.assistant_message

    confirmed_runtime = MainAgentRuntime(
        context_manager=manager,
        decision_maker=Decisions(
            [
                AgentDecision(
                    action="tool_call",
                    tool_call=ToolCall(name="execute_calendar_proposal", arguments={}),
                ),
                AgentDecision(action="final", message="已同步到 Calendar。"),
            ]
        ),
        tools=tools,
    )
    confirmed_runtime.run_turn(
        user_id="u1", conversation_id="c1", user_message="确认执行刚才的日历变更"
    )

    assert calendar.execute_calls == [
        {"user_id": "u1", "proposal_id": "proposal-1"}
    ]
    assert tools.capability_kind("execute_calendar_proposal") == "atomic_tool"
