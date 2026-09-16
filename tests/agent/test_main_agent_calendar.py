from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from career_agent.agent.context_manager import ContextManager
from career_agent.agent.main_agent_contracts import AgentDecision, CareerProfileContext, ToolCall
from career_agent.agent.main_agent_runtime import MainAgentRuntime
from career_agent.agent.main_agent_tools import MainAgentToolRegistry
from career_agent.connectors.calendar import CalendarConnectorError
from career_agent.domain.calendar import (
    CalendarChangeProposal,
    CalendarEventLink,
    CalendarEventPayload,
)
from career_agent.harness.streaming import InteractionResponse
from career_agent.storage.capability_confirmations import (
    SQLiteCapabilityConfirmationStore,
)
from career_agent.storage.context import CareerContextStore


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
            payload_hash="a" * 64, policy_epoch=1, status="pending", created_at=now,
            expires_at=now + timedelta(minutes=15),
        )
        self.execute_calls = []

    def prepare_interview_sync(self, **kwargs):
        return self.proposal

    def get_proposal(self, **kwargs):
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
    """The external write runs only after the owner presses the button.

    Before B3 the second turn executed on the model's reading of "确认执行":
    a soft consent, decided by the component whose judgement is under review.
    The preview still stops the same-turn write; the next turn now seals the
    exact proposal for the owner and runs it once on the UI-bound yes.
    """

    database = tmp_path / "context.sqlite3"
    manager = ContextManager(CareerContextStore(database))
    manager.upsert_profile(CareerProfileContext(user_id="u1"))
    seeded = manager.load_for_turn(
        user_id="u1", conversation_id="c1", user_message="seed active interview"
    )
    manager.commit_turn(
        context=seeded,
        task=seeded.task.model_copy(
            update={
                "active_interview_round_id": "interview-1",
                "tool_profile": "interview",
            }
        ),
        assistant_message="seeded",
    )
    calendar = Calendar()
    tools = MainAgentToolRegistry(calendar_service=calendar)
    first_runtime = MainAgentRuntime(
        context_manager=manager,
        decision_maker=Decisions(
            [
                AgentDecision(
                    action="tool_call",
                    tool_call=ToolCall(
                        name="prepare_interview_calendar_sync",
                        arguments={},
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

    # The model reading "yes" is not consent. Without a confirmation store the
    # write is refused outright rather than falling through to execution.
    unsealed = MainAgentRuntime(
        context_manager=manager,
        decision_maker=Decisions(
            [
                AgentDecision(
                    action="tool_call",
                    tool_call=ToolCall(name="execute_calendar_proposal", arguments={}),
                ),
                AgentDecision(action="final", message="现在无法执行。"),
            ]
        ),
        tools=tools,
    ).run_turn(
        user_id="u1", conversation_id="c1", user_message="确认执行刚才的日历变更"
    )
    assert calendar.execute_calls == []
    refusal = unsealed.context.tool_observations[-1]
    assert refusal.state == "authorization_refused"
    assert "外部系统" in refusal.message

    sealed = MainAgentRuntime(
        context_manager=manager,
        decision_maker=Decisions(
            [
                AgentDecision(
                    action="tool_call",
                    tool_call=ToolCall(name="execute_calendar_proposal", arguments={}),
                ),
                AgentDecision(action="final", message="请确认。"),
            ]
        ),
        tools=tools,
        capability_confirmation_store=SQLiteCapabilityConfirmationStore(database),
    ).run_turn(
        user_id="u1", conversation_id="c1", user_message="确认执行刚才的日历变更"
    )

    assert calendar.execute_calls == []
    assert sealed.tool_result.state == "capability_confirmation_required"
    # The owner approves the concrete event, read live from the proposal.
    assert "面试 · Acme · AI Engineer" in sealed.tool_result.message
    assert "无法由这里撤回" in sealed.tool_result.message
    gate = MainAgentRuntime._interaction_event(result=sealed, conversation_id="c1")
    assert gate is not None and gate.scope == "capability_confirmation"

    class NeverAsked:
        def decide(self, context, tool_specs):
            raise AssertionError("a confirmed action must not re-ask the model")

    confirmed = MainAgentRuntime(
        context_manager=manager,
        decision_maker=NeverAsked(),
        tools=tools,
        capability_confirmation_store=SQLiteCapabilityConfirmationStore(database),
    ).run_turn(
        user_id="u1",
        conversation_id="c1",
        user_message="确认",
        interaction_response=InteractionResponse(
            interaction_id=gate.interaction_id,
            scope="capability_confirmation",
            action="confirm",
        ),
    )

    assert confirmed.tool_result.state == "calendar_sync_complete"
    assert calendar.execute_calls == [
        {"user_id": "u1", "proposal_id": "proposal-1"}
    ]
    assert tools.capability_kind("execute_calendar_proposal") == "atomic_tool"


def test_uncertain_calendar_execution_requires_a_new_preview() -> None:
    class UncertainCalendar(Calendar):
        def execute_proposal(self, **kwargs):
            raise CalendarConnectorError(
                "GOOGLE_CALENDAR_TRANSPORT_ERROR",
                "timeout",
                outcome_unknown=True,
            )

    observation = MainAgentToolRegistry(
        calendar_service=UncertainCalendar()
    ).invoke_atomic_tool(
        "execute_calendar_proposal",
        {"user_id": "u1", "proposal_id": "proposal-1"},
    )

    assert observation.state == "calendar_write_failed"
    assert observation.disposition == "failed"
    assert observation.payload["retryable"] is False
