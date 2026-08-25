from datetime import datetime, timezone

from career_agent.agent.context_manager import ContextManager
from career_agent.agent.main_agent_contracts import (
    AgentDecision,
    CareerProfileContext,
    ToolCall,
)
from career_agent.agent.main_agent_runtime import MainAgentRuntime
from career_agent.agent.main_agent_tools import MainAgentToolRegistry
from career_agent.domain.interviews import InterviewRound
from career_agent.services.interviews import InterviewDetail
from career_agent.storage.context import CareerContextStore


class Gateway:
    def advance(self, **kwargs):
        raise AssertionError("job discovery should not run")


class Interviews:
    def __init__(self):
        now = datetime(2026, 8, 26, tzinfo=timezone.utc)
        self.interview = InterviewRound(
            id="interview-1",
            user_id="u1",
            application_id="application-1",
            sequence_number=1,
            employer_label=None,
            status="scheduled",
            scheduled_start=now,
            scheduled_end=now.replace(hour=1),
            timezone="Asia/Shanghai",
            interview_format="video",
            meeting_url="https://meet.example/one",
            created_at=now,
            updated_at=now,
        )
        self.get_calls = []

    def list_interviews(self, **kwargs):
        return (self.interview,)

    def get_interview(self, **kwargs):
        self.get_calls.append(kwargs)
        return InterviewDetail(interview=self.interview, events=())


class Decisions:
    def __init__(self):
        self.values = [
            AgentDecision(
                action="tool_call",
                tool_call=ToolCall(name="list_interviews", arguments={}),
            ),
            AgentDecision(
                action="tool_call",
                tool_call=ToolCall(
                    name="get_interview", arguments={"selection_index": 1}
                ),
            ),
            AgentDecision(action="final", message="已读取面试安排。"),
        ]

    def decide(self, context, tool_specs):
        return self.values.pop(0)


def test_main_agent_selects_interview_without_treating_sequence_as_employer_label(
    tmp_path,
) -> None:
    manager = ContextManager(CareerContextStore(tmp_path / "context.sqlite3"))
    manager.upsert_profile(CareerProfileContext(user_id="u1"))
    interviews = Interviews()
    tools = MainAgentToolRegistry(Gateway(), interview_service=interviews)
    runtime = MainAgentRuntime(
        context_manager=manager,
        decision_maker=Decisions(),
        tools=tools,
    )

    result = runtime.run_turn(
        user_id="u1", conversation_id="c1", user_message="看看我的面试"
    )

    assert tools.capability_kind("list_interviews") == "atomic_tool"
    assert interviews.get_calls[0]["interview_round_id"] == "interview-1"
    assert result.context.task.active_interview_round_id == "interview-1"
    assert result.context.task.interview_candidates[0].sequence_number == 1
    assert result.context.task.interview_candidates[0].employer_label is None
