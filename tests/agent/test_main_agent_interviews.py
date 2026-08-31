from datetime import datetime, timezone

from career_agent.agent.context_manager import ContextManager
from career_agent.agent.main_agent_contracts import (
    AgentDecision,
    CareerProfileContext,
    ToolCall,
)
from career_agent.agent.main_agent_runtime import MainAgentRuntime
from career_agent.agent.main_agent_tools import MainAgentToolRegistry
from career_agent.domain.interviews import (
    InterviewRetroQuestion,
    InterviewRetroReport,
    InterviewRound,
)
from career_agent.services.interviews import InterviewDetail
from career_agent.storage.context import CareerContextStore


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
    tools = MainAgentToolRegistry(interview_service=interviews)
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


class RetroInterviews(Interviews):
    def __init__(self):
        super().__init__()
        self.interview = self.interview.model_copy(
            update={
                "status": "completed",
                "completed_at": self.interview.updated_at,
            }
        )
        self.retro_calls = []

    def record_retro(self, **kwargs):
        self.retro_calls.append(kwargs)
        return InterviewRetroReport(
            id="retro-internal-id",
            user_id="u1",
            application_id="application-1",
            interview_round_id=self.interview.id,
            source_notes=kwargs["source_notes"],
            summary=kwargs["summary"],
            questions=kwargs["questions"],
            strengths=kwargs["strengths"],
            difficulties=kwargs["difficulties"],
            interviewer_signals=kwargs["interviewer_signals"],
            next_focus=kwargs["next_focus"],
            action_items=kwargs["action_items"],
            limitations=kwargs["limitations"],
            self_assessment=kwargs["self_assessment"],
            content_sha256="a" * 64,
            created_at=self.interview.updated_at,
        )


class Actions:
    def __init__(self):
        self.completed = []

    def complete_source_action(self, **kwargs):
        self.completed.append(kwargs)
        return None


class RetroDecisions:
    def __init__(self):
        self.values = [
            AgentDecision(
                action="tool_call",
                tool_call=ToolCall(name="list_interviews", arguments={}),
            ),
            AgentDecision(
                action="tool_call",
                tool_call=ToolCall(
                    name="record_interview_retro",
                    arguments={
                        "selection_index": 1,
                        "source_notes": "问了 RAG 评测，数据集构造没答完整。",
                        "summary": "评测数据集设计需要补强。",
                        "questions": [
                            {
                                "question": "如何评估 RAG？",
                                "answer_summary": "回答了 Recall。",
                                "self_assessment": "mixed",
                            }
                        ],
                        "difficulties": ["没讲清数据集构造"],
                        "next_focus": ["准备离线评测设计"],
                        "action_items": ["重写该题答案"],
                        "limitations": ["没有面试官书面反馈"],
                        "self_assessment": "mixed",
                    },
                ),
            ),
            AgentDecision(action="final", message="复盘报告已保存。"),
        ]

    def decide(self, context, tool_specs):
        return self.values.pop(0)


def test_main_agent_records_user_grounded_real_interview_retro(tmp_path) -> None:
    manager = ContextManager(CareerContextStore(tmp_path / "context.sqlite3"))
    manager.upsert_profile(CareerProfileContext(user_id="u1"))
    interviews = RetroInterviews()
    actions = Actions()
    tools = MainAgentToolRegistry(
        interview_service=interviews,
        action_center_service=actions,
    )
    runtime = MainAgentRuntime(
        context_manager=manager,
        decision_maker=RetroDecisions(),
        tools=tools,
    )

    result = runtime.run_turn(
        user_id="u1",
        conversation_id="c-retro",
        user_message="刚才问了 RAG 评测，数据集构造没答完整，帮我复盘。",
    )

    assert tools.capability_kind("record_interview_retro") == "atomic_tool"
    schema = next(
        item["function"]
        for item in tools.schemas()
        if item["function"]["name"] == "record_interview_retro"
    )
    assert "interview_round_id" not in schema["parameters"]["properties"]
    assert interviews.retro_calls[0]["interview_round_id"] == "interview-1"
    assert interviews.retro_calls[0]["questions"] == (
        InterviewRetroQuestion(
            question="如何评估 RAG？",
            answer_summary="回答了 Recall。",
            self_assessment="mixed",
        ),
    )
    assert actions.completed == [
        {
            "user_id": "u1",
            "action_type": "interview_retro",
            "source_id": "interview-1",
        }
    ]
    assert result.context.task.active_interview_round_id == "interview-1"
    assert result.assistant_message.startswith("# 真实面试复盘")
    assert "如何评估 RAG？" in result.assistant_message
    assert result.tool_result.resource_ref is not None
    assert result.tool_result.resource_ref.kind == "interview_retro_report"
