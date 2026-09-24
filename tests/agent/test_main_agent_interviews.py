from datetime import datetime, timezone
from types import SimpleNamespace

from career_agent.agent.context_manager import ContextManager
from career_agent.agent.main_agent_contracts import (
    AgentDecision,
    ApplicationCandidateContextItem,
    CareerProfileContext,
    ConversationTaskState,
    MainAgentContext,
    ToolCall,
    ActiveSavedJobContextItem,
    project_interview_arguments,
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


def _route_to_interview() -> AgentDecision:
    return AgentDecision(
        action="tool_call",
        tool_call=ToolCall(
            name="route_to_capability", arguments={"domain": "interview"}
        ),
    )


class Decisions:
    def __init__(self):
        self.values = [
            AgentDecision(
                action="tool_call",
                tool_call=ToolCall(name="list_interviews", arguments={}),
            ),
            _route_to_interview(),
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


def test_create_interview_selects_an_application_candidate_without_exposing_ids() -> None:
    context = MainAgentContext(
        conversation_id="c1",
        profile=CareerProfileContext(user_id="u1"),
        task=ConversationTaskState(
            active_application_id="app-active",
            application_candidates=(
                ApplicationCandidateContextItem(
                    application_id="app-1",
                    title="算法工程师",
                    company_name="甲公司",
                    status="submitted",
                ),
                ApplicationCandidateContextItem(
                    application_id="app-2",
                    title="AI 工程师",
                    company_name="乙公司",
                    status="interviewing",
                ),
            ),
        ),
        user_message="给第二个投递记录面试",
    )

    projected = project_interview_arguments(
        context,
        "create_interview",
        {
            "application_selection_index": 2,
            "details": {"interview_format": "video"},
        },
    )
    assert projected["application_id"] == "app-2"
    assert "application_selection_index" not in projected

    tools = MainAgentToolRegistry(interview_service=object())
    schema = next(
        spec["function"]
        for spec in tools.schemas()
        if spec["function"]["name"] == "create_interview"
    )
    properties = schema["parameters"]["properties"]
    assert "application_selection_index" in properties
    assert "application_id" not in properties


def test_create_interview_falls_back_to_active_and_rejects_bad_selection() -> None:
    import pytest

    context = MainAgentContext(
        conversation_id="c1",
        profile=CareerProfileContext(user_id="u1"),
        task=ConversationTaskState(
            active_application_id="app-active",
            application_candidates=(
                ApplicationCandidateContextItem(
                    application_id="app-1",
                    title="算法工程师",
                    company_name="甲公司",
                    status="submitted",
                ),
            ),
        ),
        user_message="记录面试",
    )
    arguments = {"details": {"interview_format": "video"}}
    assert project_interview_arguments(
        context, "create_interview", arguments
    )["application_id"] == "app-active"

    with pytest.raises(ValueError, match="application selection index is out of range"):
        project_interview_arguments(
            context,
            "create_interview",
            {
                "application_selection_index": 2,
                "details": {"interview_format": "video"},
            },
        )


def test_create_interview_uses_the_unique_active_saved_job_without_reasking() -> None:
    context = MainAgentContext(
        conversation_id="c1",
        profile=CareerProfileContext(user_id="u1"),
        task=ConversationTaskState(
            active_job_posting_id="job-1",
            active_jd_snapshot_id="jd-1",
            active_saved_job=ActiveSavedJobContextItem(
                job_posting_id="job-1",
                jd_snapshot_id="jd-1",
                title="多模态算法研究",
                company_name="无垠跃迁",
                jd_version=1,
            ),
        ),
        user_message="他约了我后天上午十点面试",
    )

    projected = project_interview_arguments(
        context,
        "create_interview",
        {"details": {"interview_format": "unknown"}},
    )

    assert projected["application_id"] is None
    assert projected["job_posting_id"] == "job-1"


def test_create_interview_can_materialize_tracking_from_the_active_saved_job() -> None:
    now = datetime(2026, 9, 25, 10, 0, tzinfo=timezone.utc)

    class Applications:
        def __init__(self):
            self.created = []
            self.updated = []

        def create_application(self, **arguments):
            self.created.append(arguments)
            application = SimpleNamespace(id="app-1", status="submitted")
            return SimpleNamespace(application=application, created=True)

        def get_application(self, **arguments):
            return SimpleNamespace(
                application=SimpleNamespace(id="app-1", status="submitted")
            )

        def update_application(self, **arguments):
            self.updated.append(arguments)
            return SimpleNamespace(id="app-1", status=arguments["status"])

    class InterviewService:
        def create_manual(self, **arguments):
            return InterviewRound(
                id="interview-1",
                user_id=arguments["user_id"],
                application_id=arguments["application_id"],
                sequence_number=1,
                status="scheduled",
                scheduled_start=now,
                timezone="Asia/Shanghai",
                created_at=now,
                updated_at=now,
            )

    applications = Applications()
    tools = MainAgentToolRegistry(
        application_service=applications,
        interview_service=InterviewService(),
    )

    result = tools.invoke_atomic_tool(
        "create_interview",
        {
            "user_id": "u1",
            "job_posting_id": "job-1",
            "details": {
                "scheduled_start": now.isoformat(),
                "timezone": "Asia/Shanghai",
            },
        },
    )

    assert result.state == "interview_ready"
    assert applications.created[0]["resume_version_id"] is None
    assert applications.updated[0]["status"] == "interviewing"


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
            _route_to_interview(),
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
    rendered = MainAgentRuntime._assistant_message(result.tool_result)
    assert rendered.startswith("# 真实面试复盘")
    assert "如何评估 RAG？" in rendered
    assert result.tool_result.resource_ref is not None
    assert result.tool_result.resource_ref.kind == "interview_retro_report"
    assert result.tool_result.resource_ref.title == "面试复盘"
    assert result.tool_result.resource_ref.description == "评测数据集设计需要补强。"
