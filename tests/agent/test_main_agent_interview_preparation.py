from datetime import datetime, timedelta, timezone

from career_agent.agent.summary_text import DELIVERY_SUMMARY_LIMIT
from career_agent.agent.context_manager import ContextManager
from career_agent.agent.main_agent_contracts import (
    ActionCandidateContextItem,
    AgentDecision,
    CareerProfileContext,
    ConversationTaskState,
    MainAgentContext,
    ToolCall,
    project_interview_preparation_arguments,
)
from career_agent.agent.main_agent_runtime import MainAgentRuntime
from career_agent.agent.main_agent_tools import MainAgentToolRegistry
from career_agent.domain.interview_preparation import (
    GapPreparation,
    InterviewFocusArea,
    InterviewPreparationResult,
    LikelyQuestion,
)
from career_agent.domain.interviews import InterviewRound
from career_agent.storage.context import CareerContextStore
from career_agent.storage.interview_preparations import StoredInterviewPreparation


NOW = datetime(2026, 8, 26, tzinfo=timezone.utc)


class Interviews:
    def __init__(self):
        self.interview = InterviewRound(
            id="interview-1", user_id="u1", application_id="application-1",
            sequence_number=1, status="scheduled",
            scheduled_start=NOW + timedelta(days=1), timezone="Asia/Shanghai",
            interview_format="video", created_at=NOW, updated_at=NOW,
        )

    def list_interviews(self, **kwargs):
        return (self.interview,)


class Preparations:
    def __init__(self):
        self.calls = []

    def prepare(self, **kwargs):
        self.calls.append(kwargs)
        return StoredInterviewPreparation(
            id="preparation-1", user_id="u1", interview_round_id="interview-1",
            application_id="application-1", job_posting_id="job-1",
            jd_snapshot_id="jd-1", resume_version_id="resume-version-1",
            input_fingerprint="a" * 64, worker_version="v1",
            result=InterviewPreparationResult(
                summary="重点准备 RAG 可靠性。",
                focus_areas=(InterviewFocusArea(
                    topic="检索故障恢复",
                    priority="high",
                    rationale="岗位要求建设可靠系统。",
                    jd_quote="Build reliable RAG systems.",
                ),),
                likely_questions=(LikelyQuestion(
                    question="如何设计检索降级？",
                    rationale="验证故障处理能力。",
                    answer_outline=("说明故障检测", "说明恢复目标"),
                ),),
                gaps=(GapPreparation(
                    gap="缺少大规模线上经验",
                    jd_quote="Operate at scale.",
                    honest_response_strategy="说明可迁移经验，不虚构规模。",
                ),),
                checklist=("确认会议链接",),
            ),
            created_at=NOW,
        )


class Decisions:
    def __init__(self):
        self.values = [
            AgentDecision(
                action="tool_call", tool_call=ToolCall(name="list_interviews", arguments={})
            ),
            AgentDecision(
                action="tool_call",
                tool_call=ToolCall(name="prepare_interview", arguments={"selection_index": 1}),
            ),
            AgentDecision(action="final", message="准备重点已整理。"),
        ]

    def decide(self, context, tool_specs):
        return self.values.pop(0)


def test_main_agent_selects_interview_and_persists_preparation_context(tmp_path) -> None:
    manager = ContextManager(CareerContextStore(tmp_path / "context.sqlite3"))
    manager.upsert_profile(CareerProfileContext(user_id="u1"))
    preparations = Preparations()
    tools = MainAgentToolRegistry(
        interview_service=Interviews(),
        interview_preparation_service=preparations,
    )
    runtime = MainAgentRuntime(
        context_manager=manager, decision_maker=Decisions(), tools=tools,
    )

    result = runtime.run_turn(
        user_id="u1", conversation_id="c1", user_message="帮我准备这场面试"
    )

    assert preparations.calls == [
        {"user_id": "u1", "interview_round_id": "interview-1"}
    ]
    assert result.context.task.active_interview_preparation_id == "preparation-1"
    assert result.context.task.active_resume_version_id == "resume-version-1"
    assert tools.capability_kind("prepare_interview") == "atomic_tool"
    rendered = MainAgentRuntime._assistant_message(result.tool_result)
    assert rendered.startswith("# 面试准备\n\n重点准备 RAG 可靠性。")
    assert "## 可能的问题" in rendered
    assert "确认会议链接" in rendered
    assert "preparation-1" not in rendered
    assert "jd-1" not in rendered
    assert "preparation-1" not in result.assistant_message
    assert "jd-1" not in result.assistant_message
    assert result.tool_result.resource_ref is not None
    assert result.tool_result.resource_ref.title == "面试准备"
    assert result.tool_result.resource_ref.description == "重点准备 RAG 可靠性。"

    loaded = manager.load_for_turn(
        user_id="u1",
        conversation_id="c1",
        user_message="继续",
    )
    stored_reply = loaded.recent_messages[-1]
    # History states the outcome and points at the preparation entity. The
    # material is read back from that entity, so only bounded prose enters the
    # recent window. Since F that prose is the model's own reply rather than a
    # receipt, but the property the row has to keep is unchanged: bounded, and
    # never carrying the report body into every later turn's window.
    assert stored_reply.content == "准备重点已整理。"
    assert "## 可能的问题" not in stored_reply.content
    assert len(stored_reply.content) <= DELIVERY_SUMMARY_LIMIT
    assert stored_reply.resource_refs
    assert stored_reply.resource_refs[0].kind == "interview_preparation"
    assert stored_reply.resource_refs[0].resource_id == "preparation-1"


def test_preparation_can_resolve_interview_from_action_center_selection() -> None:
    context = MainAgentContext(
        conversation_id="c1",
        profile=CareerProfileContext(user_id="u1"),
        task=ConversationTaskState(
            action_candidates=(
                ActionCandidateContextItem(
                    action_item_id="action-1",
                    action_type="interview_preparation",
                    source_type="interview_round",
                    source_id="interview-1",
                    title="准备面试",
                    status="open",
                ),
            )
        ),
        user_message="准备第一个待办",
    )

    projected = project_interview_preparation_arguments(
        context, "prepare_interview", {"action_selection_index": 1}
    )

    assert projected == {"user_id": "u1", "interview_round_id": "interview-1"}
