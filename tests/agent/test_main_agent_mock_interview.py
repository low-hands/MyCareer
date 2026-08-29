from __future__ import annotations

from datetime import datetime, timezone

import pytest

from career_agent.agent.context_manager import ContextManager
from career_agent.agent.main_agent_contracts import (
    AgentDecision,
    ApplicationCandidateContextItem,
    CareerProfileContext,
    ConversationTaskState,
    MainAgentContext,
    ToolCall,
    project_mock_interview_arguments,
)
from career_agent.agent.main_agent_runtime import MainAgentRuntime
from career_agent.agent.main_agent_tools import MainAgentToolRegistry
from career_agent.agent.mock_interview_contracts import MockInterviewGraphResult
from career_agent.agent.mock_interview_graph import (
    MockInterviewCheckpointMissingError,
    MockInterviewInputRoutingError,
)
from career_agent.domain.applications import Application
from career_agent.domain.job_discovery import JobDetail, Provenance
from career_agent.domain.mock_interviews import (
    MockInterviewQuestionResult,
    MockInterviewReport,
)
from career_agent.services.applications import ApplicationService
from career_agent.storage.applications import SQLiteApplicationStore
from career_agent.storage.context import CareerContextStore
from career_agent.storage.jobs import SQLiteJobPostingRepository
from career_agent.storage.resumes import ResumeStore


NOW = datetime(2026, 8, 27, tzinfo=timezone.utc)


class UnusedGateway:
    def advance(self, **kwargs):
        raise AssertionError("Mock interview must not enter Job Discovery")


class OneDecision:
    def __init__(self, decision: AgentDecision) -> None:
        self.decision = decision
        self.calls = 0

    def decide(self, context, tool_specs):
        self.calls += 1
        if self.calls > 1:
            raise AssertionError("Mock interview result must be presented without re-deciding")
        return self.decision


class FakeMockInterviewGraph:
    def __init__(self) -> None:
        self.starts = []
        self.resumes = []

    def start(self, request):
        self.starts.append(request)
        return MockInterviewGraphResult(
            session_id="mock-session-1",
            state="awaiting_answer",
            message="Answer the current mock interview question.",
            turn_id="turn-1",
            question="请介绍一个你亲自负责的 RAG 可靠性改进。",
        )

    def resume(self, *, user_id, session_id, answer):
        self.resumes.append(
            {"user_id": user_id, "session_id": session_id, "answer": answer}
        )
        return MockInterviewGraphResult(
            session_id=session_id,
            state="completed",
            message="Mock interview completed.",
            report_id="report-1",
            report=MockInterviewReport(
                id="report-1",
                session_id=session_id,
                completion_reason="plan_completed",
                summary="回答覆盖了核心问题，但量化验证还不够具体。",
                question_results=(
                    MockInterviewQuestionResult(
                        plan_item_number=1,
                        question="请介绍一个你亲自负责的 RAG 可靠性改进。",
                        final_rating="adequate",
                        summary="职责清楚，验证不足。",
                        follow_up_count=0,
                    ),
                ),
                strengths=("能说明个人职责",),
                development_areas=("补充验证指标",),
                practice_actions=("用 STAR 重写回答",),
                created_at=NOW,
            ),
        )

    def handle_input(self, *, user_id, session_id, message):
        return self.resume(user_id=user_id, session_id=session_id, answer=message)


def _application_setup(tmp_path):
    resumes = ResumeStore(tmp_path / "resumes.sqlite3")
    role = resumes.create_target_role(user_id="u1", title="AI Engineer", priority=1)
    _, version = resumes.import_document(
        user_id="u1",
        target_role_id=role.id,
        name="AI Resume",
        content=b"PRIVATE RESUME",
        document_format="text",
    )
    jobs = SQLiteJobPostingRepository(tmp_path / "jobs.sqlite3")
    stored = jobs.save_detail(
        user_id="u1",
        run_id="run-1",
        result_ref="ref-1",
        selection_index=1,
        detail=JobDetail(
            source_name="test",
            source_job_id="job-1",
            title="RAG Engineer",
            company_name="Acme",
            description="PRIVATE JD",
            captured_at=NOW,
            provenance=Provenance(
                source_name="test",
                source_job_id="job-1",
                captured_at=NOW,
                operation="detail",
                adapter_version="test-v1",
            ),
        ),
    )
    service = ApplicationService(
        SQLiteApplicationStore(tmp_path / "applications.sqlite3"), jobs, resumes
    )
    application = service.create_application(
        user_id="u1",
        job_posting_id=stored.posting.id,
        resume_version_id=version.id,
    ).application
    return service, application


def test_projection_selects_application_by_index_and_rejects_internal_ids() -> None:
    application = Application(
        id="app-2",
        user_id="u1",
        job_posting_id="job-2",
        jd_snapshot_id="snapshot-2",
        resume_version_id="resume-2",
        status="submitted",
        submitted_at=NOW,
        created_at=NOW,
        updated_at=NOW,
    )
    context = MainAgentContext(
        conversation_id="c1",
        profile=CareerProfileContext(user_id="u1"),
        task=ConversationTaskState(
            application_candidates=(
                ApplicationCandidateContextItem(
                    application_id=application.id,
                    title="RAG Engineer",
                    company_name="Acme",
                    status="submitted",
                ),
            )
        ),
        user_message="开始模拟面试",
    )

    projected = project_mock_interview_arguments(
        context,
        {"application_selection_index": 1, "interview_type": "technical"},
    )

    assert projected["application_id"] == "app-2"
    assert projected["interview_type"] == "technical"
    with pytest.raises(ValueError, match="internal identifiers"):
        project_mock_interview_arguments(context, {"application_id": "app-2"})


def test_runtime_starts_then_directly_resumes_active_mock_interview(tmp_path) -> None:
    service, application = _application_setup(tmp_path)
    manager = ContextManager(CareerContextStore(tmp_path / "context.sqlite3"))
    manager.upsert_profile(CareerProfileContext(user_id="u1"))
    seed = manager.load_for_turn(
        user_id="u1", conversation_id="c1", user_message="选择这次投递"
    )
    manager.commit_turn(
        context=seed,
        task=ConversationTaskState(active_application_id=application.id),
        assistant_message="已选择投递。",
    )
    graph = FakeMockInterviewGraph()
    decision_maker = OneDecision(
        AgentDecision(
            action="tool_call",
            tool_call=ToolCall(
                name="start_mock_interview",
                arguments={
                    "interview_type": "technical",
                    "max_primary_questions": 1,
                },
            ),
        )
    )
    tools = MainAgentToolRegistry(
        UnusedGateway(),
        application_service=service,
        mock_interview_graph=graph,
    )
    runtime = MainAgentRuntime(
        context_manager=manager,
        decision_maker=decision_maker,
        tools=tools,
    )

    started = runtime.run_turn(
        user_id="u1", conversation_id="c1", user_message="开始技术模拟面试"
    )

    assert decision_maker.calls == 1
    assert graph.starts[0].application_id == application.id
    assert graph.starts[0].job_posting_id == application.job_posting_id
    assert graph.starts[0].jd_snapshot_id == application.jd_snapshot_id
    assert graph.starts[0].resume_version_id == application.resume_version_id
    assert started.context.task.active_workflow == "mock_interview"
    assert started.context.task.run_id == "mock-session-1"
    assert started.assistant_message == (
        "模拟面试题：\n请介绍一个你亲自负责的 RAG 可靠性改进。"
    )
    # Mid-run the conversation has no trace of the interview at all, not even a
    # request waiting for an answer. It is held, so the whole run can be written
    # as one exchange when it ends and the stored history is never mid-turn.
    after_start = manager.load_for_turn(
        user_id="u1", conversation_id="c1", user_message="inspect"
    )
    assert [message.content for message in after_start.recent_messages] == [
        "选择这次投递",
        "已选择投递。",
    ]
    assert after_start.task.workflow_entry_message == "开始技术模拟面试"

    completed = MainAgentRuntime(
        context_manager=manager,
        decision_maker=decision_maker,
        tools=tools,
    ).run_turn(
        user_id="u1",
        conversation_id="c1",
        user_message="我负责设计离线评估集，并监控召回率。",
    )

    assert decision_maker.calls == 1
    assert graph.resumes == [
        {
            "user_id": "u1",
            "session_id": "mock-session-1",
            "answer": "我负责设计离线评估集，并监控召回率。",
        }
    ]
    assert completed.context.task.active_workflow == "none"
    assert completed.context.task.run_id is None
    assert completed.assistant_message.startswith("模拟面试完成。")
    assert "补充验证指标" in completed.assistant_message
    # The run's own turns stay out of the main thread, but its outcome does not:
    # the candidate's answer is absent while the report that closes the request
    # is present, so a later turn can act on the interview without replaying it.
    after_completion = manager.load_for_turn(
        user_id="u1", conversation_id="c1", user_message="inspect"
    )
    contents = [message.content for message in after_completion.recent_messages]
    assert contents[:3] == ["选择这次投递", "已选择投递。", "开始技术模拟面试"]
    assert contents[3].startswith("模拟面试完成。")
    assert "补充验证指标" in contents[3]
    # One request, one reply. Neither the questions nor the answers survive.
    assert len(contents) == 4
    assert not any("我负责设计离线评估集" in item for item in contents)
    assert not any("模拟面试题" in item for item in contents)


def test_start_schema_exposes_only_selection_indexes_not_internal_ids(tmp_path) -> None:
    service, _ = _application_setup(tmp_path)
    tools = MainAgentToolRegistry(
        UnusedGateway(),
        application_service=service,
        mock_interview_graph=FakeMockInterviewGraph(),
    )

    schema = next(
        item for item in tools.schemas() if item["function"]["name"] == "start_mock_interview"
    )
    properties = schema["function"]["parameters"]["properties"]
    assert "application_selection_index" in properties
    assert "interview_selection_index" in properties
    assert all(not key.endswith("_id") for key in properties)


class ReplayDecisions:
    """Answer every turn, unlike OneDecision which forbids a second call."""

    def __init__(self, decision: AgentDecision) -> None:
        self.decision = decision
        self.calls = 0

    def decide(self, context, tool_specs):
        self.calls += 1
        return self.decision


class LostCheckpointGraph(FakeMockInterviewGraph):
    """Business rows outlive the resumable thread."""

    def resume(self, *, user_id, session_id, answer):
        raise MockInterviewCheckpointMissingError("thread is gone")

    def retry(self, *, user_id, session_id):
        raise MockInterviewCheckpointMissingError("thread is gone")


class RoutingFailureGraph(FakeMockInterviewGraph):
    def handle_input(self, *, user_id, session_id, message):
        raise MockInterviewInputRoutingError(
            "INPUT_ROUTE_UNAVAILABLE",
            "router unavailable",
            retryable=True,
        )


def _runtime_with_graph(tmp_path, graph, decision_maker):
    service, application = _application_setup(tmp_path)
    manager = ContextManager(CareerContextStore(tmp_path / "context.sqlite3"))
    manager.upsert_profile(CareerProfileContext(user_id="u1"))
    seed = manager.load_for_turn(
        user_id="u1", conversation_id="c1", user_message="选择这次投递"
    )
    manager.commit_turn(
        context=seed,
        task=ConversationTaskState(active_application_id=application.id),
        assistant_message="已选择投递。",
    )
    tools = MainAgentToolRegistry(
        UnusedGateway(),
        application_service=service,
        mock_interview_graph=graph,
    )
    runtime = MainAgentRuntime(
        context_manager=manager,
        decision_maker=decision_maker,
        tools=tools,
    )
    return runtime, manager


def _start_decision() -> AgentDecision:
    return AgentDecision(
        action="tool_call",
        tool_call=ToolCall(
            name="start_mock_interview",
            arguments={"interview_type": "technical", "max_primary_questions": 1},
        ),
    )


def test_a_dead_checkpoint_hands_the_conversation_back_with_a_trace(tmp_path) -> None:
    decision_maker = ReplayDecisions(_start_decision())
    runtime, manager = _runtime_with_graph(
        tmp_path, LostCheckpointGraph(), decision_maker
    )
    runtime.run_turn(user_id="u1", conversation_id="c1", user_message="开始技术模拟面试")

    lost = runtime.run_turn(
        user_id="u1", conversation_id="c1", user_message="我的回答"
    )

    assert lost.context.task.phase == "mock_interview_checkpoint_missing"
    # The run is unreachable, so the next message must reach the decision model.
    # Keeping the slot without this would loop on the same dead thread forever.
    calls_before = decision_maker.calls
    runtime.run_turn(
        user_id="u1", conversation_id="c1", user_message="那算了，帮我看看简历"
    )
    assert decision_maker.calls > calls_before

    contents = [
        message.content
        for message in manager.load_for_turn(
            user_id="u1", conversation_id="c1", user_message="inspect"
        ).recent_messages
    ]
    # The failure is visible and the answer is not: Main Agent can explain why
    # the interview stopped without the transcript being copied over.
    assert any("执行断点已经丢失" in item for item in contents)
    assert not any("我的回答" in item for item in contents)


def test_input_routing_failure_keeps_the_question_and_does_not_store_the_input(
    tmp_path,
) -> None:
    runtime, manager = _runtime_with_graph(
        tmp_path, RoutingFailureGraph(), ReplayDecisions(_start_decision())
    )
    started = runtime.run_turn(
        user_id="u1", conversation_id="c1", user_message="开始技术模拟面试"
    )

    failed = runtime.run_turn(
        user_id="u1", conversation_id="c1", user_message="这条回答不要落库"
    )

    assert started.context.task.phase == "mock_interview_answer_required"
    assert failed.context.task.phase == "mock_interview_answer_required"
    assert failed.tool_result.state == "mock_interview_input_retry_required"
    assert "尚未保存" in failed.assistant_message
    contents = [
        item.content
        for item in manager.load_for_turn(
            user_id="u1", conversation_id="c1", user_message="inspect"
        ).recent_messages
    ]
    assert not any("这条回答不要落库" in item for item in contents)


class CancellingGraph(FakeMockInterviewGraph):
    """Ends the run without a report, the other way out of the loop."""

    def resume(self, *, user_id, session_id, answer):
        return MockInterviewGraphResult(
            session_id=session_id,
            state="cancelled",
            message="模拟面试已取消。",
        )


def test_a_cancelled_run_still_leaves_its_ending_in_the_main_thread(tmp_path) -> None:
    runtime, manager = _runtime_with_graph(
        tmp_path, CancellingGraph(), ReplayDecisions(_start_decision())
    )
    runtime.run_turn(user_id="u1", conversation_id="c1", user_message="开始技术模拟面试")

    cancelled = runtime.run_turn(
        user_id="u1", conversation_id="c1", user_message="不想练了，取消"
    )

    assert cancelled.context.task.active_workflow == "none"
    contents = [
        message.content
        for message in manager.load_for_turn(
            user_id="u1", conversation_id="c1", user_message="inspect"
        ).recent_messages
    ]
    # Every exit writes a trace, not just the one that produces a report.
    assert any("取消" in item for item in contents)
    assert not any("不想练了" in item for item in contents)


@pytest.mark.parametrize(
    "graph_factory, closing",
    [
        (FakeMockInterviewGraph, "模拟面试完成。"),
        (CancellingGraph, "模拟面试已取消。"),
        (LostCheckpointGraph, "执行断点已经丢失"),
    ],
)
def test_every_exit_pairs_the_request_and_releases_the_hold(
    tmp_path, graph_factory, closing
) -> None:
    runtime, manager = _runtime_with_graph(
        tmp_path, graph_factory(), ReplayDecisions(_start_decision())
    )
    runtime.run_turn(user_id="u1", conversation_id="c1", user_message="开始技术模拟面试")
    runtime.run_turn(user_id="u1", conversation_id="c1", user_message="我的回答")

    loaded = manager.load_for_turn(
        user_id="u1", conversation_id="c1", user_message="inspect"
    )
    contents = [message.content for message in loaded.recent_messages]
    assert contents[-2] == "开始技术模拟面试"
    assert closing in contents[-1]
    # The hold is released on every exit, including the one that keeps the slot
    # to record why the run died. A leftover request would be answered by the
    # next run's report instead of by its own.
    assert loaded.task.workflow_entry_message is None


class MaximalReportGraph(FakeMockInterviewGraph):
    """A report with every field at its contract maximum."""

    def resume(self, *, user_id, session_id, answer):
        return MockInterviewGraphResult(
            session_id=session_id,
            state="completed",
            message="Mock interview completed.",
            report_id="report-1",
            report=MockInterviewReport(
                id="report-1",
                session_id=session_id,
                completion_reason="plan_completed",
                summary="回" * 3000,
                question_results=(
                    MockInterviewQuestionResult(
                        plan_item_number=1,
                        question="q",
                        final_rating="adequate",
                        summary="s",
                        follow_up_count=0,
                    ),
                ),
                strengths=tuple("亮" * 200 for _ in range(10)),
                development_areas=tuple(f"待{index}" + "提" * 200 for index in range(10)),
                practice_actions=tuple(f"练{index}" + "习" * 200 for index in range(10)),
                created_at=NOW,
            ),
        )


def test_a_long_report_keeps_every_section_in_the_history(tmp_path) -> None:
    """The stored copy is condensed, not cut off at a fixed length.

    Truncating the screen copy would drop whole trailing sections, and those
    are the ones the next turn needs: what to work on and what to practise.
    """
    runtime, manager = _runtime_with_graph(
        tmp_path, MaximalReportGraph(), ReplayDecisions(_start_decision())
    )
    runtime.run_turn(user_id="u1", conversation_id="c1", user_message="开始技术模拟面试")
    result = runtime.run_turn(
        user_id="u1", conversation_id="c1", user_message="我的回答"
    )

    # Nothing is withheld from the candidate.
    assert "练习建议" in result.assistant_message

    loaded = manager.load_for_turn(
        user_id="u1", conversation_id="c1", user_message="inspect"
    )
    stored = loaded.recent_messages[-1].content
    assert "总结" in stored
    assert "待提升" in stored
    assert "练习建议" in stored
    assert "练0" in stored
    # Below the per-message cap, so no section was lost on the way in.
    assert len(stored) < 4000
