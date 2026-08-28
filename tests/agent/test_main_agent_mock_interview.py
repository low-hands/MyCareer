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
