from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from fastapi.testclient import TestClient

from career_agent.api.app import create_app
from career_agent.api.reads import (
    ApplicationMockInterviewsResponse,
    ApplicationView,
    CalendarAccountView,
    CalendarWorkspaceResponse,
    CompanyResearchView,
    EmailAccountView,
    EmailEventView,
    EmailWorkspaceResponse,
    ConversationMessageView,
    ConversationTranscriptResponse,
    ConversationView,
    ResumeView,
    ResumeImportResponse,
    MockInterviewSessionView,
    TargetRoleView,
    SavedJobDetailView,
    SavedJobView,
    build_workspace_reader,
)
from career_agent.domain.action_center import ActionItem, DailyBrief
from career_agent.domain.job_discovery import JobDetail, Provenance
from career_agent.domain.mock_interviews import (
    MockInterviewAnswerEvaluation,
    MockInterviewPlan,
    MockInterviewPlanItem,
    MockInterviewQuestionResult,
    MockInterviewReport,
    MockInterviewScoreDimension,
)
from career_agent.services.applications import ApplicationService
from career_agent.storage.applications import SQLiteApplicationStore
from career_agent.storage.jobs import SQLiteJobPostingRepository
from career_agent.storage.mock_interviews import SQLiteMockInterviewStore
from career_agent.storage.resumes import ResumeStore


NOW = datetime(2026, 8, 31, 2, 0, tzinfo=timezone.utc)


class _Runtime:
    def close(self) -> None:
        pass


class _ActionCenter:
    def daily_brief(self, *, user_id: str, timezone_name: str) -> DailyBrief:
        action = ActionItem(
            id="action-1",
            user_id=user_id,
            stable_key="application:app-1:follow-up",
            action_type="application_follow_up",
            source_type="application",
            source_id="app-1",
            application_id="app-1",
            title="跟进投递",
            summary="确认招聘方是否收到材料",
            due_at=NOW,
            status="open",
            created_at=NOW,
            updated_at=NOW,
        )
        return DailyBrief(
            user_id=user_id,
            timezone=timezone_name,
            generated_at=NOW,
            due_today=(action,),
        )


class _WorkspaceReader:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, int | None]] = []

    def applications(self, *, user_id: str, limit: int = 100):
        self.calls.append(("applications", user_id, limit))
        return (
            ApplicationView(
                id="app-1",
                status="interviewing",
                title="AI 产品经理",
                company_name="示例科技",
                city="北京",
                salary="25-35K",
                submitted_at=NOW,
                updated_at=NOW,
            ),
        )

    def application_mock_interviews(
        self, *, user_id: str, application_id: str, limit: int = 50
    ):
        self.calls.append((f"mock:{application_id}", user_id, limit))
        return ApplicationMockInterviewsResponse(
            application_id=application_id,
            title="AI 产品经理",
            company_name="示例科技",
            sessions=(
                MockInterviewSessionView(
                    session_id="mock-session-1",
                    status="completed",
                    interview_type="role_specific",
                    interview_type_label="专业面",
                    question_count=4,
                    max_primary_questions=6,
                    report_id="mock-report-1",
                    summary="结构清晰，需要补充量化结果。",
                    created_at=NOW,
                    completed_at=NOW,
                    updated_at=NOW,
                ),
                MockInterviewSessionView(
                    session_id="mock-session-active",
                    status="active",
                    interview_type="behavioral",
                    interview_type_label="行为面",
                    question_count=2,
                    max_primary_questions=5,
                    created_at=NOW,
                    updated_at=NOW,
                ),
            ),
        )

    def resumes(self, *, user_id: str):
        self.calls.append(("resumes", user_id, None))
        return (
            ResumeView(
                id="resume-1",
                name="AI 产品简历",
                target_role="AI 产品经理",
                status="active",
                latest_version_number=2,
                latest_version_id="resume-version-2",
                version_count=2,
                document_format="pdf",
                byte_size=4096,
                updated_at=NOW,
            ),
        )

    def jobs(self, *, user_id: str, limit: int = 100, include_dismissed: bool = False):
        self.calls.append(("jobs", user_id, limit))
        return (
            SavedJobView(
                id="job-1",
                title="AI 产品经理",
                company_name="示例科技",
                city="北京",
                salary="25-35K",
                source_name="browser_capture",
                source_url="https://example.com/job-1",
                availability_status="active",
                captured_at=NOW,
                last_checked_at=NOW,
                application_status="interviewing",
                analysis_summary="负责智能产品规划与落地。",
                responsibilities=("规划 AI 产品路线",),
                required_skills=("产品规划", "LLM"),
                preferred_qualifications=("有 Agent 产品经验",),
                clarification_questions=("团队规模是多少？",),
                analyzed_at=NOW,
            ),
            SavedJobView(
                id="job-2",
                title="智能产品经理",
                company_name="另一家公司",
                source_name="browser_capture",
                availability_status="active",
                captured_at=NOW,
                last_checked_at=NOW,
            ),
        )

    def job_count(self, *, user_id: str) -> int:
        self.calls.append(("job_count", user_id, None))
        return 2

    def conversations(self, *, user_id: str, limit: int = 50):
        self.calls.append(("conversations", user_id, limit))
        return (
            ConversationView(
                id="conversation-1",
                status="active",
                title="分析这份岗位",
                last_message_preview="这是岗位分析。",
                message_count=2,
                created_at=NOW,
                last_active_at=NOW,
            ),
        )

    def conversation_messages(
        self, *, user_id: str, conversation_id: str, limit: int = 200
    ):
        self.calls.append((f"messages:{conversation_id}", user_id, limit))
        return ConversationTranscriptResponse(
            messages=(
                ConversationMessageView(
                    role="user",
                    content="分析这份岗位",
                    created_at=NOW,
                ),
                ConversationMessageView(
                    role="assistant",
                    content="这是岗位分析。",
                    created_at=NOW,
                ),
            )
        )

    def calendar(self, *, user_id: str):
        self.calls.append(("calendar", user_id, None))
        return CalendarWorkspaceResponse(
            accounts=(
                CalendarAccountView(
                    id="calendar-1",
                    provider="google_calendar",
                    email_address="user@example.com",
                    calendar_id="primary",
                    status="active",
                    updated_at=NOW,
                ),
            ),
        )

    def research(self, *, user_id: str, limit: int = 100):
        self.calls.append(("research", user_id, limit))
        return (
            CompanyResearchView(
                id="report-1",
                company_name="示例科技",
                anchor_job_title="AI 产品经理",
                status="current",
                focus="产品线",
                summary="公开资料摘要",
                finding_count=3,
                source_count=2,
                created_at=NOW,
            ),
        )


def test_workspace_read_endpoints_scope_data_and_build_dashboard(api_keys, auth) -> None:
    reader = _WorkspaceReader()
    app = create_app(
        api_key_store_factory=lambda: api_keys,
        runtime_factory=_Runtime,
        action_center_factory=_ActionCenter,
        workspace_reader_factory=lambda: reader,
    )

    with TestClient(app) as client:
        applications = client.get("/v1/applications", headers=auth)
        jobs = client.get("/v1/jobs", headers=auth)
        conversations = client.get("/v1/conversations", headers=auth)
        transcript = client.get(
            "/v1/conversations/conversation-1/messages",
            headers=auth,
        )
        resumes = client.get("/v1/resumes", headers=auth)
        calendar = client.get("/v1/calendar", headers=auth)
        research = client.get("/v1/company-research", headers=auth)
        dashboard = client.get("/v1/dashboard", headers=auth)

    assert applications.json()[0]["title"] == "AI 产品经理"
    assert jobs.json()[1]["application_status"] is None
    assert jobs.json()[0]["required_skills"] == ["产品规划", "LLM"]
    assert conversations.json()[0]["title"] == "分析这份岗位"
    assert transcript.json()["messages"][1]["content"] == "这是岗位分析。"
    assert resumes.json()[0]["latest_version_number"] == 2
    assert calendar.json()["accounts"][0]["email_address"] == "user@example.com"
    assert "credential_ref" not in calendar.text
    assert research.json()[0]["finding_count"] == 3
    payload = dashboard.json()
    assert payload["stats"] == {
        "saved_jobs": 2,
        "applications": 1,
        "interviewing": 1,
        "offers": 0,
        "resumes": 1,
        "research_reports": 1,
        "calendar_accounts": 1,
    }
    assert payload["application_statuses"] == {"interviewing": 1}
    assert [item["id"] for item in payload["recent_jobs"]] == ["job-1", "job-2"]
    assert payload["next_actions"][0]["title"] == "跟进投递"
    assert {call[1] for call in reader.calls} == {"u1"}


def test_workspace_read_limits_are_validated_before_store_access(api_keys, auth) -> None:
    reader = _WorkspaceReader()
    app = create_app(
        api_key_store_factory=lambda: api_keys,
        runtime_factory=_Runtime,
        action_center_factory=_ActionCenter,
        workspace_reader_factory=lambda: reader,
    )

    with TestClient(app) as client:
        response = client.get(
            "/v1/company-research",
            headers=auth, params={"limit": 0},
        )

    assert response.status_code == 422
    assert reader.calls == []


def test_application_mock_interviews_are_owner_scoped_and_bounded(
    api_keys, issue_key
) -> None:
    from career_agent.storage.api_keys import WORKSPACE_READ

    reader = _WorkspaceReader()
    app = create_app(
        api_key_store_factory=lambda: api_keys,
        runtime_factory=_Runtime,
        action_center_factory=_ActionCenter,
        workspace_reader_factory=lambda: reader,
    )

    with TestClient(app) as client:
        response = client.get(
            "/v1/applications/app-1/mock-interviews",
            headers=issue_key("owner-1", WORKSPACE_READ),
            params={"limit": 10},
        )
        invalid = client.get(
            "/v1/applications/app-1/mock-interviews",
            headers=issue_key("owner-1", WORKSPACE_READ),
            params={"limit": 101},
        )

    assert response.status_code == 200
    assert response.json()["sessions"][0] == {
        "session_id": "mock-session-1",
        "status": "completed",
        "interview_type": "role_specific",
        "interview_type_label": "专业面",
        "question_count": 4,
        "max_primary_questions": 6,
        "report_id": "mock-report-1",
        "summary": "结构清晰，需要补充量化结果。",
        "created_at": NOW.isoformat().replace("+00:00", "Z"),
        "completed_at": NOW.isoformat().replace("+00:00", "Z"),
        "updated_at": NOW.isoformat().replace("+00:00", "Z"),
    }
    assert invalid.status_code == 422
    assert ("mock:app-1", "owner-1", 10) in reader.calls


def test_application_mock_interviews_hide_missing_and_other_owner_records(
    api_keys, issue_key
) -> None:
    from career_agent.services.applications import ApplicationInputNotFoundError
    from career_agent.storage.api_keys import WORKSPACE_READ

    class _OwnedReader(_WorkspaceReader):
        def application_mock_interviews(
            self, *, user_id: str, application_id: str, limit: int = 50
        ):
            if user_id != "owner-1" or application_id != "app-1":
                raise ApplicationInputNotFoundError("application")
            return ApplicationMockInterviewsResponse(
                application_id=application_id,
                title="AI 产品经理",
                company_name="示例科技",
            )

    app = create_app(
        api_key_store_factory=lambda: api_keys,
        runtime_factory=_Runtime,
        action_center_factory=_ActionCenter,
        workspace_reader_factory=_OwnedReader,
    )

    with TestClient(app) as client:
        empty = client.get(
            "/v1/applications/app-1/mock-interviews",
            headers=issue_key("owner-1", WORKSPACE_READ),
        )
        other_owner = client.get(
            "/v1/applications/app-1/mock-interviews",
            headers=issue_key("owner-2", WORKSPACE_READ),
        )
        missing = client.get(
            "/v1/applications/missing/mock-interviews",
            headers=issue_key("owner-1", WORKSPACE_READ),
        )

    assert empty.status_code == 200
    assert empty.json()["sessions"] == []
    assert other_owner.status_code == 404
    assert missing.status_code == 404


def test_application_mock_interviews_route_uses_real_owned_stores(
    tmp_path: Path, api_keys, issue_key
) -> None:
    from career_agent.storage.api_keys import WORKSPACE_READ

    args = argparse.Namespace(
        job_store=str(tmp_path / "jobs.sqlite3"),
        resume_store=str(tmp_path / "resumes.sqlite3"),
        application_store=str(tmp_path / "applications.sqlite3"),
        calendar_store=str(tmp_path / "calendar.sqlite3"),
        job_research_store=str(tmp_path / "research.sqlite3"),
        context_store=str(tmp_path / "context.sqlite3"),
        mock_interview_store=str(tmp_path / "mock.sqlite3"),
    )
    resumes = ResumeStore(Path(args.resume_store))
    role = resumes.create_target_role(
        user_id="owner-1", title="AI Engineer", priority=1
    )
    _, resume_version = resumes.import_document(
        user_id="owner-1",
        target_role_id=role.id,
        name="AI Resume",
        content=b"PRIVATE RESUME",
        document_format="text",
    )
    jobs = SQLiteJobPostingRepository(Path(args.job_store))
    stored_job = jobs.save_detail(
        user_id="owner-1",
        run_id="run-1",
        result_ref="result-1",
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
    application = ApplicationService(
        SQLiteApplicationStore(Path(args.application_store)), jobs, resumes
    ).create_application(
        user_id="owner-1",
        job_posting_id=stored_job.posting.id,
        resume_version_id=resume_version.id,
    ).application
    mock_store = SQLiteMockInterviewStore(Path(args.mock_interview_store))
    completed_session = mock_store.create_session(
        user_id="owner-1",
        application_id=application.id,
        job_posting_id=application.job_posting_id,
        jd_snapshot_id=application.jd_snapshot_id,
        resume_version_id=application.resume_version_id,
        interview_type="technical",
        max_primary_questions=2,
    )
    mock_store.save_plan(
        session=completed_session,
        plan=MockInterviewPlan(
            session_id=completed_session.id,
            summary="两道技术题",
            items=tuple(
                MockInterviewPlanItem(
                    sequence_number=index,
                    question_type="project_deep_dive",
                    difficulty="intermediate",
                    focus=f"技术主题 {index}",
                    rationale="验证真实存储读回",
                )
                for index in (1, 2)
            ),
            created_at=NOW,
        ),
    )
    completed_session = mock_store.start(session=completed_session)
    question_results = []
    for index in (1, 2):
        completed_session, turn = mock_store.ask(
            session=completed_session,
            plan_item_number=index,
            question_type="project_deep_dive",
            question=f"第 {index} 个问题",
        )
        completed_session, turn = mock_store.record_answer(
            session=completed_session,
            turn=turn,
            answer=f"第 {index} 个回答",
        )
        completed_session, _ = mock_store.record_evaluation(
            session=completed_session,
            turn=turn,
            evaluation=MockInterviewAnswerEvaluation(
                rating="adequate",
                summary=f"第 {index} 题反馈",
                dimensions=(
                    MockInterviewScoreDimension(
                        dimension="specificity",
                        score=3,
                        feedback="信息足够具体",
                    ),
                ),
                next_action="next_question",
                next_action_reason="继续下一题",
            ),
        )
        question_results.append(
            MockInterviewQuestionResult(
                plan_item_number=index,
                question=f"第 {index} 个问题",
                final_rating="adequate",
                summary=f"第 {index} 题反馈",
                follow_up_count=0,
            )
        )
    completed_session, report = mock_store.complete(
        session=completed_session,
        report=MockInterviewReport(
            id="mock-report-completed",
            session_id=completed_session.id,
            completion_reason="plan_completed",
            summary="完成两道题",
            question_results=tuple(question_results),
            created_at=NOW,
        ),
    )
    active_session = mock_store.create_session(
        user_id="owner-1",
        application_id=application.id,
        job_posting_id=application.job_posting_id,
        jd_snapshot_id=application.jd_snapshot_id,
        resume_version_id=application.resume_version_id,
        interview_type="behavioral",
        max_primary_questions=3,
    )
    mock_store.save_plan(
        session=active_session,
        plan=MockInterviewPlan(
            session_id=active_session.id,
            summary="一道行为题",
            items=(
                MockInterviewPlanItem(
                    sequence_number=1,
                    question_type="project_deep_dive",
                    difficulty="intermediate",
                    focus="行为主题",
                    rationale="验证未完成场次计数",
                ),
            ),
            created_at=NOW,
        ),
    )
    active_session = mock_store.start(session=active_session)
    active_session, active_turn = mock_store.ask(
        session=active_session,
        plan_item_number=1,
        question_type="project_deep_dive",
        question="进行中的问题",
    )
    active_session, active_turn = mock_store.record_answer(
        session=active_session,
        turn=active_turn,
        answer="进行中的回答",
    )
    active_session, _ = mock_store.record_evaluation(
        session=active_session,
        turn=active_turn,
        evaluation=MockInterviewAnswerEvaluation(
            rating="strong",
            summary="进行中的反馈",
            dimensions=(
                MockInterviewScoreDimension(
                    dimension="specificity",
                    score=4,
                    feedback="回答具体",
                ),
            ),
            next_action="next_question",
            next_action_reason="继续下一题",
        ),
    )
    app = create_app(
        api_key_store_factory=lambda: api_keys,
        runtime_factory=_Runtime,
        action_center_factory=_ActionCenter,
        workspace_reader_factory=lambda: build_workspace_reader(args),
    )

    with TestClient(app) as client:
        owned = client.get(
            f"/v1/applications/{application.id}/mock-interviews",
            headers=issue_key("owner-1", WORKSPACE_READ),
            params={"limit": 2},
        )
        bounded = client.get(
            f"/v1/applications/{application.id}/mock-interviews",
            headers=issue_key("owner-1", WORKSPACE_READ),
            params={"limit": 1},
        )
        other_owner = client.get(
            f"/v1/applications/{application.id}/mock-interviews",
            headers=issue_key("owner-2", WORKSPACE_READ),
        )
        missing = client.get(
            "/v1/applications/missing/mock-interviews",
            headers=issue_key("owner-1", WORKSPACE_READ),
        )

    assert owned.status_code == 200
    assert owned.json()["application_id"] == application.id
    sessions = {item["session_id"]: item for item in owned.json()["sessions"]}
    assert sessions[completed_session.id]["status"] == "completed"
    assert sessions[completed_session.id]["question_count"] == len(
        report.question_results
    )
    assert sessions[completed_session.id]["report_id"] == report.id
    assert sessions[active_session.id]["status"] == "active"
    assert sessions[active_session.id]["question_count"] == 1
    assert sessions[active_session.id]["report_id"] is None
    assert bounded.status_code == 200
    assert len(bounded.json()["sessions"]) == 1
    assert other_owner.status_code == 404
    assert missing.status_code == 404


def test_application_creation_requires_write_scope(api_keys, issue_key) -> None:
    from career_agent.storage.api_keys import WORKSPACE_READ, WORKSPACE_WRITE

    class _Reader(_WorkspaceReader):
        def create_application(self, **values):
            self.created = values
            return ApplicationView(
                id="app-created",
                status="submitted",
                title="AI Engineer",
                company_name="Example",
                submitted_at=NOW,
                updated_at=NOW,
            )

    reader = _Reader()
    app = create_app(
        api_key_store_factory=lambda: api_keys,
        runtime_factory=_Runtime,
        action_center_factory=_ActionCenter,
        workspace_reader_factory=lambda: reader,
    )
    payload = {
        "job_posting_id": "job-1",
        "resume_version_id": "resume-version-1",
        "submitted_at": NOW.isoformat(),
        "note": "官网投递",
    }

    with TestClient(app) as client:
        refused = client.post(
            "/v1/applications",
            headers=issue_key("u1", WORKSPACE_READ),
            json=payload,
        )
        created = client.post(
            "/v1/applications",
            headers=issue_key("u1", WORKSPACE_WRITE),
            json=payload,
        )

    assert refused.status_code == 403
    assert created.status_code == 200
    assert created.json()["id"] == "app-created"
    assert reader.created["user_id"] == "u1"
    assert reader.created["job_posting_id"] == "job-1"


def test_the_shortlist_hides_what_the_user_ruled_out_and_says_who_may_do_it(
    api_keys, issue_key
) -> None:
    """Triage is a workspace act, not a conversation, and not a read.

    ``workspace:write`` rather than ``chat:write``: tidying a list the user is
    already looking at is direct manipulation of existing records, and a
    dashboard that can do it must not thereby gain the ability to run the
    agent. Rather than a read, because ruling a job out is the one fact about
    the shortlist that cannot be derived from what is stored.
    """
    from career_agent.storage.api_keys import WORKSPACE_READ, WORKSPACE_WRITE

    class _Reader(_WorkspaceReader):
        def __init__(self) -> None:
            super().__init__()
            self.pursuit: list[tuple[str, str, str]] = []

        def set_job_pursuit(self, *, user_id, job_posting_id, status):
            self.pursuit.append((user_id, job_posting_id, status))
            return True

    reader = _Reader()
    app = create_app(
        api_key_store_factory=lambda: api_keys,
        runtime_factory=_Runtime,
        action_center_factory=_ActionCenter,
        workspace_reader_factory=lambda: reader,
    )
    writer = issue_key("u1", WORKSPACE_WRITE)
    viewer = issue_key("u1", WORKSPACE_READ)

    with TestClient(app) as client:
        refused = client.put(
            "/v1/jobs/job-1/pursuit", headers=viewer,
            json={"pursuit_status": "dismissed"},
        )
        accepted = client.put(
            "/v1/jobs/job-1/pursuit", headers=writer,
            json={"pursuit_status": "dismissed"},
        )

    assert refused.status_code == 403
    assert accepted.status_code == 200
    assert accepted.json()["changed"] is True
    # Only the write that carried the scope reached the store.
    assert reader.pursuit == [("u1", "job-1", "dismissed")]


def test_permanent_job_deletion_requires_write_scope_and_reports_conflicts(
    api_keys, issue_key
) -> None:
    from career_agent.storage.api_keys import WORKSPACE_READ, WORKSPACE_WRITE

    class _Reader(_WorkspaceReader):
        def __init__(self) -> None:
            super().__init__()
            self.deleted: list[tuple[str, str]] = []

        def delete_job(self, *, user_id, job_posting_id):
            self.deleted.append((user_id, job_posting_id))
            return {
                "job-ok": "deleted",
                "job-applied": "has_application",
            }.get(job_posting_id, "not_found")

    reader = _Reader()
    app = create_app(
        api_key_store_factory=lambda: api_keys,
        runtime_factory=_Runtime,
        action_center_factory=_ActionCenter,
        workspace_reader_factory=lambda: reader,
    )

    with TestClient(app) as client:
        refused = client.delete(
            "/v1/jobs/job-ok", headers=issue_key("u1", WORKSPACE_READ)
        )
        deleted = client.delete(
            "/v1/jobs/job-ok", headers=issue_key("u1", WORKSPACE_WRITE)
        )
        applied = client.delete(
            "/v1/jobs/job-applied", headers=issue_key("u1", WORKSPACE_WRITE)
        )
        missing = client.delete(
            "/v1/jobs/job-missing", headers=issue_key("u1", WORKSPACE_WRITE)
        )

    assert refused.status_code == 403
    assert deleted.status_code == 200
    assert deleted.json() == {"job_posting_id": "job-ok", "deleted": True}
    assert applied.status_code == 409
    assert applied.json()["detail"]["code"] == "JOB_HAS_APPLICATION"
    assert missing.status_code == 404
    assert reader.deleted == [
        ("u1", "job-ok"),
        ("u1", "job-applied"),
        ("u1", "job-missing"),
    ]


def test_conversation_deletion_is_owner_scoped_and_requires_write_scope(
    api_keys, issue_key
) -> None:
    from career_agent.storage.api_keys import WORKSPACE_READ, WORKSPACE_WRITE

    class _Reader(_WorkspaceReader):
        def __init__(self) -> None:
            super().__init__()
            self.deleted: list[tuple[str, str]] = []

        def delete_conversation(self, *, user_id, conversation_id):
            self.deleted.append((user_id, conversation_id))
            return conversation_id == "conversation-1"

    reader = _Reader()
    app = create_app(
        api_key_store_factory=lambda: api_keys,
        runtime_factory=_Runtime,
        action_center_factory=_ActionCenter,
        workspace_reader_factory=lambda: reader,
    )

    with TestClient(app) as client:
        refused = client.delete(
            "/v1/conversations/conversation-1",
            headers=issue_key("u1", WORKSPACE_READ),
        )
        deleted = client.delete(
            "/v1/conversations/conversation-1",
            headers=issue_key("u1", WORKSPACE_WRITE),
        )
        missing = client.delete(
            "/v1/conversations/missing",
            headers=issue_key("u2", WORKSPACE_WRITE),
        )

    assert refused.status_code == 403
    assert deleted.json() == {"conversation_id": "conversation-1", "deleted": True}
    assert missing.status_code == 404
    assert missing.json()["detail"]["code"] == "CONVERSATION_NOT_FOUND"
    assert reader.deleted == [("u1", "conversation-1"), ("u2", "missing")]


def test_email_workspace_exposes_safe_metadata_only(api_keys, auth) -> None:
    class _Reader(_WorkspaceReader):
        def email(self, *, user_id: str, limit: int = 100):
            self.calls.append(("email", user_id, limit))
            return EmailWorkspaceResponse(
                accounts=(
                    EmailAccountView(
                        id="mail-1",
                        provider="gmail",
                        email_address="owner@example.com",
                        status="active",
                        last_synced_at=NOW,
                    ),
                ),
                events=(
                    EmailEventView(
                        id="event-1",
                        event_type="interview_invitation",
                        status="pending_confirmation",
                        summary="邀请参加面试",
                        confidence=0.94,
                        application_id="app-1",
                        application_title="AI 产品经理",
                        company_name="示例科技",
                        occurred_at=NOW,
                    ),
                ),
            )

    reader = _Reader()
    app = create_app(
        api_key_store_factory=lambda: api_keys,
        runtime_factory=_Runtime,
        action_center_factory=_ActionCenter,
        workspace_reader_factory=lambda: reader,
    )

    with TestClient(app) as client:
        response = client.get("/v1/email", headers=auth)

    assert response.status_code == 200
    assert response.json()["events"][0]["event_type"] == "interview_invitation"
    assert "credential_ref" not in response.text
    assert "body" not in response.text
    assert reader.calls == [("email", "u1", 100)]


def test_resume_import_and_target_role_creation_require_write_scope(
    api_keys, issue_key
) -> None:
    from career_agent.storage.api_keys import WORKSPACE_READ, WORKSPACE_WRITE

    class _Reader(_WorkspaceReader):
        def create_target_role(self, *, user_id: str, title: str):
            self.calls.append((f"role:{title}", user_id, None))
            return TargetRoleView(id="role-1", title=title, priority=0, status="active")

        def import_resume(self, **values):
            self.calls.append(("import", values["user_id"], len(values["content"])))
            return ResumeImportResponse(
                resume_id="resume-1",
                resume_version_id="version-1",
                name=values["name"],
                version_number=1,
                document_format=values["document_format"],
                byte_size=len(values["content"]),
            )

    reader = _Reader()
    app = create_app(
        api_key_store_factory=lambda: api_keys,
        runtime_factory=_Runtime,
        action_center_factory=_ActionCenter,
        workspace_reader_factory=lambda: reader,
    )
    viewer = issue_key("u1", WORKSPACE_READ)
    writer = issue_key("u1", WORKSPACE_WRITE)

    with TestClient(app) as client:
        refused = client.post(
            "/v1/resumes/import",
            headers=viewer,
            data={"name": "主简历", "target_role_id": "role-1"},
            files={"file": ("resume.md", b"# Resume", "text/markdown")},
        )
        role = client.post("/v1/target-roles", headers=writer, json={"title": "AI 产品经理"})
        imported = client.post(
            "/v1/resumes/import",
            headers=writer,
            data={"name": "主简历", "target_role_id": "role-1"},
            files={"file": ("resume.md", b"# Resume", "text/markdown")},
        )
        invalid = client.post(
            "/v1/resumes/import",
            headers=writer,
            data={"name": "坏文件", "target_role_id": "role-1"},
            files={"file": ("resume.exe", b"no", "application/octet-stream")},
        )

    assert refused.status_code == 403
    assert role.json()["id"] == "role-1"
    assert imported.json()["document_format"] == "markdown"
    assert invalid.status_code == 400
    assert reader.calls == [
        ("role:AI 产品经理", "u1", None),
        ("import", "u1", len(b"# Resume")),
    ]


def test_reading_a_stored_jd_does_not_require_the_agent(api_keys, auth) -> None:
    """Two ways to reach one document, not two renderings of it.

    Until this endpoint existed, the library carried a job's *analysis* but
    never the posting, so reading a file the user already had meant spending a
    model call to have it read back. The bytes here are the same snapshot
    ``get_saved_job`` returns in conversation — that agent path hands back
    ``jd_snapshot.content`` verbatim rather than letting a model restate it,
    which is what lets both surfaces exist without disagreeing.
    """

    class _Reader(_WorkspaceReader):
        def job_detail(self, *, user_id: str, job_posting_id: str):
            self.calls.append(("job_detail", user_id, None))
            if job_posting_id != "job-1":
                return None
            return SavedJobDetailView(
                id="job-1",
                title="AI 产品经理",
                company_name="示例科技",
                source_name="browser_capture",
                availability_status="active",
                pursuit_status="open",
                jd_text="岗位职责：\n1. 规划 AI 产品路线\n2. 推动落地",
                jd_version=2,
                captured_at=NOW,
            )

    app = create_app(
        api_key_store_factory=lambda: api_keys,
        runtime_factory=_Runtime,
        action_center_factory=_ActionCenter,
        workspace_reader_factory=_Reader,
    )

    with TestClient(app) as client:
        found = client.get("/v1/jobs/job-1", headers=auth)
        missing = client.get("/v1/jobs/job-404", headers=auth)

    # Stored as captured: the line structure survives, because a JD reflowed
    # into one paragraph is a different document to read.
    assert found.json()["jd_text"].splitlines()[0] == "岗位职责："
    assert found.json()["jd_version"] == 2
    assert missing.status_code == 404


def test_a_closure_can_be_recorded_by_hand_when_the_extension_cannot(
    api_keys, issue_key
) -> None:
    """The extension was never a separate source of truth.

    It reports what the person looking at the page saw, so this route writes
    the same field with the same meaning — for the ordinary cases where the
    shortcut is unavailable: the site reworded its notice, the page needs a
    login, the extension is not installed on this machine. Without it, a job
    the user knows is gone sits in the shortlist as a live candidate.
    """
    from career_agent.storage.api_keys import WORKSPACE_READ, WORKSPACE_WRITE

    class _Reader(_WorkspaceReader):
        def __init__(self) -> None:
            super().__init__()
            self.availability: list[tuple[str, str, str]] = []

        def set_job_availability(self, *, user_id, job_posting_id, status):
            self.availability.append((user_id, job_posting_id, status))
            return True

    reader = _Reader()
    app = create_app(
        api_key_store_factory=lambda: api_keys,
        runtime_factory=_Runtime,
        action_center_factory=_ActionCenter,
        workspace_reader_factory=lambda: reader,
    )

    with TestClient(app) as client:
        refused = client.put(
            "/v1/jobs/job-1/availability",
            headers=issue_key("u1", WORKSPACE_READ),
            json={"availability_status": "closed"},
        )
        closed = client.put(
            "/v1/jobs/job-1/availability",
            headers=issue_key("u1", WORKSPACE_WRITE),
            json={"availability_status": "closed"},
        )
        # Relisting is a real answer too, so the field moves in both directions
        # and a misclick costs one click rather than being permanent.
        reopened = client.put(
            "/v1/jobs/job-1/availability",
            headers=issue_key("u1", WORKSPACE_WRITE),
            json={"availability_status": "active"},
        )

    assert refused.status_code == 403
    assert closed.status_code == 200
    assert reader.availability == [
        ("u1", "job-1", "closed"),
        ("u1", "job-1", "active"),
    ]
    assert reopened.json()["availability_status"] == "active"


def test_the_employers_state_and_the_users_decision_stay_separate_over_http(
    api_keys, issue_key
) -> None:
    """Two routes because they are two facts.

    A relisted posting and one the reader changed their mind about must not be
    the same record. Folding them into one control would make the two
    indistinguishable after the fact, and the pair is exactly what the two
    columns exist to keep apart.
    """
    from career_agent.storage.api_keys import WORKSPACE_WRITE

    class _Reader(_WorkspaceReader):
        def __init__(self) -> None:
            super().__init__()
            self.writes: list[tuple[str, str]] = []

        def set_job_availability(self, *, user_id, job_posting_id, status):
            self.writes.append(("availability", status))
            return True

        def set_job_pursuit(self, *, user_id, job_posting_id, status):
            self.writes.append(("pursuit", status))
            return True

    reader = _Reader()
    app = create_app(
        api_key_store_factory=lambda: api_keys,
        runtime_factory=_Runtime,
        action_center_factory=_ActionCenter,
        workspace_reader_factory=lambda: reader,
    )
    headers = issue_key("u1", WORKSPACE_WRITE)

    with TestClient(app) as client:
        client.put(
            "/v1/jobs/job-1/availability", headers=headers,
            json={"availability_status": "closed"},
        )
        client.put(
            "/v1/jobs/job-1/pursuit", headers=headers,
            json={"pursuit_status": "dismissed"},
        )

    assert reader.writes == [("availability", "closed"), ("pursuit", "dismissed")]


def test_job_status_updates_report_missing_jobs(api_keys, issue_key) -> None:
    from career_agent.storage.api_keys import WORKSPACE_WRITE

    class _Reader(_WorkspaceReader):
        def set_job_availability(self, *, user_id, job_posting_id, status):
            return False

        def set_job_pursuit(self, *, user_id, job_posting_id, status):
            return False

        def job_detail(self, *, user_id, job_posting_id):
            return None

    app = create_app(
        api_key_store_factory=lambda: api_keys,
        runtime_factory=_Runtime,
        action_center_factory=_ActionCenter,
        workspace_reader_factory=_Reader,
    )
    headers = issue_key("u1", WORKSPACE_WRITE)

    with TestClient(app) as client:
        availability = client.put(
            "/v1/jobs/missing/availability",
            headers=headers,
            json={"availability_status": "closed"},
        )
        pursuit = client.put(
            "/v1/jobs/missing/pursuit",
            headers=headers,
            json={"pursuit_status": "dismissed"},
        )

    assert availability.status_code == 404
    assert pursuit.status_code == 404
    assert availability.json()["detail"]["code"] == "SAVED_JOB_NOT_FOUND"
    assert pursuit.json()["detail"]["code"] == "SAVED_JOB_NOT_FOUND"
