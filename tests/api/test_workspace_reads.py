from __future__ import annotations

from datetime import datetime, timezone

from fastapi.testclient import TestClient

from career_agent.api.app import create_app
from career_agent.api.reads import (
    ApplicationView,
    CalendarAccountView,
    CalendarWorkspaceResponse,
    CompanyResearchView,
    ConversationMessageView,
    ConversationTranscriptResponse,
    ConversationView,
    ResumeView,
    SavedJobView,
)
from career_agent.domain.action_center import ActionItem, DailyBrief


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

    def resumes(self, *, user_id: str):
        self.calls.append(("resumes", user_id, None))
        return (
            ResumeView(
                id="resume-1",
                name="AI 产品简历",
                target_role="AI 产品经理",
                status="active",
                latest_version_number=2,
                version_count=2,
                document_format="pdf",
                byte_size=4096,
                updated_at=NOW,
            ),
        )

    def jobs(self, *, user_id: str, limit: int = 100):
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


def test_workspace_read_endpoints_scope_data_and_build_dashboard() -> None:
    reader = _WorkspaceReader()
    app = create_app(
        runtime_factory=_Runtime,
        action_center_factory=_ActionCenter,
        workspace_reader_factory=lambda: reader,
    )

    with TestClient(app) as client:
        applications = client.get("/v1/applications", params={"user_id": "u1"})
        jobs = client.get("/v1/jobs", params={"user_id": "u1"})
        conversations = client.get("/v1/conversations", params={"user_id": "u1"})
        transcript = client.get(
            "/v1/conversations/conversation-1/messages",
            params={"user_id": "u1"},
        )
        resumes = client.get("/v1/resumes", params={"user_id": "u1"})
        calendar = client.get("/v1/calendar", params={"user_id": "u1"})
        research = client.get("/v1/company-research", params={"user_id": "u1"})
        dashboard = client.get("/v1/dashboard", params={"user_id": "u1"})

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


def test_workspace_read_limits_are_validated_before_store_access() -> None:
    reader = _WorkspaceReader()
    app = create_app(
        runtime_factory=_Runtime,
        action_center_factory=_ActionCenter,
        workspace_reader_factory=lambda: reader,
    )

    with TestClient(app) as client:
        response = client.get(
            "/v1/company-research",
            params={"user_id": "u1", "limit": 0},
        )

    assert response.status_code == 422
    assert reader.calls == []
