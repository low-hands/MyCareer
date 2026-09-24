"""Every tool that no other test executes, run once against real stores.

``analyze_job`` shipped a result card its own contract rejected and failed every
turn that used it for eight days, because no test ever called the tool: the
tests around it scripted its result instead. The tools here were in the same
position when this file was written (measured by wrapping the registry's
dispatch over the full suite). Each is invoked through the registry against
the SQLite stores and services production wires, with only the model workers
and external connectors replaced, so a result the runtime cannot accept fails
here instead of in a user's turn.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from career_agent.agent.main_agent_tools import MainAgentToolRegistry
from career_agent.agent.openai_compatible_client import (
    AgentWorkerError,
    ProviderErrorMetadata,
)
from career_agent.domain.email_tracking import (
    EmailAssessment,
    EmailSyncBatch,
    RemoteEmailContent,
    RemoteEmailMetadata,
)
from career_agent.domain.interview_preparation import (
    EvidenceStory,
    InterviewFocusArea,
    InterviewPreparationResult,
    LikelyQuestion,
)
from career_agent.domain.interviews import InterviewDetails
from career_agent.domain.job_discovery import JobDetail, Provenance
from career_agent.domain.job_research import (
    JobResearchDraft,
    JobResearchFindingDraft,
    JobResearchSourceDraft,
)
from career_agent.services.action_center import ActionCenterService
from career_agent.services.applications import ApplicationService
from career_agent.services.calendar import CalendarService
from career_agent.services.email_tracking import EmailTrackingService
from career_agent.services.interview_preparation import InterviewPreparationService
from career_agent.services.interviews import InterviewService
from career_agent.services.job_research import (
    JobResearchExecutionError,
    JobResearchService,
)
from career_agent.storage.action_center import SQLiteActionItemStore
from career_agent.storage.applications import SQLiteApplicationStore
from career_agent.storage.calendar import SQLiteCalendarStore
from career_agent.storage.career_history import CareerHistoryStore
from career_agent.storage.email_tracking import SQLiteEmailTrackingStore
from career_agent.storage.interview_preparations import SQLiteInterviewPreparationStore
from career_agent.storage.interviews import SQLiteInterviewStore
from career_agent.storage.job_research import SQLiteJobResearchStore
from career_agent.storage.jobs import SQLiteJobPostingRepository
from career_agent.storage.resumes import ResumeStore


NOW = datetime.now(timezone.utc)
USER = "u1"


class PreparationWorker:
    def prepare(self, **kwargs):
        return InterviewPreparationResult(
            summary="重点准备 RAG 可靠性。",
            focus_areas=(InterviewFocusArea(
                topic="RAG reliability", priority="high",
                rationale="The JD asks for reliable retrieval.",
                jd_quote="Build reliable retrieval systems",
            ),),
            evidence_stories=(EvidenceStory(
                theme="Evaluation", resume_locator="Experience, bullet 1",
                resume_quote="Built production RAG systems",
                preparation_prompt="补充你用过的指标。",
            ),),
            likely_questions=(LikelyQuestion(
                question="如何评估 RAG？", rationale="岗位强调可靠性。",
                answer_outline=("说明评估方法",),
            ),),
            checklist=("确认会议链接",),
        )


class ResearchWorker:
    """Fails the first run so there is something to retry."""

    def __init__(self) -> None:
        self.failed_run_id: str | None = None

    def research(self, *, run_id, request, resume=False):
        if self.failed_run_id is None:
            self.failed_run_id = run_id
            raise AgentWorkerError(
                "JOB_RESEARCH_TRANSPORT_ERROR", "temporary failure",
                retryable=True,
                provider=ProviderErrorMetadata(category="transport", retryable=True),
            )
        return JobResearchDraft(
            summary="The role is tied to enterprise retrieval.",
            sources=(JobResearchSourceDraft(
                source_key="S1", url="https://example.com/product",
                title="Retrieval product", publisher="Acme",
                relevant_excerpt="Acme sells enterprise retrieval.",
            ),),
            findings=(JobResearchFindingDraft(
                topic="Product", statement="Acme targets enterprise retrieval.",
                evidence_type="fact", source_keys=("S1",), confidence="high",
            ),),
            open_questions=(),
        )

    def forget(self, run_id):
        return None


class Mailbox:
    provider = "gmail"

    def sync_metadata(self, *, cursor, since):
        return EmailSyncBatch(
            messages=(RemoteEmailMetadata(
                external_message_id="m1", sender="Acme Recruiting",
                subject="面试邀请", received_at=NOW,
            ),),
            next_cursor_value="h1",
        )

    def get_content(self, *, external_message_id):
        return RemoteEmailContent(
            external_message_id=external_message_id,
            text="Acme 邀请您参加 RAG Engineer 面试。",
        )


class MailboxResolver:
    def resolve(self, **kwargs):
        return Mailbox()


class PendingEmailWorker:
    classifier = "smoke_pending_v1"
    authorizes_auto_apply = False

    def __init__(self) -> None:
        self.application_id: str | None = None

    def assess(self, **kwargs):
        return EmailAssessment(
            event_type="interview_invitation",
            application_id=self.application_id,
            confidence=0.7,
            summary="识别到面试邀请，待确认。",
        )


class NoCalendarConnector:
    def resolve(self, **kwargs):
        raise AssertionError("listing must not reach the calendar provider")


@pytest.fixture
def world(tmp_path):
    jobs = SQLiteJobPostingRepository(tmp_path / "jobs.sqlite3")
    resume_path = tmp_path / "resumes.sqlite3"
    resumes = ResumeStore(resume_path)
    history = CareerHistoryStore(resume_path)
    role = resumes.create_target_role(user_id=USER, title="AI Engineer", priority=1)
    _, version = resumes.import_document(
        user_id=USER, target_role_id=role.id, name="AI Resume",
        content=b"Experience: Built production RAG systems", document_format="text",
    )
    saved = jobs.save_detail(
        user_id=USER, run_id="run-1", result_ref="ref-1", selection_index=1,
        detail=JobDetail(
            source_name="test", source_job_id="job-1",
            title="RAG Engineer", company_name="Acme",
            description="Build reliable retrieval systems. Strong Python.",
            captured_at=NOW,
            provenance=Provenance(
                source_name="test", source_job_id="job-1", captured_at=NOW,
                operation="detail", adapter_version="test-v1",
            ),
        ),
    )
    applications = ApplicationService(
        SQLiteApplicationStore(tmp_path / "applications.sqlite3"), jobs, resumes
    )
    application = applications.create_application(
        user_id=USER, job_posting_id=saved.posting.id,
        resume_version_id=version.id, submitted_at=NOW - timedelta(days=10),
    ).application
    interviews = InterviewService(
        SQLiteInterviewStore(tmp_path / "applications.sqlite3"), applications
    )
    interview = interviews.create_manual(
        user_id=USER, application_id=application.id,
        details=InterviewDetails(
            scheduled_start=NOW + timedelta(days=1),
            scheduled_end=NOW + timedelta(days=1, hours=1),
            timezone="Asia/Shanghai", interview_format="video",
        ),
    )
    email_worker = PendingEmailWorker()
    email_worker.application_id = application.id
    email_store = SQLiteEmailTrackingStore(tmp_path / "email.sqlite3")
    emails = EmailTrackingService(
        email_store, applications, MailboxResolver(), email_worker, interviews
    )
    account = email_store.add_account(
        user_id=USER, provider="gmail", email_address="user@gmail.com",
        credential_ref="env:GMAIL_SECRET",
    )
    email_event = emails.sync(user_id=USER, account_id=account.id).events_created[0]
    preparations = InterviewPreparationService(
        interviews, applications, resumes, history, PreparationWorker(),
        SQLiteInterviewPreparationStore(resume_path),
    )
    preparation = preparations.prepare(
        user_id=USER, interview_round_id=interview.id
    )
    research_worker = ResearchWorker()
    research = JobResearchService(
        jobs=jobs,
        store=SQLiteJobResearchStore(tmp_path / "research.sqlite3"),
        worker=research_worker,
    )
    with pytest.raises(JobResearchExecutionError):
        research.research(user_id=USER, job_posting_id=saved.posting.id)
    actions = ActionCenterService(
        SQLiteActionItemStore(tmp_path / "actions.sqlite3"),
        applications, emails, interviews, job_repository=jobs,
    )
    registry = MainAgentToolRegistry(
        job_repository=jobs,
        resume_store=resumes,
        career_history_store=history,
        application_service=applications,
        interview_service=interviews,
        interview_preparation_service=preparations,
        email_tracking_service=emails,
        action_center_service=actions,
        calendar_service=CalendarService(
            SQLiteCalendarStore(tmp_path / "calendar.sqlite3"),
            interviews, applications, NoCalendarConnector(),
        ),
        job_research_service=research,
    )
    return {
        "registry": registry,
        "interview_id": interview.id,
        "preparation_id": preparation.id,
        "email_event_id": email_event.id,
        "failed_run_id": research_worker.failed_run_id,
    }


def _call(world, name, **arguments):
    return world["registry"].invoke_atomic_tool(name, {"user_id": USER, **arguments})


def test_calendar_listings_run_without_touching_the_provider(world) -> None:
    assert _call(world, "list_calendar_accounts").state == "no_calendar_accounts"
    assert _call(world, "list_calendar_links").state == "no_calendar_links"


def test_action_items_can_be_listed_snoozed_and_dismissed(world) -> None:
    listed = _call(world, "list_action_items")
    assert listed.state == "action_items_found", listed.message
    items = listed.payload["items"]
    assert len(items) >= 2, [item.get("action_type") for item in items]

    snoozed = _call(
        world, "snooze_action_item",
        action_item_id=items[0]["action_item_id"],
        snoozed_until=(NOW + timedelta(days=2)).isoformat(),
    )
    assert snoozed.state == "action_item_snoozed"

    dismissed = _call(
        world, "dismiss_action_item", action_item_id=items[1]["action_item_id"]
    )
    assert dismissed.state == "action_item_resolved"


def test_an_interview_can_be_rescheduled_then_completed(world) -> None:
    updated = _call(
        world, "update_interview",
        interview_round_id=world["interview_id"],
        details={
            "change_type": "rescheduled",
            "scheduled_start": (NOW + timedelta(days=2)).isoformat(),
            "scheduled_end": (NOW + timedelta(days=2, hours=1)).isoformat(),
            "timezone": "Asia/Shanghai",
            "interview_format": "video",
        },
    )
    assert updated.state == "interview_ready", updated.message

    completed = _call(
        world, "complete_interview", interview_round_id=world["interview_id"]
    )
    assert completed.state == "interview_ready", completed.message


def test_a_stored_preparation_is_read_back_with_its_card(world) -> None:
    read = _call(
        world, "get_interview_preparation", preparation_id=world["preparation_id"]
    )
    assert read.resource_ref is not None, read.state
    assert read.resource_ref.kind == "interview_preparation"


def test_a_pending_email_event_can_be_resolved(world) -> None:
    resolved = _call(
        world, "resolve_email_event", event_id=world["email_event_id"], approve=False
    )
    assert resolved.state not in {"email_event_not_found"}, resolved.message


def test_a_failed_research_run_can_be_retried_into_a_report(world) -> None:
    retried = world["registry"].invoke_workflow(
        "retry_job_research", {"user_id": USER, "run_id": world["failed_run_id"]}
    )
    assert retried.state == "job_research_ready", retried.message
    assert retried.resource_ref is not None
    assert retried.resource_ref.kind == "job_research_report"
