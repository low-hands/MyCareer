"""Clearing applications takes everything that hangs off them with it.

Clearing used to delete only applications and their events. The interview
round of a cleared application stayed, and the action center kept generating a
retro reminder for it in the daily brief.
"""

from __future__ import annotations

from argparse import Namespace
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sqlite3

from career_agent.api.reads import WorkspaceReader
from career_agent.domain.action_center import ActionCandidate
from career_agent.domain.interviews import InterviewDetails
from career_agent.domain.job_discovery import JobDetail, Provenance
from career_agent.services.applications import ApplicationService
from career_agent.services.interviews import InterviewService
from career_agent.storage.action_center import SQLiteActionItemStore
from career_agent.storage.applications import SQLiteApplicationStore
from career_agent.storage.interview_preparations import SQLiteInterviewPreparationStore
from career_agent.storage.interviews import SQLiteInterviewStore
from career_agent.storage.jobs import SQLiteJobPostingRepository
from career_agent.storage.mock_interviews import SQLiteMockInterviewStore
from career_agent.storage.resumes import ResumeStore


NOW = datetime.now(timezone.utc)


def _paths(tmp_path) -> Namespace:
    return Namespace(**{
        name: str(tmp_path / f"{name}.sqlite3")
        for name in (
            "context_store", "resume_store", "application_store", "job_store",
            "calendar_store", "email_store", "mock_interview_store",
            "job_research_store", "action_store",
        )
    })


def _seed_application(
    paths: Namespace, *, user_id: str, source_job_id: str, mock: bool = True
):
    jobs = SQLiteJobPostingRepository(Path(paths.job_store))
    resumes = ResumeStore(Path(paths.resume_store))
    role = resumes.create_target_role(
        user_id=user_id, title=f"role-{source_job_id}", priority=1
    )
    _, version = resumes.import_document(
        user_id=user_id, content=b"Built production RAG systems",
        document_format="text", name=f"resume-{source_job_id}",
        target_role_id=role.id,
    )
    saved = jobs.save_detail(
        user_id=user_id, run_id=f"run-{source_job_id}", result_ref=source_job_id,
        selection_index=1,
        detail=JobDetail(
            source_name="test", source_job_id=source_job_id,
            title="RAG Engineer", company_name="Acme",
            description="Build reliable retrieval systems.", captured_at=NOW,
            provenance=Provenance(
                source_name="test", source_job_id=source_job_id, captured_at=NOW,
                operation="detail", adapter_version="test-v1",
            ),
        ),
    )
    applications = ApplicationService(
        SQLiteApplicationStore(Path(paths.application_store)), jobs, resumes
    )
    application = applications.create_application(
        user_id=user_id, job_posting_id=saved.posting.id,
        resume_version_id=version.id,
    ).application
    interview = InterviewService(
        SQLiteInterviewStore(Path(paths.application_store)), applications
    ).create_manual(
        user_id=user_id, application_id=application.id,
        details=InterviewDetails(scheduled_start=NOW + timedelta(days=1)),
    )
    # A user may hold one unfinished mock interview at a time.
    if mock:
        SQLiteMockInterviewStore(Path(paths.mock_interview_store)).create_session(
            user_id=user_id, application_id=application.id,
            job_posting_id=saved.posting.id, jd_snapshot_id=saved.snapshot.id,
            resume_version_id=version.id, interview_type="technical",
            interview_round_id=interview.id,
        )
    # Written directly: generating one needs a model worker, and what is under
    # test is only that the row goes with its application.
    SQLiteInterviewPreparationStore(Path(paths.resume_store))
    with sqlite3.connect(paths.resume_store) as connection:
        connection.execute(
            "INSERT INTO interview_preparations(id, user_id, interview_round_id, "
            "application_id, job_posting_id, jd_snapshot_id, resume_version_id, "
            "input_fingerprint, worker_version, result_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                f"prep-{source_job_id}", user_id, interview.id, application.id,
                saved.posting.id, saved.snapshot.id, version.id, "fp", "v1", "{}",
                NOW.isoformat(),
            ),
        )
    actions = SQLiteActionItemStore(Path(paths.action_store))
    for action_type, source_type, source_id in (
        ("interview_retro", "interview_round", interview.id),
        ("application_follow_up", "application", application.id),
    ):
        actions.upsert_candidate(
            user_id=user_id, now=NOW,
            candidate=ActionCandidate(
                stable_key=f"{action_type}:{source_id}",
                action_type=action_type, source_type=source_type,
                source_id=source_id, application_id=application.id,
                title=action_type, summary=action_type,
            ),
        )
    return application, interview


def _count(path: str, table: str, user_id: str) -> int:
    with sqlite3.connect(path) as connection:
        return connection.execute(
            f"SELECT COUNT(*) FROM {table} WHERE user_id = ?", (user_id,)
        ).fetchone()[0]


def test_clearing_applications_takes_their_dependents_and_earlier_orphans(
    tmp_path,
) -> None:
    paths = _paths(tmp_path)
    _seed_application(paths, user_id="u1", source_job_id="orphaned")
    # The state an application-only clear left behind: interview, mock
    # interview and reminders pointing at an application that is gone.
    SQLiteApplicationStore(Path(paths.application_store)).clear_user(user_id="u1")
    _seed_application(paths, user_id="u1", source_job_id="current", mock=False)
    _seed_application(paths, user_id="u2", source_job_id="someone-else")
    unrelated = SQLiteActionItemStore(Path(paths.action_store)).upsert_candidate(
        user_id="u1", now=NOW,
        candidate=ActionCandidate(
            stable_key="saved_job_review", action_type="saved_job_review",
            source_type="saved_job_library", source_id="saved_job_library",
            title="整理岗位库", summary="整理岗位库",
        ),
    )

    cleared = WorkspaceReader(paths).clear_applications(user_id="u1")

    assert cleared == 1
    for path, table in (
        (paths.application_store, "applications"),
        (paths.application_store, "interview_rounds"),
        (paths.application_store, "interview_round_events"),
        (paths.mock_interview_store, "mock_interview_sessions"),
        (paths.resume_store, "interview_preparations"),
    ):
        assert _count(path, table, "u1") == 0, table
        assert _count(path, table, "u2") > 0, table
    remaining = SQLiteActionItemStore(Path(paths.action_store)).list(
        user_id="u1", statuses=("open", "snoozed", "completed", "dismissed", "obsolete")
    )
    assert [item.id for item in remaining] == [unrelated.id]
    assert _count(paths.action_store, "action_items", "u2") == 2
