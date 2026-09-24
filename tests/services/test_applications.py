from __future__ import annotations

from datetime import datetime, timezone
import sqlite3

import pytest

from career_agent.domain.job_discovery import JobDetail, Provenance
from career_agent.services.applications import (
    ApplicationInputNotFoundError,
    ApplicationService,
    InvalidApplicationTransitionError,
)
from career_agent.storage.applications import SQLiteApplicationStore
from career_agent.storage.jobs import SQLiteJobPostingRepository
from career_agent.storage.resumes import ResumeStore


def seed_application_service(tmp_path):
    resumes = ResumeStore(tmp_path / "resumes.sqlite3")
    role = resumes.create_target_role(user_id="u1", title="AI Engineer", priority=1)
    resume, version = resumes.import_document(
        user_id="u1",
        target_role_id=role.id,
        name="AI Resume",
        content=b"PRIVATE RESUME",
        document_format="text",
    )
    jobs = SQLiteJobPostingRepository(tmp_path / "jobs.sqlite3")
    captured_at = datetime(2026, 8, 25, tzinfo=timezone.utc)
    job = jobs.save_detail(
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
            captured_at=captured_at,
            provenance=Provenance(
                source_name="test",
                source_job_id="job-1",
                captured_at=captured_at,
                operation="detail",
                adapter_version="test-v1",
            ),
        ),
    )
    store = SQLiteApplicationStore(tmp_path / "applications.sqlite3")
    return ApplicationService(store, jobs, resumes), store, job, resume, version


def test_application_service_is_idempotent_and_records_append_only_events(
    tmp_path,
) -> None:
    service, store, job, _, version = seed_application_service(tmp_path)

    first = service.create_application(
        user_id="u1",
        job_posting_id=job.posting.id,
        resume_version_id=version.id,
        note="Applied through the company site.",
    )
    repeated = service.create_application(
        user_id="u1",
        job_posting_id=job.posting.id,
        resume_version_id=version.id,
    )

    assert first.created is True
    assert repeated.created is False
    assert repeated.application.id == first.application.id
    acknowledged = service.update_application(
        user_id="u1",
        application_id=first.application.id,
        status="acknowledged",
        note="Recruiter call scheduled.",
    )
    assert acknowledged.status == "acknowledged"
    service.update_application(
        user_id="u1",
        application_id=first.application.id,
        status="acknowledged",
        note="Recruiter call completed.",
    )
    events = store.list_events(
        user_id="u1",
        application_id=first.application.id,
    )
    assert [event.event_type for event in events] == [
        "created",
        "status_changed",
        "note_added",
    ]
    assert events[0].note == "Applied through the company site."
    assert events[-1].previous_status == events[-1].new_status == "acknowledged"
    assert all(event.source == "user_reported" for event in events)
    assert first.application.jd_snapshot_id == job.snapshot.id


def test_application_can_track_a_user_report_before_the_resume_is_known(tmp_path) -> None:
    service, _, job, _, _ = seed_application_service(tmp_path)

    created = service.create_application(
        user_id="u1",
        job_posting_id=job.posting.id,
        resume_version_id=None,
        note="Interview invitation reported by the user.",
    ).application

    assert created.resume_version_id is None
    assert service.get_application(
        user_id="u1", application_id=created.id
    ).application == created


def test_application_v1_upgrade_keeps_existing_rows_and_allows_unknown_resume(
    tmp_path,
) -> None:
    path = tmp_path / "applications.sqlite3"
    now = datetime(2026, 9, 23, tzinfo=timezone.utc).isoformat()
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE schema_versions (
                component TEXT PRIMARY KEY, version INTEGER NOT NULL,
                updated_at TEXT NOT NULL
            );
            INSERT INTO schema_versions VALUES ('applications', 1, '2026-09-23');
            CREATE TABLE applications (
                id TEXT PRIMARY KEY, user_id TEXT NOT NULL,
                job_posting_id TEXT NOT NULL, jd_snapshot_id TEXT NOT NULL,
                resume_version_id TEXT NOT NULL, status TEXT NOT NULL,
                submitted_at TEXT NOT NULL, created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL, UNIQUE(user_id, job_posting_id)
            );
            CREATE TABLE application_events (
                id TEXT PRIMARY KEY,
                application_id TEXT NOT NULL REFERENCES applications(id),
                user_id TEXT NOT NULL, source TEXT NOT NULL,
                event_type TEXT NOT NULL, previous_status TEXT,
                new_status TEXT NOT NULL, note TEXT, occurred_at TEXT NOT NULL
            );
            """
        )
        connection.execute(
            "INSERT INTO applications VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("app-1", "u1", "job-1", "jd-1", "resume-1", "submitted", now, now, now),
        )

    store = SQLiteApplicationStore(path)

    restored = store.get(user_id="u1", application_id="app-1")
    assert restored is not None and restored.resume_version_id == "resume-1"
    with sqlite3.connect(path) as connection:
        resume_column = next(
            row for row in connection.execute("PRAGMA table_info(applications)")
            if row[1] == "resume_version_id"
        )
    assert resume_column[3] == 0


def test_application_service_enforces_transitions_and_prevents_duplicate_application(
    tmp_path,
) -> None:
    service, _, job, _, version = seed_application_service(tmp_path)
    created = service.create_application(
        user_id="u1",
        job_posting_id=job.posting.id,
        resume_version_id=version.id,
    ).application

    with pytest.raises(InvalidApplicationTransitionError):
        service.update_application(
            user_id="u1",
            application_id=created.id,
            status="offer",
        )
    rejected = service.update_application(
        user_id="u1",
        application_id=created.id,
        status="rejected",
    )
    assert rejected.status == "rejected"
    with pytest.raises(InvalidApplicationTransitionError):
        service.update_application(
            user_id="u1",
            application_id=created.id,
            status="interview",
        )

    repeated = service.create_application(
        user_id="u1",
        job_posting_id=job.posting.id,
        resume_version_id=version.id,
    )
    assert repeated.created is False
    assert repeated.application.id == created.id


def test_application_service_validates_owned_job_resume_and_application(tmp_path) -> None:
    service, _, job, _, version = seed_application_service(tmp_path)

    with pytest.raises(ApplicationInputNotFoundError, match="job_posting"):
        service.create_application(
            user_id="other",
            job_posting_id=job.posting.id,
            resume_version_id=version.id,
        )
    created = service.create_application(
        user_id="u1",
        job_posting_id=job.posting.id,
        resume_version_id=version.id,
    ).application
    with pytest.raises(ApplicationInputNotFoundError, match="application"):
        service.get_application(user_id="other", application_id=created.id)
    assert service.list_applications(user_id="other") == ()
