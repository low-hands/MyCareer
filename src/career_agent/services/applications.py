from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from career_agent.domain.applications import (
    Application,
    ApplicationEvent,
    ApplicationStatus,
)
from career_agent.storage.applications import SQLiteApplicationStore
from career_agent.storage.jobs import JobPostingRepository, StoredJobRecord
from career_agent.storage.resumes import ResumeStore


class ApplicationInputNotFoundError(ValueError):
    """Raised when an application or one of its owned inputs is unavailable."""


class InvalidApplicationTransitionError(ValueError):
    """Raised when a requested status transition violates pipeline rules."""


class ConcurrentApplicationUpdateError(RuntimeError):
    """Raised when an application changed between read and update."""


@dataclass(frozen=True)
class ApplicationCreation:
    application: Application
    created: bool


@dataclass(frozen=True)
class ApplicationDetail:
    application: Application
    job: StoredJobRecord
    events: tuple[ApplicationEvent, ...]


@dataclass(frozen=True)
class ApplicationSummary:
    application: Application
    job: StoredJobRecord


class ApplicationService:
    _TRANSITIONS: dict[str, frozenset[str]] = {
        "submitted": frozenset(
            {"acknowledged", "interviewing", "rejected", "withdrawn"}
        ),
        "acknowledged": frozenset({"interviewing", "rejected", "withdrawn"}),
        "interviewing": frozenset({"offer", "rejected", "withdrawn"}),
        "offer": frozenset(),
        "rejected": frozenset(),
        "withdrawn": frozenset(),
    }

    def __init__(
        self,
        application_store: SQLiteApplicationStore,
        job_repository: JobPostingRepository,
        resume_store: ResumeStore,
    ) -> None:
        self._application_store = application_store
        self._job_repository = job_repository
        self._resume_store = resume_store

    def create_application(
        self,
        *,
        user_id: str,
        job_posting_id: str,
        resume_version_id: str,
        submitted_at: datetime | None = None,
        note: str | None = None,
    ) -> ApplicationCreation:
        if not all(
            (user_id.strip(), job_posting_id.strip(), resume_version_id.strip())
        ):
            raise ValueError("Application owner, job, and resume version are required")
        job = self._job_repository.get_job(
            user_id=user_id, job_posting_id=job_posting_id
        )
        if job is None:
            raise ApplicationInputNotFoundError("job_posting")
        if self._resume_store.get_version(
            user_id=user_id, resume_version_id=resume_version_id
        ) is None:
            raise ApplicationInputNotFoundError("resume_version")
        existing = self._application_store.find_for_job(
            user_id=user_id,
            job_posting_id=job_posting_id,
        )
        if existing is not None:
            return ApplicationCreation(application=existing, created=False)
        application = self._application_store.create(
            user_id=user_id,
            job_posting_id=job_posting_id,
            jd_snapshot_id=job.snapshot.id,
            resume_version_id=resume_version_id,
            submitted_at=submitted_at or datetime.now(timezone.utc),
            note=note,
        )
        return ApplicationCreation(application=application, created=True)

    def update_application(
        self,
        *,
        user_id: str,
        application_id: str,
        status: ApplicationStatus,
        note: str | None = None,
    ) -> Application:
        application = self._application_store.get(
            user_id=user_id,
            application_id=application_id,
        )
        if application is None:
            raise ApplicationInputNotFoundError("application")
        if status == application.status:
            if note is None or not note.strip():
                raise InvalidApplicationTransitionError(
                    "An unchanged status requires a note"
                )
        elif status not in self._TRANSITIONS[application.status]:
            raise InvalidApplicationTransitionError(
                f"Cannot move application from {application.status} to {status}"
            )
        updated = self._application_store.update(
            user_id=user_id,
            application_id=application_id,
            expected_status=application.status,
            new_status=status,
            submitted_at=application.submitted_at,
            note=note,
        )
        if updated is None:
            raise ConcurrentApplicationUpdateError(
                "Application changed before this update could be saved"
            )
        return updated

    def list_applications(
        self,
        *,
        user_id: str,
        statuses: tuple[ApplicationStatus, ...] = (),
        limit: int = 20,
    ) -> tuple[ApplicationSummary, ...]:
        applications = self._application_store.list(
            user_id=user_id,
            statuses=statuses,
            limit=limit,
        )
        summaries = []
        for application in applications:
            job = self._job_repository.get_job(
                user_id=user_id,
                job_posting_id=application.job_posting_id,
            )
            if job is None:
                raise ApplicationInputNotFoundError("job_posting")
            summaries.append(ApplicationSummary(application=application, job=job))
        return tuple(summaries)

    def get_application(
        self, *, user_id: str, application_id: str
    ) -> ApplicationDetail:
        application = self._application_store.get(
            user_id=user_id,
            application_id=application_id,
        )
        if application is None:
            raise ApplicationInputNotFoundError("application")
        job = self._job_repository.get_job(
            user_id=user_id,
            job_posting_id=application.job_posting_id,
        )
        if job is None:
            raise ApplicationInputNotFoundError("job_posting")
        events = self._application_store.list_events(
            user_id=user_id,
            application_id=application_id,
        )
        return ApplicationDetail(application=application, job=job, events=events)
