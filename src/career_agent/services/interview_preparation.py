from __future__ import annotations

import hashlib
import json

from career_agent.agent.interview_preparation_contracts import (
    InterviewPreparationContext,
    InterviewPreparationWorker,
)
from career_agent.services.applications import ApplicationService
from career_agent.services.interviews import InterviewNotFoundError, InterviewService
from career_agent.services.interview_context import (
    InterviewContextInputNotFoundError,
    InterviewPreparationContextFactory,
)
from career_agent.storage.career_history import CareerHistoryStore
from career_agent.storage.interview_preparations import (
    SQLiteInterviewPreparationStore,
    StoredInterviewPreparation,
)
from career_agent.storage.resumes import ResumeStore


class InterviewPreparationInputNotFoundError(ValueError):
    pass


class InterviewPreparationNotAvailableError(ValueError):
    pass


class InterviewPreparationService:
    def __init__(
        self,
        interview_service: InterviewService,
        application_service: ApplicationService,
        resume_store: ResumeStore,
        career_history_store: CareerHistoryStore,
        worker: InterviewPreparationWorker,
        store: SQLiteInterviewPreparationStore,
        *,
        worker_version: str = "interview-preparation-v1",
        context_factory: InterviewPreparationContextFactory | None = None,
    ) -> None:
        self._interview_service = interview_service
        self._application_service = application_service
        self._worker = worker
        self._store = store
        self._worker_version = worker_version
        self._context_factory = context_factory or InterviewPreparationContextFactory(
            interviews=interview_service,
            applications=application_service,
            resumes=resume_store,
            career_history=career_history_store,
        )

    def prepare(
        self, *, user_id: str, interview_round_id: str
    ) -> StoredInterviewPreparation:
        try:
            interview = self._interview_service.get_interview(
                user_id=user_id, interview_round_id=interview_round_id
            ).interview
        except InterviewNotFoundError as error:
            raise InterviewPreparationInputNotFoundError("interview_round") from error
        if interview.status in {"cancelled", "completed"}:
            raise InterviewPreparationNotAvailableError(
                f"cannot prepare a {interview.status} interview"
            )
        application = self._application_service.get_application(
            user_id=user_id, application_id=interview.application_id
        )
        try:
            sources = self._context_factory.build(
                user_id=user_id,
                application_id=interview.application_id,
                interview_round_id=interview.id,
            )
        except InterviewContextInputNotFoundError as error:
            raise InterviewPreparationInputNotFoundError(str(error)) from error
        fingerprint = self._fingerprint(
            context=sources.context,
            jd_snapshot_id=application.application.jd_snapshot_id,
            resume_version_id=application.application.resume_version_id,
        )
        cached = self._store.find(
            user_id=user_id,
            interview_round_id=interview.id,
            input_fingerprint=fingerprint,
            worker_version=self._worker_version,
        )
        if cached is not None:
            return cached
        result = self._worker.prepare(
            document=sources.document,
            context=sources.context,
        )
        return self._store.save(
            user_id=user_id,
            interview_round_id=interview.id,
            application_id=application.application.id,
            job_posting_id=application.application.job_posting_id,
            jd_snapshot_id=application.application.jd_snapshot_id,
            resume_version_id=application.application.resume_version_id,
            input_fingerprint=fingerprint,
            worker_version=self._worker_version,
            result=result,
        )

    def get(
        self, *, user_id: str, preparation_id: str
    ) -> StoredInterviewPreparation:
        stored = self._store.get(
            user_id=user_id, preparation_id=preparation_id
        )
        if stored is None:
            raise InterviewPreparationInputNotFoundError("interview_preparation")
        return stored

    @staticmethod
    def _fingerprint(
        *,
        context: InterviewPreparationContext,
        jd_snapshot_id: str,
        resume_version_id: str,
    ) -> str:
        canonical = json.dumps(
            {
                "context": context.model_dump(mode="json"),
                "jd_snapshot_id": jd_snapshot_id,
                "resume_version_id": resume_version_id,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode()).hexdigest()
