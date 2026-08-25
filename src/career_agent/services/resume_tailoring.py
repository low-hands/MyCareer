from __future__ import annotations

from career_agent.agent.resume_job_match_contracts import ConfirmedResumeFact
from career_agent.agent.resume_tailoring_contracts import ResumeTailoringWorker
from career_agent.services.resume_job_match import ResumeJobMatchInputNotFoundError
from career_agent.storage.career_history import CareerHistoryStore
from career_agent.storage.jobs import JobPostingRepository
from career_agent.storage.resumes import ResumeStore
from career_agent.storage.resume_job_matches import SQLiteResumeJobMatchStore
from career_agent.storage.resume_tailoring import (
    SQLiteResumeTailoringDraftStore,
    StoredResumeTailoringDraft,
)


class ResumeTailoringDraftNotFoundError(ValueError):
    """Raised when a draft is missing, expired, or belongs to another user."""


class ResumeTailoringService:
    def __init__(
        self,
        resume_store: ResumeStore,
        job_repository: JobPostingRepository,
        career_history_store: CareerHistoryStore,
        match_store: SQLiteResumeJobMatchStore,
        draft_store: SQLiteResumeTailoringDraftStore,
        worker: ResumeTailoringWorker,
        *,
        worker_version: str = "resume-tailoring-v1",
    ) -> None:
        self._resume_store = resume_store
        self._job_repository = job_repository
        self._career_history_store = career_history_store
        self._match_store = match_store
        self._draft_store = draft_store
        self._worker = worker
        self._worker_version = worker_version

    def create_draft(
        self,
        *,
        user_id: str,
        match_id: str,
        tailoring_goal: str | None = None,
    ) -> StoredResumeTailoringDraft:
        if not user_id.strip() or not match_id.strip():
            raise ValueError("user_id and match_id are required")
        stored_match = self._match_store.get(user_id=user_id, match_id=match_id)
        if stored_match is None:
            raise ResumeJobMatchInputNotFoundError("match")
        document = self._resume_store.read_version_document(
            user_id=user_id,
            resume_version_id=stored_match.resume_version_id,
        )
        if document is None:
            raise ResumeJobMatchInputNotFoundError("resume_version")
        job = self._job_repository.get_job(
            user_id=user_id,
            job_posting_id=stored_match.job_posting_id,
        )
        if job is None or job.snapshot.id != stored_match.jd_snapshot_id:
            raise ResumeJobMatchInputNotFoundError("job_posting")
        confirmed_facts = tuple(
            ConfirmedResumeFact(
                claim=evidence.claim,
                source_locator=evidence.source_locator,
                source_quote=evidence.source_quote,
            )
            for evidence in self._career_history_store.list_evidence(
                user_id=user_id,
                verification_status="confirmed",
                source_resume_version_id=stored_match.resume_version_id,
            )
            if evidence.source_locator is not None and evidence.source_quote is not None
        )
        result = self._worker.tailor(
            document=document,
            jd_text=job.snapshot.content,
            match_result=stored_match.result,
            confirmed_facts=confirmed_facts,
            tailoring_goal=tailoring_goal,
        )
        return self._draft_store.create(
            user_id=user_id,
            match_id=match_id,
            tailoring_goal=tailoring_goal,
            worker_version=self._worker_version,
            result=result,
        )

    def get_draft(
        self,
        *,
        user_id: str,
        draft_id: str,
    ) -> StoredResumeTailoringDraft:
        draft = self._draft_store.get(user_id=user_id, draft_id=draft_id)
        if draft is None:
            raise ResumeTailoringDraftNotFoundError(
                "Resume tailoring draft not found, expired, or belongs to another user"
            )
        return draft
