from __future__ import annotations

import hashlib
import json

from career_agent.agent.resume_job_match_contracts import (
    ConfirmedResumeFact,
    ResumeJobMatchWorker,
)
from career_agent.storage.career_history import CareerHistoryStore
from career_agent.storage.jobs import JobPostingRepository
from career_agent.storage.resumes import ResumeStore
from career_agent.storage.resume_job_matches import (
    SQLiteResumeJobMatchStore,
    StoredResumeJobMatch,
)


class ResumeJobMatchInputNotFoundError(ValueError):
    """Raised when either user-owned input cannot be resolved."""

    def __init__(self, input_kind: str) -> None:
        self.input_kind = input_kind
        super().__init__(f"{input_kind} not found or does not belong to the current user")


class ResumeJobMatchService:
    """Keeps complete resume/JD documents behind a narrow matching boundary."""

    def __init__(
        self,
        resume_store: ResumeStore,
        job_repository: JobPostingRepository,
        career_history_store: CareerHistoryStore,
        worker: ResumeJobMatchWorker,
        match_store: SQLiteResumeJobMatchStore,
        *,
        matcher_version: str = "resume-job-match-v1",
    ) -> None:
        self._resume_store = resume_store
        self._job_repository = job_repository
        self._career_history_store = career_history_store
        self._worker = worker
        self._match_store = match_store
        self._matcher_version = matcher_version

    def match(
        self,
        *,
        user_id: str,
        resume_version_id: str,
        job_posting_id: str,
    ) -> StoredResumeJobMatch:
        if not user_id.strip() or not resume_version_id.strip() or not job_posting_id.strip():
            raise ValueError("user_id, resume_version_id, and job_posting_id are required")
        document = self._resume_store.read_version_document(
            user_id=user_id,
            resume_version_id=resume_version_id,
        )
        if document is None:
            raise ResumeJobMatchInputNotFoundError("resume_version")
        job = self._job_repository.get_job(
            user_id=user_id,
            job_posting_id=job_posting_id,
        )
        if job is None:
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
                source_resume_version_id=resume_version_id,
            )
            if evidence.source_locator is not None
            and evidence.source_quote is not None
        )
        evidence_fingerprint = self._evidence_fingerprint(confirmed_facts)
        cached = self._match_store.find(
            user_id=user_id,
            resume_version_id=resume_version_id,
            jd_snapshot_id=job.snapshot.id,
            matcher_version=self._matcher_version,
            evidence_fingerprint=evidence_fingerprint,
        )
        if cached is not None:
            return cached
        result = self._worker.match(
            document=document,
            jd_text=job.snapshot.content,
            confirmed_facts=confirmed_facts,
        )
        return self._match_store.save(
            user_id=user_id,
            resume_version_id=resume_version_id,
            job_posting_id=job_posting_id,
            jd_snapshot_id=job.snapshot.id,
            matcher_version=self._matcher_version,
            evidence_fingerprint=evidence_fingerprint,
            result=result,
        )

    def get_match(self, *, user_id: str, match_id: str) -> StoredResumeJobMatch:
        if not user_id.strip() or not match_id.strip():
            raise ValueError("user_id and match_id are required")
        stored = self._match_store.get(user_id=user_id, match_id=match_id)
        if stored is None:
            raise ResumeJobMatchInputNotFoundError("match")
        return stored

    @staticmethod
    def _evidence_fingerprint(facts: tuple[ConfirmedResumeFact, ...]) -> str:
        serialized = json.dumps(
            [fact.model_dump(mode="json") for fact in facts],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
