from __future__ import annotations

from career_agent.agent.job_analysis_contracts import JobAnalysisWorker
from career_agent.domain.job_discovery import content_fingerprint
from career_agent.storage.jobs import (
    JDAnalysisPayload,
    JobPostingRepository,
    StoredJDAnalysis,
)

JOB_ANALYZER_VERSION = "job-analysis-v1"


class JobAnalysisInputNotFoundError(ValueError):
    """Raised when the saved job or pinned JD snapshot cannot be resolved."""

    def __init__(self, input_kind: str) -> None:
        self.input_kind = input_kind
        super().__init__(f"{input_kind} not found or does not belong to the current user")


class JobAnalysisService:
    """Analyzes one immutable JD snapshot and caches the result per snapshot."""

    def __init__(
        self,
        job_repository: JobPostingRepository,
        worker: JobAnalysisWorker,
        *,
        analyzer_version: str = JOB_ANALYZER_VERSION,
    ) -> None:
        self._job_repository = job_repository
        self._worker = worker
        self._analyzer_version = analyzer_version

    @property
    def analyzer_version(self) -> str:
        return self._analyzer_version

    def analyze(
        self,
        *,
        user_id: str,
        job_posting_id: str,
        jd_snapshot_id: str | None = None,
    ) -> StoredJDAnalysis:
        if not user_id.strip() or not job_posting_id.strip():
            raise ValueError("user_id and job_posting_id are required")
        job = self._job_repository.get_job(user_id=user_id, job_posting_id=job_posting_id)
        if job is None:
            raise JobAnalysisInputNotFoundError("job_posting")
        snapshot = job.snapshot
        if jd_snapshot_id is not None and jd_snapshot_id != snapshot.id:
            pinned = self._job_repository.get_snapshot(user_id=user_id, jd_snapshot_id=jd_snapshot_id)
            if pinned is None or pinned.job_posting_id != job.posting.id:
                raise JobAnalysisInputNotFoundError("jd_snapshot")
            snapshot = pinned
        fingerprint = content_fingerprint(
            job.posting.title, job.posting.company_name, snapshot.content
        )
        cached = self._job_repository.find_analysis(
            user_id=user_id,
            jd_snapshot_id=snapshot.id,
            analyzer_version=self._analyzer_version,
            content_fingerprint=fingerprint,
        )
        if cached is not None:
            return cached
        result = self._worker.analyze(jd_text=snapshot.content)
        return self._job_repository.save_analysis(
            user_id=user_id,
            jd_snapshot_id=snapshot.id,
            analyzer_version=self._analyzer_version,
            analysis=JDAnalysisPayload.from_result(result),
            content_fingerprint=fingerprint,
        )
