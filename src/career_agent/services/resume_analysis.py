from __future__ import annotations

from career_agent.agent.resume_analysis_contracts import (
    ResumeAnalysisResult,
    ResumeAnalysisWorker,
)
from career_agent.storage.resumes import ResumeStore


class ResumeVersionNotFoundError(ValueError):
    """Raised when a resume version is missing or belongs to another user."""


class ResumeAnalysisService:
    """Loads one user-owned resume version and delegates its analysis."""

    def __init__(self, store: ResumeStore, worker: ResumeAnalysisWorker) -> None:
        self._store = store
        self._worker = worker

    def analyze_version(
        self,
        *,
        user_id: str,
        resume_version_id: str,
    ) -> ResumeAnalysisResult:
        if not user_id.strip() or not resume_version_id.strip():
            raise ValueError("user_id and resume_version_id are required")
        document = self._store.read_version_document(
            user_id=user_id,
            resume_version_id=resume_version_id,
        )
        if document is None:
            raise ResumeVersionNotFoundError(
                "Resume version not found or does not belong to the current user"
            )
        return self._worker.analyze(document)
