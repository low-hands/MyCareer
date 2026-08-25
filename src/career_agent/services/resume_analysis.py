from __future__ import annotations

from datetime import datetime
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict

from career_agent.agent.resume_analysis_contracts import (
    ResumeAnalysisResult,
    ResumeAnalysisWorker,
)
from career_agent.storage.career_history import (
    CareerHistoryImportResult,
    CareerHistoryStore,
)
from career_agent.storage.resumes import ResumeStore


class ResumeVersionNotFoundError(ValueError):
    """Raised when a resume version is missing or belongs to another user."""


class ResumeAnalysisNotFoundError(ValueError):
    """Raised when an analysis is expired, missing, or belongs to another user."""


class ResumeAnalysisDraft(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    user_id: str
    resume_version_id: str
    status: Literal["pending", "confirmed", "rejected"] = "pending"
    result: ResumeAnalysisResult
    created_at: datetime
    updated_at: datetime
    expires_at: datetime


class ResumeAnalysisDraftStore(Protocol):
    def create(
        self,
        *,
        user_id: str,
        resume_version_id: str,
        result: ResumeAnalysisResult,
    ) -> ResumeAnalysisDraft: ...

    def get(self, *, user_id: str, analysis_id: str) -> ResumeAnalysisDraft | None: ...

    def mark_confirmed(
        self, *, user_id: str, analysis_id: str
    ) -> ResumeAnalysisDraft: ...


class ResumeAnalysisService:
    """Loads one user-owned resume version and delegates its analysis."""

    def __init__(
        self,
        store: ResumeStore,
        worker: ResumeAnalysisWorker,
        draft_store: ResumeAnalysisDraftStore,
        career_history_store: CareerHistoryStore,
    ) -> None:
        self._store = store
        self._worker = worker
        self._draft_store = draft_store
        self._career_history_store = career_history_store

    def analyze_version(
        self,
        *,
        user_id: str,
        resume_version_id: str,
    ) -> ResumeAnalysisDraft:
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
        result = self._worker.analyze(document)
        return self._draft_store.create(
            user_id=user_id,
            resume_version_id=resume_version_id,
            result=result,
        )

    def get_analysis(
        self, *, user_id: str, analysis_id: str
    ) -> ResumeAnalysisDraft:
        draft = self._draft_store.get(user_id=user_id, analysis_id=analysis_id)
        if draft is None:
            raise ResumeAnalysisNotFoundError(
                "Resume analysis not found, expired, or does not belong to the current user"
            )
        return draft

    def confirm_analysis(
        self, *, user_id: str, analysis_id: str
    ) -> CareerHistoryImportResult:
        draft = self.get_analysis(user_id=user_id, analysis_id=analysis_id)
        if draft.status == "rejected":
            raise ValueError("Rejected resume analysis cannot be confirmed")
        imported = self._career_history_store.import_confirmed_resume_analysis(
            user_id=user_id,
            analysis_id=draft.id,
            resume_version_id=draft.resume_version_id,
            result=draft.result,
        )
        self._draft_store.mark_confirmed(user_id=user_id, analysis_id=draft.id)
        return imported
