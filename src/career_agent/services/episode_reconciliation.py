from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from career_agent.domain.applications import Application
from career_agent.domain.episodes import CareerEpisodeDraft, EpisodeResourceRef
from career_agent.domain.interviews import InterviewRound
from career_agent.domain.job_research import JobResearchReport
from career_agent.domain.mock_interviews import (
    MockInterviewReport,
    MockInterviewSession,
    MockInterviewStatus,
)
from career_agent.storage.episodes import SQLiteCareerEpisodeStore


class ApplicationEpisodeSource(Protocol):
    def list(
        self, *, user_id: str, statuses: tuple = (), limit: int = 20
    ) -> tuple[Application, ...]: ...


class InterviewEpisodeSource(Protocol):
    def list(
        self,
        *,
        user_id: str,
        application_id: str | None = None,
        statuses: tuple = (),
        limit: int = 50,
    ) -> tuple[InterviewRound, ...]: ...


class MockInterviewEpisodeSource(Protocol):
    def list_sessions(
        self,
        *,
        user_id: str,
        application_id: str | None = None,
        statuses: tuple[MockInterviewStatus, ...] = (),
        limit: int = 50,
    ) -> tuple[MockInterviewSession, ...]: ...

    def list_reports(
        self,
        *,
        user_id: str,
        session_ids: tuple[str, ...],
    ) -> tuple[MockInterviewReport, ...]: ...


class JobResearchEpisodeSource(Protocol):
    def list_reports(
        self, *, user_id: str, limit: int = 50
    ) -> tuple[JobResearchReport, ...]: ...


@dataclass(frozen=True)
class EpisodeReconciliationResult:
    scanned: int
    inserted: int


class EpisodeReconciliationLimitError(RuntimeError):
    """A source exceeded the bounded recovery scan and was not reconciled."""


class EpisodeReconciler:
    """Rebuild L1 rows from durable domain state after a split-store failure."""

    _MAX_SOURCE_ROWS = 10_000
    _SCAN_LIMIT = _MAX_SOURCE_ROWS + 1

    def __init__(
        self,
        *,
        episodes: SQLiteCareerEpisodeStore,
        applications: ApplicationEpisodeSource,
        interviews: InterviewEpisodeSource,
        mock_interviews: MockInterviewEpisodeSource,
        job_research: JobResearchEpisodeSource,
    ) -> None:
        self._episodes = episodes
        self._applications = applications
        self._interviews = interviews
        self._mock_interviews = mock_interviews
        self._job_research = job_research

    def reconcile_user(self, *, user_id: str) -> EpisodeReconciliationResult:
        before = self._episodes.list_source_keys(user_id=user_id)
        drafts = (
            *self._application_drafts(user_id),
            *self._interview_drafts(user_id),
            *self._mock_interview_drafts(user_id),
            *self._job_research_drafts(user_id),
        )
        missing = tuple(
            draft
            for draft in drafts
            if (draft.kind, draft.source_run_id) not in before
        )
        # Reconciliation projects authoritative domain rows, not the current
        # conversation's career-memory window. It therefore supplies no career
        # scope bindings; only turn-derived episodes receive those bindings.
        inserted = len(self._episodes.upsert_many(missing)) if missing else 0
        return EpisodeReconciliationResult(
            scanned=len(drafts),
            inserted=inserted,
        )

    def _application_drafts(
        self, user_id: str
    ) -> tuple[CareerEpisodeDraft, ...]:
        rows = self._applications.list(user_id=user_id, limit=self._SCAN_LIMIT)
        self._require_complete("applications", len(rows))
        return tuple(
            CareerEpisodeDraft(
                user_id=user_id,
                kind="application",
                source_run_id=row.id,
                occurred_at=row.updated_at,
                title="投递记录",
                summary=f"投递状态：{row.status}。",
            )
            for row in rows
        )

    def _interview_drafts(
        self, user_id: str
    ) -> tuple[CareerEpisodeDraft, ...]:
        rows = self._interviews.list(user_id=user_id, limit=self._SCAN_LIMIT)
        self._require_complete("interviews", len(rows))
        return tuple(
            CareerEpisodeDraft(
                user_id=user_id,
                kind="interview_round",
                source_run_id=row.id,
                occurred_at=row.updated_at,
                title=(
                    f"{row.employer_label} · 第 {row.sequence_number} 轮面试"
                    if row.employer_label
                    else f"第 {row.sequence_number} 轮面试"
                )[:80],
                summary=f"面试状态：{row.status}。",
            )
            for row in rows
        )

    def _mock_interview_drafts(
        self, user_id: str
    ) -> tuple[CareerEpisodeDraft, ...]:
        rows = self._mock_interviews.list_sessions(
            user_id=user_id,
            statuses=("completed", "cancelled"),
            limit=self._SCAN_LIMIT,
        )
        self._require_complete("mock_interviews", len(rows))
        reports_by_session = {
            report.session_id: report
            for report in self._mock_interviews.list_reports(
                user_id=user_id,
                session_ids=tuple(row.id for row in rows),
            )
        }
        drafts: list[CareerEpisodeDraft] = []
        for row in rows:
            report = reports_by_session.get(row.id)
            refs = (
                (
                    EpisodeResourceRef(
                        kind="mock_interview_report",
                        resource_id=report.id,
                        title="模拟面试报告",
                    ),
                )
                if report is not None
                else ()
            )
            drafts.append(
                CareerEpisodeDraft(
                    user_id=user_id,
                    kind="mock_interview",
                    source_run_id=row.id,
                    occurred_at=row.completed_at or row.updated_at,
                    title="模拟面试",
                    summary=(
                        report.summary[:400]
                        if report is not None
                        else (
                            "模拟面试已完成。"
                            if row.status == "completed"
                            else "模拟面试已取消。"
                        )
                    ),
                    resource_refs=refs,
                )
            )
        return tuple(drafts)

    def _job_research_drafts(
        self, user_id: str
    ) -> tuple[CareerEpisodeDraft, ...]:
        rows = self._job_research.list_reports(
            user_id=user_id,
            limit=self._SCAN_LIMIT,
        )
        self._require_complete("job_research", len(rows))
        return tuple(
            CareerEpisodeDraft(
                user_id=user_id,
                kind="job_research",
                source_run_id=row.run_id,
                occurred_at=row.created_at,
                title="岗位研究",
                summary=row.summary[:400],
                resource_refs=(
                    EpisodeResourceRef(
                        kind="job_research_report",
                        resource_id=row.id,
                        title="岗位研究报告",
                    ),
                ),
            )
            for row in rows
        )

    @classmethod
    def _require_complete(cls, source: str, row_count: int) -> None:
        if row_count > cls._MAX_SOURCE_ROWS:
            raise EpisodeReconciliationLimitError(
                f"{source} recovery exceeds {cls._MAX_SOURCE_ROWS} rows; "
                "refusing a silently incomplete L1 reconciliation"
            )
