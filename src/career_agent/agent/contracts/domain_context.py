from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field

from career_agent.agent.contracts.candidates import (
    ActionCandidateContextItem,
    ActiveSavedJobContextItem,
    ApplicationCandidateContextItem,
    CalendarAccountCandidateContextItem,
    EmailEventCandidateContextItem,
    InterviewCandidateContextItem,
    ResumeCandidateContextItem,
    ResumeVersionCandidateContextItem,
    SavedJobCandidateContextItem,
    TargetRoleCandidateContextItem,
)
from career_agent.domain.applications import ApplicationStatus
from career_agent.domain.job_discovery import ContractModel


class ApplicationTaskContext(ContractModel):
    """Conversation-local references owned by application tracking."""

    active_id: str | None = None
    active_status: ApplicationStatus | None = None
    candidates: tuple[ApplicationCandidateContextItem, ...] = ()
    email_event_candidates: tuple[EmailEventCandidateContextItem, ...] = ()
    email_sync_phase: str | None = None


class InterviewTaskContext(ContractModel):
    """Conversation-local references owned by interview and calendar work."""

    active_round_id: str | None = None
    candidates: tuple[InterviewCandidateContextItem, ...] = ()
    active_preparation_id: str | None = None
    active_calendar_proposal_id: str | None = None
    calendar_proposal_expires_at: datetime | None = None
    calendar_account_candidates: tuple[
        CalendarAccountCandidateContextItem, ...
    ] = ()


class ActionCenterTaskContext(ContractModel):
    active_id: str | None = None
    candidates: tuple[ActionCandidateContextItem, ...] = ()


class JobTaskContext(ContractModel):
    active_posting_id: str | None = None
    active_jd_snapshot_id: str | None = None
    active_saved_job: ActiveSavedJobContextItem | None = None
    saved_job_candidates: tuple[SavedJobCandidateContextItem, ...] = ()
    target_role_candidates: tuple[TargetRoleCandidateContextItem, ...] = ()
    active_analysis_id: str | None = None
    active_analysis_jd_snapshot_id: str | None = None
    analysis_status: Literal["ready"] | None = None
    active_research_run_id: str | None = None
    active_research_report_id: str | None = None
    research_status: Literal["current", "outdated", "failed"] | None = None


class ResumeTaskContext(ContractModel):
    active_job_match_id: str | None = None
    job_match_status: Literal["ready"] | None = None
    active_tailoring_draft_id: str | None = None
    tailoring_status: Literal[
        "pending", "in_review", "reviewed", "finalized", "superseded"
    ] | None = None
    active_version_id: str | None = None
    active_artifact_id: str | None = None
    candidates: tuple[ResumeCandidateContextItem, ...] = ()
    version_candidates: tuple[ResumeVersionCandidateContextItem, ...] = ()


class DomainTaskContext(ContractModel):
    """Typed domain state retained across capability-profile switches.

    Domains are siblings rather than a union: a cross-domain plan needs the
    selected job, resume, and application references at the same time. The
    mutually exclusive concern is which domain is active, not whether the
    other domains may retain bounded context.
    """

    application: ApplicationTaskContext = Field(
        default_factory=ApplicationTaskContext
    )
    interview: InterviewTaskContext = Field(default_factory=InterviewTaskContext)
    action_center: ActionCenterTaskContext = Field(
        default_factory=ActionCenterTaskContext
    )
    job: JobTaskContext = Field(default_factory=JobTaskContext)
    resume: ResumeTaskContext = Field(default_factory=ResumeTaskContext)
