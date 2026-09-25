from __future__ import annotations

from datetime import datetime
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from career_agent.domain.interview_preparation import InterviewPreparationResult
from career_agent.storage.resumes import StoredResumeDocument


class PreparationConfirmedFact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    claim: str
    source_locator: str
    source_quote: str


class InterviewLogisticsContext(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    employer_label: str | None = None
    scheduled_start: datetime | None = None
    scheduled_end: datetime | None = None
    timezone: str | None = None
    interview_format: str
    location: str | None = None
    meeting_url: str | None = None


class PriorInterviewQuestionContext(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    question: str
    answer_summary: str | None = None
    self_assessment: Literal["strong", "mixed", "weak", "uncertain"]
    notes: str | None = None


class PriorInterviewRetroContext(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    sequence_number: int = Field(ge=1)
    employer_label: str | None = None
    completed_at: datetime | None = None
    summary: str
    questions: tuple[PriorInterviewQuestionContext, ...] = ()
    strengths: tuple[str, ...] = ()
    difficulties: tuple[str, ...] = ()
    interviewer_signals: tuple[str, ...] = ()
    next_focus: tuple[str, ...] = ()
    action_items: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    self_assessment: Literal["strong", "mixed", "weak", "uncertain"]


class CompanyResearchFindingContext(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    topic: str
    statement: str
    evidence_type: Literal["fact", "inference", "unknown"]
    confidence: Literal["high", "medium", "low"]


class CompanyResearchContext(BaseModel):
    """Public business context from the user's own company research report.

    It says what the company does and where it stands, never how it
    interviews: interview style comes only from the mock-interview skill's
    company profiles.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    researched_at: datetime
    outdated: bool
    summary: str
    findings: tuple[CompanyResearchFindingContext, ...] = ()


class InterviewPreparationContext(BaseModel):
    """Shared structured context for prep briefs and mock-interview planning."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    company_name: str
    role_title: str
    jd_text: str
    logistics: InterviewLogisticsContext | None = None
    confirmed_facts: tuple[PreparationConfirmedFact, ...] = ()
    prior_retros: tuple[PriorInterviewRetroContext, ...] = ()
    company_research: CompanyResearchContext | None = None


class InterviewPreparationWorker(Protocol):
    def prepare(
        self,
        *,
        document: StoredResumeDocument,
        context: InterviewPreparationContext,
    ) -> InterviewPreparationResult: ...
