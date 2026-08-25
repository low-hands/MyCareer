from __future__ import annotations

from datetime import datetime
from typing import Protocol

from pydantic import BaseModel, ConfigDict

from career_agent.domain.interview_preparation import InterviewPreparationResult
from career_agent.storage.resumes import StoredResumeDocument


class PreparationConfirmedFact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    claim: str
    source_locator: str
    source_quote: str


class InterviewPreparationContext(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    employer_label: str | None = None
    scheduled_start: datetime | None = None
    scheduled_end: datetime | None = None
    timezone: str | None = None
    interview_format: str
    location: str | None = None
    meeting_url: str | None = None


class InterviewPreparationWorker(Protocol):
    def prepare(
        self,
        *,
        document: StoredResumeDocument,
        jd_text: str,
        interview: InterviewPreparationContext,
        confirmed_facts: tuple[PreparationConfirmedFact, ...] = (),
    ) -> InterviewPreparationResult: ...
