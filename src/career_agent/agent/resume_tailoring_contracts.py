from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from career_agent.agent.resume_job_match_contracts import (
    ConfirmedResumeFact,
    ResumeJobMatchResult,
)

if TYPE_CHECKING:
    from career_agent.storage.resumes import StoredResumeDocument


class ResumeTailoringContract(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )


class TailoringSupportEvidence(ResumeTailoringContract):
    source_locator: str = Field(min_length=1, max_length=300)
    source_quote: str = Field(min_length=1, max_length=500)


class ResumeTailoringChange(ResumeTailoringContract):
    target_locator: str = Field(min_length=1, max_length=300)
    original_quote: str | None = Field(default=None, min_length=1, max_length=500)
    proposed_text: str = Field(min_length=1, max_length=2000)
    rationale: str = Field(min_length=1, max_length=1500)
    addresses_requirements: tuple[str, ...] = Field(default=(), max_length=5)
    support_evidence: tuple[TailoringSupportEvidence, ...] = Field(
        min_length=1,
        max_length=5,
    )

    @model_validator(mode="after")
    def reject_noop_change(self) -> ResumeTailoringChange:
        if self.original_quote == self.proposed_text:
            raise ValueError("proposed text must differ from the original quote")
        return self


class ResumeTailoringResult(ResumeTailoringContract):
    strategy_summary: str = Field(min_length=1, max_length=3000)
    changes: tuple[ResumeTailoringChange, ...] = Field(default=(), max_length=30)
    preserved_strengths: tuple[str, ...] = Field(default=(), max_length=10)
    unresolved_gaps: tuple[str, ...] = Field(default=(), max_length=10)
    clarification_questions: tuple[str, ...] = Field(default=(), max_length=10)
    warnings: tuple[str, ...] = Field(default=(), max_length=10)


class ResumeTailoringWorker(Protocol):
    def tailor(
        self,
        *,
        document: StoredResumeDocument,
        jd_text: str,
        match_result: ResumeJobMatchResult,
        confirmed_facts: tuple[ConfirmedResumeFact, ...] = (),
        tailoring_goal: str | None = None,
    ) -> ResumeTailoringResult: ...
