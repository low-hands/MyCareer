from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

if TYPE_CHECKING:
    from career_agent.storage.resumes import StoredResumeDocument


class ResumeAnalysisContract(BaseModel):
    """Base contract for untrusted, model-produced resume analysis."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )


class ExtractedCareerEvidence(ResumeAnalysisContract):
    """A resume claim together with the text needed to verify it."""

    claim: str = Field(min_length=1)
    source_locator: str = Field(min_length=1)
    source_quote: str = Field(min_length=1)


class ExtractedCareerRecord(ResumeAnalysisContract):
    """An unconfirmed career record extracted from a resume."""

    record_type: Literal[
        "education",
        "work",
        "internship",
        "project",
        "certification",
    ]
    organization: str | None = Field(default=None, min_length=1)
    title: str = Field(min_length=1)
    start_year: int | None = Field(default=None, ge=1900, le=2200)
    start_month: int | None = Field(default=None, ge=1, le=12)
    end_year: int | None = Field(default=None, ge=1900, le=2200)
    end_month: int | None = Field(default=None, ge=1, le=12)
    is_current: bool = False
    source_locator: str = Field(min_length=1)
    source_quote: str = Field(min_length=1)
    evidence: tuple[ExtractedCareerEvidence, ...] = ()

    @model_validator(mode="after")
    def validate_period(self) -> ExtractedCareerRecord:
        if self.start_month is not None and self.start_year is None:
            raise ValueError("start_month requires start_year")
        if self.end_month is not None and self.end_year is None:
            raise ValueError("end_month requires end_year")
        if self.is_current and (self.end_year is not None or self.end_month is not None):
            raise ValueError("current records cannot have an end date")
        if self.start_year is not None and self.end_year is not None:
            start = (self.start_year, self.start_month or 1)
            end = (self.end_year, self.end_month or 12)
            if end < start:
                raise ValueError("end date cannot be before start date")
        return self


class ResumeAnalysisResult(ResumeAnalysisContract):
    """Structured output returned by a resume-analysis worker."""

    records: tuple[ExtractedCareerRecord, ...] = ()
    clarification_questions: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


class ResumeAnalysisWorker(Protocol):
    """Provider-independent boundary for reading and analysing a resume file."""

    def analyze(self, document: StoredResumeDocument) -> ResumeAnalysisResult: ...
