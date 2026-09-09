from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

if TYPE_CHECKING:
    from career_agent.storage.resumes import StoredResumeDocument


class ResumeJobMatchContract(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )


class ConfirmedResumeFact(ResumeJobMatchContract):
    """A confirmed extraction from the exact resume version being matched."""

    claim: str = Field(min_length=1, max_length=1000)
    source_locator: str = Field(min_length=1, max_length=300)
    source_quote: str = Field(min_length=1, max_length=500)


class IntentStateAnchor(ResumeJobMatchContract):
    """One current preference, carrying when it was last confirmed.

    Staleness is reported as the corroboration event itself rather than as a
    decayed score, so an anchor is never withheld for having aged and the
    reader is told what it is judging.
    """

    scope_key: str = Field(min_length=1, max_length=300)
    pref_scope: str = Field(min_length=1, max_length=120)
    value: str = Field(min_length=1, max_length=2000)
    valid_from: datetime
    last_confirmed_at: datetime


class IntentStateTransition(ResumeJobMatchContract):
    scope_key: str = Field(min_length=1, max_length=300)
    pref_scope: str = Field(min_length=1, max_length=120)
    old_value: str = Field(min_length=1, max_length=2000)
    new_value: str = Field(min_length=1, max_length=2000)
    old_valid_from: datetime
    new_valid_from: datetime
    last_confirmed_at: datetime

    @model_validator(mode="after")
    def chronology_is_forward(self) -> "IntentStateTransition":
        if self.new_valid_from <= self.old_valid_from:
            raise ValueError("state transition chronology must move forward")
        return self


class ResumeMatchEvidence(ResumeJobMatchContract):
    source_locator: str = Field(min_length=1, max_length=300)
    source_quote: str = Field(min_length=1, max_length=500)


class RequirementAssessment(ResumeJobMatchContract):
    requirement: str = Field(min_length=1, max_length=1000)
    jd_quote: str = Field(min_length=1, max_length=500)
    status: Literal["matched", "partial", "missing", "unclear"]
    rationale: str = Field(min_length=1, max_length=2000)
    resume_evidence: tuple[ResumeMatchEvidence, ...] = Field(default=(), max_length=3)

    @model_validator(mode="after")
    def require_grounding_for_positive_match(self) -> RequirementAssessment:
        if self.status in {"matched", "partial"} and not self.resume_evidence:
            raise ValueError("matched and partial requirements need resume evidence")
        if self.status in {"missing", "unclear"} and self.resume_evidence:
            raise ValueError("missing and unclear requirements cannot cite resume evidence")
        return self


class ResumeJobMatchResult(ResumeJobMatchContract):
    overall_fit: Literal["strong", "moderate", "weak", "insufficient_evidence"]
    summary: str = Field(min_length=1, max_length=3000)
    requirements: tuple[RequirementAssessment, ...] = Field(default=(), max_length=40)
    recommendations: tuple[str, ...] = Field(default=(), max_length=10)
    clarification_questions: tuple[str, ...] = Field(default=(), max_length=10)
    limitations: tuple[str, ...] = Field(default=(), max_length=10)


class ResumeJobMatchStateFinding(ResumeJobMatchContract):
    scope_key: str = Field(min_length=1, max_length=300)
    pref_scope: str = Field(min_length=1, max_length=120)
    old_value: str = Field(min_length=1, max_length=2000)
    new_value: str = Field(min_length=1, max_length=2000)
    status: Literal["current", "stale", "unknown"]
    material: bool
    rationale: str = Field(min_length=1, max_length=2000)


class ResumeJobMatchAuditProposal(ResumeJobMatchContract):
    findings: tuple[ResumeJobMatchStateFinding, ...] = Field(
        default=(),
        max_length=50,
    )
    repaired_result: ResumeJobMatchResult


class ResumeJobMatchWorker(Protocol):
    def match(
        self,
        *,
        document: StoredResumeDocument,
        jd_text: str,
        confirmed_facts: tuple[ConfirmedResumeFact, ...] = (),
        intent_states: tuple[IntentStateAnchor, ...] = (),
    ) -> ResumeJobMatchResult: ...
