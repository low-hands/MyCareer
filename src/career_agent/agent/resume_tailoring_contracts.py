from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Protocol

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


class AcceptedTailoringChange(ResumeTailoringContract):
    change_index: int = Field(ge=1)
    change: ResumeTailoringChange


class FinalizedResumeDocument(ResumeTailoringContract):
    markdown: str = Field(min_length=1, max_length=500_000)
    applied_change_indices: tuple[int, ...] = Field(min_length=1, max_length=30)
    warnings: tuple[str, ...] = Field(default=(), max_length=10)


class ResumeReviewIssue(ResumeTailoringContract):
    category: Literal[
        "unsupported_fact",
        "meaning_changed",
        "jd_misalignment",
        "important_detail_lost",
        "keyword_stuffing",
        "unclear_expression",
        "change_set_mismatch",
    ]
    severity: Literal["blocking", "warning"]
    change_index: int | None = Field(default=None, ge=1)
    source_quote: str | None = Field(default=None, min_length=1, max_length=500)
    explanation: str = Field(min_length=1, max_length=1500)
    revision_instruction: str | None = Field(
        default=None, min_length=1, max_length=1000
    )


class ResumeReviewResult(ResumeTailoringContract):
    verdict: Literal["pass", "revise", "block"]
    summary: str = Field(min_length=1, max_length=2000)
    issues: tuple[ResumeReviewIssue, ...] = Field(default=(), max_length=30)

    @model_validator(mode="after")
    def enforce_verdict_consistency(self) -> ResumeReviewResult:
        has_blocking = any(issue.severity == "blocking" for issue in self.issues)
        if self.verdict == "pass" and has_blocking:
            raise ValueError("pass verdict cannot contain blocking issues")
        if self.verdict in {"revise", "block"} and not has_blocking:
            raise ValueError("revise and block verdicts require a blocking issue")
        return self


class ResumeReviewAttempt(ResumeTailoringContract):
    attempt_number: int = Field(ge=1)
    draft_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    result: ResumeReviewResult


class ResumeReviewTrace(ResumeTailoringContract):
    status: Literal["passed", "blocked"]
    attempts: tuple[ResumeReviewAttempt, ...] = Field(min_length=1, max_length=3)
    stop_reason: Literal[
        "passed",
        "reviewer_blocked",
        "revision_limit",
        "no_progress",
        "cycle_detected",
    ]


class ResumeTailoringWorker(Protocol):
    def tailor(
        self,
        *,
        document: StoredResumeDocument,
        jd_text: str,
        match_result: ResumeJobMatchResult,
        confirmed_facts: tuple[ConfirmedResumeFact, ...] = (),
        tailoring_goal: str | None = None,
        user_feedback: str | None = None,
        review_feedback: tuple[str, ...] = (),
        previous_draft: ResumeTailoringResult | None = None,
    ) -> ResumeTailoringResult: ...


class ResumeFinalizationWorker(Protocol):
    def finalize(
        self,
        *,
        document: StoredResumeDocument,
        accepted_changes: tuple[AcceptedTailoringChange, ...],
        confirmed_facts: tuple[ConfirmedResumeFact, ...] = (),
    ) -> FinalizedResumeDocument: ...


class ResumeTailoringReviewer(Protocol):
    def review_draft(
        self,
        *,
        document: StoredResumeDocument,
        jd_text: str,
        match_result: ResumeJobMatchResult,
        draft: ResumeTailoringResult,
        confirmed_facts: tuple[ConfirmedResumeFact, ...] = (),
    ) -> ResumeReviewResult: ...

    def review_final(
        self,
        *,
        document: StoredResumeDocument,
        accepted_changes: tuple[AcceptedTailoringChange, ...],
        finalized: FinalizedResumeDocument,
        confirmed_facts: tuple[ConfirmedResumeFact, ...] = (),
    ) -> ResumeReviewResult: ...
