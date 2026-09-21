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


class GapAdjacentEvidence(ResumeTailoringContract):
    source_locator: str = Field(min_length=1, max_length=300)
    source_quote: str = Field(min_length=1, max_length=500)
    relevance: str = Field(min_length=1, max_length=1000)


class GapLearningPlan(ResumeTailoringContract):
    objective: str = Field(min_length=1, max_length=1000)
    resource_directions: tuple[str, ...] = Field(min_length=1, max_length=5)
    minimum_acceptable_level: str = Field(min_length=1, max_length=1000)
    estimated_effort: str | None = Field(default=None, min_length=1, max_length=300)


class GapInterviewTalkingPoint(ResumeTailoringContract):
    acknowledge_gap: str = Field(min_length=1, max_length=1000)
    bridge_to_evidence: str | None = Field(default=None, min_length=1, max_length=1000)
    close_with_action: str = Field(min_length=1, max_length=1000)


class GapMitigation(ResumeTailoringContract):
    gap: str = Field(min_length=1, max_length=1000)
    requirement_id: str | None = Field(
        default=None,
        pattern=r"^job_requirement_[a-f0-9]{20}$",
    )
    gap_type: Literal["hard_blocker", "strengthenable"]
    priority: Literal["P0", "P1", "P2"]
    rationale: str = Field(min_length=1, max_length=1500)
    adjacent_experience: tuple[GapAdjacentEvidence, ...] = Field(
        default=(),
        max_length=5,
    )
    alternative_evidence: tuple[str, ...] = Field(default=(), max_length=5)
    next_action: str = Field(min_length=1, max_length=1000)
    learning_plan: GapLearningPlan | None = None
    interview_talking_point: GapInterviewTalkingPoint


class ResumeTailoringResult(ResumeTailoringContract):
    strategy_summary: str = Field(min_length=1, max_length=3000)
    changes: tuple[ResumeTailoringChange, ...] = Field(default=(), max_length=30)
    preserved_strengths: tuple[str, ...] = Field(default=(), max_length=10)
    # Bounded like ``changes`` rather than like the other short lists: the skill
    # requires every missing or unclear requirement to stay an unresolved gap,
    # so gaps grow with how weak the match is. A cap of 10 rejected a real
    # weak-match draft that listed 11, throwing away the whole tailoring run —
    # the cap punished exactly the honesty the skill asks for.
    unresolved_gaps: tuple[str, ...] = Field(default=(), max_length=30)
    # Optional on read so pre-096 drafts remain loadable. Newly generated drafts
    # are required by the service boundary to cover every unresolved gap.
    gap_mitigations: tuple[GapMitigation, ...] = Field(default=(), max_length=30)
    clarification_questions: tuple[str, ...] = Field(default=(), max_length=10)
    warnings: tuple[str, ...] = Field(default=(), max_length=10)

    @model_validator(mode="after")
    def bind_mitigations_to_gaps(self) -> ResumeTailoringResult:
        if not self.gap_mitigations:
            return self
        gaps = tuple(item.gap for item in self.gap_mitigations)
        if len(set(gaps)) != len(gaps):
            raise ValueError("gap mitigations must reference unique unresolved gaps")
        if set(gaps) != set(self.unresolved_gaps):
            raise ValueError("gap mitigations must cover every unresolved gap exactly once")
        return self


def gap_mitigation_errors(
    result: ResumeTailoringResult,
    match_result: ResumeJobMatchResult,
) -> tuple[str, ...]:
    """Validate new drafts without making legacy stored drafts unreadable."""
    errors: list[str] = []
    if result.unresolved_gaps and not result.gap_mitigations:
        errors.append("every unresolved gap requires a mitigation")

    assessments = {
        item.requirement_id: item
        for item in match_result.requirements
        if item.requirement_id is not None
    }
    linked_ids = {
        item.requirement_id
        for item in result.gap_mitigations
        if item.requirement_id is not None
    }
    if len(linked_ids) != sum(
        item.requirement_id is not None for item in result.gap_mitigations
    ):
        errors.append("a requirement can have only one gap mitigation")

    expected_ids = {
        item.requirement_id
        for item in match_result.requirements
        if item.requirement_id is not None
        and item.status in {"missing", "unclear"}
    }
    omitted = expected_ids - linked_ids
    if omitted:
        errors.append(
            "missing mitigations for requirement IDs: " + ", ".join(sorted(omitted))
        )

    for mitigation in result.gap_mitigations:
        if mitigation.requirement_id is None:
            if mitigation.gap_type == "hard_blocker":
                errors.append(
                    f"unbound gap cannot be a hard blocker: {mitigation.gap}"
                )
            continue
        assessment = assessments.get(mitigation.requirement_id)
        if assessment is None:
            errors.append(f"unknown requirement ID: {mitigation.requirement_id}")
            continue
        if assessment.status not in {"missing", "unclear"}:
            errors.append(
                f"supported requirement cannot be an unresolved gap: {mitigation.requirement_id}"
            )
            continue
        is_hard_blocker = (
            assessment.tier == "S"
            and assessment.kind == "fact"
            and assessment.status == "missing"
        )
        if is_hard_blocker and mitigation.gap_type != "hard_blocker":
            errors.append(
                f"S/fact/missing requirement must be a hard blocker: {mitigation.requirement_id}"
            )
        if not is_hard_blocker and mitigation.gap_type == "hard_blocker":
            errors.append(
                f"only S/fact/missing can be a hard blocker: {mitigation.requirement_id}"
            )
        if mitigation.gap_type == "hard_blocker" and mitigation.priority != "P0":
            errors.append(
                f"hard blocker must have P0 priority: {mitigation.requirement_id}"
            )
    return tuple(errors)


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
