from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from career_agent.agent.resume_job_match_contracts import (
    ConfirmedResumeFact,
    ResumeJobMatchResult,
    is_confirmed_hard_gate_assessment,
)


EvidenceQuality = Literal["exact", "normalized", "ocr_unverified"]

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
    evidence_quality: EvidenceQuality = "exact"
    page: int | None = Field(default=None, ge=1)


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
    evidence_quality: EvidenceQuality = "exact"
    page: int | None = Field(default=None, ge=1)


class GapLearningPlan(ResumeTailoringContract):
    objective: str = Field(min_length=1, max_length=1000)
    resource_directions: tuple[str, ...] = Field(min_length=1, max_length=5)
    minimum_acceptable_level: str = Field(min_length=1, max_length=1000)
    estimated_effort: str | None = Field(default=None, min_length=1, max_length=300)


class GapInterviewTalkingPoint(ResumeTailoringContract):
    acknowledge_gap: str = Field(min_length=1, max_length=1000)
    bridge_to_evidence: str | None = Field(default=None, min_length=1, max_length=1000)
    close_with_action: str = Field(min_length=1, max_length=1000)


class GapAlternativeEvidence(ResumeTailoringContract):
    description: str = Field(min_length=1, max_length=1000)
    status: Literal["existing", "planned"]
    source_locator: str | None = Field(default=None, min_length=1, max_length=300)
    source_quote: str | None = Field(default=None, min_length=1, max_length=500)
    evidence_quality: EvidenceQuality = "exact"
    page: int | None = Field(default=None, ge=1)
    acceptance_criteria: str | None = Field(
        default=None,
        min_length=1,
        max_length=1000,
    )

    @model_validator(mode="after")
    def require_proof_or_acceptance_criteria(self) -> GapAlternativeEvidence:
        if self.status == "existing":
            if self.source_locator is None or self.source_quote is None:
                raise ValueError("existing alternative evidence requires a source quote")
            if self.acceptance_criteria is not None:
                raise ValueError("existing alternative evidence cannot use acceptance criteria")
        else:
            if self.acceptance_criteria is None:
                raise ValueError("planned alternative evidence requires acceptance criteria")
            if self.source_locator is not None or self.source_quote is not None:
                raise ValueError("planned alternative evidence cannot claim a source quote")
            if self.evidence_quality != "exact" or self.page is not None:
                raise ValueError("planned alternative evidence cannot carry source metadata")
        return self


class GapMitigation(ResumeTailoringContract):
    # ``gap`` and ``resolution_mode`` remain optional while old stored drafts
    # are read. New drafts are canonicalized from the bound requirement and
    # rejected at the service boundary if the resolution mode is absent.
    gap: str | None = Field(default=None, min_length=1, max_length=1000)
    requirement_id: str | None = Field(
        default=None,
        pattern=r"^job_requirement_[a-f0-9]{20}$",
    )
    gap_type: Literal["hard_blocker", "strengthenable"]
    priority: Literal["P0", "P1", "P2"]
    resolution_mode: Literal[
        "clarify",
        "provide_evidence",
        "build_artifact",
        "learn",
    ] | None = None
    rationale: str | None = Field(default=None, min_length=1, max_length=1500)
    adjacent_experience: tuple[GapAdjacentEvidence, ...] = Field(
        default=(),
        max_length=5,
    )
    # Strings are accepted only to keep pre-migration JSON readable. New
    # drafts must emit the structured existing/planned form.
    alternative_evidence: tuple[GapAlternativeEvidence | str, ...] = Field(
        default=(),
        max_length=5,
    )
    next_action: str = Field(min_length=1, max_length=1000)
    learning_plan: GapLearningPlan | None = None
    clarification_question: str | None = Field(default=None, min_length=1, max_length=1000)
    interview_talking_point: GapInterviewTalkingPoint | None = None

    @model_validator(mode="after")
    def enforce_conditional_fields(self) -> GapMitigation:
        # Legacy records predate resolution_mode and retain their wider shape.
        if self.resolution_mode is None:
            return self
        if self.rationale is None:
            raise ValueError("new mitigation requires a rationale")
        if self.interview_talking_point is None:
            raise ValueError("new mitigation requires interview talking points")
        if self.resolution_mode == "clarify":
            if self.clarification_question is None:
                raise ValueError("clarify mitigation requires a clarification question")
            if self.learning_plan is not None:
                raise ValueError("clarify mitigation cannot include a learning plan")
            if self.adjacent_experience or self.alternative_evidence:
                raise ValueError("clarify mitigation cannot include evidence plans")
        elif self.resolution_mode == "provide_evidence":
            if not (self.adjacent_experience or self.alternative_evidence):
                raise ValueError("provide_evidence mitigation requires evidence")
            if self.learning_plan is not None or self.clarification_question is not None:
                raise ValueError("provide_evidence mitigation cannot include learning or clarification")
        elif self.resolution_mode == "build_artifact":
            planned = tuple(
                item for item in self.alternative_evidence
                if isinstance(item, GapAlternativeEvidence) and item.status == "planned"
            )
            if not planned:
                raise ValueError("build_artifact mitigation requires a planned artifact")
            if self.learning_plan is not None or self.clarification_question is not None:
                raise ValueError("build_artifact mitigation cannot include learning or clarification")
        elif self.resolution_mode == "learn":
            if self.learning_plan is None:
                raise ValueError("learn mitigation requires a learning plan")
            if self.clarification_question is not None:
                raise ValueError("learn mitigation cannot include a clarification question")
        return self


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
    # Optional on read so pre-096 drafts remain loadable. For new drafts this is
    # a compatibility projection rebuilt from stable-ID mitigations; it is not
    # the source of truth for coverage.
    gap_mitigations: tuple[GapMitigation, ...] = Field(default=(), max_length=30)
    clarification_questions: tuple[str, ...] = Field(default=(), max_length=10)
    warnings: tuple[str, ...] = Field(default=(), max_length=10)



def canonicalize_gap_mitigations(
    result: ResumeTailoringResult,
    match_result: ResumeJobMatchResult,
) -> ResumeTailoringResult:
    """Copy requirement text from the authoritative match into bound plans."""
    assessments = {
        item.requirement_id: item
        for item in match_result.requirements
        if item.requirement_id is not None
    }
    mitigations = tuple(
        item.model_copy(
            update={"gap": assessments[item.requirement_id].requirement}
        )
        if item.requirement_id in assessments
        else item
        for item in result.gap_mitigations
    )
    if not mitigations:
        return result
    gaps = tuple(item.gap for item in mitigations if item.gap is not None)
    return result.model_copy(
        update={
            "gap_mitigations": mitigations,
            # Compatibility projection for consumers and pre-096 storage
            # readers. Coverage itself is based only on stable IDs.
            "unresolved_gaps": gaps,
        }
    )


def gap_mitigation_errors(
    result: ResumeTailoringResult,
    match_result: ResumeJobMatchResult,
) -> tuple[str, ...]:
    """Validate new drafts without making legacy stored drafts unreadable."""
    errors: list[str] = []
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
    if result.unresolved_gaps and not result.gap_mitigations:
        errors.append(
            "unresolved gap text cannot replace requirement-ID mitigations"
        )

    for mitigation in result.gap_mitigations:
        if mitigation.requirement_id is None:
            errors.append("new gap mitigations require an authoritative requirement ID")
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
        if mitigation.resolution_mode is None:
            errors.append(
                f"gap mitigation requires a resolution mode: {mitigation.requirement_id}"
            )
        if mitigation.rationale is None:
            errors.append(f"new mitigation requires a rationale: {mitigation.requirement_id}")
        if mitigation.interview_talking_point is None:
            errors.append(
                "new mitigation requires interview talking points: "
                f"{mitigation.requirement_id}"
            )
        if any(isinstance(item, str) for item in mitigation.alternative_evidence):
            errors.append(
                "new alternative evidence must declare existing or planned status: "
                f"{mitigation.requirement_id}"
            )
        # Pre-096 persisted matches have no classification metadata. Preserve
        # their historical validation semantics while requiring an explicit
        # confirmed gate for all newly-produced results.
        legacy_classification = (
            assessment.tier_confidence is None
            and assessment.tier_rationale is None
            and assessment.tier_evidence is None
            and assessment.classification_status is None
        )
        is_hard_blocker = (
            assessment.tier == "S"
            and assessment.kind == "fact"
            and assessment.status == "missing"
            and (
                legacy_classification
                or (
                    is_confirmed_hard_gate_assessment(assessment)
                )
            )
        )
        if is_hard_blocker and mitigation.gap_type != "hard_blocker":
            errors.append(
                f"S/fact/missing requirement must be a hard blocker: {mitigation.requirement_id}"
            )
        if not is_hard_blocker and mitigation.gap_type == "hard_blocker":
            errors.append(
                f"only S/fact/missing can be a hard blocker: {mitigation.requirement_id}"
            )
        is_preapplication_clarification = (
            assessment.tier == "S"
            and assessment.kind == "fact"
            and assessment.status == "unclear"
            and mitigation.resolution_mode == "clarify"
        )
        if mitigation.gap_type == "hard_blocker" and mitigation.priority != "P0":
            errors.append(
                f"hard blocker must have P0 priority: {mitigation.requirement_id}"
            )
        if mitigation.priority == "P0" and not (
            is_hard_blocker or is_preapplication_clarification
        ):
            errors.append(
                "P0 is reserved for hard blockers or S/fact/unclear clarification: "
                f"{mitigation.requirement_id}"
            )
        if assessment.status == "unclear":
            if mitigation.resolution_mode != "clarify":
                errors.append(
                    f"unclear requirement must use clarify mode: {mitigation.requirement_id}"
                )
            if mitigation.learning_plan is not None:
                errors.append(
                    f"unclear requirement cannot prescribe learning: {mitigation.requirement_id}"
                )
        if mitigation.resolution_mode == "learn" and mitigation.learning_plan is None:
            errors.append(
                f"learn mode requires a learning plan: {mitigation.requirement_id}"
            )
        if mitigation.resolution_mode != "learn" and mitigation.learning_plan is not None:
            errors.append(
                "learning plan is allowed only in learn mode: "
                f"{mitigation.requirement_id}"
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
