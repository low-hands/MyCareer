from __future__ import annotations

from dataclasses import dataclass
import hashlib
from io import BytesIO
from typing import Literal, TypedDict

from pypdf import PdfReader

from langgraph.graph import END, START, StateGraph

from career_agent.agent.resume_job_match_contracts import (
    ConfirmedResumeFact,
    ResumeJobMatchResult,
)
from career_agent.agent.resume_tailoring_contracts import (
    EvidenceQuality,
    ResumeReviewAttempt,
    ResumeReviewIssue,
    ResumeReviewResult,
    ResumeReviewTrace,
    ResumeTailoringResult,
    ResumeTailoringReviewer,
    ResumeTailoringWorker,
    canonicalize_gap_mitigations,
    gap_mitigation_errors,
)
from career_agent.storage.resumes import StoredResumeDocument


@dataclass(frozen=True)
class EvidenceCheck:
    matched: bool
    quality: EvidenceQuality
    page: int | None
    reason: str


class ResumeTailoringReviewState(TypedDict, total=False):
    document: StoredResumeDocument
    jd_text: str
    match_result: ResumeJobMatchResult
    confirmed_facts: tuple[ConfirmedResumeFact, ...]
    tailoring_goal: str | None
    user_feedback: str | None
    previous_draft: ResumeTailoringResult | None
    review_feedback: tuple[str, ...]
    draft: ResumeTailoringResult
    review: ResumeReviewResult
    attempts: tuple[ResumeReviewAttempt, ...]
    fingerprints: tuple[str, ...]
    revisions: int
    status: Literal["running", "passed", "blocked"]
    stop_reason: Literal[
        "passed",
        "reviewer_blocked",
        "revision_limit",
        "no_progress",
        "cycle_detected",
    ]


class ResumeTailoringReviewOutcome:
    def __init__(self, *, draft: ResumeTailoringResult, trace: ResumeReviewTrace) -> None:
        self.draft = draft
        self.trace = trace


class ResumeTailoringReviewGraph:
    """Bounded writer-evaluator loop for a resume-tailoring draft."""

    def __init__(
        self,
        worker: ResumeTailoringWorker,
        reviewer: ResumeTailoringReviewer,
        *,
        # Keep the default interactive loop to one repair. The second repair
        # usually adds a full model round trip for a low-probability edge case;
        # offline callers can still request two explicitly.
        max_revisions: int = 1,
    ) -> None:
        if not 0 <= max_revisions <= 2:
            raise ValueError("max_revisions must be between zero and two")
        self._worker = worker
        self._reviewer = reviewer
        self._max_revisions = max_revisions

        graph = StateGraph(ResumeTailoringReviewState)
        graph.add_node("write", self._write)
        graph.add_node("evaluate", self._evaluate)
        graph.add_node("record", self._record)
        graph.add_node("revise", self._revise)
        graph.add_edge(START, "write")
        graph.add_edge("write", "evaluate")
        graph.add_edge("evaluate", "record")
        graph.add_conditional_edges(
            "record",
            self._after_record,
            {"revise": "revise", "finish": END},
        )
        graph.add_edge("revise", "write")
        self._graph = graph.compile()

    def run(
        self,
        *,
        document: StoredResumeDocument,
        jd_text: str,
        match_result: ResumeJobMatchResult,
        confirmed_facts: tuple[ConfirmedResumeFact, ...] = (),
        tailoring_goal: str | None = None,
        previous_draft: ResumeTailoringResult | None = None,
        user_feedback: str | None = None,
    ) -> ResumeTailoringReviewOutcome:
        state = self._graph.invoke(
            {
                "document": document,
                "jd_text": jd_text,
                "match_result": match_result,
                "confirmed_facts": confirmed_facts,
                "tailoring_goal": tailoring_goal,
                "previous_draft": previous_draft,
                "user_feedback": user_feedback,
                "review_feedback": (),
                "attempts": (),
                "fingerprints": (),
                "revisions": 0,
                "status": "running",
            },
            {"recursion_limit": 16},
        )
        return ResumeTailoringReviewOutcome(
            draft=state["draft"],
            trace=ResumeReviewTrace(
                status=state["status"],
                attempts=state["attempts"],
                stop_reason=state["stop_reason"],
            ),
        )

    def _write(self, state: ResumeTailoringReviewState) -> ResumeTailoringReviewState:
        # On a revision the draft under review is the one the feedback is about,
        # so it — not the caller's starting point — is what the writer revises.
        # The writer is told to change only the stated issues and keep every
        # unchallenged change; handing it the caller's draft (None on a fresh
        # request) asks it to preserve something it was never shown, so it
        # rewrites blind and the feedback's change indices point at nothing.
        draft = self._worker.tailor(
            document=state["document"],
            jd_text=state["jd_text"],
            match_result=state["match_result"],
            confirmed_facts=state["confirmed_facts"],
            tailoring_goal=state.get("tailoring_goal"),
            user_feedback=state.get("user_feedback"),
            review_feedback=state.get("review_feedback", ()),
            previous_draft=state.get("draft") or state.get("previous_draft"),
        )
        draft = self._canonicalize_evidence(draft, state["document"])
        return {
            "draft": canonicalize_gap_mitigations(draft, state["match_result"])
        }

    def _evaluate(self, state: ResumeTailoringReviewState) -> ResumeTailoringReviewState:
        fingerprint = self._fingerprint(state["draft"])
        previous = state.get("fingerprints", ())
        if previous and fingerprint == previous[-1]:
            return {"review": self._stalled_review("The revision did not change the draft.")}
        if fingerprint in previous:
            return {"review": self._stalled_review("The draft returned to an earlier state.")}

        deterministic = self._validate_grounding(
            document=state["document"],
            match_result=state["match_result"],
            confirmed_facts=state["confirmed_facts"],
            draft=state["draft"],
        )
        if any(issue.severity == "blocking" for issue in deterministic):
            return {
                "review": ResumeReviewResult(
                    verdict="revise",
                    summary="Deterministic grounding checks found blocking issues.",
                    issues=deterministic,
                )
            }

        reviewed = self._reviewer.review_draft(
            document=state["document"],
            jd_text=state["jd_text"],
            match_result=state["match_result"],
            draft=state["draft"],
            confirmed_facts=state["confirmed_facts"],
        )
        if deterministic:
            reviewed = reviewed.model_copy(
                update={"issues": (*deterministic, *reviewed.issues)}
            )
        return {"review": reviewed}

    def _record(self, state: ResumeTailoringReviewState) -> ResumeTailoringReviewState:
        fingerprint = self._fingerprint(state["draft"])
        review = state["review"]
        attempts = (
            *state.get("attempts", ()),
            ResumeReviewAttempt(
                attempt_number=len(state.get("attempts", ())) + 1,
                draft_fingerprint=fingerprint,
                result=review,
            ),
        )
        fingerprints = (*state.get("fingerprints", ()), fingerprint)
        if review.verdict == "pass":
            return {
                "attempts": attempts,
                "fingerprints": fingerprints,
                "status": "passed",
                "stop_reason": "passed",
            }
        if review.verdict == "block":
            return {
                "attempts": attempts,
                "fingerprints": fingerprints,
                "status": "blocked",
                "stop_reason": "reviewer_blocked",
            }
        if state.get("fingerprints") and fingerprint == state["fingerprints"][-1]:
            reason = "no_progress"
        elif fingerprint in state.get("fingerprints", ()):
            reason = "cycle_detected"
        elif state.get("revisions", 0) >= self._max_revisions:
            reason = "revision_limit"
        else:
            feedback = tuple(
                issue.explanation + (
                    f" Required correction: {issue.revision_instruction}"
                    if issue.revision_instruction else ""
                )
                for issue in review.issues
                if issue.severity == "blocking"
            )
            return {
                "attempts": attempts,
                "fingerprints": fingerprints,
                "review_feedback": feedback,
                "status": "running",
            }
        return {
            "attempts": attempts,
            "fingerprints": fingerprints,
            "status": "blocked",
            "stop_reason": reason,
        }

    @staticmethod
    def _after_record(state: ResumeTailoringReviewState) -> Literal["revise", "finish"]:
        return "revise" if state["status"] == "running" else "finish"

    @staticmethod
    def _revise(state: ResumeTailoringReviewState) -> ResumeTailoringReviewState:
        return {"revisions": state.get("revisions", 0) + 1}

    @staticmethod
    def _fingerprint(draft: ResumeTailoringResult) -> str:
        return hashlib.sha256(
            draft.model_dump_json().encode("utf-8")
        ).hexdigest()

    @staticmethod
    def _stalled_review(explanation: str) -> ResumeReviewResult:
        return ResumeReviewResult(
            verdict="revise",
            summary=explanation,
            issues=(
                ResumeReviewIssue(
                    category="unclear_expression",
                    severity="blocking",
                    explanation=explanation,
                    revision_instruction="Stop automatic revision and ask the user for direction.",
                ),
            ),
        )

    @classmethod
    def _canonicalize_evidence(
        cls,
        draft: ResumeTailoringResult,
        document: StoredResumeDocument,
    ) -> ResumeTailoringResult:
        """Write back the quality/page actually observed by the server."""
        changes = []
        for change in draft.changes:
            evidence_items = []
            for evidence in change.support_evidence:
                check = cls._check_evidence(
                    document=document,
                    quote=cls._normalize(evidence.source_quote),
                    declared_quality=evidence.evidence_quality,
                    page=evidence.page,
                )
                update = {}
                if check.matched or check.quality == "ocr_unverified":
                    update = {
                        "evidence_quality": check.quality,
                        "page": check.page,
                    }
                evidence_items.append(evidence.model_copy(update=update))
            changes.append(change.model_copy(update={"support_evidence": tuple(evidence_items)}))

        mitigations = []
        for mitigation in draft.gap_mitigations:
            adjacent = []
            for evidence in getattr(mitigation, "adjacent_experience", ()):
                check = cls._check_evidence(
                    document=document,
                    quote=cls._normalize(evidence.source_quote),
                    declared_quality=evidence.evidence_quality,
                    page=evidence.page,
                )
                update = {}
                if check.matched or check.quality == "ocr_unverified":
                    update = {"evidence_quality": check.quality, "page": check.page}
                adjacent.append(evidence.model_copy(update=update))
            alternatives = []
            for evidence in getattr(mitigation, "alternative_evidence", ()):
                if isinstance(evidence, str) or evidence.status != "existing":
                    alternatives.append(evidence)
                    continue
                check = cls._check_evidence(
                    document=document,
                    quote=cls._normalize(evidence.source_quote or ""),
                    declared_quality=evidence.evidence_quality,
                    page=evidence.page,
                )
                update = {}
                if check.matched or check.quality == "ocr_unverified":
                    update = {"evidence_quality": check.quality, "page": check.page}
                alternatives.append(evidence.model_copy(update=update))
            mitigations.append(
                mitigation.model_copy(
                    update={
                        "adjacent_experience": tuple(adjacent),
                        "alternative_evidence": tuple(alternatives),
                    }
                )
            )
        return draft.model_copy(
            update={"changes": tuple(changes), "gap_mitigations": tuple(mitigations)}
        )

    @classmethod
    def _validate_grounding(
        cls,
        *,
        document: StoredResumeDocument,
        match_result: ResumeJobMatchResult,
        confirmed_facts: tuple[ConfirmedResumeFact, ...],
        draft: ResumeTailoringResult,
        include_gap_mitigations: bool = True,
    ) -> tuple[ResumeReviewIssue, ...]:
        resume_text, readable = cls._resume_text(document)
        # A readable document that yields no text is a genuine scan: grounding is
        # unverifiable, so warn. An unreadable document proves nothing and must
        # still block, otherwise a corrupt upload silently buys a free pass.
        unverifiable = resume_text is None and readable

        issues: list[ResumeReviewIssue] = []
        for explanation in (
            gap_mitigation_errors(draft, match_result)
            if include_gap_mitigations else ()
        ):
            issues.append(
                ResumeReviewIssue(
                    category="jd_misalignment",
                    severity="blocking",
                    explanation=explanation,
                    revision_instruction=(
                        "Rebuild gap mitigations from the authoritative requirement IDs, "
                        "tiers, kinds, and statuses."
                    ),
                )
            )
        locators: dict[str, int] = {}
        for index, change in enumerate(draft.changes, start=1):
            locator = cls._normalize(change.target_locator)
            if locator in locators:
                issues.append(
                    ResumeReviewIssue(
                        category="change_set_mismatch",
                        severity="blocking",
                        change_index=index,
                        explanation=(
                            f"Changes {locators[locator]} and {index} target the same location."
                        ),
                        revision_instruction="Merge duplicate changes into one proposal.",
                    )
                )
            else:
                locators[locator] = index
            for evidence in change.support_evidence:
                quote = cls._normalize(evidence.source_quote)
                check = cls._check_evidence(
                    document=document,
                    quote=quote,
                    declared_quality=evidence.evidence_quality,
                    page=evidence.page,
                )
                if evidence.page is not None and not check.matched:
                    issues.append(
                        ResumeReviewIssue(
                            category="unsupported_fact",
                            severity="blocking",
                            change_index=index,
                            source_quote=evidence.source_quote,
                            explanation=(
                                "The support quote could not be verified on the cited page "
                                f"({check.reason})."
                            ),
                            revision_instruction="Cite the page containing the exact quote or remove the change.",
                        )
                    )
                    continue
                if evidence.evidence_quality == "ocr_unverified":
                    issues.append(
                        ResumeReviewIssue(
                            category="unsupported_fact",
                            severity="blocking",
                            change_index=index,
                            source_quote=evidence.source_quote,
                            explanation="OCR-unverified evidence cannot be used to write an automatic resume change.",
                            revision_instruction="Confirm the quote in the source document or remove the change.",
                        )
                    )
                    continue
                if check.matched:
                    continue
                if unverifiable:
                    # A scan may be genuine, but without a text layer it cannot
                    # authorize an automatic resume write.
                    issues.append(
                        ResumeReviewIssue(
                            category="unsupported_fact",
                            severity="blocking",
                            change_index=index,
                            source_quote=evidence.source_quote,
                            explanation="The exact resume version has no extractable text, so this support quote is OCR-unverified and cannot authorize an automatic resume write.",
                            revision_instruction="Confirm the quote against the original resume before accepting the change.",
                        )
                    )
                else:
                    issues.append(
                        ResumeReviewIssue(
                            category="unsupported_fact",
                            severity="blocking",
                            change_index=index,
                            source_quote=evidence.source_quote,
                            explanation="The cited support quote is absent from the exact resume version.",
                            revision_instruction="Use an exact quote from the resume or remove the change.",
                        )
                    )
        for mitigation in (draft.gap_mitigations if include_gap_mitigations else ()):
            for evidence in getattr(mitigation, "adjacent_experience", ()):
                quote = cls._normalize(evidence.source_quote)
                check = cls._check_evidence(
                    document=document,
                    quote=quote,
                    declared_quality=evidence.evidence_quality,
                    page=evidence.page,
                )
                if (
                    evidence.page is not None
                    and not check.matched
                    and check.reason != "no_reliable_text_layer"
                ):
                    issues.append(
                        ResumeReviewIssue(
                            category="unsupported_fact",
                            severity="blocking",
                            source_quote=evidence.source_quote,
                            explanation=(
                                "Adjacent evidence could not be verified on the cited page "
                                f"({check.reason})."
                            ),
                            revision_instruction="Cite the page containing the quote or remove this evidence.",
                        )
                    )
                    continue
                if (
                    check.reason == "no_reliable_text_layer"
                    or evidence.evidence_quality == "ocr_unverified"
                ):
                    issues.append(
                        ResumeReviewIssue(
                            category="unsupported_fact",
                            severity="warning",
                            source_quote=evidence.source_quote,
                            explanation="OCR-unverified adjacent evidence is a candidate only and needs user confirmation.",
                            revision_instruction="Confirm the quote before using it in interview language.",
                        )
                    )
                    continue
                if check.matched:
                    continue
                if unverifiable:
                    issues.append(
                        ResumeReviewIssue(
                            category="unsupported_fact",
                            severity="warning",
                            source_quote=evidence.source_quote,
                            explanation=(
                                f"Adjacent evidence for gap '{mitigation.gap}' could not "
                                "be verified because the resume has no extractable text."
                            ),
                            revision_instruction=(
                                "Confirm the quote against the original resume before "
                                "using it in an interview."
                            ),
                        )
                    )
                else:
                    issues.append(
                        ResumeReviewIssue(
                            category="unsupported_fact",
                            severity="blocking",
                            source_quote=evidence.source_quote,
                            explanation=(
                                f"Adjacent evidence for gap '{mitigation.gap}' is absent "
                                "from the exact resume version."
                            ),
                            revision_instruction=(
                                "Use an exact resume quote or remove the adjacent evidence."
                            ),
                        )
                    )
            for evidence in getattr(mitigation, "alternative_evidence", ()):
                if isinstance(evidence, str) or evidence.status != "existing":
                    continue
                quote = cls._normalize(evidence.source_quote or "")
                check = cls._check_evidence(
                    document=document,
                    quote=quote,
                    declared_quality=evidence.evidence_quality,
                    page=evidence.page,
                )
                if (
                    evidence.page is not None
                    and not check.matched
                    and check.reason != "no_reliable_text_layer"
                ):
                    issues.append(
                        ResumeReviewIssue(
                            category="unsupported_fact",
                            severity="blocking",
                            source_quote=evidence.source_quote,
                            explanation=(
                                "Existing alternative evidence could not be verified on "
                                f"the cited page ({check.reason})."
                            ),
                            revision_instruction="Cite the page containing the quote or mark the material as planned.",
                        )
                    )
                    continue
                if (
                    check.reason == "no_reliable_text_layer"
                    or evidence.evidence_quality == "ocr_unverified"
                ):
                    issues.append(
                        ResumeReviewIssue(
                            category="unsupported_fact",
                            severity="warning",
                            source_quote=evidence.source_quote,
                            explanation="OCR-unverified existing material is a candidate only and needs user confirmation.",
                            revision_instruction="Confirm the material or mark it as planned.",
                        )
                    )
                    continue
                if check.matched:
                    continue
                if unverifiable:
                    issues.append(
                        ResumeReviewIssue(
                            category="unsupported_fact",
                            severity="warning",
                            source_quote=evidence.source_quote,
                            explanation=(
                                f"Existing alternative evidence for gap '{mitigation.gap}' "
                                "could not be verified because the resume has no "
                                "extractable text."
                            ),
                            revision_instruction=(
                                "Confirm the material against the original resume or mark "
                                "it as planned."
                            ),
                        )
                    )
                else:
                    issues.append(
                        ResumeReviewIssue(
                            category="unsupported_fact",
                            severity="blocking",
                            source_quote=evidence.source_quote,
                            explanation=(
                                f"Existing alternative evidence for gap '{mitigation.gap}' "
                                "is absent from the exact resume version."
                            ),
                            revision_instruction=(
                                "Use an exact resume quote or mark the material as planned."
                            ),
                        )
                    )
        return tuple(issues)

    @classmethod
    def _resume_pages(cls, document: StoredResumeDocument) -> tuple[tuple[str, ...] | None, bool]:
        """Return (normalized text or None, whether the document was readable).

        The two are independent: a readable scan has no text, while an unreadable
        upload has neither. Callers must treat those cases differently.
        """

        if document.document_format == "pdf":
            try:
                reader = PdfReader(BytesIO(document.raw_bytes), strict=False)
                extracted = tuple(
                    cls._normalize(page.extract_text() or "") for page in reader.pages
                )
            except Exception:
                return None, False
            return extracted, True
        try:
            decoded = document.raw_bytes.decode("utf-8-sig")
        except UnicodeDecodeError:
            return None, False
        return (cls._normalize(decoded),), True

    @classmethod
    def _resume_text(cls, document: StoredResumeDocument) -> tuple[str | None, bool]:
        pages, readable = cls._resume_pages(document)
        if pages is None:
            return None, readable
        return " ".join(pages) or None, readable

    @staticmethod
    def _normalize(value: str) -> str:
        return " ".join(value.casefold().split())

    @classmethod
    def _quote_in_text(
        cls,
        quote: str,
        resume_text: str,
        *,
        allow_pdf_layout_match: bool,
    ) -> bool:
        if quote in resume_text:
            return True
        if not allow_pdf_layout_match:
            return False
        # PDF extraction frequently inserts line-break whitespace or soft/
        # visible hyphens inside one source phrase. This deliberately narrow
        # relaxed layer handles only those layout artifacts; it is not fuzzy
        # semantic matching and therefore cannot turn a paraphrase into proof.
        def layout_key(value: str) -> str:
            return "".join(
                character
                for character in value
                if not character.isspace() and character not in {"-", "\u00ad"}
            )

        relaxed_quote = layout_key(quote)
        return len(relaxed_quote) >= 8 and relaxed_quote in layout_key(resume_text)

    @classmethod
    def _check_evidence(
        cls,
        *,
        document: StoredResumeDocument,
        quote: str,
        declared_quality: EvidenceQuality,
        page: int | None,
    ) -> EvidenceCheck:
        pages, readable = cls._resume_pages(document)
        if pages is None:
            return EvidenceCheck(
                False,
                "ocr_unverified" if readable else declared_quality,
                page,
                "no_reliable_text_layer" if readable else "document_unreadable",
            )
        if page is not None:
            if page < 1 or page > len(pages):
                return EvidenceCheck(
                    False,
                    "ocr_unverified" if not any(pages) else declared_quality,
                    page,
                    "page_out_of_range",
                )
            if not pages[page - 1]:
                return EvidenceCheck(False, "ocr_unverified", page, "no_reliable_text_layer")
            candidates = ((page, pages[page - 1]),)
        else:
            candidates = tuple(enumerate(pages, start=1))
        if not any(pages):
            return EvidenceCheck(
                False,
                "ocr_unverified",
                page,
                "no_reliable_text_layer",
            )
        for matched_page, page_text in candidates:
            if quote in page_text:
                return EvidenceCheck(True, "exact", matched_page, "exact_text_match")
        if document.document_format == "pdf":
            for matched_page, page_text in candidates:
                if cls._quote_in_text(
                    quote,
                    page_text,
                    allow_pdf_layout_match=True,
                ):
                    return EvidenceCheck(
                        True,
                        "normalized",
                        matched_page,
                        "hyphenation_normalized",
                    )
        return EvidenceCheck(
            False,
            "ocr_unverified" if declared_quality == "ocr_unverified" else declared_quality,
            page,
            "ocr_unverified" if declared_quality == "ocr_unverified" else "quote_not_found",
        )
