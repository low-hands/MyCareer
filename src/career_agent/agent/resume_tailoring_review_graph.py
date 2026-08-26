from __future__ import annotations

import hashlib
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from career_agent.agent.resume_job_match_contracts import (
    ConfirmedResumeFact,
    ResumeJobMatchResult,
)
from career_agent.agent.resume_tailoring_contracts import (
    ResumeReviewAttempt,
    ResumeReviewIssue,
    ResumeReviewResult,
    ResumeReviewTrace,
    ResumeTailoringResult,
    ResumeTailoringReviewer,
    ResumeTailoringWorker,
)
from career_agent.storage.resumes import StoredResumeDocument


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
        max_revisions: int = 2,
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
        draft = self._worker.tailor(
            document=state["document"],
            jd_text=state["jd_text"],
            match_result=state["match_result"],
            confirmed_facts=state["confirmed_facts"],
            tailoring_goal=state.get("tailoring_goal"),
            user_feedback=state.get("user_feedback"),
            review_feedback=state.get("review_feedback", ()),
            previous_draft=state.get("previous_draft"),
        )
        return {"draft": draft}

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
                issue.revision_instruction or issue.explanation
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
    def _validate_grounding(
        cls,
        *,
        document: StoredResumeDocument,
        match_result: ResumeJobMatchResult,
        confirmed_facts: tuple[ConfirmedResumeFact, ...],
        draft: ResumeTailoringResult,
    ) -> tuple[ResumeReviewIssue, ...]:
        known_quotes = {
            cls._normalize(fact.source_quote) for fact in confirmed_facts
        }
        known_quotes.update(
            cls._normalize(evidence.source_quote)
            for requirement in match_result.requirements
            for evidence in requirement.resume_evidence
        )
        resume_text: str | None = None
        if document.document_format != "pdf":
            try:
                resume_text = cls._normalize(document.raw_bytes.decode("utf-8-sig"))
            except UnicodeDecodeError:
                resume_text = None

        issues: list[ResumeReviewIssue] = []
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
                grounded = quote in known_quotes or (
                    resume_text is not None and quote in resume_text
                )
                if document.document_format != "pdf" and not grounded:
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
        return tuple(issues)

    @staticmethod
    def _normalize(value: str) -> str:
        return " ".join(value.casefold().split())
