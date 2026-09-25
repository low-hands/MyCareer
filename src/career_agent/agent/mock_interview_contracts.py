from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from career_agent.agent.resume_job_match_contracts import ConfirmedResumeFact
from career_agent.agent.interview_preparation_contracts import InterviewPreparationContext
from career_agent.domain.mock_interviews import (
    MockInterviewAnswerEvaluation,
    MockInterviewPlan,
    MockInterviewPlanItem,
    MockInterviewQuestionResult,
    MockInterviewReport,
    MockInterviewScoreDimension,
    MockInterviewSession,
    MockInterviewTurn,
    MockInterviewType,
)

if TYPE_CHECKING:
    from career_agent.storage.resumes import StoredResumeDocument


MockInterviewCompletionReason = Literal[
    "plan_completed",
    "user_ended",
    "time_limit",
    "safety_stop",
]


def _clip_to_declared_lengths(model: type[BaseModel], value: Any) -> Any:
    """Cut every list field of raw model output to the length its schema allows.

    The schema is sent with ``strict=False``, so a model may return a sixth
    strength or a ninth key fact. Each such overflow used to fail the whole
    structured response (seen live on end-of-interview evaluations); the
    extra entries are the least important ones, so they are dropped instead.
    """
    if not isinstance(value, dict):
        return value
    clipped = dict(value)
    for name, field in model.model_fields.items():
        items = clipped.get(name)
        limit = next(
            (getattr(meta, "max_length", None) for meta in field.metadata
             if getattr(meta, "max_length", None) is not None),
            None,
        )
        if isinstance(items, list) and limit is not None:
            clipped[name] = items[:limit]
    return clipped


class MockInterviewWorkerContract(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )


class MockInterviewStartRequest(MockInterviewWorkerContract):
    user_id: str = Field(min_length=1)
    application_id: str | None = Field(default=None, min_length=1)
    job_posting_id: str | None = Field(default=None, min_length=1)
    jd_snapshot_id: str | None = Field(default=None, min_length=1)
    resume_version_id: str | None = Field(default=None, min_length=1)
    target_role: str | None = Field(default=None, min_length=1, max_length=300)
    target_company: str | None = Field(default=None, min_length=1, max_length=200)
    company_research_report_id: str | None = Field(default=None, min_length=1)
    interview_type: MockInterviewType
    interview_round_id: str | None = Field(default=None, min_length=1)
    max_primary_questions: int = Field(default=10, ge=1, le=20)
    max_follow_ups_per_question: int = Field(default=2, ge=0, le=5)
    conversation_id: str | None = Field(default=None, min_length=1)


class MockInterviewGraphResult(MockInterviewWorkerContract):
    session_id: str = Field(min_length=1)
    state: Literal["awaiting_answer", "running", "completed", "cancelled"]
    message: str = Field(min_length=1)
    turn_id: str | None = Field(default=None, min_length=1)
    question: str | None = Field(default=None, min_length=1, max_length=2000)
    evaluation: MockInterviewAnswerEvaluation | None = None
    report_id: str | None = Field(default=None, min_length=1)
    report: MockInterviewReport | None = None
    # What the questions are based on (resume, job or company, research); set
    # only on the opening question.
    resume_basis: str | None = Field(default=None, min_length=1, max_length=600)


class MockInterviewQuestionSummary(MockInterviewWorkerContract):
    """One primary question's row in a finished run's index."""

    plan_item_number: int = Field(ge=1)
    question: str = Field(min_length=1)
    rating: str = Field(min_length=1)
    follow_up_count: int = Field(default=0, ge=0)


class MockInterviewResultView(MockInterviewWorkerContract):
    """A finished run as the readback tool found it in the store.

    Carried in the observation payload so the runtime can render the index
    without the tool layer composing prose, the same way job research and
    interview preparation already travel. The report's own summary rides along
    because the index is read to decide what to re-open, and the summary is
    what makes that decision.
    """

    interview_type: str = Field(min_length=1)
    status: str = Field(min_length=1)
    questions: tuple[MockInterviewQuestionSummary, ...] = ()
    answered_count: int = Field(default=0, ge=0)
    report_id: str | None = Field(default=None, min_length=1)
    report_summary: str | None = Field(default=None, min_length=1)


class MockInterviewExchange(MockInterviewWorkerContract):
    """One question and what came back for it, verbatim."""

    turn_type: Literal["primary", "follow_up"]
    question: str = Field(min_length=1)
    answer: str | None = Field(default=None, min_length=1)
    rating: str | None = Field(default=None, min_length=1)
    evaluation_summary: str | None = Field(default=None, min_length=1)


class MockInterviewQuestionView(MockInterviewWorkerContract):
    """One plan item read back in full, its follow-ups included."""

    question_number: int = Field(ge=1)
    exchanges: tuple[MockInterviewExchange, ...] = Field(min_length=1)


class MockInterviewInputDecision(MockInterviewWorkerContract):
    """A local routing decision made before an answer can be persisted."""

    action: Literal["answer", "cancel"]


class MockInterviewPlanDraft(MockInterviewWorkerContract):
    """Model-authored plan content before the workflow adds identity and time."""

    summary: str = Field(min_length=1, max_length=2000)
    items: tuple[MockInterviewPlanItem, ...] = Field(min_length=1, max_length=20)
    limitations: tuple[str, ...] = Field(default=(), max_length=10)
    company_style_profile: str | None = Field(
        default=None,
        max_length=200,
        description=(
            "The exact company.md profile heading the plan followed, or null "
            "when no company profile was applied."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def trim_item_evidence(cls, value: Any) -> Any:
        """Keep at most three quotes per item, and only paired resume quotes.

        Models quote generously; five resume quotes, or one more locator than
        quote, failed the whole plan and with it the interview. Evidence is
        supporting context for a question the item already states, so the
        extra lines are dropped rather than the plan.
        """
        if not isinstance(value, dict) or not isinstance(value.get("items"), list):
            return value
        items = []
        for item in value["items"]:
            if isinstance(item, dict):
                item = dict(item)
                if isinstance(item.get("jd_quotes"), list):
                    item["jd_quotes"] = item["jd_quotes"][:3]
                locators, quotes = item.get("resume_locators"), item.get("resume_quotes")
                if isinstance(locators, list) and isinstance(quotes, list):
                    paired = min(len(locators), len(quotes), 3)
                    item["resume_locators"], item["resume_quotes"] = locators[:paired], quotes[:paired]
            items.append(item)
        return {**value, "items": items}

    @model_validator(mode="after")
    def require_contiguous_sequence(self) -> MockInterviewPlanDraft:
        expected = tuple(range(1, len(self.items) + 1))
        actual = tuple(item.sequence_number for item in self.items)
        if actual != expected:
            raise ValueError("plan item sequence numbers must be contiguous and ordered")
        return self


class MockInterviewQuestionDraft(MockInterviewWorkerContract):
    """Exactly one primary question; follow-ups come from the evaluation contract."""

    question: str = Field(min_length=1, max_length=2000)


class MockInterviewFollowUpDecision(MockInterviewWorkerContract):
    """The only per-answer model decision: probe this answer once more, or move on.

    Scoring happens once, per question, when the interview ends; deciding a
    follow-up needs a judgment about the answer but not a written assessment.
    """

    next_action: Literal["follow_up", "next_question", "finish"]
    follow_up_question: str | None = Field(default=None, min_length=1, max_length=1500)

    @model_validator(mode="after")
    def validate_follow_up(self) -> MockInterviewFollowUpDecision:
        if (self.next_action == "follow_up") != (self.follow_up_question is not None):
            raise ValueError("a follow_up decision needs exactly one follow_up_question")
        return self


class MockInterviewQuestionEvaluationDraft(MockInterviewWorkerContract):
    """End-of-interview assessment of one primary question and its follow-ups."""

    rating: Literal["strong", "adequate", "weak", "insufficient_evidence"]
    summary: str = Field(min_length=1, max_length=1500)
    dimensions: tuple[MockInterviewScoreDimension, ...] = Field(min_length=1, max_length=6)
    strengths: tuple[str, ...] = Field(default=(), max_length=8)
    improvements: tuple[str, ...] = Field(default=(), max_length=8)
    unsupported_claims: tuple[str, ...] = Field(default=(), max_length=5)
    key_facts: tuple[str, ...] = Field(default=(), max_length=8)

    @model_validator(mode="before")
    @classmethod
    def fit_model_output(cls, value: Any) -> Any:
        # Duplicate dimensions would fail the stored evaluation's uniqueness
        # check later; keep the first assessment of each.
        if isinstance(value, dict) and isinstance(value.get("dimensions"), list):
            seen: set[object] = set()
            unique = []
            for item in value["dimensions"]:
                key = item.get("dimension") if isinstance(item, dict) else item
                if key not in seen:
                    seen.add(key)
                    unique.append(item)
            value = {**value, "dimensions": unique}
        return _clip_to_declared_lengths(cls, value)


class MockInterviewReportSynthesisDraft(MockInterviewWorkerContract):
    """What the report model writes; per-question results are assembled by code."""

    summary: str = Field(min_length=1, max_length=3000)
    strengths: tuple[str, ...] = Field(default=(), max_length=10)
    development_areas: tuple[str, ...] = Field(default=(), max_length=10)
    practice_actions: tuple[str, ...] = Field(default=(), max_length=10)
    limitations: tuple[str, ...] = Field(default=(), max_length=10)
    # Facts from different questions that cannot both be true.
    consistency_issues: tuple[str, ...] = Field(default=(), max_length=5)

    @model_validator(mode="before")
    @classmethod
    def fit_model_output(cls, value: Any) -> Any:
        return _clip_to_declared_lengths(cls, value)


class MockInterviewReportDraft(MockInterviewWorkerContract):
    """Observed performance content before the workflow adds report metadata."""

    summary: str = Field(min_length=1, max_length=3000)
    question_results: tuple[MockInterviewQuestionResult, ...] = Field(
        min_length=1,
        max_length=20,
    )
    strengths: tuple[str, ...] = Field(default=(), max_length=10)
    development_areas: tuple[str, ...] = Field(default=(), max_length=10)
    practice_actions: tuple[str, ...] = Field(default=(), max_length=10)
    limitations: tuple[str, ...] = Field(default=(), max_length=10)

    @model_validator(mode="after")
    def require_unique_question_results(self) -> MockInterviewReportDraft:
        item_numbers = [item.plan_item_number for item in self.question_results]
        if len(set(item_numbers)) != len(item_numbers):
            raise ValueError("report question results must be unique by plan item")
        return self


class MockInterviewPlanningWorker(Protocol):
    def plan(
        self,
        *,
        session: MockInterviewSession,
        document: StoredResumeDocument | None,
        context: InterviewPreparationContext,
    ) -> MockInterviewPlanDraft: ...


class MockInterviewInputRoutingWorker(Protocol):
    def route_input(
        self,
        *,
        session: MockInterviewSession,
        turn: MockInterviewTurn,
        user_message: str,
    ) -> MockInterviewInputDecision: ...


class MockInterviewQuestionWorker(Protocol):
    def ask(
        self,
        *,
        session: MockInterviewSession,
        plan: MockInterviewPlan,
        plan_item: MockInterviewPlanItem,
        prior_turns: tuple[MockInterviewTurn, ...] = (),
        document: StoredResumeDocument | None,
        jd_text: str,
        company_name: str = "",
        role_title: str = "",
        confirmed_facts: tuple[ConfirmedResumeFact, ...] = (),
    ) -> MockInterviewQuestionDraft: ...


class MockInterviewFollowUpWorker(Protocol):
    def decide_follow_up(
        self,
        *,
        session: MockInterviewSession,
        plan_item: MockInterviewPlanItem,
        turns: tuple[MockInterviewTurn, ...],
        follow_ups_remaining: int,
        document: StoredResumeDocument | None,
        jd_text: str,
    ) -> MockInterviewFollowUpDecision: ...


class MockInterviewEvaluationWorker(Protocol):
    def evaluate(
        self,
        *,
        session: MockInterviewSession,
        plan_item: MockInterviewPlanItem,
        turns: tuple[MockInterviewTurn, ...],
        document: StoredResumeDocument | None,
        jd_text: str,
        confirmed_facts: tuple[ConfirmedResumeFact, ...] = (),
    ) -> MockInterviewAnswerEvaluation: ...


class MockInterviewReportingWorker(Protocol):
    def report(
        self,
        *,
        session: MockInterviewSession,
        plan: MockInterviewPlan,
        turns: tuple[MockInterviewTurn, ...],
        completion_reason: MockInterviewCompletionReason,
        document: StoredResumeDocument | None,
        jd_text: str,
        confirmed_facts: tuple[ConfirmedResumeFact, ...] = (),
    ) -> MockInterviewReportDraft: ...


class MockInterviewWorker(
    MockInterviewPlanningWorker,
    MockInterviewInputRoutingWorker,
    MockInterviewQuestionWorker,
    MockInterviewFollowUpWorker,
    MockInterviewEvaluationWorker,
    MockInterviewReportingWorker,
    Protocol,
):
    """One implementation may provide every isolated model operation."""
