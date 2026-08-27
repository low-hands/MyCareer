from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from career_agent.agent.resume_job_match_contracts import ConfirmedResumeFact
from career_agent.domain.mock_interviews import (
    MockInterviewAnswerEvaluation,
    MockInterviewPlan,
    MockInterviewPlanItem,
    MockInterviewQuestionResult,
    MockInterviewReport,
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


class MockInterviewWorkerContract(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )


class MockInterviewStartRequest(MockInterviewWorkerContract):
    user_id: str = Field(min_length=1)
    application_id: str = Field(min_length=1)
    job_posting_id: str = Field(min_length=1)
    jd_snapshot_id: str = Field(min_length=1)
    resume_version_id: str = Field(min_length=1)
    interview_type: MockInterviewType
    interview_round_id: str | None = Field(default=None, min_length=1)
    max_primary_questions: int = Field(default=6, ge=1, le=20)
    max_follow_ups_per_question: int = Field(default=2, ge=0, le=5)


class MockInterviewGraphResult(MockInterviewWorkerContract):
    session_id: str = Field(min_length=1)
    state: Literal["awaiting_answer", "running", "completed", "cancelled"]
    message: str = Field(min_length=1)
    turn_id: str | None = Field(default=None, min_length=1)
    question: str | None = Field(default=None, min_length=1, max_length=2000)
    evaluation: MockInterviewAnswerEvaluation | None = None
    report_id: str | None = Field(default=None, min_length=1)
    report: MockInterviewReport | None = None


class MockInterviewPlanDraft(MockInterviewWorkerContract):
    """Model-authored plan content before the workflow adds identity and time."""

    summary: str = Field(min_length=1, max_length=2000)
    items: tuple[MockInterviewPlanItem, ...] = Field(min_length=1, max_length=20)
    limitations: tuple[str, ...] = Field(default=(), max_length=10)

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
        document: StoredResumeDocument,
        jd_text: str,
        confirmed_facts: tuple[ConfirmedResumeFact, ...] = (),
    ) -> MockInterviewPlanDraft: ...


class MockInterviewQuestionWorker(Protocol):
    def ask(
        self,
        *,
        session: MockInterviewSession,
        plan: MockInterviewPlan,
        plan_item: MockInterviewPlanItem,
        prior_turns: tuple[MockInterviewTurn, ...] = (),
        document: StoredResumeDocument,
        jd_text: str,
        confirmed_facts: tuple[ConfirmedResumeFact, ...] = (),
    ) -> MockInterviewQuestionDraft: ...


class MockInterviewEvaluationWorker(Protocol):
    def evaluate(
        self,
        *,
        session: MockInterviewSession,
        plan_item: MockInterviewPlanItem,
        turn: MockInterviewTurn,
        prior_turns: tuple[MockInterviewTurn, ...] = (),
        document: StoredResumeDocument,
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
        document: StoredResumeDocument,
        jd_text: str,
        confirmed_facts: tuple[ConfirmedResumeFact, ...] = (),
    ) -> MockInterviewReportDraft: ...


class MockInterviewWorker(
    MockInterviewPlanningWorker,
    MockInterviewQuestionWorker,
    MockInterviewEvaluationWorker,
    MockInterviewReportingWorker,
    Protocol,
):
    """One implementation may provide all four isolated model operations."""
