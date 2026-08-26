from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


MockInterviewType = Literal[
    "technical",
    "role_specific",
    "behavioral",
    "hr",
    "mixed",
]
MockInterviewStatus = Literal[
    "created",
    "active",
    "paused",
    "completed",
    "cancelled",
]
MockInterviewDifficulty = Literal["introductory", "intermediate", "advanced"]
MockInterviewQuestionType = Literal[
    "introduction",
    "knowledge",
    "problem_solving",
    "system_design",
    "project_deep_dive",
    "role_scenario",
    "behavioral",
    "motivation",
    "career_planning",
]


class MockInterviewContract(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )


class MockInterviewSession(MockInterviewContract):
    id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    application_id: str = Field(min_length=1)
    interview_round_id: str | None = Field(default=None, min_length=1)
    job_posting_id: str = Field(min_length=1)
    jd_snapshot_id: str = Field(min_length=1)
    resume_version_id: str = Field(min_length=1)
    interview_type: MockInterviewType
    status: MockInterviewStatus = "created"
    max_primary_questions: int = Field(default=6, ge=1, le=20)
    max_follow_ups_per_question: int = Field(default=2, ge=0, le=5)
    current_plan_item: int = Field(default=0, ge=0)
    current_turn_id: str | None = Field(default=None, min_length=1)
    created_at: datetime
    started_at: datetime | None = None
    paused_at: datetime | None = None
    completed_at: datetime | None = None
    updated_at: datetime

    @model_validator(mode="after")
    def validate_lifecycle(self) -> MockInterviewSession:
        if self.updated_at < self.created_at:
            raise ValueError("updated_at cannot precede created_at")
        if self.started_at is not None and self.started_at < self.created_at:
            raise ValueError("started_at cannot precede created_at")
        if (
            self.paused_at is not None
            and self.started_at is not None
            and self.paused_at < self.started_at
        ):
            raise ValueError("paused_at cannot precede started_at")
        if (
            self.completed_at is not None
            and self.started_at is not None
            and self.completed_at < self.started_at
        ):
            raise ValueError("completed_at cannot precede started_at")
        if self.current_plan_item > self.max_primary_questions:
            raise ValueError("current_plan_item exceeds max_primary_questions")
        if self.status in {"active", "paused", "completed"} and self.started_at is None:
            raise ValueError("started, paused, and completed sessions require started_at")
        if self.status == "paused" and self.paused_at is None:
            raise ValueError("paused sessions require paused_at")
        if self.status != "paused" and self.paused_at is not None:
            raise ValueError("only paused sessions can have paused_at")
        if self.status == "completed" and self.completed_at is None:
            raise ValueError("completed sessions require completed_at")
        if self.status != "completed" and self.completed_at is not None:
            raise ValueError("only completed sessions can have completed_at")
        if self.status != "active" and self.current_turn_id is not None:
            raise ValueError("only active sessions can await a current turn")
        return self


class MockInterviewPlanItem(MockInterviewContract):
    sequence_number: int = Field(ge=1)
    question_type: MockInterviewQuestionType
    difficulty: MockInterviewDifficulty
    focus: str = Field(min_length=1, max_length=500)
    rationale: str = Field(min_length=1, max_length=1000)
    jd_quotes: tuple[str, ...] = Field(default=(), max_length=3)
    resume_locators: tuple[str, ...] = Field(default=(), max_length=3)
    resume_quotes: tuple[str, ...] = Field(default=(), max_length=3)

    @model_validator(mode="after")
    def align_resume_evidence(self) -> MockInterviewPlanItem:
        if len(self.resume_locators) != len(self.resume_quotes):
            raise ValueError("resume locators and quotes must have equal length")
        return self


class MockInterviewPlan(MockInterviewContract):
    session_id: str = Field(min_length=1)
    summary: str = Field(min_length=1, max_length=2000)
    items: tuple[MockInterviewPlanItem, ...] = Field(min_length=1, max_length=20)
    limitations: tuple[str, ...] = Field(default=(), max_length=10)
    created_at: datetime

    @model_validator(mode="after")
    def require_contiguous_sequence(self) -> MockInterviewPlan:
        expected = tuple(range(1, len(self.items) + 1))
        actual = tuple(item.sequence_number for item in self.items)
        if actual != expected:
            raise ValueError("plan item sequence numbers must be contiguous and ordered")
        return self


class MockInterviewScoreDimension(MockInterviewContract):
    dimension: Literal[
        "accuracy",
        "relevance",
        "specificity",
        "structure",
        "reasoning",
        "communication",
    ]
    score: int = Field(ge=1, le=5)
    feedback: str = Field(min_length=1, max_length=1000)


class MockInterviewAnswerEvaluation(MockInterviewContract):
    rating: Literal["strong", "adequate", "weak", "insufficient_evidence"]
    summary: str = Field(min_length=1, max_length=1500)
    dimensions: tuple[MockInterviewScoreDimension, ...] = Field(
        min_length=1, max_length=6
    )
    strengths: tuple[str, ...] = Field(default=(), max_length=8)
    improvements: tuple[str, ...] = Field(default=(), max_length=8)
    unsupported_claims: tuple[str, ...] = Field(default=(), max_length=5)
    next_action: Literal["follow_up", "next_question", "finish"]
    next_action_reason: str = Field(min_length=1, max_length=1000)
    follow_up_question: str | None = Field(default=None, min_length=1, max_length=1500)

    @model_validator(mode="after")
    def validate_follow_up(self) -> MockInterviewAnswerEvaluation:
        if self.next_action == "follow_up" and self.follow_up_question is None:
            raise ValueError("follow_up action requires a follow_up_question")
        if self.next_action != "follow_up" and self.follow_up_question is not None:
            raise ValueError("only follow_up action can include a follow_up_question")
        dimensions = [item.dimension for item in self.dimensions]
        if len(set(dimensions)) != len(dimensions):
            raise ValueError("evaluation dimensions must be unique")
        return self


class MockInterviewTurn(MockInterviewContract):
    id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    sequence_number: int = Field(ge=1)
    plan_item_number: int = Field(ge=1)
    turn_type: Literal["primary", "follow_up"]
    parent_turn_id: str | None = Field(default=None, min_length=1)
    question_type: MockInterviewQuestionType
    question: str = Field(min_length=1, max_length=2000)
    answer: str | None = Field(default=None, min_length=1, max_length=20_000)
    evaluation: MockInterviewAnswerEvaluation | None = None
    status: Literal["awaiting_answer", "evaluated"] = "awaiting_answer"
    asked_at: datetime
    answered_at: datetime | None = None
    evaluated_at: datetime | None = None

    @model_validator(mode="after")
    def validate_turn(self) -> MockInterviewTurn:
        if self.turn_type == "follow_up" and self.parent_turn_id is None:
            raise ValueError("follow-up turns require parent_turn_id")
        if self.turn_type == "primary" and self.parent_turn_id is not None:
            raise ValueError("primary turns cannot have parent_turn_id")
        if self.status == "awaiting_answer":
            if any(
                value is not None
                for value in (
                    self.answer,
                    self.evaluation,
                    self.answered_at,
                    self.evaluated_at,
                )
            ):
                raise ValueError("awaiting turns cannot contain answer or evaluation data")
        else:
            if any(
                value is None
                for value in (
                    self.answer,
                    self.evaluation,
                    self.answered_at,
                    self.evaluated_at,
                )
            ):
                raise ValueError("evaluated turns require answer, evaluation, and timestamps")
            if self.answered_at < self.asked_at:
                raise ValueError("answered_at cannot precede asked_at")
            if self.evaluated_at < self.answered_at:
                raise ValueError("evaluated_at cannot precede answered_at")
        return self


class MockInterviewQuestionResult(MockInterviewContract):
    plan_item_number: int = Field(ge=1)
    question: str = Field(min_length=1, max_length=2000)
    final_rating: Literal["strong", "adequate", "weak", "insufficient_evidence"]
    summary: str = Field(min_length=1, max_length=1500)
    follow_up_count: int = Field(ge=0, le=5)


class MockInterviewReport(MockInterviewContract):
    id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    completion_reason: Literal[
        "plan_completed",
        "user_ended",
        "time_limit",
        "safety_stop",
    ]
    summary: str = Field(min_length=1, max_length=3000)
    question_results: tuple[MockInterviewQuestionResult, ...] = Field(
        min_length=1, max_length=20
    )
    strengths: tuple[str, ...] = Field(default=(), max_length=10)
    development_areas: tuple[str, ...] = Field(default=(), max_length=10)
    practice_actions: tuple[str, ...] = Field(default=(), max_length=10)
    limitations: tuple[str, ...] = Field(default=(), max_length=10)
    created_at: datetime

    @model_validator(mode="after")
    def require_unique_question_results(self) -> MockInterviewReport:
        item_numbers = [item.plan_item_number for item in self.question_results]
        if len(set(item_numbers)) != len(item_numbers):
            raise ValueError("report question results must be unique by plan item")
        return self
