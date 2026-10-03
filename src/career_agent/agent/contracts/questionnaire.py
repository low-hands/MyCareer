"""Bounded questions and answers for one resumable task clarification."""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class QuestionnaireContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class QuestionOption(QuestionnaireContract):
    value: str = Field(pattern=r"^[a-z][a-z0-9_]{0,39}$")
    label: str = Field(min_length=1, max_length=100)
    meaning: Literal["choice", "none", "other"] = "choice"


class UserQuestion(QuestionnaireContract):
    question_id: str = Field(pattern=r"^q[1-8]$")
    prompt: str = Field(min_length=1, max_length=500)
    kind: Literal["single", "multiple", "free_text"]
    options: tuple[QuestionOption, ...] = Field(default=(), max_length=12)
    allow_free_text: bool = False
    allow_skip: bool = True

    @model_validator(mode="after")
    def validate_options(self) -> "UserQuestion":
        if self.kind == "free_text" and self.options:
            raise ValueError("free_text question cannot have options")
        if self.kind != "free_text" and not self.options:
            raise ValueError("selection question requires options")
        values = [option.value for option in self.options]
        if len(values) != len(set(values)):
            raise ValueError("duplicate option value")
        if sum(option.meaning == "none" for option in self.options) > 1:
            raise ValueError("multiple none options")
        if sum(option.meaning == "other" for option in self.options) > 1:
            raise ValueError("multiple other options")
        return self


class QuestionAnswer(QuestionnaireContract):
    question_id: str = Field(pattern=r"^q[1-8]$")
    selected_values: tuple[str, ...] = Field(default=(), max_length=12)
    free_text: str | None = Field(default=None, max_length=1000)
    skipped: bool = False

    @model_validator(mode="after")
    def validate_shape(self) -> "QuestionAnswer":
        if self.skipped and (self.selected_values or self.free_text):
            raise ValueError("skipped answer cannot contain content")
        if len(self.selected_values) != len(set(self.selected_values)):
            raise ValueError("duplicate selected option")
        return self


class PendingQuestionnaire(QuestionnaireContract):
    interaction_id: str = Field(pattern=r"^interaction_[a-f0-9]{20}$")
    prompt: str = Field(min_length=1, max_length=5000)
    questions: tuple[UserQuestion, ...] = Field(min_length=2, max_length=8)
    created_at: datetime
    expires_at: datetime
    active_workflow: str
    resume_version_id: str | None = None
    job_posting_id: str | None = None
    jd_snapshot_id: str | None = None

    @model_validator(mode="after")
    def validate_questions(self) -> "PendingQuestionnaire":
        ids = tuple(question.question_id for question in self.questions)
        if ids != tuple(f"q{index}" for index in range(1, len(ids) + 1)):
            raise ValueError("question ids must be contiguous and ordered")
        if self.expires_at <= self.created_at:
            raise ValueError("questionnaire expiry must follow creation")
        return self

    def validate_answers(self, answers: tuple[QuestionAnswer, ...]) -> None:
        if tuple(answer.question_id for answer in answers) != tuple(
            question.question_id for question in self.questions
        ):
            raise ValueError("answers must match questions once, in order")
        for question, answer in zip(self.questions, answers):
            if answer.skipped:
                if not question.allow_skip:
                    raise ValueError("required question skipped")
                continue
            allowed = {option.value: option for option in question.options}
            if any(value not in allowed for value in answer.selected_values):
                raise ValueError("unknown option")
            if question.kind == "free_text":
                if answer.selected_values or not answer.free_text:
                    raise ValueError("free_text answer required")
                continue
            elif question.kind == "single":
                if len(answer.selected_values) != 1:
                    raise ValueError("single answer requires one option")
            elif not answer.selected_values:
                raise ValueError("multiple answer requires an option")
            if any(allowed[value].meaning == "none" for value in answer.selected_values):
                if len(answer.selected_values) != 1 or answer.free_text:
                    raise ValueError("none is exclusive")
            needs_text = any(allowed[value].meaning == "other" for value in answer.selected_values)
            if answer.free_text and not (question.allow_free_text or needs_text):
                raise ValueError("free text not allowed")
            if needs_text and not answer.free_text:
                raise ValueError("other requires text")
