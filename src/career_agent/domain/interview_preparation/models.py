from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class InterviewPreparationContract(BaseModel):
    model_config = ConfigDict(
        extra="forbid", frozen=True, str_strip_whitespace=True
    )


class InterviewFocusArea(InterviewPreparationContract):
    topic: str = Field(min_length=1, max_length=300)
    priority: Literal["high", "medium", "low"]
    rationale: str = Field(min_length=1, max_length=1000)
    jd_quote: str = Field(min_length=1, max_length=500)


class EvidenceStory(InterviewPreparationContract):
    theme: str = Field(min_length=1, max_length=300)
    resume_locator: str = Field(min_length=1, max_length=500)
    resume_quote: str = Field(min_length=1, max_length=500)
    preparation_prompt: str = Field(min_length=1, max_length=1000)


class LikelyQuestion(InterviewPreparationContract):
    question: str = Field(min_length=1, max_length=1000)
    rationale: str = Field(min_length=1, max_length=1000)
    answer_outline: tuple[str, ...] = Field(default=(), max_length=8)
    follow_ups: tuple[str, ...] = Field(default=(), max_length=5)


class GapPreparation(InterviewPreparationContract):
    gap: str = Field(min_length=1, max_length=500)
    jd_quote: str = Field(min_length=1, max_length=500)
    honest_response_strategy: str = Field(min_length=1, max_length=1500)


class QuestionToAsk(InterviewPreparationContract):
    question: str = Field(min_length=1, max_length=1000)
    rationale: str = Field(min_length=1, max_length=1000)


class InterviewPreparationResult(InterviewPreparationContract):
    summary: str = Field(min_length=1, max_length=2000)
    focus_areas: tuple[InterviewFocusArea, ...] = Field(default=(), max_length=12)
    evidence_stories: tuple[EvidenceStory, ...] = Field(default=(), max_length=10)
    likely_questions: tuple[LikelyQuestion, ...] = Field(default=(), max_length=20)
    gaps: tuple[GapPreparation, ...] = Field(default=(), max_length=10)
    questions_to_ask: tuple[QuestionToAsk, ...] = Field(default=(), max_length=10)
    checklist: tuple[str, ...] = Field(default=(), max_length=20)
    limitations: tuple[str, ...] = Field(default=(), max_length=10)
