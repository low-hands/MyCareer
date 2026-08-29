from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


InterviewStatus = Literal["identified", "scheduled", "completed", "cancelled"]
InterviewChangeType = Literal[
    "invited",
    "rescheduled",
    "details_updated",
    "cancelled",
]
InterviewSelfAssessment = Literal["strong", "mixed", "weak", "uncertain"]


class InterviewContract(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )


class InterviewDetails(InterviewContract):
    change_type: InterviewChangeType = "invited"
    employer_label: str | None = Field(default=None, min_length=1, max_length=200)
    scheduled_start: datetime | None = None
    scheduled_end: datetime | None = None
    timezone: str | None = Field(default=None, min_length=1, max_length=100)
    interview_format: Literal["video", "phone", "onsite", "unknown"] = "unknown"
    location: str | None = Field(default=None, min_length=1, max_length=1000)
    meeting_url: str | None = Field(default=None, min_length=1, max_length=2000)
    contact_summary: str | None = Field(default=None, min_length=1, max_length=500)

    @model_validator(mode="after")
    def validate_schedule(self) -> "InterviewDetails":
        if self.scheduled_end is not None and self.scheduled_start is None:
            raise ValueError("scheduled_end requires scheduled_start")
        if (
            self.scheduled_start is not None
            and self.scheduled_end is not None
        ):
            start_aware = self.scheduled_start.tzinfo is not None
            end_aware = self.scheduled_end.tzinfo is not None
            if start_aware != end_aware:
                raise ValueError("schedule datetimes must use compatible timezone forms")
            if self.scheduled_end <= self.scheduled_start:
                raise ValueError("scheduled_end must be after scheduled_start")
        if (
            self.scheduled_start is not None
            and self.scheduled_start.tzinfo is None
            and self.timezone is None
        ):
            raise ValueError("naive scheduled_start requires timezone")
        return self


class InterviewRound(InterviewContract):
    id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    application_id: str = Field(min_length=1)
    sequence_number: int = Field(ge=1)
    employer_label: str | None = Field(default=None, min_length=1, max_length=200)
    status: InterviewStatus
    scheduled_start: datetime | None = None
    scheduled_end: datetime | None = None
    timezone: str | None = Field(default=None, min_length=1, max_length=100)
    interview_format: Literal["video", "phone", "onsite", "unknown"] = "unknown"
    location: str | None = Field(default=None, min_length=1, max_length=1000)
    meeting_url: str | None = Field(default=None, min_length=1, max_length=2000)
    contact_summary: str | None = Field(default=None, min_length=1, max_length=500)
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None = None

    @model_validator(mode="after")
    def validate_lifecycle(self) -> "InterviewRound":
        if self.status == "scheduled" and self.scheduled_start is None:
            raise ValueError("scheduled interviews require scheduled_start")
        if self.status == "completed" and self.completed_at is None:
            raise ValueError("completed interviews require completed_at")
        if self.status != "completed" and self.completed_at is not None:
            raise ValueError("only completed interviews can have completed_at")
        return self


class InterviewRoundEvent(InterviewContract):
    id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    application_id: str = Field(min_length=1)
    interview_round_id: str = Field(min_length=1)
    source: Literal["email_sync", "user_reported", "system"]
    event_type: Literal[
        "created",
        "rescheduled",
        "details_updated",
        "cancelled",
        "completed",
        "corrected",
    ]
    email_event_id: str | None = Field(default=None, min_length=1)
    source_thread_id: str | None = Field(default=None, min_length=1)
    details: InterviewDetails
    occurred_at: datetime


class InterviewRetroQuestion(InterviewContract):
    question: str = Field(min_length=1, max_length=2000)
    answer_summary: str | None = Field(default=None, min_length=1, max_length=5000)
    self_assessment: InterviewSelfAssessment = "uncertain"
    notes: str | None = Field(default=None, min_length=1, max_length=2000)


class InterviewRetroReport(InterviewContract):
    """A versioned post-interview report grounded only in the user's recollection."""

    id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    application_id: str = Field(min_length=1)
    interview_round_id: str = Field(min_length=1)
    source_notes: str = Field(min_length=1, max_length=20_000)
    summary: str = Field(min_length=1, max_length=5000)
    questions: tuple[InterviewRetroQuestion, ...] = Field(default=(), max_length=30)
    strengths: tuple[str, ...] = Field(default=(), max_length=20)
    difficulties: tuple[str, ...] = Field(default=(), max_length=20)
    interviewer_signals: tuple[str, ...] = Field(default=(), max_length=20)
    next_focus: tuple[str, ...] = Field(default=(), max_length=20)
    action_items: tuple[str, ...] = Field(default=(), max_length=20)
    limitations: tuple[str, ...] = Field(default=(), max_length=20)
    self_assessment: InterviewSelfAssessment = "uncertain"
    content_sha256: str = Field(min_length=64, max_length=64)
    created_at: datetime
