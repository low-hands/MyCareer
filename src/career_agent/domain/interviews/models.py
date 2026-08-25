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
