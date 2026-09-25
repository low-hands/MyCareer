from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


ApplicationStatus = Literal[
    "submitted",
    "acknowledged",
    "interviewing",
    "interview_completed",
    "offer",
    "rejected",
    "withdrawn",
]
TerminalApplicationStatus = Literal["offer", "rejected", "withdrawn"]


class ApplicationContract(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )


class Application(ApplicationContract):
    id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    job_posting_id: str = Field(min_length=1)
    jd_snapshot_id: str = Field(min_length=1)
    resume_version_id: str | None = Field(default=None, min_length=1)
    status: ApplicationStatus
    submitted_at: datetime
    created_at: datetime
    updated_at: datetime


class ApplicationEvent(ApplicationContract):
    id: str = Field(min_length=1)
    application_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    source: Literal["user_reported", "email_sync", "system"]
    event_type: Literal["created", "status_changed", "note_added", "resume_version_changed"]
    previous_status: ApplicationStatus | None = None
    new_status: ApplicationStatus
    note: str | None = Field(default=None, min_length=1, max_length=2000)
    occurred_at: datetime

    @model_validator(mode="after")
    def validate_event(self) -> "ApplicationEvent":
        if self.event_type == "created" and self.previous_status is not None:
            raise ValueError("created events cannot have a previous status")
        if self.event_type == "status_changed" and (
            self.previous_status is None or self.previous_status == self.new_status
        ):
            raise ValueError("status changes require two different statuses")
        if self.event_type == "note_added" and (
            self.previous_status != self.new_status or self.note is None
        ):
            raise ValueError("note events require an unchanged status and a note")
        if self.event_type == "resume_version_changed" and self.previous_status != self.new_status:
            raise ValueError("resume version events require an unchanged status")
        return self
