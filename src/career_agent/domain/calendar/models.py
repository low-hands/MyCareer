from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


CalendarProvider = Literal["google"]
CalendarOperation = Literal["create", "update", "cancel"]
CalendarProposalStatus = Literal[
    "pending", "executed", "failed", "expired", "superseded"
]


class CalendarContract(BaseModel):
    model_config = ConfigDict(
        extra="forbid", frozen=True, str_strip_whitespace=True
    )


class CalendarAccount(CalendarContract):
    id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    provider: CalendarProvider
    email_address: str = Field(min_length=3, max_length=320)
    calendar_id: str = Field(min_length=1, max_length=1000)
    credential_ref: str = Field(min_length=1, max_length=500)
    status: Literal["active", "disabled"] = "active"
    created_at: datetime
    updated_at: datetime


class CalendarEventPayload(CalendarContract):
    title: str = Field(min_length=1, max_length=500)
    description: str = Field(min_length=1, max_length=4000)
    start_at: datetime
    end_at: datetime
    timezone: str = Field(min_length=1, max_length=100)
    location: str | None = Field(default=None, min_length=1, max_length=1000)

    @model_validator(mode="after")
    def validate_schedule(self) -> "CalendarEventPayload":
        if self.start_at.tzinfo is None or self.end_at.tzinfo is None:
            raise ValueError("calendar event timestamps require timezone offsets")
        if self.end_at <= self.start_at:
            raise ValueError("calendar event end_at must be after start_at")
        return self


class CalendarChangeProposal(CalendarContract):
    id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    calendar_account_id: str = Field(min_length=1)
    interview_round_id: str = Field(min_length=1)
    operation: CalendarOperation
    external_event_id: str = Field(min_length=1, max_length=1024)
    payload: CalendarEventPayload | None = None
    payload_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: CalendarProposalStatus
    created_at: datetime
    expires_at: datetime
    executed_at: datetime | None = None
    error_code: str | None = Field(default=None, min_length=1, max_length=200)
    error_detail: str | None = Field(default=None, min_length=1, max_length=2000)

    @model_validator(mode="after")
    def validate_lifecycle(self) -> "CalendarChangeProposal":
        if self.operation in {"create", "update"} and self.payload is None:
            raise ValueError("calendar create and update require a payload")
        if self.expires_at <= self.created_at:
            raise ValueError("calendar proposal must expire after creation")
        if self.status == "executed" and self.executed_at is None:
            raise ValueError("executed calendar proposal requires executed_at")
        if self.status != "executed" and self.executed_at is not None:
            raise ValueError("only executed calendar proposal has executed_at")
        if self.status == "failed" and self.error_code is None:
            raise ValueError("failed calendar proposal requires error_code")
        return self


class CalendarEventLink(CalendarContract):
    id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    calendar_account_id: str = Field(min_length=1)
    interview_round_id: str = Field(min_length=1)
    external_event_id: str = Field(min_length=1, max_length=1024)
    external_etag: str | None = Field(default=None, min_length=1, max_length=1000)
    external_html_link: str | None = Field(default=None, min_length=1, max_length=2000)
    status: Literal["active", "cancelled"]
    last_payload_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    created_at: datetime
    updated_at: datetime


class CalendarChangeEvent(CalendarContract):
    id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    proposal_id: str = Field(min_length=1)
    interview_round_id: str = Field(min_length=1)
    event_type: Literal[
        "proposed", "executed", "failed", "expired", "superseded"
    ]
    occurred_at: datetime
    detail: str | None = Field(default=None, min_length=1, max_length=2000)
