from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from career_agent.domain.interviews import InterviewDetails


EmailProvider = Literal["gmail", "qq"]
EmailEventType = Literal[
    "acknowledgement",
    "interview_invitation",
    "rejection",
    "offer",
    "material_request",
    "unclear",
]
EmailEventStatus = Literal["pending_confirmation", "applied", "dismissed"]


class EmailTrackingContract(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )


class EmailAccount(EmailTrackingContract):
    id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    provider: EmailProvider
    email_address: str = Field(min_length=3)
    credential_ref: str = Field(min_length=1)
    status: Literal["active", "needs_reauthorization", "disabled"] = "active"
    created_at: datetime
    updated_at: datetime


class EmailSyncCursor(EmailTrackingContract):
    account_id: str = Field(min_length=1)
    cursor_type: Literal["gmail_history_id", "imap_uid"]
    value: str = Field(min_length=1)
    uid_validity: str | None = Field(default=None, min_length=1)
    updated_at: datetime

    @model_validator(mode="after")
    def validate_provider_cursor(self) -> "EmailSyncCursor":
        if self.cursor_type == "gmail_history_id" and self.uid_validity is not None:
            raise ValueError("Gmail history cursors cannot have UIDVALIDITY")
        return self


class EmailMessage(EmailTrackingContract):
    id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    account_id: str = Field(min_length=1)
    provider: EmailProvider
    external_message_id: str = Field(min_length=1)
    external_thread_id: str | None = Field(default=None, min_length=1)
    sender: str = Field(min_length=1, max_length=1000)
    subject: str = Field(min_length=1, max_length=2000)
    received_at: datetime
    content_sha256: str | None = Field(default=None, min_length=64, max_length=64)
    encrypted_content_ref: str | None = Field(default=None, min_length=1)
    application_id: str | None = Field(default=None, min_length=1)
    classification: EmailEventType | None = None
    candidate: bool
    created_at: datetime


class EmailEvent(EmailTrackingContract):
    id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    email_message_id: str = Field(min_length=1)
    application_id: str | None = Field(default=None, min_length=1)
    event_type: EmailEventType
    status: EmailEventStatus
    confidence: float = Field(ge=0.0, le=1.0)
    classifier: str = Field(min_length=1, max_length=200)
    summary: str = Field(min_length=1, max_length=2000)
    interview_details: InterviewDetails | None = None
    occurred_at: datetime
    created_at: datetime
    resolved_at: datetime | None = None

    @model_validator(mode="after")
    def validate_resolution(self) -> "EmailEvent":
        if self.status == "pending_confirmation" and self.resolved_at is not None:
            raise ValueError("pending events cannot be resolved")
        if self.status != "pending_confirmation" and self.resolved_at is None:
            raise ValueError("resolved events require resolved_at")
        if self.status == "applied" and self.application_id is None:
            raise ValueError("applied events require an application")
        if self.interview_details is not None and self.event_type != "interview_invitation":
            raise ValueError("interview details require an interview invitation event")
        return self


class RemoteEmailMetadata(EmailTrackingContract):
    external_message_id: str = Field(min_length=1)
    external_thread_id: str | None = Field(default=None, min_length=1)
    sender: str = Field(min_length=1)
    subject: str = Field(min_length=1)
    received_at: datetime


class RemoteEmailContent(EmailTrackingContract):
    external_message_id: str = Field(min_length=1)
    text: str = Field(min_length=1)


class EmailSyncBatch(EmailTrackingContract):
    messages: tuple[RemoteEmailMetadata, ...] = ()
    next_cursor_value: str = Field(min_length=1)
    uid_validity: str | None = Field(default=None, min_length=1)


class ApplicationEmailCandidate(EmailTrackingContract):
    application_id: str = Field(min_length=1)
    company_name: str = Field(min_length=1)
    job_title: str = Field(min_length=1)
    status: str = Field(min_length=1)
    submitted_at: datetime


class EmailAssessment(EmailTrackingContract):
    event_type: EmailEventType
    application_id: str | None = Field(default=None, min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    summary: str = Field(min_length=1, max_length=2000)
    interview_details: InterviewDetails | None = None

    @model_validator(mode="after")
    def validate_interview_details(self) -> "EmailAssessment":
        if self.interview_details is not None and self.event_type != "interview_invitation":
            raise ValueError("interview details require an interview invitation event")
        return self
