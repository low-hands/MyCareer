from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class BodyReference(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SavedJobBodySource(BodyReference):
    kind: Literal["saved_job"] = "saved_job"
    job_posting_id: str = Field(min_length=1)


class ResumeAnalysisBodySource(BodyReference):
    kind: Literal["resume_analysis"] = "resume_analysis"
    analysis_id: str = Field(min_length=1)
    expires_at: datetime

    @field_validator("expires_at")
    @classmethod
    def timezone_required(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("expires_at must be timezone-aware")
        return value


class MockInterviewBodySource(BodyReference):
    kind: Literal["mock_interview_question"] = "mock_interview_question"
    session_id: str = Field(min_length=1)
    question_number: int = Field(ge=1)


DeliveredBodySource = Annotated[
    SavedJobBodySource | ResumeAnalysisBodySource | MockInterviewBodySource,
    Field(discriminator="kind"),
]


class BodyDependency(BodyReference):
    kind: Literal["job", "application", "interview_round", "email_event"]
    resource_id: str = Field(min_length=1)
