from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class BodyReference(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SavedJobBodySource(BodyReference):
    kind: Literal["saved_job"] = "saved_job"
    job_posting_id: str = Field(min_length=1)


class MockInterviewBodySource(BodyReference):
    kind: Literal["mock_interview_question"] = "mock_interview_question"
    session_id: str = Field(min_length=1)
    question_number: int = Field(ge=1)


DeliveredBodySource = Annotated[
    SavedJobBodySource | MockInterviewBodySource,
    Field(discriminator="kind"),
]


class BodyDependency(BodyReference):
    kind: Literal["job", "application", "interview_round", "email_event"]
    resource_id: str = Field(min_length=1)
