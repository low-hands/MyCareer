from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class ResumeArtifactReference(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    resume_version_id: str = Field(min_length=1)
    filename: str = Field(min_length=1, max_length=180)
    media_type: str = Field(min_length=1, max_length=100)
    byte_size: int = Field(ge=1)
    created_at: datetime


class ResumeArtifactDelivery(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    reference: ResumeArtifactReference
    content: bytes = Field(min_length=1)
