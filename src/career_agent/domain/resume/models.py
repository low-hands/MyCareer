from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ResumeContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TargetRole(ResumeContract):
    id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    priority: int = Field(ge=0)
    status: Literal["active"] = "active"
    created_at: datetime
    updated_at: datetime


class Resume(ResumeContract):
    id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    target_role_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    status: Literal["active"] = "active"
    latest_version_id: str = Field(min_length=1)
    created_at: datetime
    updated_at: datetime


class ResumeVersion(ResumeContract):
    id: str = Field(min_length=1)
    resume_id: str = Field(min_length=1)
    version_number: int = Field(ge=1)
    source_type: Literal["user_import", "agent_tailoring"] = "user_import"
    document_format: Literal["pdf", "text", "markdown"]
    content_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    byte_size: int = Field(ge=1)
    created_at: datetime
