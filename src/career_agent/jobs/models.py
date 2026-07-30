"""Source-grounded domain models for job postings."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, field_validator, model_validator

JobDetailLevel = Literal["summary", "full"]


class JobPosting(BaseModel):
    """One concrete vacancy retrieved from an external or manual source."""

    job_id: str
    tenant_id: str

    source: str
    external_id: str = ""
    detail_locator: str = ""
    source_url: str = ""

    title: str
    company_name: str
    location: str = ""
    employment_type: str = ""
    salary: str = ""
    experience_required: str = ""
    education_required: str = ""
    company_industry: str = ""
    company_size: str = ""
    company_stage: str = ""
    description: str = ""

    detail_level: JobDetailLevel = "summary"
    fetched_at: datetime
    created_at: datetime
    updated_at: datetime

    @field_validator("job_id", "tenant_id", "source", "title", "company_name")
    @classmethod
    def _required_text_must_not_be_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must not be blank")
        return normalized

    @model_validator(mode="after")
    def _full_posting_requires_description(self) -> JobPosting:
        if self.detail_level == "full" and not self.description.strip():
            raise ValueError("full job posting requires a description")
        return self
