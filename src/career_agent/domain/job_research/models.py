from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


ResearchEvidenceType = Literal["fact", "inference", "unknown"]
ResearchConfidence = Literal["high", "medium", "low"]
JobResearchRunStatus = Literal["running", "completed", "failed", "cancelled"]
JobResearchReportStatus = Literal["current", "outdated", "superseded"]


class JobResearchContract(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )


class JobResearchScope(JobResearchContract):
    focus: str | None = Field(default=None, min_length=1, max_length=1000)
    user_provided_context: str | None = Field(
        default=None,
        min_length=1,
        max_length=4000,
    )
    max_sources: int = Field(default=8, ge=2, le=15)


class JobResearchSourceDraft(JobResearchContract):
    source_key: str = Field(pattern=r"^[A-Z][A-Z0-9_-]{0,15}$")
    url: str = Field(min_length=1, max_length=3000)
    title: str = Field(min_length=1, max_length=1000)
    publisher: str | None = Field(default=None, min_length=1, max_length=300)
    published_at: date | None = None
    relevant_excerpt: str = Field(min_length=1, max_length=2000)


class JobResearchFindingDraft(JobResearchContract):
    topic: str = Field(min_length=1, max_length=300)
    statement: str = Field(min_length=1, max_length=2000)
    evidence_type: ResearchEvidenceType
    source_keys: tuple[str, ...] = Field(default=(), max_length=8)
    confidence: ResearchConfidence

    @model_validator(mode="after")
    def require_citations_for_supported_claims(self) -> JobResearchFindingDraft:
        if self.evidence_type in {"fact", "inference"} and not self.source_keys:
            raise ValueError("facts and inferences require at least one source key")
        if self.evidence_type == "unknown" and self.source_keys:
            raise ValueError("unknown findings cannot cite sources as support")
        if len(set(self.source_keys)) != len(self.source_keys):
            raise ValueError("finding source keys must be unique")
        return self


class JobResearchDraft(JobResearchContract):
    summary: str = Field(min_length=1, max_length=4000)
    sources: tuple[JobResearchSourceDraft, ...] = Field(min_length=1, max_length=15)
    findings: tuple[JobResearchFindingDraft, ...] = Field(
        min_length=1,
        max_length=40,
    )
    open_questions: tuple[str, ...] = Field(default=(), max_length=15)
    limitations: tuple[str, ...] = Field(default=(), max_length=15)

    @model_validator(mode="after")
    def require_unique_source_keys(self) -> JobResearchDraft:
        keys = tuple(source.source_key for source in self.sources)
        if len(set(keys)) != len(keys):
            raise ValueError("research source keys must be unique")
        return self


class JobResearchSource(JobResearchContract):
    id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    source_key: str = Field(pattern=r"^[A-Z][A-Z0-9_-]{0,15}$")
    url: str = Field(min_length=1, max_length=3000)
    normalized_url: str = Field(min_length=1, max_length=3000)
    title: str = Field(min_length=1, max_length=1000)
    publisher: str | None = Field(default=None, min_length=1, max_length=300)
    published_at: date | None = None
    retrieved_at: datetime
    relevant_excerpt: str = Field(min_length=1, max_length=2000)
    content_sha256: str = Field(min_length=64, max_length=64)


class JobResearchFinding(JobResearchContract):
    topic: str = Field(min_length=1, max_length=300)
    statement: str = Field(min_length=1, max_length=2000)
    evidence_type: ResearchEvidenceType
    source_keys: tuple[str, ...] = Field(default=(), max_length=8)
    confidence: ResearchConfidence


class JobResearchReport(JobResearchContract):
    id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    job_posting_id: str = Field(min_length=1)
    jd_snapshot_id: str = Field(min_length=1)
    status: JobResearchReportStatus
    scope: JobResearchScope
    summary: str = Field(min_length=1, max_length=4000)
    findings: tuple[JobResearchFinding, ...] = Field(min_length=1, max_length=40)
    open_questions: tuple[str, ...] = Field(default=(), max_length=15)
    limitations: tuple[str, ...] = Field(default=(), max_length=15)
    created_at: datetime


class JobResearchRun(JobResearchContract):
    id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    job_posting_id: str = Field(min_length=1)
    jd_snapshot_id: str = Field(min_length=1)
    scope: JobResearchScope
    status: JobResearchRunStatus
    input_fingerprint: str = Field(min_length=64, max_length=64)
    worker_version: str = Field(min_length=1, max_length=200)
    report_id: str | None = Field(default=None, min_length=1)
    error_code: str | None = Field(default=None, min_length=1, max_length=200)
    error_detail: str | None = Field(default=None, min_length=1, max_length=2000)
    started_at: datetime
    completed_at: datetime | None = None
    updated_at: datetime
