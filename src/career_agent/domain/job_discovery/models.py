from __future__ import annotations

from datetime import datetime
from enum import Enum
import hashlib
import re
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, model_validator


class DomainError(Exception):
    code = "DOMAIN_ERROR"
    recoverable = False


class IncompleteJobDetail(DomainError):
    code = "INCOMPLETE_JD"
    recoverable = True


class TargetRoleRequired(DomainError):
    code = "TARGET_ROLE_REQUIRED"
    recoverable = True


class DuplicateDecisionRequired(DomainError):
    code = "DUPLICATE_DECISION_REQUIRED"
    recoverable = True


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class WaitlistStatus(str, Enum):
    ACTIVE = "active"
    ARCHIVED = "archived"


ArchiveReason = Literal["user_removed", "source_expired", "source_removed", "duplicate_merged"]
DuplicateRule = Literal["source_id", "company_title", "content"]


class Provenance(ContractModel):
    source_name: str
    captured_at: datetime
    operation: str
    adapter_version: str
    source_job_id: str | None = None
    source_url: str | None = None

    @model_validator(mode="after")
    def require_identity(self) -> "Provenance":
        if not self.source_name or not self.operation or not self.adapter_version:
            raise ValueError("source_name, operation, and adapter_version are required")
        return self


class SearchResult(ContractModel):
    result_ref: str
    source_name: str
    title: str
    company_name: str
    captured_at: datetime
    provenance: Provenance
    source_url: str | None = None
    security_id: str | None = None
    source_job_id: str | None = None
    city: str | None = None
    salary: str | None = None
    experience: str | None = None
    education: str | None = None
    labels: tuple[str, ...] = ()


class JobDetail(ContractModel):
    source_name: str
    title: str
    company_name: str
    description: str
    captured_at: datetime
    provenance: Provenance
    source_job_id: str | None = None
    security_id: str | None = None
    city: str | None = None
    salary: str | None = None
    experience: str | None = None
    education: str | None = None
    labels: tuple[str, ...] = ()
    source_url: str | None = None
    content_origin: Literal["boss_detail", "user_provided"] = "boss_detail"


class JobPosting(ContractModel):
    id: str
    user_id: str
    title: str
    company_name: str
    source_name: str
    source_job_id: str | None
    source_url: str | None
    external_status: str
    persisted_at: datetime
    last_seen_at: datetime
    latest_snapshot_id: str
    company_title_fingerprint: str
    content_fingerprint: str


class JDSnapshot(ContractModel):
    id: str
    job_posting_id: str
    version: int
    content: str
    content_hash: str
    captured_at: datetime
    provenance: Provenance
    normalizer_version: str


class WaitlistItem(ContractModel):
    id: str
    user_id: str
    job_posting_id: str
    target_role_id: str
    target_role_title: str
    status: WaitlistStatus
    added_at: datetime
    archived_at: datetime | None = None
    archive_reason: ArchiveReason | None = None
    note: str | None = None

    @model_validator(mode="after")
    def require_consistent_state(self) -> "WaitlistItem":
        if not self.user_id or not self.job_posting_id or not self.target_role_id:
            raise ValueError("waitlist ownership and target role are required")
        if self.status is WaitlistStatus.ACTIVE and (self.archived_at is not None or self.archive_reason is not None):
            raise ValueError("active waitlist items cannot have archive metadata")
        if self.status is WaitlistStatus.ARCHIVED and (self.archived_at is None or self.archive_reason is None):
            raise ValueError("archived waitlist items require archive metadata")
        return self

    def archive(self, reason: ArchiveReason, at: datetime) -> "WaitlistItem":
        if self.status is WaitlistStatus.ARCHIVED:
            return self
        return self.model_copy(update={"status": WaitlistStatus.ARCHIVED, "archived_at": at, "archive_reason": reason})


class DuplicateCandidate(ContractModel):
    job_posting_id: str
    rule: DuplicateRule
    title: str
    company_name: str


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


def normalize_jd(description: str) -> str:
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in description.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    normalized: list[str] = []
    previous_blank = True
    for line in lines:
        blank = not line
        if blank and previous_blank:
            continue
        normalized.append(line)
        previous_blank = blank
    return "\n".join(normalized).strip()


def jd_content_hash(description: str) -> str:
    return hashlib.sha256(normalize_jd(description).encode("utf-8")).hexdigest()


def content_fingerprint(title: str, company_name: str, description: str) -> str:
    value = "␟".join((title.casefold().strip(), company_name.casefold().strip(), normalize_jd(description)))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def company_title_fingerprint(title: str, company_name: str) -> str:
    value = "␟".join((title.casefold().strip(), company_name.casefold().strip()))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def validate_job_detail(detail: JobDetail) -> str:
    if not detail.title.strip():
        raise IncompleteJobDetail("Job title is required before promotion.")
    if not detail.company_name.strip():
        raise IncompleteJobDetail("Company name is required before promotion.")
    description = normalize_jd(detail.description)
    if not description:
        raise IncompleteJobDetail("A non-empty job description is required before promotion.")
    return description
