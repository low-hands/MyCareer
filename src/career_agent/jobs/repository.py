"""Persistence contract for source-grounded job postings."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Protocol

from career_agent.jobs.models import JobPosting


class JobPostingNotFoundError(LookupError):
    """Raised when a posting is missing or inaccessible to the tenant."""


class JobPostingRepository(Protocol):
    """Tenant-scoped persistence contract for normalized vacancies."""

    def upsert(self, posting: JobPosting) -> JobPosting: ...

    def get(self, tenant_id: str, job_id: str) -> JobPosting | None: ...

    def get_many(
        self,
        tenant_id: str,
        job_ids: Sequence[str],
    ) -> list[JobPosting]: ...

    def save_full_detail(
        self,
        *,
        tenant_id: str,
        job_id: str,
        description: str,
        fetched_at: datetime | None = None,
    ) -> JobPosting: ...
