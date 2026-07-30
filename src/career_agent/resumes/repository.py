"""Persistence contracts for resume pools."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

from career_agent.resumes.models import ResumeData, ResumeVersion


class DuplicateResumeError(ValueError):
    """Raised when the same file already exists in a tenant's role pool."""


class ResumeNotFoundError(LookupError):
    """Raised when a tenant cannot access the requested resume version."""


class ResumeVersionRepository(Protocol):
    """Tenant-scoped persistence contract.

    Every method requires ``tenant_id`` so callers cannot accidentally perform
    an unscoped cross-tenant query.
    """

    def create_version(
        self,
        *,
        version_id: str,
        tenant_id: str,
        role_type: str,
        note: str,
        original_filename: str,
        stored_file_path: str,
        file_hash: str,
        parsed_data: ResumeData,
        created_at: datetime | None = None,
    ) -> ResumeVersion: ...

    def find_by_hash(
        self,
        tenant_id: str,
        role_type: str,
        file_hash: str,
    ) -> ResumeVersion | None: ...

    def list_versions(
        self,
        tenant_id: str,
        role_type: str | None = None,
    ) -> list[ResumeVersion]: ...

    def list_role_types(self, tenant_id: str) -> list[str]: ...

    def get_version(
        self,
        tenant_id: str,
        version_id: str,
    ) -> ResumeVersion | None: ...

    def set_active_version(
        self,
        tenant_id: str,
        version_id: str,
    ) -> None: ...

    def get_active_version(self, tenant_id: str) -> ResumeVersion | None: ...
