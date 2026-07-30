"""Persistence contract for immutable search-run snapshots."""

from __future__ import annotations

from typing import Protocol

from career_agent.jobs.search_models import SearchRun


class DuplicateSearchRunError(ValueError):
    """Raised when a run ID has already been persisted."""


class SearchRunRepository(Protocol):
    """Read-only persistence contract for immutable SearchRun records.

    Deliberately provides no update or delete methods — a SearchRun is
    created once and never modified.  Re-ordering or filtering produces
    a *new* SearchRun.
    """

    def create(self, run: SearchRun) -> SearchRun: ...

    def get(self, tenant_id: str, run_id: str) -> SearchRun | None: ...

    def get_latest(self, tenant_id: str, thread_id: str) -> SearchRun | None: ...

    def list_by_thread(
        self,
        tenant_id: str,
        thread_id: str,
        *,
        limit: int = 20,
    ) -> list[SearchRun]: ...
