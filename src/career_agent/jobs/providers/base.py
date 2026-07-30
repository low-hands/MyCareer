"""Provider-neutral contracts for retrieving source-grounded vacancies."""

from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, Field, model_validator

from career_agent.jobs.models import JobPosting
from career_agent.jobs.search_models import SearchCriteria


class AsyncTool(Protocol):
    """Small callable boundary shared by LangChain and test tools."""

    async def ainvoke(self, arguments: dict[str, Any]) -> Any: ...


class JobProviderError(RuntimeError):
    """Normalized provider/MCP failure with recovery metadata."""

    def __init__(
        self,
        *,
        code: str,
        message: str,
        recoverable: bool = False,
        recovery_action: str | None = None,
    ) -> None:
        self.code = code
        self.message = message
        self.recoverable = recoverable
        self.recovery_action = recovery_action
        super().__init__(f"{code}: {message}")


class ProviderSearchResult(BaseModel, frozen=True):
    """Normalized result of one logical provider search."""

    postings: tuple[JobPosting, ...]
    candidate_count: int = Field(ge=0)
    provider_total_count: int = Field(ge=0)
    has_more: bool = False
    warnings: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _candidate_count_matches_postings(self) -> ProviderSearchResult:
        if self.candidate_count != len(self.postings):
            raise ValueError("candidate_count must equal len(postings)")
        return self


class JobSearchProvider(Protocol):
    async def search(
        self,
        *,
        tenant_id: str,
        criteria: SearchCriteria,
        page: int = 1,
    ) -> ProviderSearchResult: ...

    async def get_detail(self, posting: JobPosting) -> JobPosting: ...
