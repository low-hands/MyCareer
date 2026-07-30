"""End-to-end application service for grounded job discovery."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from career_agent.jobs.providers.base import JobSearchProvider
from career_agent.jobs.repository import JobPostingRepository
from career_agent.jobs.search_models import (
    ResolvedSearchResult,
    SearchCriteria,
    SearchResolution,
    SearchResultSelector,
)
from career_agent.jobs.search_service import SearchRunService


class DisplayedJob(BaseModel, frozen=True):
    position: int = Field(ge=1)
    title: str
    company_name: str
    location: str = ""
    salary: str = ""


class PublishedSearchResults(BaseModel, frozen=True):
    status: Literal["published"] = "published"
    run_id: str
    candidate_count: int = Field(ge=1)
    provider_total_count: int = Field(ge=0)
    results: tuple[DisplayedJob, ...] = Field(min_length=1)
    warnings: tuple[str, ...] = ()


class NoSearchResults(BaseModel, frozen=True):
    status: Literal["no_results"] = "no_results"
    message: str
    warnings: tuple[str, ...] = ()


class JobDiscoveryService:
    """Persist provider results before exposing them to an agent."""

    def __init__(
        self,
        *,
        provider: JobSearchProvider,
        posting_repository: JobPostingRepository,
        run_service: SearchRunService,
    ) -> None:
        self._provider = provider
        self._postings = posting_repository
        self._runs = run_service

    async def search_and_publish(
        self,
        *,
        tenant_id: str,
        thread_id: str,
        user_query: str,
        criteria: SearchCriteria,
        max_results: int = 10,
    ) -> PublishedSearchResults | NoSearchResults:
        if max_results < 1:
            raise ValueError("max_results must be positive")

        provider_result = await self._provider.search(
            tenant_id=tenant_id,
            criteria=criteria,
        )
        persisted = [self._postings.upsert(posting) for posting in provider_result.postings]
        if not persisted:
            return NoSearchResults(
                message="No grounded job postings matched the search criteria.",
                warnings=provider_result.warnings,
            )

        displayed = persisted[:max_results]
        run = self._runs.publish_results(
            tenant_id=tenant_id,
            thread_id=thread_id,
            user_query=user_query,
            criteria=criteria,
            job_ids=[posting.job_id for posting in displayed],
            candidate_count=provider_result.candidate_count,
        )
        return PublishedSearchResults(
            run_id=run.run_id,
            candidate_count=provider_result.candidate_count,
            provider_total_count=provider_result.provider_total_count,
            results=tuple(
                DisplayedJob(
                    position=position,
                    title=posting.title,
                    company_name=posting.company_name,
                    location=posting.location,
                    salary=posting.salary,
                )
                for position, posting in enumerate(displayed, start=1)
            ),
            warnings=provider_result.warnings,
        )

    async def resolve_and_get_detail(
        self,
        *,
        tenant_id: str,
        thread_id: str,
        selector: SearchResultSelector,
    ) -> SearchResolution:
        resolution = self._runs.resolve_reference(
            tenant_id=tenant_id,
            thread_id=thread_id,
            selector=selector,
        )
        if not isinstance(resolution, ResolvedSearchResult):
            return resolution

        detailed = await self._provider.get_detail(resolution.posting)
        persisted = self._postings.upsert(detailed)
        return resolution.model_copy(update={"posting": persisted})
