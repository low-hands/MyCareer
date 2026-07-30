"""Tests for task-scoped Agent tools around job discovery services."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from career_agent.jobs.agent_tools import (
    JobDiscoveryToolContext,
    build_job_discovery_tools,
)
from career_agent.jobs.discovery_service import JobDiscoveryService
from career_agent.jobs.models import JobPosting
from career_agent.jobs.providers.base import ProviderSearchResult
from career_agent.jobs.search_models import SearchCriteria
from career_agent.jobs.search_service import SearchRunService
from career_agent.jobs.search_sqlite_repo import SQLiteSearchRunRepository
from career_agent.jobs.sqlite_repo import SQLiteJobPostingRepository


class StubProvider:
    def __init__(self, postings: list[JobPosting]) -> None:
        self.postings = postings
        self.criteria: SearchCriteria | None = None
        self.detail_calls: list[str] = []

    async def search(
        self,
        *,
        tenant_id: str,
        criteria: SearchCriteria,
        page: int = 1,
    ) -> ProviderSearchResult:
        self.criteria = criteria
        return ProviderSearchResult(
            postings=tuple(self.postings),
            candidate_count=len(self.postings),
            provider_total_count=len(self.postings),
        )

    async def get_detail(self, posting: JobPosting) -> JobPosting:
        self.detail_calls.append(posting.job_id)
        now = datetime.now(UTC)
        return posting.model_copy(
            update={
                "description": f"{posting.title} 的完整 JD",
                "detail_level": "full",
                "fetched_at": now,
                "updated_at": now,
            }
        )


def _posting(
    *,
    job_id: str,
    external_id: str,
    location: str,
) -> JobPosting:
    now = datetime(2026, 7, 30, tzinfo=UTC)
    return JobPosting(
        job_id=job_id,
        tenant_id="tenant-a",
        source="boss",
        external_id=external_id,
        detail_locator=f"security-{external_id}",
        title="Agent 开发工程师",
        company_name="腾讯科技",
        location=location,
        salary="30-50K",
        fetched_at=now,
        created_at=now,
        updated_at=now,
    )


def _tools(
    tmp_path: Path,
) -> tuple[dict[str, object], StubProvider]:
    database_path = tmp_path / "career-agent.db"
    postings = SQLiteJobPostingRepository(database_path)
    runs = SQLiteSearchRunRepository(database_path)
    provider = StubProvider(
        [
            _posting(job_id="proposed-1", external_id="boss-1", location="深圳"),
            _posting(job_id="proposed-2", external_id="boss-2", location="北京"),
        ]
    )
    service = JobDiscoveryService(
        provider=provider,
        posting_repository=postings,
        run_service=SearchRunService(
            run_repository=runs,
            posting_repository=postings,
        ),
    )
    built = build_job_discovery_tools(
        service=service,
        context=JobDiscoveryToolContext(
            tenant_id="tenant-a",
            thread_id="thread-1",
            user_query="帮我找腾讯的 Agent 岗位",
        ),
    )
    return {tool.name: tool for tool in built}, provider


@pytest.mark.asyncio
async def test_search_tool_uses_search_criteria_schema_and_publishes_results(
    tmp_path: Path,
) -> None:
    tools, provider = _tools(tmp_path)

    raw = await tools["career_search_jobs"].ainvoke(
        {
            "query": "Agent 开发",
            "cities": ["深圳", "北京"],
        }
    )
    result = json.loads(raw)

    assert result["status"] == "published"
    assert [item["position"] for item in result["results"]] == [1, 2]
    assert provider.criteria == SearchCriteria(
        query="Agent 开发",
        cities=["深圳", "北京"],
    )
    assert "tenant_id" not in tools["career_search_jobs"].args
    assert "thread_id" not in tools["career_search_jobs"].args


@pytest.mark.asyncio
async def test_reference_tool_returns_ambiguity_without_fetching_detail(
    tmp_path: Path,
) -> None:
    tools, provider = _tools(tmp_path)
    await tools["career_search_jobs"].ainvoke({"query": "Agent"})

    raw = await tools["career_get_job_detail"].ainvoke({"company_name": "腾讯"})
    result = json.loads(raw)

    assert result["status"] == "ambiguous"
    assert provider.detail_calls == []
    assert all("job_id" not in item for item in result["candidates"])


@pytest.mark.asyncio
async def test_reference_tool_fetches_detail_only_after_unique_resolution(
    tmp_path: Path,
) -> None:
    tools, provider = _tools(tmp_path)
    await tools["career_search_jobs"].ainvoke({"query": "Agent"})

    raw = await tools["career_get_job_detail"].ainvoke(
        {
            "company_name": "腾讯",
            "location": "北京",
        }
    )
    result = json.loads(raw)

    assert result["status"] == "resolved"
    assert result["posting"]["detail_level"] == "full"
    assert result["posting"]["description"].endswith("完整 JD")
    assert provider.detail_calls == ["proposed-2"]
