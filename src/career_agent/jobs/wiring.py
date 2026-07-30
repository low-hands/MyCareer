"""Composition helpers that keep raw provider tools behind Career tools."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from langchain_core.tools import BaseTool

from career_agent.contracts import ToolPermission
from career_agent.jobs.agent_tools import (
    JobDiscoveryToolContext,
    build_job_discovery_tools,
)
from career_agent.jobs.discovery_service import JobDiscoveryService
from career_agent.jobs.providers.boss_mcp import boss_provider_from_tools
from career_agent.jobs.search_service import SearchRunService
from career_agent.jobs.search_sqlite_repo import SQLiteSearchRunRepository
from career_agent.jobs.sqlite_repo import SQLiteJobPostingRepository
from career_agent.tools import ToolRegistry


def build_job_discovery_service(
    *,
    mcp_tools: Iterable[BaseTool],
    database_path: str | Path,
) -> JobDiscoveryService:
    """Build the provider-backed service without exposing raw MCP tools."""

    posting_repository = SQLiteJobPostingRepository(database_path)
    run_repository = SQLiteSearchRunRepository(database_path)
    run_service = SearchRunService(
        run_repository=run_repository,
        posting_repository=posting_repository,
    )
    provider = boss_provider_from_tools(mcp_tools)
    return JobDiscoveryService(
        provider=provider,
        posting_repository=posting_repository,
        run_service=run_service,
    )


def build_job_discovery_registry(
    *,
    service: JobDiscoveryService,
    context: JobDiscoveryToolContext,
) -> ToolRegistry:
    """Expose only request-scoped Career tools to the model."""

    registry = ToolRegistry()
    for tool in build_job_discovery_tools(service=service, context=context):
        registry.register(tool, permission=ToolPermission.READ)
    return registry
