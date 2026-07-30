from pathlib import Path

from langchain_core.tools import StructuredTool

from career_agent.jobs.agent_tools import JobDiscoveryToolContext
from career_agent.jobs.wiring import (
    build_job_discovery_registry,
    build_job_discovery_service,
)


def _mcp_tool(name: str) -> StructuredTool:
    async def invoke(**arguments: object) -> str:
        return '{"ok": true, "data": {"count": 0, "jobs": []}}'

    return StructuredTool.from_function(
        coroutine=invoke,
        name=name,
        description=f"Fake {name}",
    )


def test_cli_wiring_exposes_only_owned_career_tools(tmp_path: Path) -> None:
    mcp_tools = [_mcp_tool("boss_export"), _mcp_tool("boss_detail"), _mcp_tool("boss_other")]
    service = build_job_discovery_service(
        mcp_tools=mcp_tools,
        database_path=tmp_path / "career-agent.db",
    )

    registry = build_job_discovery_registry(
        service=service,
        context=JobDiscoveryToolContext(
            tenant_id="tenant-a",
            thread_id="thread-a",
            user_query="北京的 Agent 开发岗位",
        ),
    )

    exposed_names = {entry.tool.name for entry in registry.describe()}
    assert exposed_names == {"career_search_jobs", "career_get_job_detail"}
    assert not exposed_names.intersection({"boss_export", "boss_detail", "boss_other"})


def test_cli_wiring_initializes_shared_job_search_schema(tmp_path: Path) -> None:
    database_path = tmp_path / "career-agent.db"

    build_job_discovery_service(
        mcp_tools=[_mcp_tool("boss_export"), _mcp_tool("boss_detail")],
        database_path=database_path,
    )

    import sqlite3

    with sqlite3.connect(database_path) as connection:
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert {"job_postings", "search_runs", "search_run_items"} <= tables
