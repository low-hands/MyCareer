from datetime import datetime, timezone

from career_agent.agent.main_agent_contracts import (
    CareerProfileContext,
    MainAgentContext,
)
from career_agent.agent.main_agent_runtime import MainAgentRuntime
from career_agent.agent.main_agent_tools import MainAgentToolRegistry
from career_agent.domain.episodes import CareerEpisodeDraft, EpisodeResourceRef
from career_agent.storage.episodes import SQLiteCareerEpisodeStore


def _episode(
    *,
    source_run_id: str,
    kind: str,
    occurred_at: datetime,
    title: str,
    summary: str,
    resource_kind: str | None = None,
) -> CareerEpisodeDraft:
    return CareerEpisodeDraft(
        user_id="u1",
        kind=kind,
        source_run_id=source_run_id,
        occurred_at=occurred_at,
        title=title,
        summary=summary,
        conversation_id="conversation-1",
        resource_refs=(
            (
                EpisodeResourceRef(
                    kind=resource_kind,
                    resource_id=source_run_id,
                    title=title,
                ),
            )
            if resource_kind is not None
            else ()
        ),
    )


def test_episode_search_is_offered_only_with_the_l1_store(tmp_path) -> None:
    def names(registry: MainAgentToolRegistry) -> set[str]:
        return {item["function"]["name"] for item in registry.schemas()}

    assert "search_career_episodes" not in names(MainAgentToolRegistry())
    assert "search_career_episodes" in names(
        MainAgentToolRegistry(
            episode_store=SQLiteCareerEpisodeStore(tmp_path / "context.sqlite3")
        )
    )


def test_episode_search_filters_time_and_kind_and_returns_pointers(tmp_path) -> None:
    store = SQLiteCareerEpisodeStore(tmp_path / "context.sqlite3")
    store.upsert(
        _episode(
            source_run_id="application-1",
            kind="application",
            occurred_at=datetime(2026, 8, 1, tzinfo=timezone.utc),
            title="Acme · ML Engineer",
            summary="没有继续投递，因为岗位已关闭。",
        )
    )
    store.upsert(
        _episode(
            source_run_id="mock-1",
            kind="mock_interview",
            occurred_at=datetime(2026, 9, 1, tzinfo=timezone.utc),
            title="Acme mock interview",
            summary="系统设计回答缺少容量估算。",
            resource_kind="mock_interview_report",
        )
    )
    tools = MainAgentToolRegistry(episode_store=store)

    result = tools.invoke_atomic_tool(
        "search_career_episodes",
        {
            "user_id": "u1",
            "query": "Acme",
            "start_datetime": "2026-08-15T00:00:00Z",
            "kinds": ["mock_interview"],
            "top_k": 5,
        },
    )

    assert result.state == "career_episode_search_found"
    assert result.facts["returned"] == 1
    assert result.payload["items"][0]["resource_refs"] == [
        {
            "kind": "mock_interview_report",
            "resource_id": "mock-1",
            "title": "Acme mock interview",
        }
    ]
    assert [item.resource_id for item in result.resource_refs] == ["mock-1"]


def test_runtime_injects_the_owner_for_both_memory_search_layers() -> None:
    context = MainAgentContext(
        conversation_id="conversation-1",
        profile=CareerProfileContext(user_id="u1"),
        user_message="Why did I skip Acme?",
    )

    episode_arguments = MainAgentRuntime._project_atomic_tool_arguments(
        context,
        "search_career_episodes",
        {"query": "Acme", "top_k": 3},
    )
    semantic_arguments = MainAgentRuntime._project_atomic_tool_arguments(
        context,
        "search_career_memory",
        {"query": "retrieval"},
    )

    assert episode_arguments["user_id"] == "u1"
    assert episode_arguments["top_k"] == 3
    assert semantic_arguments["user_id"] == "u1"
