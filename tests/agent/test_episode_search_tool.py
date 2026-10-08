from datetime import datetime, timezone

import pytest

from career_agent.agent.contracts.context import MainAgentContext
from career_agent.agent.contracts.observations import EpisodeProjectionContext
from career_agent.agent.contracts.profile import CareerProfileContext
from career_agent.agent.runtime.main_agent_runtime import MainAgentRuntime
from career_agent.agent.runtime.ports import project_atomic_tool_arguments
from career_agent.agent.capabilities.registry import MainAgentToolRegistry
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
    accessed = store.get_by_source(
        user_id="u1",
        kind="mock_interview",
        source_run_id="mock-1",
    )
    assert accessed is not None
    assert accessed.access_count == 1


def test_episode_search_reports_unmatched_company_without_hiding_other_hits(tmp_path) -> None:
    store = SQLiteCareerEpisodeStore(tmp_path / "context.sqlite3")
    store.upsert(_episode(
        source_run_id="interview-1",
        kind="interview_round",
        occurred_at=datetime(2026, 9, 1, tzinfo=timezone.utc),
        title="美团 · 配送算法工程师 · 第 1 轮面试",
        summary="完成面试。",
    ))
    result = MainAgentToolRegistry(episode_store=store).invoke_atomic_tool(
        "search_career_episodes", {"user_id": "u1", "query": "饿了么 面试"}
    )
    assert result.state == "career_episode_search_found"
    assert result.payload["items"][0]["title"].startswith("美团")
    assert "饿了么" in result.payload["unmatched_terms"]
    assert "面试" in result.payload["matched_terms"]


def test_episode_search_alias_does_not_negate_full_name_match(tmp_path) -> None:
    store = SQLiteCareerEpisodeStore(tmp_path / "context.sqlite3")
    store.upsert(_episode(
        source_run_id="interview-1",
        kind="interview_round",
        occurred_at=datetime(2026, 9, 1, tzinfo=timezone.utc),
        title="腾讯 · 后台开发工程师 · 第 1 轮面试",
        summary="完成面试。",
    ))
    result = MainAgentToolRegistry(episode_store=store).invoke_atomic_tool(
        "search_career_episodes", {"user_id": "u1", "query": "腾讯 鹅厂"}
    )
    assert result.state == "career_episode_search_found"
    assert result.payload["matched_terms"] == ["腾讯"]
    assert result.payload["unmatched_terms"] == ["鹅厂"]
    assert result.payload["items"][0]["title"].startswith("腾讯")


def test_sentence_query_does_not_report_unmatched_fragments(tmp_path) -> None:
    store = SQLiteCareerEpisodeStore(tmp_path / "context.sqlite3")
    store.upsert(_episode(
        source_run_id="interview-1",
        kind="interview_round",
        occurred_at=datetime(2026, 9, 1, tzinfo=timezone.utc),
        title="美团 · 配送算法工程师 · 第 1 轮面试",
        summary="完成面试。",
    ))
    result = MainAgentToolRegistry(episode_store=store).invoke_atomic_tool(
        "search_career_episodes", {"user_id": "u1", "query": "我在美团的面试怎么样"}
    )
    assert result.state == "career_episode_search_found"
    assert result.payload["matched_terms"] == []
    assert result.payload["partially_matched_terms"] == []
    assert result.payload["unmatched_terms"] == []


@pytest.mark.parametrize("query", ["快手", "ByteDance", "PDD", "拼夕夕", "饿了么", "华为技术"])
def test_single_name_query_reports_missing_name(tmp_path, query: str) -> None:
    store = SQLiteCareerEpisodeStore(tmp_path / "context.sqlite3")
    store.upsert(_episode(
        source_run_id="research-1",
        kind="job_research",
        occurred_at=datetime(2026, 9, 1, tzinfo=timezone.utc),
        title="美团 · 公司调研",
        summary="完成配送研究。",
    ))
    result = MainAgentToolRegistry(episode_store=store).invoke_atomic_tool(
        "search_career_episodes", {"user_id": "u1", "query": query}
    )
    assert result.payload["matched_terms"] == []
    assert result.payload["partially_matched_terms"] == []
    assert result.payload["unmatched_terms"] == [query.casefold()]


@pytest.mark.parametrize("name", ["京东物流", "蚂蚁集团"])
def test_name_fragment_is_not_reported_as_full_match(tmp_path, name: str) -> None:
    store = SQLiteCareerEpisodeStore(tmp_path / "context.sqlite3")
    store.upsert(_episode(
        source_run_id="research-1",
        kind="job_research",
        occurred_at=datetime(2026, 9, 1, tzinfo=timezone.utc),
        title="美团 · 公司调研",
        summary="物流集团的配送研究。",
    ))
    result = MainAgentToolRegistry(episode_store=store).invoke_atomic_tool(
        "search_career_episodes", {"user_id": "u1", "query": f"{name} 调研"}
    )
    assert name not in result.payload["matched_terms"]
    assert name in result.payload["partially_matched_terms"]
    assert name not in result.payload["unmatched_terms"]


def test_episode_term_feedback_uses_the_same_kind_and_time_filters(tmp_path) -> None:
    store = SQLiteCareerEpisodeStore(tmp_path / "context.sqlite3")
    store.upsert(_episode(
        source_run_id="research-1",
        kind="job_research",
        occurred_at=datetime(2026, 8, 1, tzinfo=timezone.utc),
        title="美团 · 公司调研",
        summary="完成调研。",
    ))
    store.upsert(_episode(
        source_run_id="interview-1",
        kind="interview_round",
        occurred_at=datetime(2026, 9, 1, tzinfo=timezone.utc),
        title="腾讯 · 后台开发工程师 · 第 1 轮面试",
        summary="完成面试。",
    ))
    result = MainAgentToolRegistry(episode_store=store).invoke_atomic_tool(
        "search_career_episodes",
        {"user_id": "u1", "query": "美团 面试", "kinds": ["interview_round"]},
    )
    assert result.payload["matched_terms"] == ["面试"]
    assert result.payload["unmatched_terms"] == ["美团"]
    timed = MainAgentToolRegistry(episode_store=store).invoke_atomic_tool(
        "search_career_episodes",
        {
            "user_id": "u1",
            "query": "美团 调研",
            "start_datetime": "2026-08-15T00:00:00Z",
        },
    )
    assert timed.payload["matched_terms"] == []
    assert timed.payload["unmatched_terms"] == ["美团", "调研"]


def test_projected_detail_ref_can_be_expanded_and_unprojected_ref_is_rejected(
    tmp_path,
) -> None:
    store = SQLiteCareerEpisodeStore(tmp_path / "context.sqlite3")
    episode = store.upsert(
        _episode(
            source_run_id="mock-1",
            kind="mock_interview",
            occurred_at=datetime(2026, 9, 1, tzinfo=timezone.utc),
            title="System design mock",
            summary="Capacity planning needed more detail.",
        )
    )
    detail_ref = f"episode:{episode.id}"
    context = MainAgentContext(
        conversation_id="conversation-2",
        profile=CareerProfileContext(user_id="u1"),
        career_episodes=(
            EpisodeProjectionContext(
                detail_ref=detail_ref,
                kind=episode.kind,
                occurred_at=episode.occurred_at,
                title=episode.title,
                synopsis=episode.summary,
            ),
        ),
        user_message="展开这次复盘",
    )

    arguments = project_atomic_tool_arguments(
        context,
        "search_career_episodes",
        {"detail_ref": detail_ref},
    )
    result = MainAgentToolRegistry(episode_store=store).invoke_atomic_tool(
        "search_career_episodes",
        arguments,
    )

    assert result.state == "career_episode_search_found"
    assert result.payload["items"][0]["title"] == episode.title
    refreshed = store.get(user_id="u1", episode_id=episode.id)
    assert refreshed is not None
    assert refreshed.access_count == 1
    with pytest.raises(ValueError, match="not projected"):
        project_atomic_tool_arguments(
            context,
            "search_career_episodes",
            {"detail_ref": "episode:career_episode_" + "0" * 32},
        )


def test_runtime_injects_the_owner_for_both_memory_search_layers() -> None:
    context = MainAgentContext(
        conversation_id="conversation-1",
        profile=CareerProfileContext(user_id="u1"),
        user_message="Why did I skip Acme?",
    )

    episode_arguments = project_atomic_tool_arguments(
        context,
        "search_career_episodes",
        {"query": "Acme", "top_k": 3},
    )
    semantic_arguments = project_atomic_tool_arguments(
        context,
        "search_career_memory",
        {"query": "retrieval"},
    )

    assert episode_arguments["user_id"] == "u1"
    assert episode_arguments["top_k"] == 3
    assert semantic_arguments["user_id"] == "u1"
