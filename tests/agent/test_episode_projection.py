from datetime import datetime, timedelta, timezone

from career_agent.agent.context_manager import ContextManager
from career_agent.agent.main_agent_contracts import (
    PREFERENCE_EPISODE_CHAR_BUDGET,
)
from career_agent.domain.episodes import CareerEpisodeDraft, EpisodeResourceRef
from career_agent.storage.context import CareerContextStore
from career_agent.storage.episodes import SQLiteCareerEpisodeStore


def _episode(
    *,
    source_run_id: str,
    conversation_id: str,
    title: str,
    summary: str,
    age_days: int = 1,
) -> CareerEpisodeDraft:
    return CareerEpisodeDraft(
        user_id="u1",
        kind="mock_interview",
        source_run_id=source_run_id,
        occurred_at=datetime.now(timezone.utc) - timedelta(days=age_days),
        title=title,
        summary=summary,
        conversation_id=conversation_id,
        resource_refs=(
            EpisodeResourceRef(
                kind="mock_interview",
                resource_id=source_run_id,
                title=title[:80],
            ),
        ),
    )


def test_related_episode_is_projected_across_sessions_with_detail_ref(
    tmp_path,
) -> None:
    path = tmp_path / "context.sqlite3"
    context_store = CareerContextStore(path)
    episode_store = SQLiteCareerEpisodeStore(path)
    stored = episode_store.upsert(
        _episode(
            source_run_id="mock-1",
            conversation_id="old-conversation",
            title="系统设计模拟面试",
            summary="容量估算环节需要补充峰值流量和存储量推导。",
        )
    )
    manager = ContextManager(context_store, episode_store=episode_store)

    context = manager.load_for_turn(
        user_id="u1",
        conversation_id="new-conversation",
        user_message="容量估算应该怎么准备？",
    )
    projected = context.model_context()

    assert [item.title for item in context.career_episodes] == [
        "系统设计模拟面试"
    ]
    assert "渐进披露目录" in projected["career_episodes"]
    assert "容量估算环节" in projected["career_episodes"]
    assert f"detail_ref=episode:{stored.id}" in projected["career_episodes"]
    refreshed = episode_store.get(user_id="u1", episode_id=stored.id)
    assert refreshed is not None
    assert refreshed.access_count == 1

    unrelated = manager.load_for_turn(
        user_id="u1",
        conversation_id="another-conversation",
        user_message="帮我看看薪资范围",
    )
    assert unrelated.career_episodes == ()
    assert "career_episodes" not in unrelated.model_context()


def test_episode_and_preference_projection_share_the_character_budget(
    tmp_path,
) -> None:
    path = tmp_path / "context.sqlite3"
    context_store = CareerContextStore(path)
    episode_store = SQLiteCareerEpisodeStore(path)
    for index in range(5):
        episode_store.upsert(
            _episode(
                source_run_id=f"mock-{index}",
                conversation_id=f"old-{index}",
                title=f"分布式训练复盘 {index}",
                summary=(
                    "分布式训练中需要解释数据并行、梯度同步、故障恢复和性能分析。"
                    * 5
                ),
                age_days=index + 1,
            )
        )
    manager = ContextManager(context_store, episode_store=episode_store)

    projected = manager.load_for_turn(
        user_id="u1",
        conversation_id="new",
        user_message="回顾一下分布式训练面试",
    ).model_context()

    combined = (
        projected["free_text_preferences"]
        + projected.get("career_episodes", "")
    )
    assert len(combined) <= PREFERENCE_EPISODE_CHAR_BUDGET
    assert 1 <= projected["career_episodes"].count("detail_ref=") <= 5
