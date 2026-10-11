from datetime import datetime
import json
from pathlib import Path

import pytest

from career_agent.agent.capabilities.registry import MainAgentToolRegistry
from career_agent.agent.presentation.episode_dates import relative_episode_date
from career_agent.agent.context.manager import ContextManager
from career_agent.storage.context import CareerContextStore
from career_agent.domain.episodes import CareerEpisodeDraft
from career_agent.storage.episodes import SQLiteCareerEpisodeStore

FIXTURE = json.loads((Path(__file__).parents[2] / "evals/episode_retrieval/temporal_v1/year_labels.json").read_text(encoding="utf-8"))
CLOCK = datetime.fromisoformat(FIXTURE["clock"])


@pytest.mark.parametrize("case", FIXTURE["cases"])
def test_frozen_date_labels_use_reference_timezone(case):
    assert relative_episode_date(datetime.fromisoformat(case["occurred_at"]), now=CLOCK) == case["expected"]


@pytest.mark.parametrize("detail", [False, True])
def test_search_and_detail_deliver_label_and_reference_without_changing_date(tmp_path, detail):
    store = SQLiteCareerEpisodeStore(tmp_path / "context.sqlite3")
    date = "2026-03-15T12:00:00+08:00"
    episode = store.upsert(CareerEpisodeDraft(
        user_id="u1", kind="application", source_run_id="spring",
        title="字节跳动投递", summary="投递后收到拒信。", occurred_at=datetime.fromisoformat(date),
    ))
    args = {"detail_ref": f"episode:{episode.id}"} if detail else {"query": "字节跳动"}
    result = MainAgentToolRegistry(episode_store=store, episode_clock=lambda: CLOCK).invoke_atomic_tool(
        "search_career_episodes", {"user_id": "u1", **args},
    )
    item = result.payload["items"][0]
    assert datetime.fromisoformat(item["occurred_at"]) == datetime.fromisoformat(date)
    assert item["relative_year"] == "今年"
    assert result.payload["relative_year_reference"] == FIXTURE["clock"]
    assert "今年" in result.payload["body"]


def test_default_label_clock_uses_runtime_timezone(tmp_path, monkeypatch):
    monkeypatch.setenv("CAREER_AGENT_TIMEZONE", "Asia/Shanghai")
    from career_agent.agent.runtime import decision_messages
    monkeypatch.setattr(decision_messages, "runtime_clock", lambda: {
        "now": FIXTURE["clock"], "timezone": "Asia/Shanghai",
    })
    store = SQLiteCareerEpisodeStore(tmp_path / "context.sqlite3")
    store.upsert(CareerEpisodeDraft(
        user_id="u1", kind="application", source_run_id="year-boundary",
        title="字节跳动投递", summary="已投递。",
        occurred_at=datetime.fromisoformat("2025-12-31T16:30:00+00:00"),
    ))
    result = MainAgentToolRegistry(episode_store=store).invoke_atomic_tool(
        "search_career_episodes", {"user_id": "u1", "query": "字节跳动"},
    )
    assert result.payload["items"][0]["relative_year"] == "今年"


def test_naive_date_is_not_silently_interpreted_in_host_timezone():
    with pytest.raises(ValueError, match="timezone-aware"):
        relative_episode_date(datetime(2026, 3, 15), now=CLOCK)


def test_projection_shows_date_and_year_in_data_context(tmp_path):
    store = SQLiteCareerEpisodeStore(tmp_path / "context.sqlite3")
    store.upsert(CareerEpisodeDraft(
        user_id="u1", kind="application", source_run_id="projected",
        title="字节跳动投递", summary="投递后收到拒信。",
        occurred_at=datetime.fromisoformat("2026-03-12T12:00:00+08:00"),
        conversation_id="old",
    ))
    manager = ContextManager(CareerContextStore(store.path), episode_store=store, clock=lambda: CLOCK)
    context = manager.load_for_turn(user_id="u1", conversation_id="new", user_message="去年投的字节")
    assert context.career_episodes
    assert context.career_episodes[0].relative_year == "今年"
    rendered = context.model_context()["career_episodes"]
    assert "2026-03-12" in rendered and "（今年）" in rendered
    assert "detail_ref=episode:" in rendered
