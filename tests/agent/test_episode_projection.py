from datetime import datetime, timedelta, timezone

import pytest

from career_agent.agent import main_agent_contracts
from career_agent.agent.context_manager import ContextManager
from career_agent.agent.main_agent_contracts import (
    PREFERENCE_EPISODE_CHAR_BUDGET,
    AgentDecision,
    CareerProfileContext,
    ConversationTaskState,
    FreeTextPreferenceContext,
    ToolCall,
    ToolObservation,
)
from career_agent.agent.main_agent_runtime import MainAgentRuntime
from career_agent.agent.main_agent_tools import MainAgentToolRegistry
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
    # Loading a context does not count as exposure — the runtime marks it once,
    # just before the model is shown the projection.
    loaded = episode_store.get(user_id="u1", episode_id=stored.id)
    assert loaded is not None
    assert loaded.access_count == 0
    manager.mark_episodes_projected(user_id="u1", context=context)
    refreshed = episode_store.get(user_id="u1", episode_id=stored.id)
    assert refreshed is not None
    assert refreshed.access_count == 1
    manager.mark_episodes_projected(
        user_id="u1",
        context=context.model_copy(update={"career_episodes": ()}),
    )
    assert (
        episode_store.get(user_id="u1", episode_id=stored.id).access_count == 1
    )

    unrelated = manager.load_for_turn(
        user_id="u1",
        conversation_id="another-conversation",
        user_message="帮我看看薪资范围",
    )
    assert unrelated.career_episodes == ()
    assert "career_episodes" not in unrelated.model_context()


class _Decisions:
    def __init__(self, *decisions) -> None:
        self._decisions = list(decisions)
        self.contexts: list[object] = []

    def decide(self, context, tool_specs):
        self.contexts.append(context)
        if not self._decisions:
            raise AssertionError("more decisions were requested than expected")
        return self._decisions.pop(0)


class _NoteWritingRegistry(MainAgentToolRegistry):
    """One write whose state makes ``observe`` reload the turn's context."""

    def capability_kind(self, name):
        return "atomic_tool"

    def invoke_atomic_tool(self, name, arguments):
        return ToolObservation(
            tool_name=name,
            state="working_notes_updated",
            message="已更新工作笔记。",
            execution_outcome="committed",
        )


def _episode_runtime(tmp_path, *decisions):
    path = tmp_path / "context.sqlite3"
    episode_store = SQLiteCareerEpisodeStore(path)
    stored = episode_store.upsert(
        _episode(
            source_run_id="mock-1",
            conversation_id="old-conversation",
            title="系统设计模拟面试",
            summary="容量估算环节需要补充峰值流量和存储量推导。",
        )
    )
    manager = ContextManager(
        CareerContextStore(path), episode_store=episode_store
    )
    manager.upsert_profile(CareerProfileContext(user_id="u1"))

    class Runtime(MainAgentRuntime):
        @staticmethod
        def _project_atomic_tool_arguments(context, name, arguments):
            return {"user_id": context.profile.user_id, **arguments}

    decision_maker = _Decisions(*decisions)
    runtime = Runtime(
        context_manager=manager,
        decision_maker=decision_maker,
        tools=_NoteWritingRegistry(),
    )
    return runtime, manager, episode_store, stored, decision_maker


def test_a_turn_counts_one_exposure_however_many_times_it_decides(
    tmp_path,
) -> None:
    """Access counts what the model saw, not how often the runtime built it."""

    runtime, _, episode_store, stored, decisions = _episode_runtime(
        tmp_path,
        AgentDecision(
            action="tool_call",
            tool_call=ToolCall(name="update_working_notes", arguments={}),
        ),
        AgentDecision(action="final", message="已记录。"),
    )

    runtime.run_turn(
        user_id="u1",
        conversation_id="new-conversation",
        user_message="容量估算应该怎么准备？",
    )

    # Two decisions, and the write reloaded the context in between: the episode
    # was in front of the model throughout one turn, so it counts once.
    assert len(decisions.contexts) == 2
    marked = episode_store.get(user_id="u1", episode_id=stored.id)
    assert marked is not None
    assert marked.access_count == 1

    runtime._decision_maker = _Decisions(
        AgentDecision(action="final", message="好的。")
    )
    runtime.run_turn(
        user_id="u1",
        conversation_id="new-conversation",
        user_message="再说说容量估算",
    )

    assert (
        episode_store.get(user_id="u1", episode_id=stored.id).access_count == 2
    )


def test_a_workflow_owned_turn_never_marks_an_episode_exposed(tmp_path) -> None:
    runtime, manager, episode_store, stored, _ = _episode_runtime(tmp_path)
    manager._store.upsert_task(
        user_id="u1",
        conversation_id="new-conversation",
        task=ConversationTaskState(
            active_workflow="mock_interview",
            run_id="mock-running",
            phase="mock_interview_answer_required",
        ),
    )

    # The turn is routed to the workflow and dies inside it, on this stub
    # registry rather than on anything under test. That is the point: it never
    # reaches ``decide``, so no episode was put in front of a model.
    with pytest.raises(ValueError, match="handle_mock_interview_input"):
        runtime.run_turn(
            user_id="u1",
            conversation_id="new-conversation",
            user_message="容量估算应该怎么准备？",
        )

    unmarked = episode_store.get(user_id="u1", episode_id=stored.id)
    assert unmarked is not None
    assert unmarked.access_count == 0


def test_an_unpressured_load_builds_its_context_once(tmp_path, monkeypatch) -> None:
    class Worker:
        def summarize(self, *, previous, messages):  # pragma: no cover - guarded
            raise AssertionError("an unpressured turn must not compact")

    path = tmp_path / "context.sqlite3"
    manager = ContextManager(CareerContextStore(path), summary_worker=Worker())
    manager.upsert_profile(CareerProfileContext(user_id="u1"))
    manager.configure_request_token_estimator(lambda context: (1, 1000))
    builds = 0
    original = ContextManager._build_context

    def counting(self, **kwargs):
        nonlocal builds
        builds += 1
        return original(self, **kwargs)

    monkeypatch.setattr(ContextManager, "_build_context", counting)

    manager.load_for_turn(
        user_id="u1",
        conversation_id="c1",
        user_message="容量估算应该怎么准备？",
    )

    # Measuring pressure already built the whole context, and nothing compacted
    # or expired afterwards, so the load returns that one instead of repeating a
    # full build with two FTS rankings and a notes read.
    assert builds == 1


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


def test_a_catalogue_with_no_entry_left_is_dropped_rather_than_shown(
    tmp_path, monkeypatch
) -> None:
    """A count of hidden lines is not a catalogue.

    The real shared budget always leaves room for one entry, so the guard is
    driven here by shrinking it: what is under test is that the guard looks for
    an entry rather than counting lines, now that bounding can add one of its own.
    """

    path = tmp_path / "context.sqlite3"
    episode_store = SQLiteCareerEpisodeStore(path)
    episode_store.upsert(
        _episode(
            source_run_id="mock-1",
            conversation_id="old-conversation",
            title="系统设计模拟面试",
            summary="容量估算环节需要补充峰值流量和存储量推导。" * 5,
        )
    )
    manager = ContextManager(
        CareerContextStore(path), episode_store=episode_store
    )
    context = manager.load_for_turn(
        user_id="u1",
        conversation_id="new-conversation",
        user_message="容量估算应该怎么准备？",
    )
    monkeypatch.setattr(
        main_agent_contracts, "PREFERENCE_EPISODE_CHAR_BUDGET", 145
    )

    projected = context.model_context()

    assert context.career_episodes != ()
    assert "career_episodes" not in projected
