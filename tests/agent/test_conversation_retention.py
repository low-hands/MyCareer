"""Reclaiming summarised conversation history, only when asked.

Summarising used to delete the messages it covered, which kept the file small at
the cost of making a bad summary unrecoverable. The originals now stay until an
operator asks for them to go, and a threshold notice is how they find out that
asking is worth it.
"""

from __future__ import annotations

import pytest

from career_agent.agent.conversation_memory_contracts import (
    ConversationSummaryContent,
)
from career_agent.agent.openai_compatible_client import AgentWorkerError
from career_agent.agent.main_agent_contracts import (
    CareerProfileContext,
    ConversationTaskState,
)
from career_agent.storage.context import CareerContextStore

from test_context_manager import RecordingSummaryWorker, manager


class _FailingWorker:
    """Configured but never succeeds, so no summary is written automatically."""

    def summarize(self, *, previous, messages):
        raise AgentWorkerError("SUMMARY_UNAVAILABLE", "unavailable", retryable=True)


def _talk(context_manager, *, turns: int, conversation_id: str = "c1") -> None:
    for index in range(turns):
        context = context_manager.load_for_turn(
            user_id="u1",
            conversation_id=conversation_id,
            user_message=f"user-{index}",
        )
        context_manager.commit_turn(
            context=context,
            task=ConversationTaskState(),
            assistant_message=f"assistant-{index}",
        )


def test_pruning_removes_covered_rows_and_leaves_the_read_window_intact(
    tmp_path,
) -> None:
    """The read window survives a prune, because no summary covers it yet."""
    context_manager = manager(
        tmp_path,
        limit=4,
        summary_worker=RecordingSummaryWorker(),
        max_recent_context_chars=32,
    )
    _talk(context_manager, turns=20)
    before = context_manager.load_for_turn(
        user_id="u1", conversation_id="c1", user_message="next"
    )

    deleted = context_manager._store.prune_compacted_messages(
        user_id="u1", conversation_id="c1"
    )

    assert deleted > 0
    after = context_manager.load_for_turn(
        user_id="u1", conversation_id="c1", user_message="next"
    )
    assert [message.content for message in after.recent_messages] == [
        message.content for message in before.recent_messages
    ]
    assert after.conversation_summary == before.conversation_summary
    assert context_manager._store.count_compacted_messages(
        user_id="u1", conversation_id="c1"
    ) == (0, 0)


def test_pruning_one_conversation_leaves_the_others_alone(tmp_path) -> None:
    """Scoping matters: a summary in one session says nothing about another."""
    context_manager = manager(
        tmp_path,
        limit=4,
        summary_worker=RecordingSummaryWorker(),
        max_recent_context_chars=32,
    )
    _talk(context_manager, turns=12, conversation_id="c1")
    _talk(context_manager, turns=12, conversation_id="c2")
    other_before, _ = context_manager._store.count_compacted_messages(
        user_id="u1", conversation_id="c2"
    )

    context_manager._store.prune_compacted_messages(user_id="u1", conversation_id="c1")

    assert context_manager._store.count_compacted_messages(
        user_id="u1", conversation_id="c1"
    ) == (0, 0)
    assert (
        context_manager._store.count_compacted_messages(
            user_id="u1", conversation_id="c2"
        )[0]
        == other_before
    )
    # Omitting the conversation covers every one of them.
    context_manager._store.prune_compacted_messages(user_id="u1")
    assert context_manager._store.count_compacted_messages(user_id="u1") == (0, 0)


def test_pruning_never_touches_a_message_no_summary_covers(tmp_path) -> None:
    """The bound comes from the stored summary, not from the caller.

    A turn committed after the summary was written is not covered by it, so it
    has to survive a prune that runs immediately afterwards.
    """
    store = CareerContextStore(tmp_path / "context.sqlite3")
    # A worker that always fails: the summariser is configured, so the unrelated
    # newest-N trim stays off, but no summary is written except the one below.
    context_manager = manager(tmp_path, limit=4, summary_worker=_FailingWorker())
    context_manager.upsert_profile(CareerProfileContext(user_id="u1"))
    _talk(context_manager, turns=3)
    store.compact_conversation_summary(
        user_id="u1",
        conversation_id="c1",
        expected_previous_through_sequence=0,
        content=ConversationSummaryContent(
            user_goals=("g",),
            confirmed_decisions=(),
            unresolved_questions=(),
            active_constraints=(),
        ),
        through_sequence=2,
    )
    _talk(context_manager, turns=1)

    store.prune_compacted_messages(user_id="u1", conversation_id="c1")

    remaining = store.list_messages_after(
        user_id="u1", conversation_id="c1", after_sequence=0, limit=50
    )
    assert [message.sequence for message in remaining] == [3, 4, 5, 6, 7, 8]


def test_the_notice_appears_only_past_the_threshold_and_clears_after_pruning(
    tmp_path,
) -> None:
    """The operator is told once it is worth acting on, and not before."""
    context_manager = manager(
        tmp_path,
        limit=4,
        summary_worker=RecordingSummaryWorker(),
        max_recent_context_chars=32,
        compacted_message_warning_threshold=20,
    )
    _talk(context_manager, turns=8)

    assert context_manager.compacted_message_notice(user_id="u1") is None

    _talk(context_manager, turns=12)
    notice = context_manager.compacted_message_notice(user_id="u1")

    assert notice is not None
    assert "context prune" in notice
    context_manager._store.prune_compacted_messages(user_id="u1")
    assert context_manager.compacted_message_notice(user_id="u1") is None


def test_a_nonpositive_threshold_is_rejected(tmp_path) -> None:
    """Zero would warn on every turn, which trains the operator to ignore it."""
    with pytest.raises(ValueError):
        manager(tmp_path, compacted_message_warning_threshold=0)
