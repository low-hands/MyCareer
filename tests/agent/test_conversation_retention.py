"""Conversation compression remains reconstructable from original messages."""

from __future__ import annotations

from career_agent.agent.main_agent_contracts import ConversationTaskState

from test_context_manager import RecordingSummaryWorker, manager


def test_compaction_never_deletes_covered_messages(tmp_path) -> None:
    context_manager = manager(
        tmp_path,
        limit=4,
        summary_worker=RecordingSummaryWorker(),
        max_recent_context_chars=32,
    )
    for index in range(20):
        context = context_manager.load_for_turn(
            user_id="u1",
            conversation_id="c1",
            user_message=f"user-{index}",
        )
        context_manager.commit_turn(
            context=context,
            task=ConversationTaskState(),
            assistant_message=f"assistant-{index}",
        )

    summary = context_manager._store.get_conversation_summary(
        user_id="u1",
        conversation_id="c1",
    )
    messages = context_manager._store.list_messages_after(
        user_id="u1",
        conversation_id="c1",
        after_sequence=0,
        limit=100,
    )

    assert summary is not None
    assert summary.through_sequence > 0
    assert [message.sequence for message in messages] == list(range(1, 41))
