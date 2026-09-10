from datetime import datetime, timezone

from career_agent.agent.context_manager import ContextManager
from career_agent.agent.conversation_memory_contracts import ConversationSummaryContent
from career_agent.agent.main_agent_contracts import (
    ConversationMessageContext,
    ConversationTaskState,
)
from career_agent.domain.episodes import CareerEpisodeDraft
from career_agent.storage.context import CareerContextStore
from career_agent.storage.episodes import SQLiteCareerEpisodeStore


def _commit(
    store: CareerContextStore,
    *,
    conversation_id: str,
    user_message: str,
    assistant_message: str,
    scope_keys: tuple[str, ...] = (),
) -> None:
    now = datetime.now(timezone.utc)
    store.commit_turn(
        user_id="u1",
        conversation_id=conversation_id,
        task=ConversationTaskState(),
        user_message=ConversationMessageContext(
            role="user", content=user_message, created_at=now
        ),
        assistant_message=ConversationMessageContext(
            role="assistant", content=assistant_message, created_at=now
        ),
        memory_scope_keys=scope_keys,
    )


def test_purge_derived_memory_clears_indexes_and_sets_summary_floor(tmp_path) -> None:
    path = tmp_path / "context.sqlite3"
    store = CareerContextStore(path)
    manager = ContextManager(store)
    manager.load_for_turn(
        user_id="u1",
        conversation_id="c1",
        user_message="My deleted internship was at Private Corp.",
    )
    scope_key = "career_evidence/record-1/claim"
    _commit(
        store,
        conversation_id="c1",
        user_message="My deleted internship was at Private Corp.",
        assistant_message="I will remember the Private Corp. internship.",
        scope_keys=(scope_key,),
    )
    assert store.compact_conversation_summary(
        user_id="u1",
        conversation_id="c1",
        expected_previous_through_sequence=0,
        content=ConversationSummaryContent(
            confirmed_decisions=("Private Corp. internship",),
        ),
        through_sequence=2,
    )
    episodes = SQLiteCareerEpisodeStore(path)
    episode = episodes.upsert(
        CareerEpisodeDraft(
            user_id="u1",
            kind="mock_interview",
            source_run_id="mock-1",
            occurred_at=datetime.now(timezone.utc),
            title="Private Corp. interview",
            summary="Discussed the deleted internship.",
            conversation_id="c1",
        ),
        memory_scope_keys=(scope_key,),
    )

    result = store.purge_derived_memory(user_id="u1", scope_key=scope_key)

    assert result == {
        "conversation_fragments": 2,
        "conversation_summaries": 1,
        "career_episodes": 1,
        "affected_conversations": 1,
        "memory_review_items": 0,
    }
    assert store.get_conversation_summary(user_id="u1", conversation_id="c1") is None
    assert store.list_messages_after(
        user_id="u1",
        conversation_id="c1",
        after_sequence=0,
        limit=10,
    ) == ()
    assert store.read_conversation_span(
        user_id="u1",
        conversation_id="c1",
        from_sequence=1,
        through_sequence=2,
    ).messages == ()
    assert episodes.get_by_source(
        user_id="u1",
        kind="mock_interview",
        source_run_id=episode.source_run_id,
    ) is None
    assert episodes.search(user_id="u1", query="Private") == ()

    next_context = manager.load_for_turn(
        user_id="u1",
        conversation_id="c1",
        user_message="Start fresh.",
    )
    assert next_context.recent_messages == ()
    manager.commit_turn(
        context=next_context,
        task=ConversationTaskState(),
        assistant_message="Fresh context acknowledged.",
    )
    assert [
        item.sequence
        for item in store.list_messages_after(
            user_id="u1",
            conversation_id="c1",
            after_sequence=0,
            limit=10,
        )
    ] == [3, 4]


def test_scope_purge_does_not_hide_an_unrelated_conversation(tmp_path) -> None:
    path = tmp_path / "context.sqlite3"
    store = CareerContextStore(path)
    manager = ContextManager(store)
    scope_key = "career_evidence/record-1/claim"
    for conversation_id, user_message, assistant_message, scopes in (
        ("conv-a", "SECRET-SALARY-0", "salary noted", (scope_key,)),
        ("conv-b", "Unrelated interview", "interview noted", ()),
    ):
        manager.load_for_turn(
            user_id="u1",
            conversation_id=conversation_id,
            user_message=user_message,
        )
        _commit(
            store,
            conversation_id=conversation_id,
            user_message=user_message,
            assistant_message=assistant_message,
            scope_keys=scopes,
        )
    _commit(
        store,
        conversation_id="conv-a",
        user_message="Later planning only",
        assistant_message="planning noted",
    )

    store.purge_derived_memory(user_id="u1", scope_key=scope_key)

    conversations = {
        item.conversation_id: item for item in store.list_conversations(user_id="u1")
    }
    assert conversations["conv-b"].title == "Unrelated interview"
    assert conversations["conv-b"].message_count == 2
    assert conversations["conv-a"].title == "Later planning only"
    assert conversations["conv-a"].message_count == 2
    assert store.read_conversation_span(
        user_id="u1",
        conversation_id="conv-b",
        from_sequence=1,
        through_sequence=2,
    ).total == 2
    assert [
        message.content
        for message in store.read_conversation_span(
            user_id="u1",
            conversation_id="conv-a",
            from_sequence=1,
            through_sequence=4,
        ).messages
    ] == ["Later planning only", "planning noted"]


def test_summary_rebuild_uses_retained_unsuppressed_messages(tmp_path) -> None:
    store = CareerContextStore(tmp_path / "context.sqlite3")
    first_scope = "career_evidence/record-1/claim"
    second_scope = "career_evidence/record-2/claim"
    later_scope = "career_evidence/record-3/claim"
    _commit(
        store,
        conversation_id="c1",
        user_message="Delete this memory",
        assistant_message="First memory noted",
        scope_keys=(first_scope,),
    )
    _commit(
        store,
        conversation_id="c1",
        user_message="Keep this memory",
        assistant_message="Second memory noted",
        scope_keys=(second_scope,),
    )
    assert store.compact_conversation_summary(
        user_id="u1",
        conversation_id="c1",
        expected_previous_through_sequence=0,
        content=ConversationSummaryContent(
            confirmed_decisions=("First and second memories",),
        ),
        through_sequence=4,
    )

    store.purge_derived_memory(user_id="u1", scope_key=first_scope)

    assert store.get_conversation_summary(
        user_id="u1", conversation_id="c1"
    ) is None
    assert [
        message.content
        for message in store.list_messages_after(
            user_id="u1",
            conversation_id="c1",
            after_sequence=0,
            limit=10,
        )
    ] == ["Keep this memory", "Second memory noted"]
    assert store.compact_conversation_summary(
        user_id="u1",
        conversation_id="c1",
        expected_previous_through_sequence=0,
        content=ConversationSummaryContent(
            confirmed_decisions=("Second memory",),
        ),
        through_sequence=4,
    )
    _commit(
        store,
        conversation_id="c1",
        user_message="Later disposable memory",
        assistant_message="Later memory noted",
        scope_keys=(later_scope,),
    )

    store.purge_derived_memory(user_id="u1", scope_key=later_scope)

    rebuilt = store.get_conversation_summary(
        user_id="u1", conversation_id="c1"
    )
    assert rebuilt is not None
    assert rebuilt.content.confirmed_decisions == ("Second memory",)


def test_v8_migration_drops_ignored_legacy_cutoff_table(tmp_path) -> None:
    path = tmp_path / "context.sqlite3"
    store = CareerContextStore(path)
    manager = ContextManager(store)
    manager.load_for_turn(
        user_id="u1",
        conversation_id="c1",
        user_message="Keep this history",
    )
    _commit(
        store,
        conversation_id="c1",
        user_message="Keep this history",
        assistant_message="history kept",
    )
    with store._connect() as connection:
        connection.execute(
            """
            CREATE TABLE memory_deletion_cutoffs (
                user_id TEXT NOT NULL,
                conversation_id TEXT NOT NULL,
                through_sequence INTEGER NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(user_id, conversation_id)
            )
            """
        )
        connection.execute(
            """
            INSERT INTO memory_deletion_cutoffs(
                user_id, conversation_id, through_sequence, updated_at
            ) VALUES (?, ?, ?, ?)
            """,
            ("u1", "c1", 2, datetime.now(timezone.utc).isoformat()),
        )
        connection.execute(
            "UPDATE schema_versions SET version = 7 WHERE component = 'agent_context'"
        )

    store = CareerContextStore(path)

    span = store.read_conversation_span(
        user_id="u1",
        conversation_id="c1",
        from_sequence=1,
        through_sequence=2,
    )
    assert span.total == 2
    assert [message.content for message in span.messages] == [
        "Keep this history",
        "history kept",
    ]
    with store._connect() as connection:
        assert connection.execute(
            """
            SELECT 1 FROM sqlite_master
            WHERE type = 'table' AND name = 'memory_deletion_cutoffs'
            """
        ).fetchone() is None


def test_free_text_marker_does_not_hide_unbound_unrelated_text(tmp_path) -> None:
    path = tmp_path / "context.sqlite3"
    store = CareerContextStore(path)
    _commit(
        store,
        conversation_id="c1",
        user_message="Private Corp. is still my current employer.",
        assistant_message="noted",
    )

    store.purge_derived_memory(
        user_id="u1",
        scope_key="career_evidence/other/claim",
        lineage_markers=(
            "Private Corp.",
            "This is a deliberately long claim that is still free text.",
        ),
    )

    span = store.read_conversation_span(
        user_id="u1",
        conversation_id="c1",
        from_sequence=1,
        through_sequence=2,
    )
    assert span.total == 2


def test_opaque_marker_hides_matching_unbound_legacy_message(tmp_path) -> None:
    path = tmp_path / "context.sqlite3"
    store = CareerContextStore(path)
    marker = "detail_" + "a" * 24
    _commit(
        store,
        conversation_id="c1",
        user_message=f"legacy readback {marker}",
        assistant_message="unrelated reply",
    )

    store.purge_derived_memory(
        user_id="u1",
        scope_key="career_evidence/other/claim",
        lineage_markers=(marker,),
    )

    span = store.read_conversation_span(
        user_id="u1",
        conversation_id="c1",
        from_sequence=1,
        through_sequence=2,
    )
    assert [message.content for message in span.messages] == ["unrelated reply"]


def test_scope_purge_allows_backdated_unrelated_episode(tmp_path) -> None:
    path = tmp_path / "context.sqlite3"
    context = CareerContextStore(path)
    episodes = SQLiteCareerEpisodeStore(path)
    scope_key = "career_evidence/record-1/claim"
    deleted = CareerEpisodeDraft(
        user_id="u1",
        kind="interview_round",
        source_run_id="round-deleted",
        occurred_at=datetime(2026, 9, 7, tzinfo=timezone.utc),
        title="Deleted claim interview",
        summary="Derived from the claim.",
    )
    assert episodes.upsert(deleted, memory_scope_keys=(scope_key,)) is not None

    context.purge_derived_memory(user_id="u1", scope_key=scope_key)

    assert episodes.upsert(
        deleted.model_copy(update={"summary": "Delayed stale re-derivation."}),
        memory_scope_keys=(scope_key,),
    ) is None
    assert episodes.get_by_source(
        user_id="u1",
        kind="interview_round",
        source_run_id="round-deleted",
    ) is None

    backdated = deleted.model_copy(
        update={
            "source_run_id": "round-unrelated",
            "title": "Yesterday's unrelated interview",
            "summary": "Independent event added after cleanup.",
        }
    )
    assert episodes.upsert(backdated) is not None
    assert episodes.get_by_source(
        user_id="u1",
        kind="interview_round",
        source_run_id="round-unrelated",
    ) is not None
