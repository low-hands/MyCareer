from datetime import datetime, timedelta, timezone
import sqlite3

import pytest

from career_agent.domain.memory_scope import CanonicalScope, ScopeProposal
from career_agent.evaluation.memory_metrics import summarize_unresolved_key_backlog
from career_agent.services.canonical_scope import CanonicalScopeResolver
from career_agent.services.memory_scope import (
    MemoryScopeWriteGate,
    UnresolvedMemoryScopeError,
)
from career_agent.storage.scope_resolution import SQLiteScopeResolutionStore


def _unresolved_proposal(
    value: str = "主要工作地点是杭州",
    *,
    conversation_id: str | None = None,
) -> ScopeProposal:
    return ScopeProposal(
        user_id="u1",
        conversation_id=conversation_id,
        family="career_evidence",
        subject_id="career_record_1",
        relation="主要工作地点",
        proposed_value=value,
        source_kind="career_evidence",
        source_id="career_evidence_1",
    )


def test_unresolved_writes_fail_closed_and_enqueue_idempotently(tmp_path) -> None:
    store = SQLiteScopeResolutionStore(tmp_path / "context.sqlite3")
    gate = MemoryScopeWriteGate(CanonicalScopeResolver(), store)

    first = gate.admit(_unresolved_proposal())
    second = gate.admit(_unresolved_proposal())

    assert first.admitted is False
    assert first.queue_item is not None
    assert second.queue_item is not None
    assert second.queue_item.id == first.queue_item.id
    assert len(store.list_open(user_id="u1")) == 1
    assert [event.event_type for event in store.list_events(
        user_id="u1", queue_item_id=first.queue_item.id
    )] == ["enqueued"]


def test_duplicate_proposal_keeps_one_item_and_routes_to_latest_conversation(
    tmp_path,
) -> None:
    store = SQLiteScopeResolutionStore(tmp_path / "context.sqlite3")
    gate = MemoryScopeWriteGate(CanonicalScopeResolver(), store)

    first = gate.admit(_unresolved_proposal(conversation_id="conversation-1"))
    second = gate.admit(_unresolved_proposal(conversation_id="conversation-2"))

    assert first.queue_item is not None
    assert second.queue_item is not None
    assert second.queue_item.id == first.queue_item.id
    assert second.queue_item.conversation_id == "conversation-2"
    assert len(store.list_open(user_id="u1")) == 1


def test_require_raises_only_after_the_candidate_is_durable(tmp_path) -> None:
    store = SQLiteScopeResolutionStore(tmp_path / "context.sqlite3")
    gate = MemoryScopeWriteGate(CanonicalScopeResolver(), store)

    with pytest.raises(UnresolvedMemoryScopeError) as error:
        gate.require(_unresolved_proposal())

    assert store.get(
        user_id="u1", queue_item_id=error.value.queue_item.id
    ) is not None


def test_clarification_then_resolution_preserves_an_audit_chain(tmp_path) -> None:
    store = SQLiteScopeResolutionStore(tmp_path / "context.sqlite3")
    item = store.enqueue(CanonicalScopeResolver().resolve(_unresolved_proposal()))

    requested = store.request_clarification(
        user_id="u1", queue_item_id=item.id
    )
    resolved = store.resolve(
        user_id="u1",
        queue_item_id=item.id,
        canonical_scope=CanonicalScope(
            family="career_evidence",
            subject_id="career_record_1",
            relation="work_city",
            scope_key="career_evidence/career_record_1/work_city",
        ),
    )

    assert requested.status == "clarification_requested"
    assert requested.clarification_attempts == 1
    assert resolved.status == "resolved"
    assert resolved.resolved_scope_key.endswith("/work_city")
    assert store.list_open(user_id="u1") == ()
    assert [event.event_type for event in store.list_events(
        user_id="u1", queue_item_id=item.id
    )] == ["enqueued", "clarification_requested", "resolved"]


def test_terminal_items_cannot_be_reinterpreted(tmp_path) -> None:
    store = SQLiteScopeResolutionStore(tmp_path / "context.sqlite3")
    item = store.enqueue(CanonicalScopeResolver().resolve(_unresolved_proposal()))
    store.dismiss(user_id="u1", queue_item_id=item.id, reason="Not memory")

    with pytest.raises(ValueError, match="cannot be resolved"):
        store.resolve(
            user_id="u1",
            queue_item_id=item.id,
            canonical_scope=CanonicalScope(
                family="career_evidence",
                subject_id="career_record_1",
                relation="work_city",
                scope_key="career_evidence/career_record_1/work_city",
            ),
        )


def test_backlog_metrics_expose_count_age_and_resolution_latency(tmp_path) -> None:
    store = SQLiteScopeResolutionStore(tmp_path / "context.sqlite3")
    first = store.enqueue(CanonicalScopeResolver().resolve(_unresolved_proposal()))
    second = store.enqueue(
        CanonicalScopeResolver().resolve(
            _unresolved_proposal("工作城市最终需要确认")
        )
    )
    store.request_clarification(user_id="u1", queue_item_id=second.id)
    store.resolve(
        user_id="u1",
        queue_item_id=first.id,
        canonical_scope=CanonicalScope(
            family="career_evidence",
            subject_id="career_record_1",
            relation="work_city",
            scope_key="career_evidence/career_record_1/work_city",
        ),
    )
    items = store.list_all(user_id="u1")
    summary = summarize_unresolved_key_backlog(
        items,
        now=max(item.updated_at for item in items) + timedelta(seconds=10),
    )

    assert summary.open_count == 1
    assert summary.clarification_requested_count == 1
    assert summary.oldest_open_age_seconds is not None
    assert summary.oldest_open_age_seconds >= 10
    assert summary.resolved_count == 1
    assert summary.median_resolution_latency_seconds is not None


def test_scope_store_is_a_separate_component_in_the_context_file(tmp_path) -> None:
    path = tmp_path / "context.sqlite3"
    SQLiteScopeResolutionStore(path)

    with sqlite3.connect(path) as connection:
        version = connection.execute(
            "SELECT version FROM schema_versions WHERE component = 'memory_scope'"
        ).fetchone()[0]
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }

    assert version == 2
    assert {"scope_resolution_queue", "scope_resolution_events"} <= tables


def test_v1_scope_queue_adds_conversation_routing_without_losing_items(
    tmp_path,
) -> None:
    path = tmp_path / "context.sqlite3"
    store = SQLiteScopeResolutionStore(path)
    item = store.enqueue(CanonicalScopeResolver().resolve(_unresolved_proposal()))
    with sqlite3.connect(path) as connection:
        connection.execute(
            "ALTER TABLE scope_resolution_queue DROP COLUMN conversation_id"
        )
        connection.execute(
            "UPDATE schema_versions SET version = 1 "
            "WHERE component = 'memory_scope'"
        )

    reopened = SQLiteScopeResolutionStore(path)

    assert reopened.get(user_id="u1", queue_item_id=item.id) is not None
    with sqlite3.connect(path) as connection:
        columns = {
            row[1]
            for row in connection.execute(
                "PRAGMA table_info(scope_resolution_queue)"
            )
        }
    assert "conversation_id" in columns
