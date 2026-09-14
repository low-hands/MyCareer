from datetime import datetime, timedelta, timezone
from pathlib import Path
import sqlite3

import pytest

from career_agent.agent.context_manager import ContextManager
from career_agent.agent.delivered_body_contracts import BodyDependency
from career_agent.agent.main_agent_contracts import (
    ConversationMessageContext,
    ConversationTaskState,
)
from career_agent.agent.main_agent_runtime import MainAgentRuntime
from career_agent.agent.main_agent_tools import MainAgentToolRegistry
from career_agent.harness.streaming import ContentDeltaEvent, TurnCompletedEvent
from career_agent.storage.context import CareerContextStore, DeliveredBodyDraft
from career_agent.storage.turn_receipts import SQLiteTurnReceiptStore


KEY = {"user_id": "u1", "conversation_id": "c1", "request_id": "request-1"}
SCOPE = "career_evidence/private/claim"


class _NoDecisions:
    def decide(self, context, tool_names):
        raise AssertionError("a committed request must not execute another turn")


def _runtime(path: Path) -> MainAgentRuntime:
    return MainAgentRuntime(
        context_manager=ContextManager(CareerContextStore(path)),
        decision_maker=_NoDecisions(),
        tools=MainAgentToolRegistry(),
        turn_receipt_store=SQLiteTurnReceiptStore(path),
    )


def _commit(
    store: CareerContextStore,
    *,
    conversation_id: str = "c1",
    scope_keys: tuple[str, ...] = (),
    dependencies: tuple[BodyDependency, ...] = (),
    turn_id: str | None = None,
) -> None:
    ContextManager(store).load_for_turn(
        user_id="u1", conversation_id=conversation_id, user_message="查看简报"
    )
    now = datetime.now(timezone.utc)
    store.commit_turn(
        user_id="u1",
        conversation_id=conversation_id,
        task=ConversationTaskState(),
        user_message=ConversationMessageContext(
            role="user", content="查看简报", created_at=now
        ),
        assistant_message=ConversationMessageContext(
            role="assistant", content="简报已展示", created_at=now
        ),
        memory_scope_keys=scope_keys,
        turn_id=turn_id,
        assistant_bodies=(
            DeliveredBodyDraft(
                kind="daily_brief_ready",
                title="每日简报",
                retention="snapshot",
                body="PRIVATE BODY",
                dependencies=dependencies,
            ),
        ),
    )


@pytest.mark.parametrize("purge_first", [False, True])
def test_memory_purge_physically_removes_bodies_in_both_commit_orders(
    tmp_path: Path, purge_first: bool
) -> None:
    path = tmp_path / "context.sqlite3"
    store = CareerContextStore(path)
    receipts = SQLiteTurnReceiptStore(path)
    receipts.begin(**KEY, turn_id="turn-1")
    if purge_first:
        store.purge_derived_memory(user_id="u1", scope_key=SCOPE)
    _commit(store, scope_keys=(SCOPE,), turn_id="turn-1")
    if not purge_first:
        store.purge_derived_memory(user_id="u1", scope_key=SCOPE)
    receipts.commit(
        **KEY,
        turn_id="turn-1",
        events=(
            ContentDeltaEvent(delta="PRIVATE BODY"),
            TurnCompletedEvent(turn_id="turn-1"),
        ),
    )

    assert store.list_delivered_body_references("u1", "c1", from_sequence=1) == ()
    assert (
        store.list_messages_after(
            user_id="u1", conversation_id="c1", after_sequence=0, limit=10
        )
        == ()
    )
    with sqlite3.connect(path) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM conversation_delivered_bodies"
        ).fetchone() == (0,)
        assert connection.execute(
            "SELECT status, events_json FROM turn_receipts"
        ).fetchone() == ("COMMITTED", "[]")
    replay = receipts.begin(**KEY, turn_id="turn-2")
    assert replay.turn_id == "turn-1"
    assert replay.content_status == "deleted"


def _settle_receipts(
    receipts: SQLiteTurnReceiptStore, *keys: tuple[str, str]
) -> None:
    for conversation_id, request_id in keys:
        key = dict(KEY, conversation_id=conversation_id, request_id=request_id)
        receipts.begin(**key, turn_id=request_id)
        receipts.commit(
            **key, turn_id=request_id, events=(ContentDeltaEvent(delta="PRIVATE"),)
        )


def test_memory_purge_redacts_only_the_receipts_of_the_turns_it_hid(
    tmp_path: Path,
) -> None:
    path = tmp_path / "context.sqlite3"
    store, receipts = CareerContextStore(path), SQLiteTurnReceiptStore(path)
    _commit(store, scope_keys=(SCOPE,), turn_id="r1")
    _commit(store, turn_id="r2")
    _commit(store, conversation_id="other", turn_id="r3")
    _settle_receipts(receipts, ("c1", "r1"), ("c1", "r2"), ("other", "r3"))
    store.purge_derived_memory(user_id="u1", scope_key=SCOPE)
    hidden = receipts.get(**dict(KEY, request_id="r1"))
    assert hidden.status == "COMMITTED" and hidden.content_status == "deleted"
    assert hidden.events == ()
    for conversation_id, request_id in (("c1", "r2"), ("other", "r3")):
        kept = receipts.get(
            **dict(KEY, conversation_id=conversation_id, request_id=request_id)
        )
        assert kept.content_status == "available" and kept.events
    assert len(store.list_delivered_body_references("u1", "c1", from_sequence=1)) == 1
    assert (
        len(store.list_delivered_body_references("u1", "other", from_sequence=1)) == 1
    )


def test_deleting_a_dependency_redacts_only_the_turn_that_showed_it(
    tmp_path: Path,
) -> None:
    path = tmp_path / "context.sqlite3"
    store, receipts = CareerContextStore(path), SQLiteTurnReceiptStore(path)
    dependency = BodyDependency(kind="job", resource_id="job-1")
    _commit(store, dependencies=(dependency,), turn_id="r1")
    _commit(store, turn_id="r2")
    _settle_receipts(receipts, ("c1", "r1"), ("c1", "r2"))
    store.purge_delivered_body_dependency(user_id="u1", dependency=dependency)
    assert receipts.get(**dict(KEY, request_id="r1")).content_status == "deleted"
    assert receipts.get(**dict(KEY, request_id="r2")).content_status == "available"
    # A later turn in the same conversation is untouched by the earlier deletion.
    _settle_receipts(receipts, ("c1", "r4"))
    assert receipts.get(**dict(KEY, request_id="r4")).events


def test_dependency_deleted_mid_turn_redacts_that_running_turn_only(
    tmp_path: Path,
) -> None:
    path = tmp_path / "context.sqlite3"
    store, receipts = CareerContextStore(path), SQLiteTurnReceiptStore(path)
    dependency = BodyDependency(kind="job", resource_id="job-1")
    _commit(store, turn_id="r1")
    _settle_receipts(receipts, ("c1", "r1"))
    receipts.begin(**dict(KEY, request_id="r2"), turn_id="r2")
    store.purge_delivered_body_dependency(user_id="u1", dependency=dependency)
    _commit(store, dependencies=(dependency,), turn_id="r2")
    receipts.commit(
        **dict(KEY, request_id="r2"),
        turn_id="r2",
        events=(ContentDeltaEvent(delta="PRIVATE BODY"),),
    )
    assert receipts.get(**dict(KEY, request_id="r2")).content_status == "deleted"
    assert receipts.get(**dict(KEY, request_id="r1")).content_status == "available"


def test_rows_without_a_turn_fall_back_to_the_conversation_up_to_now(
    tmp_path: Path,
) -> None:
    path = tmp_path / "context.sqlite3"
    store, receipts = CareerContextStore(path), SQLiteTurnReceiptStore(path)
    _commit(store, scope_keys=(SCOPE,))
    _settle_receipts(receipts, ("c1", "r1"))
    store.purge_derived_memory(user_id="u1", scope_key=SCOPE)
    assert receipts.get(**dict(KEY, request_id="r1")).content_status == "deleted"
    _settle_receipts(receipts, ("c1", "r2"))
    assert receipts.get(**dict(KEY, request_id="r2")).content_status == "available"


def test_conversation_deletion_keeps_receipt_key_before_or_after_settlement(
    tmp_path: Path,
) -> None:
    path = tmp_path / "context.sqlite3"
    store, receipts = CareerContextStore(path), SQLiteTurnReceiptStore(path)
    _commit(store)
    body_id = store.list_delivered_body_references("u1", "c1", from_sequence=1)[
        0
    ].body_id
    receipts.begin(**KEY, turn_id="turn-1")
    assert store.delete_conversation(user_id="u1", conversation_id="c1")
    receipts.commit(
        **KEY, turn_id="turn-1", events=(ContentDeltaEvent(delta="PRIVATE"),)
    )
    assert store.get_delivered_body("u1", body_id) is None
    replay = receipts.begin(**KEY, turn_id="turn-2")
    assert replay.status == "COMMITTED"
    assert replay.turn_id == "turn-1"
    assert replay.events == ()
    events = []
    result = _runtime(path).run_turn(
        **KEY, user_message="重试", event_sink=events.append
    )
    assert result.assistant_message == "内容已删除。"
    assert [event.type for event in events] == [
        "turn_started",
        "content_delta",
        "turn_completed",
    ]
    assert events[-1].turn_id == "turn-1"


@pytest.mark.parametrize("source_deadline", [False, True])
def test_receipt_expiry_preserves_turn_identity_and_physically_clears_events(
    tmp_path: Path, source_deadline: bool
) -> None:
    path = tmp_path / "context.sqlite3"
    store = SQLiteTurnReceiptStore(path)
    now = datetime.now(timezone.utc)
    deadline = now + (timedelta(minutes=5) if source_deadline else timedelta(hours=24))
    store.begin(**KEY, turn_id="turn-1", now=now)
    store.commit(
        **KEY,
        turn_id="turn-1",
        now=now,
        events=(ContentDeltaEvent(delta="PRIVATE"),),
        body_expires_at=deadline if source_deadline else None,
    )
    assert store.get(**KEY, now=deadline - timedelta(seconds=1)).events
    replay = store.begin(**KEY, turn_id="turn-2", now=deadline)
    assert replay.status == "COMMITTED" and replay.turn_id == "turn-1"
    assert replay.content_status == "expired" and replay.events == ()
    with sqlite3.connect(path) as connection:
        assert connection.execute(
            "SELECT events_json FROM turn_receipts"
        ).fetchone() == ("[]",)
    events = []
    result = _runtime(path).run_turn(
        **KEY, user_message="重试", event_sink=events.append
    )
    assert "过期" in result.assistant_message
    assert "PRIVATE" not in result.assistant_message
    assert events[-1].type == "turn_completed"


def test_deleted_dependency_prevents_late_snapshot_insertion(tmp_path: Path) -> None:
    path = tmp_path / "context.sqlite3"
    store = CareerContextStore(path)
    dependency = BodyDependency(kind="job", resource_id="job-1")
    store.purge_delivered_body_dependency(user_id="u1", dependency=dependency)
    _commit(store, dependencies=(dependency,))
    assert store.list_delivered_body_references("u1", "c1", from_sequence=1) == ()
    assert (
        len(
            store.list_messages_after(
                user_id="u1", conversation_id="c1", after_sequence=0, limit=10
            )
        )
        == 2
    )
    with sqlite3.connect(path) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM conversation_delivered_bodies"
        ).fetchone() == (0,)


def test_upgrade_removes_legacy_copies_without_losing_committed_receipts(
    tmp_path: Path,
) -> None:
    path = tmp_path / "context.sqlite3"
    store, receipts = CareerContextStore(path), SQLiteTurnReceiptStore(path)
    _commit(store)
    receipts.begin(**KEY, turn_id="turn-1")
    receipts.commit(
        **KEY, turn_id="turn-1", events=(ContentDeltaEvent(delta="PRIVATE"),)
    )
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE schema_versions SET version = 15 WHERE component = 'agent_context'"
        )
        connection.execute(
            "UPDATE schema_versions SET version = 1 WHERE component = 'turn_receipts'"
        )
        connection.execute(
            "UPDATE conversation_delivered_bodies SET kind = 'resume_analysis_ready'"
        )
        for name in ("retention", "source_json", "dependencies_json"):
            connection.execute(
                f"ALTER TABLE conversation_delivered_bodies DROP COLUMN {name}"
            )
        for name in ("content_status", "body_expires_at"):
            connection.execute(f"ALTER TABLE turn_receipts DROP COLUMN {name}")
    upgraded = CareerContextStore(path)
    assert upgraded.list_delivered_body_references("u1", "c1", from_sequence=1) == ()
    receipt = SQLiteTurnReceiptStore(path).get(**KEY)
    assert receipt.status == "COMMITTED" and receipt.turn_id == "turn-1"
    assert receipt.content_status == "deleted"
    assert receipt.events == ()
