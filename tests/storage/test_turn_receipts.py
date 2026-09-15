from datetime import datetime, timezone
from pathlib import Path

from career_agent.harness.streaming import (
    ContentDeltaEvent,
    ProgressEvent,
    ReportReadyEvent,
    TurnCompletedEvent,
)
from career_agent.storage.turn_receipts import (
    REPLAYED_EVENT_TYPES,
    SQLiteTurnReceiptStore,
    coalesce_content,
)


NOW = datetime(2026, 9, 13, tzinfo=timezone.utc)
KEY = {"user_id": "u1", "conversation_id": "c1", "request_id": "request-1"}


def test_the_first_attempt_owns_the_key_and_a_committed_key_hands_back_its_answer(
    tmp_path: Path,
) -> None:
    store = SQLiteTurnReceiptStore(tmp_path / "context.sqlite3")

    assert store.begin(**KEY, turn_id="turn-1", now=NOW) is None
    running = store.begin(**KEY, turn_id="turn-2", now=NOW)
    assert running is not None
    assert running.status == "RUNNING"
    assert running.turn_id == "turn-1"

    store.commit(
        **KEY,
        turn_id="turn-1",
        events=(
            ContentDeltaEvent(delta="已记录"),
            ContentDeltaEvent(delta="投递。"),
            ReportReadyEvent(
                kind="job_research_report",
                resource_id="rr-1",
                status_at_delivery="current",
            ),
            TurnCompletedEvent(turn_id="turn-1"),
        ),
        now=NOW,
    )

    replay = store.begin(**KEY, turn_id="turn-3", now=NOW)
    assert replay is not None
    assert replay.status == "COMMITTED"
    assert replay.turn_id == "turn-1"
    assert replay.settled_at == NOW
    assert [event.type for event in replay.events] == [
        "content_delta",
        "report_ready",
        "turn_completed",
    ]
    assert replay.events[0].delta == "已记录投递。"


def test_a_failed_attempt_releases_the_key_to_the_next_attempt(tmp_path: Path) -> None:
    store = SQLiteTurnReceiptStore(tmp_path / "context.sqlite3")
    assert store.begin(**KEY, turn_id="turn-1", now=NOW) is None
    store.fail(**KEY, turn_id="turn-1", now=NOW)

    assert store.begin(**KEY, turn_id="turn-2", now=NOW) is None
    owner = store.get(**KEY)
    assert owner is not None
    assert owner.turn_id == "turn-2"
    assert owner.status == "RUNNING"


def test_only_the_owning_attempt_can_settle_a_receipt(tmp_path: Path) -> None:
    store = SQLiteTurnReceiptStore(tmp_path / "context.sqlite3")
    assert store.begin(**KEY, turn_id="turn-1", now=NOW) is None

    store.commit(**KEY, turn_id="turn-9", events=(TurnCompletedEvent(turn_id="turn-9"),))
    assert store.get(**KEY).status == "RUNNING"

    store.commit(**KEY, turn_id="turn-1", events=(TurnCompletedEvent(turn_id="turn-1"),))
    store.fail(**KEY, turn_id="turn-1")
    assert store.get(**KEY).status == "COMMITTED"


def test_orphaned_running_receipts_are_failed_and_release_their_keys(tmp_path: Path) -> None:
    """A process that dies mid-turn leaves RUNNING rows nobody will ever settle."""

    store = SQLiteTurnReceiptStore(tmp_path / "context.sqlite3")
    assert store.begin(**KEY, turn_id="turn-1", now=NOW) is None
    other = {**KEY, "conversation_id": "c2", "request_id": "request-2"}
    assert store.begin(**other, turn_id="turn-2", now=NOW) is None
    done = {**KEY, "request_id": "request-3"}
    assert store.begin(**done, turn_id="turn-3", now=NOW) is None
    store.commit(**done, turn_id="turn-3", events=(TurnCompletedEvent(turn_id="turn-3"),), now=NOW)

    # A new process starts: the lock proves nothing is executing.
    recovered = store.fail_orphaned_running(now=NOW)

    assert [(item.turn_id, item.status) for item in recovered] == [
        ("turn-1", "RUNNING"),
        ("turn-2", "RUNNING"),
    ]
    assert store.get(**KEY).status == "FAILED"
    assert store.get(**KEY).settled_at == NOW
    assert store.get(**other).status == "FAILED"
    assert store.get(**done).status == "COMMITTED"

    # The key is free again, exactly as after an in-process failure.
    assert store.begin(**KEY, turn_id="turn-4", now=NOW) is None
    fresh = store.get(**KEY)
    assert fresh.turn_id == "turn-4"
    assert store.fail_orphaned_running(now=NOW) == (fresh,)


def test_recovery_with_nothing_running_changes_nothing(tmp_path: Path) -> None:
    store = SQLiteTurnReceiptStore(tmp_path / "context.sqlite3")
    assert store.begin(**KEY, turn_id="turn-1", now=NOW) is None
    store.commit(**KEY, turn_id="turn-1", events=(TurnCompletedEvent(turn_id="turn-1"),), now=NOW)

    assert store.fail_orphaned_running(now=NOW) == ()
    assert store.get(**KEY).status == "COMMITTED"


def test_keys_are_scoped_to_user_and_conversation(tmp_path: Path) -> None:
    store = SQLiteTurnReceiptStore(tmp_path / "context.sqlite3")
    assert store.begin(**KEY, turn_id="turn-1", now=NOW) is None
    assert store.begin(**{**KEY, "conversation_id": "c2"}, turn_id="turn-2", now=NOW) is None
    assert store.begin(**{**KEY, "user_id": "u2"}, turn_id="turn-3", now=NOW) is None


def test_coalescing_keeps_order_and_drops_nothing_but_chunk_boundaries() -> None:
    events = coalesce_content(
        (
            ContentDeltaEvent(delta="a"),
            ContentDeltaEvent(delta="b\n\n"),
            TurnCompletedEvent(turn_id="t"),
        )
    )
    assert [event.type for event in events] == ["content_delta", "turn_completed"]
    assert events[0].delta == "ab\n\n"
    assert ProgressEvent not in REPLAYED_EVENT_TYPES
