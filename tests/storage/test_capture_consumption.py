from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sqlite3

from career_agent.storage.job_captures import JobCaptureIntent, SQLiteJobCaptureStore


def intent_for(store: SQLiteJobCaptureStore) -> JobCaptureIntent:
    return store.create_intent(
        user_id="u1", conversation_id="c1", source_turn_id="source-turn",
        platform="boss", keyword="AI", city=None,
    )


def record(store: SQLiteJobCaptureStore, intent: JobCaptureIntent, snapshot: str = "s1"):
    return store.record_capture(
        intent=intent, job_posting_id="j1", jd_snapshot_id=snapshot,
        title="AI", company_name="Example",
    )


def test_consumption_pins_one_snapshot_and_replays_after_expiry_and_ack(tmp_path: Path):
    store = SQLiteJobCaptureStore(tmp_path / "jobs.sqlite3")
    intent = intent_for(store)
    first = record(store, intent)
    assert first is not None and first.created
    store.acknowledge_event(user_id="u1", event_id=first.event.id)
    store.settle_continuation(
        user_id="u1", event_id=first.event.id, status="completed", turn_id="new-turn",
    )
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            "UPDATE job_capture_intents SET expires_at = ?",
            ((datetime.now(timezone.utc) - timedelta(days=1)).isoformat(),),
        )
    reopened = SQLiteJobCaptureStore(store.path)
    consumed = reopened.get_intent(user_id="u1", intent_id=intent.id)
    assert consumed is not None
    assert consumed.source_turn_id == "source-turn"
    assert consumed.consumed_at is not None
    assert consumed.consumed_event_id == first.event.id
    assert reopened.get_live_intent(user_id="u1", intent_id=intent.id) is None
    replay = record(reopened, intent)
    assert replay is not None and not replay.created
    assert replay.event.id == first.event.id
    assert replay.event.continuation_turn_id == "new-turn"
    assert record(reopened, intent, "s2") is None
    assert reopened.list_pending_continuations() == ()


def test_concurrent_saves_consume_the_intent_once(tmp_path: Path):
    path = tmp_path / "jobs.sqlite3"
    store = SQLiteJobCaptureStore(path)
    intent = intent_for(store)

    def save(snapshot: str):
        return record(SQLiteJobCaptureStore(path), intent, snapshot)

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(save, ["s1", "s2", "s1", "s2"]))
    recorded = [result for result in results if result is not None]
    assert sum(result.created for result in recorded) == 1
    assert len({result.event.id for result in recorded}) == 1
    assert len(store.list_pending_continuations()) == 1


def test_record_revalidates_owner_conversation_and_expiry(tmp_path: Path):
    store = SQLiteJobCaptureStore(tmp_path / "jobs.sqlite3")
    intent = intent_for(store)
    assert record(store, intent.model_copy(update={"user_id": "u2"})) is None
    assert record(store, intent.model_copy(update={"conversation_id": "c2"})) is None
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            "UPDATE job_capture_intents SET expires_at = ?",
            ((datetime.now(timezone.utc) - timedelta(days=1)).isoformat(),),
        )
    assert record(store, intent) is None
    assert store.list_pending_continuations() == ()


def test_ack_cannot_cancel_execution_and_retry_is_owner_scoped(tmp_path: Path):
    store = SQLiteJobCaptureStore(tmp_path / "jobs.sqlite3")
    saved = record(store, intent_for(store))
    assert saved is not None
    store.acknowledge_event(user_id="u1", event_id=saved.event.id)
    assert len(store.list_pending_continuations()) == 1
    store.settle_continuation(user_id="u1", event_id=saved.event.id, status="failed")
    assert not store.retry_continuation(user_id="u2", event_id=saved.event.id)
    assert store.retry_continuation(user_id="u1", event_id=saved.event.id)
    assert not store.retry_continuation(user_id="u1", event_id=saved.event.id)
    assert len(store.list_pending_events(user_id="u1")) == 1


# The job_captures v1 schema exactly as committed (f2bba10..a2ce819), registered
# at version 1 like every database that build created.
_COMMITTED_V1 = """
    CREATE TABLE schema_versions (
        component TEXT PRIMARY KEY, version INTEGER NOT NULL,
        updated_at TEXT NOT NULL DEFAULT (datetime('now'))
    );
    INSERT INTO schema_versions(component, version) VALUES ('job_captures', 1);
    CREATE TABLE job_capture_intents (
        id TEXT PRIMARY KEY, user_id TEXT NOT NULL, conversation_id TEXT NOT NULL,
        platform TEXT NOT NULL, keyword TEXT NOT NULL, city TEXT,
        created_at TEXT NOT NULL, expires_at TEXT NOT NULL
    );
    CREATE INDEX job_capture_intents_user_idx ON job_capture_intents(user_id, expires_at);
    CREATE TABLE job_captured_events (
        id TEXT PRIMARY KEY, user_id TEXT NOT NULL, conversation_id TEXT NOT NULL,
        intent_id TEXT NOT NULL, job_posting_id TEXT NOT NULL,
        jd_snapshot_id TEXT NOT NULL, title TEXT NOT NULL, company_name TEXT NOT NULL,
        created_at TEXT NOT NULL, acknowledged_at TEXT,
        UNIQUE(intent_id, job_posting_id),
        FOREIGN KEY(intent_id) REFERENCES job_capture_intents(id)
    );
    CREATE INDEX job_captured_events_pending_idx
        ON job_captured_events(user_id, acknowledged_at, created_at);
"""


def test_v1_database_upgrades_its_rows_to_v2(tmp_path: Path):
    path = tmp_path / "jobs.sqlite3"
    now = datetime.now(timezone.utc)
    later = (now + timedelta(seconds=1)).isoformat()
    expires = (now + timedelta(hours=1)).isoformat()
    with sqlite3.connect(path) as connection:
        connection.executescript(_COMMITTED_V1)
        for intent in ("delivered", "waiting", "unused"):
            connection.execute(
                "INSERT INTO job_capture_intents VALUES (?, 'u1', 'c1', 'boss', 'AI', NULL, ?, ?)",
                (intent, now.isoformat(), expires),
            )
        rows = (
            # A page-delivered event, then a second save under the same intent.
            ("event-1", "delivered", "job-1", "snap-1", now.isoformat(), now.isoformat()),
            ("event-2", "delivered", "job-2", "snap-2", later, None),
            # Saved while the page was closed: never delivered.
            ("event-3", "waiting", "job-3", "snap-3", now.isoformat(), None),
        )
        connection.executemany(
            "INSERT INTO job_captured_events VALUES (?, 'u1', 'c1', ?, ?, ?, 'AI', 'X', ?, ?)",
            rows,
        )
    for _ in range(2):  # the second open must be a no-op
        store = SQLiteJobCaptureStore(path)
        delivered = store.get_intent(user_id="u1", intent_id="delivered")
        waiting = store.get_intent(user_id="u1", intent_id="waiting")
        unused = store.get_intent(user_id="u1", intent_id="unused")
        assert delivered is not None and delivered.consumed_event_id == "event-1"
        assert waiting is not None and waiting.consumed_event_id == "event-3"
        assert unused is not None and unused.consumed_at is None
        statuses = {
            event_id: store.get_event(user_id="u1", event_id=event_id)
            for event_id in ("event-1", "event-2", "event-3")
        }
        assert {key: event.continuation_status for key, event in statuses.items()} == {
            "event-1": "completed", "event-2": "discarded", "event-3": "pending",
        }
        assert statuses["event-1"].title == "AI" and statuses["event-1"].company_name == "X"
        assert statuses["event-3"].jd_snapshot_id == "snap-3"
        assert [event.id for event in store.list_pending_continuations()] == ["event-3"]
    with sqlite3.connect(path) as connection:
        assert connection.execute(
            "SELECT version FROM schema_versions WHERE component = 'job_captures'"
        ).fetchone() == (2,)
        indexes = {row[1] for row in connection.execute(
            "SELECT type, name FROM sqlite_master WHERE type = 'index'"
        )}
        assert {"job_capture_intents_user_idx", "job_captured_events_pending_idx"} <= indexes
        # v2 keys events by snapshot: a new JD version of the same posting is a
        # new event, the same snapshot is still a duplicate.
        insert = (
            "INSERT INTO job_captured_events (id, user_id, conversation_id, intent_id, "
            "job_posting_id, jd_snapshot_id, title, company_name, created_at) "
            "VALUES (?, 'u1', 'c1', 'waiting', 'job-3', ?, 'AI', 'X', ?)"
        )
        connection.execute(insert, ("event-4", "snap-3b", later))
        try:
            connection.execute(insert, ("event-5", "snap-3", later))
        except sqlite3.IntegrityError:
            pass
        else:
            raise AssertionError("duplicate (intent, snapshot) was accepted")
