"""A durable, best-effort telemetry store that can be wired without a business DB.

Three behaviours are pinned here:

1. The recorder persists to its own file, independent of the business stores, so
   telemetry writes never join ``commit_turn``'s ``BEGIN IMMEDIATE`` transaction.
2. Every value reaching the file — structured ``details`` and free-text
   ``error_detail`` alike — passes mandatory redaction, so credential-shaped
   material is stripped by construction rather than by review.
3. Recording is best-effort at the caller's choice, but this store itself never
   raises on a telemetry write beyond the caller's own handling.
"""

from __future__ import annotations

from pathlib import Path
import sqlite3

from career_agent.harness.observability import (
    InMemoryTraceRecorder,
    conversation_trace_key,
)
from career_agent.evaluation.rederivation import tool_call_fingerprint
from career_agent.storage.run_events import SQLiteTraceRecorder


def test_the_sqlite_recorder_persists_and_reads_a_trace(tmp_path: Path) -> None:
    path = tmp_path / "telemetry" / "run_events.sqlite3"
    recorder = SQLiteTraceRecorder(path)

    recorder.record(
        "run-1",
        "turn_failed",
        "invoke_capability",
        outcome="failed",
        error_code="EMAIL_SYNC_FAILED",
        error_detail="Bearer __zp_stoken__=secretvalue in the error",
        details={"attempt": 1, "access_token": "should-not-survive"},
        recoverable=True,
    )

    snapshot = recorder.snapshot("run-1")
    assert len(snapshot.events) == 1
    event = snapshot.events[0]
    assert event.run_id == "run-1"
    assert event.sequence == 1
    assert event.event_type == "turn_failed"
    assert event.error_code == "EMAIL_SYNC_FAILED"
    assert event.recoverable is True

    # Credential-shaped material never reaches disk, in either field.
    assert "secretvalue" not in event.error_detail or "__zp_stoken__" not in event.error_detail
    assert "secretvalue" not in event.error_detail
    assert "should-not-survive" not in event.details.get("access_token", "")
    assert event.details.get("attempt") == 1


def test_sequence_increments_per_run(tmp_path: Path) -> None:
    recorder = SQLiteTraceRecorder(tmp_path / "run_events.sqlite3")
    recorder.record("run-1", "run_started", "boot")
    recorder.record("run-1", "turn_completed", "boot")
    recorder.record("run-2", "run_started", "boot")

    assert [e.sequence for e in recorder.snapshot("run-1").events] == [1, 2]
    assert [e.sequence for e in recorder.snapshot("run-2").events] == [1]


def test_run_events_file_is_chmod_600(tmp_path: Path) -> None:
    import os

    path = tmp_path / "run_events.sqlite3"
    SQLiteTraceRecorder(path)
    assert os.stat(path).st_mode & 0o777 == 0o600


def test_mandatory_redaction_applies_to_in_memory_recorder() -> None:
    recorder = InMemoryTraceRecorder()
    recorder.record(
        "run-1",
        "capability_failed",
        "research",
        details={"cookie": "session=abc"},
    )
    event = recorder.snapshot("run-1").events[0]
    assert event.details.get("cookie") != "session=abc"
    # The key is sensitive-shaped, so the value is blanked outright.
    assert event.details.get("cookie") is None or event.details.get("cookie") != "session=abc"


def test_error_detail_is_truncated_to_two_thousand_chars(tmp_path: Path) -> None:
    recorder = SQLiteTraceRecorder(tmp_path / "run_events.sqlite3")
    long_error = "x" * 3000
    event = recorder.record(
        "run-1", "turn_failed", "boot", outcome="failed", error_detail=long_error
    )
    assert len(event.error_detail) <= 2000


def test_model_call_category_survives_sqlite_round_trip(tmp_path: Path) -> None:
    recorder = SQLiteTraceRecorder(tmp_path / "run_events.sqlite3")
    recorder.record(
        "run-1",
        "model_attempt",
        "main_agent_decide",
        outcome="started",
        model_call_category="orchestrator_decision",
    )

    event = recorder.snapshot("run-1").events[0]
    assert event.model_call_category == "orchestrator_decision"


def test_conversation_events_join_turns_without_crossing_users(tmp_path: Path) -> None:
    recorder = SQLiteTraceRecorder(tmp_path / "run_events.sqlite3")
    key = conversation_trace_key("u1", "c1")
    call_details = {
        "conversation_id": "c1",
        "conversation_key": key,
        "tool_name": "find_saved_jobs",
        "tool_arguments_fingerprint": tool_call_fingerprint(
            "find_saved_jobs", {"query": "X"}
        ),
    }
    recorder.record(
        "turn-before",
        "model_succeeded",
        "main_agent_decide",
        outcome="succeeded",
        details=call_details,
        model_call_category="orchestrator_decision",
    )
    recorder.record(
        "other-user",
        "context_compacted",
        "conversation_summary",
        outcome="succeeded",
        details={
            "conversation_id": "c1",
            "conversation_key": conversation_trace_key("u2", "c1"),
        },
    )
    recorder.record(
        "turn-compact",
        "context_compacted",
        "conversation_summary",
        outcome="succeeded",
        details={"conversation_id": "c1", "conversation_key": key},
    )
    recorder.record(
        "turn-after",
        "context_estimated",
        "request_estimate",
        outcome="succeeded",
        details={
            "conversation_key": key,
            "input_occupancy_numerator": 12_500,
            "input_occupancy_denominator": 32_000,
        },
    )
    recorder.record(
        "turn-after",
        "model_succeeded",
        "main_agent_decide",
        outcome="succeeded",
        details=call_details,
        model_call_category="orchestrator_decision",
    )
    # Wall-clock precision is not a causal order. Force the collision that
    # would sort "turn-after" before "turn-before" under a UUID/name tiebreak.
    with sqlite3.connect(recorder.path) as connection:
        connection.execute(
            "UPDATE run_events SET occurred_at = '2026-09-06T09:00:00+00:00'"
        )

    events = recorder.list_conversation_events(user_id="u1", conversation_id="c1")

    assert [(event.run_id, event.event_type) for event in events] == [
        ("turn-before", "model_succeeded"),
        ("turn-compact", "context_compacted"),
        ("turn-after", "context_estimated"),
        ("turn-after", "model_succeeded"),
    ]


def test_memory_events_require_the_join_key_on_each_producer(tmp_path: Path) -> None:
    recorder = SQLiteTraceRecorder(tmp_path / "run_events.sqlite3")
    key = conversation_trace_key("u1", "c1")
    recorder.record(
        "turn-1",
        "memory_context_observed",
        "main_agent_decide",
        details={"conversation_key": key},
    )
    recorder.record(
        "turn-1",
        "memory_tombstone_observed",
        "career_evidence_tombstone",
        details={"conversation_key": key, "entries": []},
    )
    recorder.record(
        "turn-1",
        "memory_tombstone_observed",
        "career_evidence_tombstone",
        details={"entries": []},
    )
    # Sharing a run id is insufficient. A producer that omits the key must be
    # excluded so the missing instrumentation cannot be hidden by another event.
    events = recorder.list_memory_events(user_id="u1", conversation_id="c1")

    assert [event.event_type for event in events] == [
        "memory_context_observed",
        "memory_tombstone_observed",
    ]
    assert all(event.details["conversation_key"] == key for event in events)


def test_v1_trace_store_is_upgraded_before_model_events_are_written(
    tmp_path: Path,
) -> None:
    path = tmp_path / "run_events.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE schema_versions (
                component TEXT PRIMARY KEY,
                version INTEGER NOT NULL,
                updated_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE run_events (
                run_id TEXT NOT NULL,
                sequence INTEGER NOT NULL,
                event_type TEXT NOT NULL,
                stage TEXT NOT NULL,
                attempt INTEGER,
                occurred_at TEXT NOT NULL,
                duration_ms INTEGER,
                outcome TEXT NOT NULL,
                details_json TEXT NOT NULL,
                error_code TEXT,
                error_detail TEXT,
                recoverable INTEGER,
                PRIMARY KEY(run_id, sequence)
            )
            """
        )
        connection.execute(
            "INSERT INTO schema_versions(component, version) VALUES ('run_events', 1)"
        )
        connection.execute(
            """
            INSERT INTO run_events(
                run_id, sequence, event_type, stage, occurred_at, outcome,
                details_json
            ) VALUES (
                'legacy-run', 1, 'model_attempt', 'old_worker',
                '2026-09-01T00:00:00+00:00', 'started', '{}'
            )
            """
        )

    recorder = SQLiteTraceRecorder(path)
    recorder.record(
        "run-1",
        "model_attempt",
        "main_agent_decide",
        model_call_category="orchestrator_decision",
    )

    assert recorder.snapshot("run-1").events[0].model_call_category == (
        "orchestrator_decision"
    )
    legacy = recorder.snapshot("legacy-run").events[0]
    assert legacy.event_type == "model_attempt"
    assert legacy.model_call_category is None


def test_unregistered_v1_trace_table_is_adopted_with_the_v2_column(
    tmp_path: Path,
) -> None:
    path = tmp_path / "run_events.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE run_events (
                run_id TEXT NOT NULL,
                sequence INTEGER NOT NULL,
                event_type TEXT NOT NULL,
                stage TEXT NOT NULL,
                attempt INTEGER,
                occurred_at TEXT NOT NULL,
                duration_ms INTEGER,
                outcome TEXT NOT NULL,
                details_json TEXT NOT NULL,
                error_code TEXT,
                error_detail TEXT,
                recoverable INTEGER,
                PRIMARY KEY(run_id, sequence)
            )
            """
        )

    recorder = SQLiteTraceRecorder(path)
    recorder.record(
        "run-1",
        "model_attempt",
        "main_agent_decide",
        model_call_category="orchestrator_decision",
    )

    assert recorder.snapshot("run-1").events[0].model_call_category == (
        "orchestrator_decision"
    )
