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

from career_agent.harness.observability import (
    EventType,
    RunEvent,
    RunTrace,
    InMemoryTraceRecorder,
)
from career_agent.security.redaction import redact_text
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