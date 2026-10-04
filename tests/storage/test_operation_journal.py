from __future__ import annotations

from datetime import datetime, timezone
import sqlite3

from career_agent.storage.operation_journal import SQLiteOperationJournal


def test_operation_journal_records_the_execution_lifecycle(tmp_path) -> None:
    journal = SQLiteOperationJournal(tmp_path / "operations.sqlite3")
    operation, created = journal.prepare(
        user_id="user-1",
        conversation_id="conversation-1",
        turn_id="turn-1",
        anchor="request-1",
        request_id="request-1",
        write_slot=0,
        tool_name="create_application",
        fingerprint="fingerprint-1",
        policy_epoch=1,
        replay_allowed=True,
    )

    assert created is True
    assert operation.phase == "PREPARED"
    assert operation.attempt_count == 0
    assert operation.operation_id == operation.action_id
    assert operation.turn_id == "turn-1"

    running = journal.mark_running(action_id=operation.operation_id)
    assert running.phase == "RUNNING"
    assert running.attempt_count == 1
    assert running.last_attempt_at is not None

    committed = journal.succeed(
        action_id=operation.operation_id,
        output={"application_id": "application-1"},
    )
    assert committed.status == "SUCCEEDED"
    assert committed.phase == "EFFECT_COMMITTED"
    assert committed.effect_committed_at is not None
    assert committed.completed_at is None
    assert journal.list_incomplete(user_id="user-1") == (committed,)

    completed = journal.complete_for_anchor(
        user_id="user-1",
        conversation_id="conversation-1",
        anchor="request-1",
    )[0]
    assert completed.phase == "COMPLETED"
    assert completed.completed_at is not None
    assert journal.list_incomplete(user_id="user-1") == ()


def test_uncertain_operation_is_visible_to_reconciliation(tmp_path) -> None:
    journal = SQLiteOperationJournal(tmp_path / "operations.sqlite3")
    operation, _ = journal.prepare(
        user_id="user-1",
        conversation_id="conversation-1",
        turn_id="turn-1",
        anchor="request-1",
        request_id="request-1",
        write_slot=0,
        tool_name="create_interview",
        fingerprint="fingerprint-1",
        policy_epoch=1,
    )
    journal.mark_running(action_id=operation.operation_id)

    uncertain = journal.require_reconciliation(
        action_id=operation.operation_id,
        error_code="TIMEOUT",
        error_detail="The provider did not confirm the outcome.",
    )

    assert uncertain.status == "PENDING"
    assert uncertain.phase == "RECONCILIATION_REQUIRED"
    assert uncertain.error_code == "TIMEOUT"
    assert journal.list_pending(user_id="user-1") == (uncertain,)


def test_version_one_action_rows_upgrade_in_place(tmp_path) -> None:
    path = tmp_path / "operations.sqlite3"
    settled_at = datetime(2026, 1, 2, tzinfo=timezone.utc).isoformat()
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE schema_versions (
                component TEXT PRIMARY KEY,
                version INTEGER NOT NULL,
                updated_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
            INSERT INTO schema_versions(component, version)
            VALUES ('action_executions', 1);
            CREATE TABLE action_executions (
                action_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                conversation_id TEXT NOT NULL,
                anchor TEXT NOT NULL,
                request_id TEXT,
                write_slot INTEGER NOT NULL,
                tool_name TEXT NOT NULL,
                fingerprint TEXT NOT NULL,
                policy_epoch INTEGER NOT NULL,
                retry_safe INTEGER NOT NULL,
                status TEXT NOT NULL,
                output_json TEXT NOT NULL,
                error_code TEXT,
                error_detail TEXT,
                started_at TEXT NOT NULL,
                settled_at TEXT,
                UNIQUE(user_id, conversation_id, anchor, write_slot)
            );
            """
        )
        connection.execute(
            """
            INSERT INTO action_executions VALUES (
                'action-1', 'user-1', 'conversation-1', 'request-1',
                'request-1', 0, 'create_application', 'fingerprint-1',
                1, 1, 'SUCCEEDED', '{"application_id": "application-1"}',
                NULL, NULL, ?, ?
            )
            """,
            (settled_at, settled_at),
        )

    operation = SQLiteOperationJournal(path).get(action_id="action-1")

    assert operation is not None
    assert operation.turn_id == "request-1"
    assert operation.phase == "COMPLETED"
    assert operation.attempt_count == 1
    assert operation.effect_committed_at == datetime.fromisoformat(settled_at)
    assert operation.completed_at == datetime.fromisoformat(settled_at)
