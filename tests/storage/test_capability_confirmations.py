from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import sqlite3
import subprocess
import sys

import pytest

from career_agent.storage.capability_confirmations import (
    CapabilityConfirmationInProgressError,
    SQLiteCapabilityConfirmationStore,
)


def test_confirmation_separates_owner_approval_from_execution_outcome(tmp_path):
    store = SQLiteCapabilityConfirmationStore(tmp_path / "context.sqlite3")
    sealed = store.seal(
        user_id="u1",
        conversation_id="c1",
        capability="create_application",
        display_summary="创建投递记录",
        arguments={"user_id": "u1", "job_posting_id": "j1"},
        policy_revision=3,
    )

    claimed = store.claim(confirmation_id=sealed.confirmation_id, user_id="u1")
    assert claimed.status == "APPLYING"
    assert claimed.attempt_count == 1
    assert store.cancel(confirmation_id=sealed.confirmation_id, user_id="u1") is False
    assert store.get(sealed.confirmation_id).status == "APPLYING"
    with pytest.raises(CapabilityConfirmationInProgressError):
        store.claim(confirmation_id=sealed.confirmation_id, user_id="u1")

    settled = store.settle(
        confirmation_id=sealed.confirmation_id,
        user_id="u1",
        status="EXECUTED",
    )
    assert settled.status == "EXECUTED"


def test_expired_execution_lease_reuses_the_same_confirmation_identity(tmp_path):
    store = SQLiteCapabilityConfirmationStore(tmp_path / "context.sqlite3")
    sealed = store.seal(
        user_id="u1", conversation_id="c1", capability="create_application",
        display_summary="创建投递记录",
        arguments={"job": 1}, policy_revision=0,
    )
    store.claim(
        confirmation_id=sealed.confirmation_id, user_id="u1", lease_seconds=1
    )
    with sqlite3.connect(tmp_path / "context.sqlite3") as connection:
        connection.execute(
            "UPDATE capability_confirmations SET lease_expires_at=? WHERE confirmation_id=?",
            ((datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat(), sealed.confirmation_id),
        )

    reclaimed = store.claim(confirmation_id=sealed.confirmation_id, user_id="u1")
    assert reclaimed.confirmation_id == sealed.confirmation_id
    assert reclaimed.attempt_count == 2


def test_policy_revision_change_replaces_an_old_pending_seal(tmp_path):
    store = SQLiteCapabilityConfirmationStore(tmp_path / "context.sqlite3")
    first = store.seal(
        user_id="u1", conversation_id="c1", capability="create_application",
        display_summary="创建投递记录",
        arguments={"job": 1}, policy_revision=1,
    )
    second = store.seal(
        user_id="u1", conversation_id="c1", capability="create_application",
        display_summary="创建投递记录",
        arguments={"job": 1}, policy_revision=2,
    )

    assert second.confirmation_id != first.confirmation_id
    assert store.get(first.confirmation_id).status == "CANCELLED"
    assert second.policy_revision == 2


def test_stale_policy_confirmation_stays_auditable_but_is_not_renderable(tmp_path):
    store = SQLiteCapabilityConfirmationStore(tmp_path / "context.sqlite3")
    sealed = store.seal(
        user_id="u1", conversation_id="c1", capability="create_application",
        display_summary="创建投递记录", arguments={"job": 1}, policy_revision=1,
    )

    visible = store.pending_for_conversation(
        user_id="u1", conversation_id="c1", policy_revision=2
    )

    assert visible == ()
    assert store.get(sealed.confirmation_id).status == "PENDING"


def test_listing_pending_confirmations_must_choose_a_policy_view(tmp_path):
    store = SQLiteCapabilityConfirmationStore(tmp_path / "context.sqlite3")

    with pytest.raises(TypeError, match="policy_revision"):
        store.pending_for_conversation(user_id="u1", conversation_id="c1")


def test_an_execution_lease_survives_abrupt_process_death(tmp_path):
    path = tmp_path / "context.sqlite3"
    script = """
import sys, time
from career_agent.storage.capability_confirmations import SQLiteCapabilityConfirmationStore
store = SQLiteCapabilityConfirmationStore(sys.argv[1])
sealed = store.seal(
    user_id='u1', conversation_id='c1', capability='create_application',
    display_summary='创建投递记录', arguments={'job': 1}, policy_revision=1,
)
store.claim(confirmation_id=sealed.confirmation_id, user_id='u1')
print(sealed.confirmation_id, flush=True)
time.sleep(30)
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path.cwd() / "src")
    process = subprocess.Popen(
        [sys.executable, "-c", script, str(path)],
        stdout=subprocess.PIPE,
        text=True,
        env=environment,
    )
    assert process.stdout is not None
    confirmation_id = process.stdout.readline().strip()
    process.kill()
    process.wait(timeout=5)

    persisted = SQLiteCapabilityConfirmationStore(path).get(confirmation_id)
    assert persisted.status == "APPLYING"
    assert persisted.lease_expires_at is not None


def test_v1_consumed_confirmation_migrates_to_unknown_not_executed(tmp_path):
    path = tmp_path / "context.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE schema_versions(component TEXT PRIMARY KEY, version INTEGER, updated_at TEXT)"
        )
        connection.execute(
            "INSERT INTO schema_versions VALUES ('capability_confirmations', 1, datetime('now'))"
        )
        connection.execute(
            """
            CREATE TABLE capability_confirmations(
                confirmation_id TEXT PRIMARY KEY, user_id TEXT NOT NULL,
                conversation_id TEXT NOT NULL, capability TEXT NOT NULL,
                arguments_json TEXT NOT NULL, arguments_hash TEXT NOT NULL,
                status TEXT NOT NULL, created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL, settled_at TEXT
            )
            """
        )
        now = datetime.now(timezone.utc).isoformat()
        connection.execute(
            "INSERT INTO capability_confirmations VALUES "
            "('old', 'u1', 'c1', 'create_application', '{}', ?, 'CONFIRMED', ?, ?, ?)",
            ("a" * 64, now, now, now),
        )

    migrated = SQLiteCapabilityConfirmationStore(path).get("old")
    assert migrated.status == "RECONCILIATION_REQUIRED"
