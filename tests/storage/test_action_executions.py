from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys

import pytest

from career_agent.storage.action_executions import (
    ActionExecutionConflictError,
    ActionExecutionReconciliationRequiredError,
    SQLiteActionExecutionStore,
)


NOW = datetime(2026, 9, 4, tzinfo=timezone.utc)


def _prepare(store: SQLiteActionExecutionStore, **updates):
    values = {
        "user_id": "u1",
        "conversation_id": "c1",
        "anchor": "request-1",
        "request_id": "request-1",
        "write_slot": 0,
        "tool_name": "create_application",
        "fingerprint": "a" * 64,
        "policy_epoch": 1,
        "replay_allowed": True,
        "now": NOW,
    }
    values.update(updates)
    return store.prepare(**values)


def test_a_slot_allocates_one_stable_action_id_and_separates_its_fingerprint(
    tmp_path: Path,
) -> None:
    store = SQLiteActionExecutionStore(tmp_path / "context.sqlite3")

    first, created = _prepare(store)
    replay, replay_created = _prepare(store)

    assert created is True
    assert replay_created is False
    assert replay.action_id == first.action_id
    # Different arguments in a slot whose outcome is still unknown: refused, and
    # told which pending action is in the way. Refusal alone would strand the
    # row — nothing could then establish whether its effect reached the outside.
    with pytest.raises(ActionExecutionReconciliationRequiredError) as blocked:
        _prepare(store, fingerprint="b" * 64)
    assert blocked.value.execution.action_id == first.action_id
    assert blocked.value.execution.status == "PENDING"


@pytest.mark.parametrize("settle", ("succeed", "fail"))
def test_a_settled_slot_refuses_a_different_write_without_offering_recovery(
    tmp_path: Path, settle
) -> None:
    """Only a pending slot has anything left to resolve.

    Once an action is settled its outcome is known, so a different write in the
    same slot is simply refused. Reporting it as reconcilable would send an
    operator looking for external state that has already been accounted for.
    """
    store = SQLiteActionExecutionStore(tmp_path / "context.sqlite3")
    prepared, _ = _prepare(store)
    if settle == "succeed":
        store.succeed(action_id=prepared.action_id, output={"application_id": "app-1"})
    else:
        store.fail(
            action_id=prepared.action_id,
            error_code="TOOL_FAILED",
            error_detail="the service refused",
        )

    with pytest.raises(ActionExecutionConflictError) as refused:
        _prepare(store, fingerprint="b" * 64)

    assert not isinstance(refused.value, ActionExecutionReconciliationRequiredError)


def test_a_policy_change_does_not_strand_an_action_that_may_have_run(
    tmp_path: Path,
) -> None:
    """The boundary this distinction exists for.

    A policy epoch governs whether a *new* action may start. Applying it to a
    prepared one would permanently strand exactly the executions it means to
    govern: the row stays pending, the external effect stays unknown, and no
    path advances it.

    Calendar already draws this line one layer up — an epoch change supersedes a
    proposal only while it is still ``pending``, and an ``executing`` or
    ``reconciliation_required`` proposal falls through to recovery regardless.
    """
    store = SQLiteActionExecutionStore(tmp_path / "context.sqlite3")
    prepared, _ = _prepare(store, policy_epoch=1)

    with pytest.raises(ActionExecutionReconciliationRequiredError) as blocked:
        _prepare(store, policy_epoch=2)

    assert blocked.value.execution.action_id == prepared.action_id
    # And the row is still reachable: reconciliation can settle it either way.
    assert store.list_pending()[0].action_id == prepared.action_id
    settled = store.fail(
        action_id=prepared.action_id,
        error_code="POLICY_EPOCH_CHANGED",
        error_detail=(
            "reconciliation confirmed the write never reached the provider; "
            "the approving policy has since changed, so a fresh attempt needs a "
            "new request identity"
        ),
    )
    assert settled.status == "FAILED"
    assert store.list_pending() == ()


def test_an_unkeyed_turn_is_explicitly_retry_unsafe(tmp_path: Path) -> None:
    store = SQLiteActionExecutionStore(tmp_path / "context.sqlite3")

    execution, _ = _prepare(
        store,
        anchor="ephemeral-turn-id",
        request_id=None,
    )

    assert execution.retry_safe is False


def test_a_stable_request_does_not_make_an_unsafe_capability_replayable(
    tmp_path: Path,
) -> None:
    store = SQLiteActionExecutionStore(tmp_path / "context.sqlite3")

    execution, _ = _prepare(
        store,
        tool_name="create_interview",
        replay_allowed=False,
    )

    assert execution.request_id == "request-1"
    assert execution.retry_safe is False


def test_a_prepared_action_survives_an_abrupt_process_death(tmp_path: Path) -> None:
    path = tmp_path / "context.sqlite3"
    script = """
import sys
import time
from pathlib import Path
from career_agent.storage.action_executions import SQLiteActionExecutionStore

store = SQLiteActionExecutionStore(Path(sys.argv[1]))
store.prepare(
    user_id="u1", conversation_id="c1", anchor="request-killed",
    request_id="request-killed", write_slot=0,
    tool_name="create_application", fingerprint="a" * 64,
    policy_epoch=1, replay_allowed=True,
)
print("prepared", flush=True)
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
    assert process.stdout.readline().strip() == "prepared"
    process.kill()
    process.wait(timeout=5)

    pending = SQLiteActionExecutionStore(path).list_pending(user_id="u1")
    assert len(pending) == 1
    assert pending[0].anchor == "request-killed"
    assert pending[0].status == "PENDING"


def test_a_success_receipt_rejects_result_bodies(tmp_path: Path) -> None:
    store = SQLiteActionExecutionStore(tmp_path / "context.sqlite3")
    execution, _ = _prepare(store)

    with pytest.raises(ValueError, match="scalar receipts"):
        store.succeed(action_id=execution.action_id, output={"body": {"x": 1}})
