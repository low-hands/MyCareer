"""Recording what reconciliation found, and refusing to guess when it did not.

``reconcile`` lists actions that started and never settled. This is the other
half: writing down what an investigation established, so the row stops being
pending and task state can be repaired from it.

Nothing here re-runs anything. An external write cannot be rolled back or safely
repeated from a CLI, so the only honest operations are to look and to record.
"""

from __future__ import annotations

from io import StringIO
import json
from pathlib import Path

import pytest

from career_agent.cli import main
from career_agent.storage.action_executions import SQLiteActionExecutionStore
from career_agent.storage.capability_confirmations import SQLiteCapabilityConfirmationStore


def _pending(tmp_path: Path, tool_name: str = "create_application"):
    store = SQLiteActionExecutionStore(tmp_path / "context.sqlite3")
    execution, _ = store.prepare(
        user_id="u1",
        conversation_id="c1",
        anchor="request-1",
        request_id="request-1",
        write_slot=0,
        tool_name=tool_name,
        fingerprint="a" * 64,
        policy_epoch=1,
    )
    return store, execution


def _settle(tmp_path: Path, *arguments: str) -> tuple[int, dict]:
    output = StringIO()
    code = main(
        [
            "actions",
            "settle",
            "--context-store",
            str(tmp_path / "context.sqlite3"),
            *arguments,
        ],
        stdout=output,
        stderr=StringIO(),
    )
    return code, json.loads(output.getvalue())


def test_a_confirmed_effect_is_recorded_with_the_ids_that_repair_task_state(
    tmp_path: Path,
) -> None:
    """Reconciliation found the write did happen, so the row settles succeeded.

    Policy is not re-checked at this point: the effect already exists in the
    outside world, and refusing to record it because some rule has since changed
    would leave the row pending forever — stranding exactly the execution the
    rule was meant to govern.
    """
    store, execution = _pending(tmp_path)

    code, settled = _settle(
        tmp_path,
        "--action-id",
        execution.action_id,
        "--executed",
        "--output",
        "application_id=app-1",
        "--output",
        "job_posting_id=job-1",
        "--output",
        "resume_version_id=resume-1",
        "--output",
        "status=submitted",
    )

    assert code == 0
    assert settled["status"] == "SUCCEEDED"
    assert settled["output"] == {
        "__result_state__": "application_ready",
        "application_id": "app-1",
        "job_posting_id": "job-1",
        "resume_version_id": "resume-1",
        "status": "submitted",
    }
    assert store.list_pending() == ()


def test_a_success_without_identifiers_is_refused(tmp_path: Path) -> None:
    """The identifiers are the reason to record a success at all.

    A crashed turn's reducer never ran, so ``active_*_id`` is empty. Repairing
    it is what the receipt is for; a success with nothing in it closes the row
    while leaving the state it was supposed to repair still broken.
    """
    store, execution = _pending(tmp_path)

    code, refused = _settle(
        tmp_path, "--action-id", execution.action_id, "--executed"
    )

    assert code != 0
    assert "--output" in refused["error"]
    # Still pending: a refused settle must not half-close the row.
    assert store.list_pending()[0].action_id == execution.action_id


def test_an_undeclared_receipt_can_close_reconciliation_but_cannot_drive_state(
    tmp_path: Path,
) -> None:
    store, execution = _pending(tmp_path, tool_name="create_interview")

    code, settled = _settle(
        tmp_path, "--action-id", execution.action_id, "--executed"
    )

    assert code == 0
    assert settled["output"] == {"__result_state__": "action_reconciled"}
    assert store.list_pending() == ()


def test_a_confirmed_non_effect_is_terminal_and_says_a_new_identity_is_needed(
    tmp_path: Path,
) -> None:
    """The row closes, and the reason has to survive in it.

    Whoever retries later reuses the request id, hits this terminal row, and has
    only ``error_detail`` to learn why an identity that once worked now fails.
    Without the sentence the failure reads as an unrelated problem.
    """
    store, execution = _pending(tmp_path, tool_name="execute_calendar_proposal")

    code, settled = _settle(
        tmp_path,
        "--action-id",
        execution.action_id,
        "--not-executed",
        "--reason",
        "Google Calendar 上没有对应事件",
    )

    assert code == 0
    assert settled["status"] == "FAILED"
    assert settled["error_code"] == "RECONCILED_NOT_EXECUTED"
    assert store.list_pending() == ()
    closed = SQLiteActionExecutionStore(tmp_path / "context.sqlite3")
    assert closed.list_pending() == ()


def test_a_non_effect_without_a_reason_is_refused(tmp_path: Path) -> None:
    store, execution = _pending(tmp_path)

    code, refused = _settle(
        tmp_path, "--action-id", execution.action_id, "--not-executed"
    )

    assert code != 0
    assert "--reason" in refused["error"]
    assert store.list_pending()[0].action_id == execution.action_id


@pytest.mark.parametrize("second", ("--executed", "--not-executed"))
def test_an_already_settled_action_cannot_be_settled_again(
    tmp_path: Path, second
) -> None:
    """Settling twice would let a later reading overwrite an established fact.

    The second call is not a correction — the operator has no way to know the
    row was already closed except by being told.
    """
    _, execution = _pending(tmp_path)
    _settle(
        tmp_path,
        "--action-id",
        execution.action_id,
        "--executed",
        "--output",
        "application_id=app-1",
        "--output",
        "job_posting_id=job-1",
        "--output",
        "resume_version_id=resume-1",
        "--output",
        "status=submitted",
    )

    arguments = ["--action-id", execution.action_id, second]
    arguments += (
        [
            "--output", "application_id=app-2",
            "--output", "job_posting_id=job-1",
            "--output", "resume_version_id=resume-1",
            "--output", "status=submitted",
        ]
        if second == "--executed"
        else ["--reason", "second look"]
    )
    code, response = _settle(tmp_path, *arguments)

    # Refused, not silently ignored. Returning the earlier outcome with a zero
    # exit code would read as "your finding was recorded" — which is how a
    # second, contradictory investigation gets lost.
    assert code != 0
    assert "no longer pending" in response["error"]


def test_settling_an_approved_action_also_closes_its_confirmation(tmp_path: Path):
    path = tmp_path / "context.sqlite3"
    confirmations = SQLiteCapabilityConfirmationStore(path)
    confirmation = confirmations.seal(
        user_id="u1", conversation_id="c1", capability="create_application",
        display_summary="创建投递记录", arguments={"job": 1}, policy_revision=1,
    )
    confirmations.claim(confirmation_id=confirmation.confirmation_id, user_id="u1")
    actions = SQLiteActionExecutionStore(path)
    execution, _ = actions.prepare(
        user_id="u1", conversation_id="c1",
        anchor=f"confirmation:{confirmation.confirmation_id}",
        request_id=f"confirmation:{confirmation.confirmation_id}",
        write_slot=0, tool_name="create_application", fingerprint="a" * 64,
        policy_epoch=1,
    )

    code, _ = _settle(
        tmp_path, "--action-id", execution.action_id, "--not-executed",
        "--reason", "本地自然键确认没有记录",
    )

    assert code == 0
    assert confirmations.get(confirmation.confirmation_id).status == "FAILED"
