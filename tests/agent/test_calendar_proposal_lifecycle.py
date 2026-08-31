"""The pending-Calendar-preview slot is released by every state that ends one.

The slot existed before its existence reached the model, so a stale id was
invisible and cost nothing. Projecting ``has_active_calendar_proposal`` changed
that: a preview that has already been executed, expired, superseded, or failed
now reads to the model as one still waiting for approval, and the model is
directed by policy to act on exactly that.

``calendar_write_failed`` is in the clearing set on evidence, not preference:
``CalendarService.execute_proposal`` marks the proposal ``failed`` before
re-raising, and it requires ``pending`` to run at all, so no retry can succeed.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from career_agent.agent.main_agent_contracts import (
    ConversationTaskState,
    ToolObservation,
)
from career_agent.agent.main_agent_reducers import reduce_task_state

_NOW = datetime(2026, 8, 31, 12, tzinfo=timezone.utc)


def _pending() -> ConversationTaskState:
    return ConversationTaskState(
        active_calendar_proposal_id="proposal-1",
        active_calendar_proposal_expires_at=_NOW,
        active_interview_round_id="round-1",
    )


def _observation(state: str, **payload) -> ToolObservation:
    return ToolObservation(
        tool_name="execute_calendar_proposal",
        state=state,
        message="x",
        payload=payload,
    )


@pytest.mark.parametrize(
    "state",
    [
        "calendar_sync_complete",
        "calendar_proposal_not_found",
        "calendar_approval_invalid",
        "calendar_write_failed",
    ],
)
def test_a_state_that_ends_a_preview_releases_the_slot(state) -> None:
    # The completion payload carries the proposal back, which is how the stale
    # id used to be rewritten rather than cleared.
    task = reduce_task_state(
        _pending(),
        _observation(state, proposal_id="proposal-1", expires_at=_NOW.isoformat()),
    )

    assert task.active_calendar_proposal_id is None
    assert task.active_calendar_proposal_expires_at is None
    assert task.active_resource_flags()["has_active_calendar_proposal"] is False
    # The interview it belonged to is still what the conversation is about.
    assert task.active_interview_round_id == "round-1"


@pytest.mark.parametrize(
    "state", ["calendar_approval_required", "calendar_proposal_ready"]
)
def test_a_state_that_leaves_a_preview_waiting_claims_the_slot(state) -> None:
    task = reduce_task_state(
        ConversationTaskState(),
        _observation(
            state,
            proposal_id="proposal-2",
            expires_at=_NOW.isoformat(),
            interview_round_id="round-9",
        ),
    )

    assert task.active_calendar_proposal_id == "proposal-2"
    assert task.active_calendar_proposal_expires_at == _NOW
    assert task.active_resource_flags()["has_active_calendar_proposal"] is True
    assert task.active_interview_round_id == "round-9"


def test_executing_then_preparing_again_does_not_resurrect_the_old_preview() -> None:
    """The sequence a second sync goes through, end to end."""
    task = reduce_task_state(
        _pending(), _observation("calendar_sync_complete", proposal_id="proposal-1")
    )
    assert task.active_calendar_proposal_id is None

    task = reduce_task_state(
        task,
        _observation(
            "calendar_approval_required",
            proposal_id="proposal-2",
            expires_at=_NOW.isoformat(),
        ),
    )

    assert task.active_calendar_proposal_id == "proposal-2"


def test_the_pending_set_and_the_registered_states_stay_in_step() -> None:
    """A state that ends a preview but reaches no reducer clears nothing.

    Three of them did exactly that before this change — not_found, invalid, and
    write_failed were unregistered, so the slot survived them.
    """
    from career_agent.agent.main_agent_reducers import (
        _PENDING_CALENDAR_PROPOSAL_STATES,
        ATOMIC_TASK_REDUCERS,
    )

    registered = {
        state
        for entry in ATOMIC_TASK_REDUCERS.values()
        for state in entry.states
        if state.startswith("calendar_")
    }
    ending = {
        "calendar_sync_complete",
        "calendar_proposal_not_found",
        "calendar_approval_invalid",
        "calendar_write_failed",
    }
    assert _PENDING_CALENDAR_PROPOSAL_STATES <= registered
    assert ending <= registered
    assert not (_PENDING_CALENDAR_PROPOSAL_STATES & ending)
