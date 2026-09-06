"""What the model is told about the objects it is directed to act on.

Withholding internal ids from the model is correct and load-bearing: the model
can only point at things it was shown. Withholding their *existence* was not
intended and broke a capability outright — the Calendar approval gate, the one
place in this project with a real two-phase commit, could not be reached,
because nothing in the projection said a preview was pending.

These tests hold both halves at once: existence crosses for every active object,
and no identifier ever does.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from career_agent.agent.main_agent_contracts import (
    CareerProfileContext,
    ConversationTaskState,
    MainAgentContext,
)

_NOW = datetime(2026, 8, 31, tzinfo=timezone.utc)


def _active_id_fields() -> tuple[str, ...]:
    return tuple(
        name
        for name in ConversationTaskState.model_fields
        if name.startswith("active_") and name.endswith("_id")
    )


def _populated_task() -> ConversationTaskState:
    """Every active object set, so a leak has something to leak."""
    return ConversationTaskState(
        **{name: f"secret-{name}" for name in _active_id_fields()},
        active_calendar_proposal_expires_at=_NOW + timedelta(minutes=30),
    )


def _projection(task: ConversationTaskState) -> dict:
    return MainAgentContext(
        conversation_id="c1",
        profile=CareerProfileContext(user_id="u1"),
        task=task,
        user_message="继续",
    ).model_context()


def test_every_active_object_reports_whether_it_exists() -> None:
    """Derived from the field names, so a new one cannot be forgotten."""
    projected = _projection(_populated_task())["task"]
    fields = _active_id_fields()
    assert len(fields) >= 13
    for name in fields:
        flag = f"has_{name[: -len('_id')]}"
        assert projected[flag] is True, f"{name} has no existence flag"


def test_an_absent_object_reports_false_rather_than_being_omitted() -> None:
    """Silence is not the same claim as 'there is none'.

    A missing key leaves the model to guess; ``false`` is a statement it can act
    on, which is the difference between asking a needless question and not.
    """
    projected = _projection(ConversationTaskState())["task"]
    for name in _active_id_fields():
        assert projected[f"has_{name[: -len('_id')]}"] is False


def test_no_internal_identifier_reaches_the_model() -> None:
    """The half that was already right, held while the other half changed."""
    task = _populated_task()
    raw = json.dumps(_projection(task), ensure_ascii=False, sort_keys=True)
    for name in _active_id_fields():
        assert f"secret-{name}" not in raw
    assert "active_calendar_proposal_id" not in raw


def test_the_calendar_preview_deadline_crosses_but_its_id_does_not() -> None:
    """A timestamp names no object, and expiry decides execute versus re-prepare."""
    projected = _projection(_populated_task())["task"]
    assert projected["has_active_calendar_proposal"] is True
    assert projected["active_calendar_proposal_expires_at"] == (
        (_NOW + timedelta(minutes=30)).isoformat()
    )


def test_the_prompt_tells_the_model_these_flags_are_the_only_signal() -> None:
    """The projection and the policy have to refer to the same thing.

    The prompt directs the model at "the active object" in a dozen places. With
    the flags present but unmentioned, a model could still read absence of an id
    as absence of the object, which is exactly the wrong inference.
    """
    from career_agent.agent.openai_compatible_main_agent import (
        OpenAICompatibleMainAgentDecisionMaker,
    )

    prompt = OpenAICompatibleMainAgentDecisionMaker._system_prompt()
    assert "has_active_*" in prompt
    assert "active_calendar_proposal_expires_at" in prompt
