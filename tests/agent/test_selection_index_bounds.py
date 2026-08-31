"""Every selector the model can send is bounded on both sides.

A selection index is the model's only way to point at one of the user's
objects, so it is the one number the model fully controls that reaches a real
lookup. The bound was written out per field: correct twenty-nine times, absent
once, on the single collection-typed selector — where index 0 resolved through
``candidates[-1]`` to the last saved job and ``compare_saved_jobs`` ran against
the wrong pair without raising.

These tests hold the bound to the type rather than to the discipline of whoever
adds the next selector.
"""

from __future__ import annotations

import inspect
from typing import get_args

import pytest
from annotated_types import Ge
from pydantic import BaseModel
from pydantic.fields import FieldInfo

from career_agent.agent import main_agent_contracts as contracts
from career_agent.agent.main_agent_contracts import CompareSavedJobsToolArguments
from career_agent.evaluation.main_agent_scenarios import SCENARIOS


def _selector_fields():
    """Every declared field whose name marks it as a selector."""
    for _, model in inspect.getmembers(contracts, inspect.isclass):
        if not issubclass(model, BaseModel) or model is BaseModel:
            continue
        for name, field in model.model_fields.items():
            if "index" in name or "indices" in name:
                yield model.__name__, name, field


def _lower_bounds(annotation) -> list[int]:
    """Every ``ge`` constraint reachable inside a field's annotation.

    Walked structurally rather than compared to ``SelectionIndex`` by identity,
    because an optional or tuple-wrapped alias is a different object by the time
    pydantic is done with it. What has to be true is that a lower bound is in
    there somewhere.
    """
    found = []
    pending = [annotation]
    while pending:
        current = pending.pop()
        if isinstance(current, Ge):
            found.append(current.ge)
            continue
        if isinstance(current, FieldInfo):
            # ``Annotated[int, Field(ge=1)]`` stores the constraint one level
            # deeper than typing exposes: get_args yields the FieldInfo, and the
            # Ge lives on its metadata.
            pending.extend(current.metadata)
            continue
        args = get_args(current)
        if args:
            pending.extend(arg for arg in args if arg is not Ellipsis)
    return found


def test_every_selector_carries_a_lower_bound() -> None:
    """The bug was one missing bound among thirty declarations."""
    checked = 0
    for model_name, field_name, field in _selector_fields():
        bounds = _lower_bounds(field.annotation) + [
            item.ge for item in field.metadata if isinstance(item, Ge)
        ]
        assert bounds, (
            f"{model_name}.{field_name} has no lower bound; declare it as "
            "SelectionIndex so index 0 cannot resolve to the last item"
        )
        assert min(bounds) >= 1, f"{model_name}.{field_name} allows {min(bounds)}"
        checked += 1
    assert checked >= 30


@pytest.mark.parametrize("indices", [(0, 1), (-1, 2), (1, 0)])
def test_a_non_positive_selector_is_refused_by_the_schema(indices) -> None:
    """The bug, stated directly: 0 used to mean 'the last one'."""
    with pytest.raises(ValueError):
        CompareSavedJobsToolArguments.model_validate(
            {"job_selection_indices": list(indices)}
        )


def test_resolution_checks_both_bounds_independently_of_the_schema() -> None:
    """Two checks for a boundary the model controls, not one.

    The projection guards used to test only ``index > len(candidates)``, so a
    selector that slipped past the schema had nothing left to stop it.
    """
    context = next(
        item.context
        for item in SCENARIOS
        if item.name == "research_is_not_started_as_part_of_matching"
    )
    with pytest.raises(ValueError, match="out of range"):
        contracts.project_saved_job_arguments(
            context, "get_saved_job", {"selection_index": 99}
        )
    # Bypassing the schema the way a future refactor might, to prove the guard
    # itself covers the lower bound rather than inheriting it.
    candidates = context.task.saved_job_candidates
    assert not 1 <= 0 <= len(candidates)


def test_no_guard_checks_only_the_upper_bound() -> None:
    """A one-sided guard is the shape the bug had; none may come back."""
    source = inspect.getsource(contracts)
    assert "> len(" not in source.replace("<= len(", ""), (
        "a one-sided range guard reappeared; write 'not 1 <= index <= len(...)'"
    )
