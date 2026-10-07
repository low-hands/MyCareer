"""Trajectory fixtures must apply the same task migrations as stored state."""

from __future__ import annotations

import pytest

from career_agent.evaluation.main_agent_scenarios import SCENARIOS
from career_agent.evaluation.trajectory import (
    TrajectoryStep, advance_trajectory_context,
)


@pytest.mark.parametrize(
    ("name", "step_index", "field"),
    (
        ("stated_intent_is_proposed_before_it_is_recorded", 1,
         "target_role_candidates"),
        ("a_repeated_call_is_not_reissued_after_an_observation", 1,
         "saved_job_candidates"),
        ("tool_selection_combines_job_analysis_and_resume_match", 1,
         "active_job_analysis_id"),
    ),
)
def test_legacy_task_fields_are_applied_to_domain_context(
    name, step_index, field,
) -> None:
    scenario = next(item for item in SCENARIOS if item.name == name)
    updated = advance_trajectory_context(
        scenario.context, scenario.steps[step_index],
    )
    assert getattr(updated.task, field) == scenario.steps[step_index].task_update[field]
    assert field not in updated.task.model_dump(mode="python")


@pytest.mark.parametrize(
    ("patch", "message"),
    (
        ({"saved_jobs_candidates": ()}, "Extra inputs are not permitted"),
        ({"domain_context": {"job": {"saved_jobs_candidates": ()}}},
         "unknown trajectory task_update field"),
    ),
)
def test_unknown_trajectory_task_update_fails_loudly(patch, message) -> None:
    with pytest.raises(ValueError, match=message):
        advance_trajectory_context(
            SCENARIOS[0].context, TrajectoryStep(task_update=patch),
        )
