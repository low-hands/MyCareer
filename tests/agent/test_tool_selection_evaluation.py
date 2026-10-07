"""Model-free coverage for the active capability selection policy."""

from career_agent.evaluation.tool_selection import (
    RuntimeSearchSelector,
    evaluate_tool_selection,
)
from career_agent.evaluation.tool_selection_scenarios import (
    SELECTION_DEV,
    SELECTION_HOLDOUT,
)
from career_agent.evaluation.trajectory import trajectory_tool_specs


def test_development_selection_baseline() -> None:
    specs = trajectory_tool_specs()
    report = evaluate_tool_selection(
        SELECTION_DEV,
        selector=RuntimeSearchSelector(specs, intent_enabled=True),
    )
    assert (report.covered_steps, report.demand_steps) == (51, 62)
    assert all(step.schema_tokens_proxy > 0 for step in report.steps)
    assert all(step.offered_names <= step.selected_names for step in report.steps)


def test_frozen_holdout_remains_evaluable() -> None:
    specs = trajectory_tool_specs()
    report = evaluate_tool_selection(
        SELECTION_HOLDOUT,
        selector=RuntimeSearchSelector(specs, intent_enabled=True),
    )
    assert report.demand_steps > 0
    assert len(report.steps) >= report.demand_steps
