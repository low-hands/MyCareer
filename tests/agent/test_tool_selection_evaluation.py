"""Model-free coverage for the active capability selection policy."""

from career_agent.agent.contracts.context import MainAgentContext
from career_agent.agent.contracts.observations import DecisionObservation
from career_agent.agent.contracts.profile import CareerProfileContext
from career_agent.agent.contracts.task_state import ConversationTaskState
from career_agent.evaluation.tool_selection import (
    RuntimeSearchSelector,
    SelectionCase,
    ToolOffer,
    evaluate_tool_selection,
)
from career_agent.evaluation.tool_selection_scenarios import (
    SELECTION_DEV,
    SELECTION_HOLDOUT,
)
from career_agent.evaluation.trajectory import (
    TrajectoryScenario,
    TrajectoryStep,
    trajectory_tool_specs,
)


class _FixedOfferSelector(RuntimeSearchSelector):
    def __init__(self, name: str) -> None:
        schema = next(
            item for item in trajectory_tool_specs()
            if item["function"]["name"] == name
        )
        self.offer = ToolOffer(
            names=frozenset({name}),
            selected_names=frozenset({name}),
            schemas=(schema,),
            sources={name: "test"},
        )

    def select(self, context: MainAgentContext, prior: object | None):
        return self.offer, prior


def _case(
    *, task: ConversationTaskState, observation: DecisionObservation | None = None,
) -> SelectionCase:
    context = MainAgentContext(
        conversation_id="selection-counts",
        profile=CareerProfileContext(user_id="selection-counts"),
        user_message="请帮我处理当前岗位",
        task=task,
        tool_observations=(observation,) if observation else (),
    )
    return SelectionCase(
        TrajectoryScenario(
            name="selection-counts", policy="Count inappropriate offers.",
            context=context, steps=(TrajectoryStep(),),
        ),
        "dev", "control", frozenset(),
    )


def test_development_selection_baseline() -> None:
    specs = trajectory_tool_specs()
    report = evaluate_tool_selection(
        SELECTION_DEV,
        selector=RuntimeSearchSelector(specs),
    )
    assert (report.covered_steps, report.demand_steps) == (51, 62)
    assert all(step.schema_tokens_proxy > 0 for step in report.steps)
    assert all(step.offered_names <= step.selected_names for step in report.steps)


def test_frozen_holdout_remains_evaluable() -> None:
    specs = trajectory_tool_specs()
    report = evaluate_tool_selection(
        SELECTION_HOLDOUT,
        selector=RuntimeSearchSelector(specs),
    )
    assert report.demand_steps > 0
    assert len(report.steps) >= report.demand_steps


def test_unreachable_offer_is_counted() -> None:
    report = evaluate_tool_selection(
        (_case(task=ConversationTaskState()),),
        selector=_FixedOfferSelector("create_application"),
    )

    assert report.steps[0].unreachable_offered == frozenset({"create_application"})
    assert report.unreachable_offer_count == 1


def test_waiting_reoffer_is_counted() -> None:
    report = evaluate_tool_selection(
        (_case(
            task=ConversationTaskState(active_job_posting_id="job-1"),
            observation=DecisionObservation(
                tool_name="create_application",
                state="capability_confirmation_required",
                message="需要确认。", arguments={},
            ),
        ),),
        selector=_FixedOfferSelector("create_application"),
    )

    assert report.steps[0].waiting_reoffered == frozenset({"create_application"})
    assert report.waiting_reoffer_count == 1


def test_unrequested_write_offer_is_counted() -> None:
    report = evaluate_tool_selection(
        (_case(task=ConversationTaskState(active_job_posting_id="job-1")),),
        selector=_FixedOfferSelector("create_application"),
    )

    assert report.steps[0].unrequested_writes == frozenset({"create_application"})
    assert report.unrequested_write_offer_count == 1
