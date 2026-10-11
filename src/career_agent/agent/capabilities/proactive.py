"""Reviewed, transient capability offers for search-mode model calls."""

from __future__ import annotations

from career_agent.agent.capabilities.catalog import CAPABILITIES
from career_agent.agent.capabilities.reachability import reachable
from career_agent.agent.capabilities.waiting import WAITING_FOR_USER_STATES
from career_agent.agent.contracts.context import MainAgentContext
from career_agent.agent.contracts.observations import DecisionObservation
from career_agent.agent.presentation.delivery_policy import is_failed


# A bound object supports a direct readback. These names are offered only while
# the binding is present; they are not added to durable loaded_capabilities.
BOUND_RESOURCE_READS = (
    ("active_job_posting_id", "get_saved_job"),
    ("active_application_id", "get_application"),
    ("active_interview_round_id", "get_interview"),
    ("active_resume_version_id", "get_resume_metadata"),
    ("active_job_research_report_id", "get_job_research"),
    ("active_resume_job_match_id", "get_resume_job_match"),
    ("active_resume_tailoring_draft_id", "get_resume_tailoring_draft"),
    ("active_interview_preparation_id", "get_interview_preparation"),
    ("active_calendar_proposal_id", "get_calendar_proposal"),
)

_NON_SUCCESS_RESULT_STATES = frozenset({
    "authorization_refused",
    "invalid_input",
    "conversation_span_unavailable",
    "working_notes_stale",
})


def eligible_successors(observation: DecisionObservation) -> tuple[str, ...]:
    if not succeeded(observation):
        return ()
    return CAPABILITIES[observation.tool_name].successors


def succeeded(observation: DecisionObservation) -> bool:
    """Use the same success boundary for retained tools and W successors."""
    return retainable_tool_result(
        observation.tool_name,
        observation.state,
        observation.disposition or "completed",
        observation.execution_outcome,
    )


def retainable_tool_result(
    name: str, state: str, disposition: str = "completed",
    execution_outcome: str | None = None,
) -> bool:
    """A completed model-callable tool stays loaded in search mode."""
    descriptor = CAPABILITIES.get(name)
    return bool(
        descriptor is not None
        and descriptor.model_callable
        and descriptor.effect != "CONTROL"
        and disposition == "completed"
        and execution_outcome not in {"not_committed", "unknown"}
        and state not in _NON_SUCCESS_RESULT_STATES
        and state not in WAITING_FOR_USER_STATES
        and not is_failed(state)
    )


def proactive_tool_names(context: MainAgentContext) -> tuple[str, ...]:
    """Current bound reads and all eligible successors from this user turn."""
    task = context.task
    names = set(context.turn_proactive_capabilities)
    names.update({
        name for field, name in BOUND_RESOURCE_READS
        if getattr(task, field) is not None and CAPABILITIES[name].effect == "READ"
    })
    # Search, skill and other intervening calls must not hide an earlier
    # successor. The context manager starts each user turn with empty turn state.
    for observation in context.tool_observations:
        names.update(eligible_successors(observation))
    return tuple(
        name for name in CAPABILITIES
        if name in names and CAPABILITIES[name].model_callable and reachable(name, task)
    )
