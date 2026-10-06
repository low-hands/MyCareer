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


def eligible_successors(observation: DecisionObservation) -> tuple[str, ...]:
    if (
        observation.state in WAITING_FOR_USER_STATES
        or is_failed(observation.state)
        or observation.tool_name not in CAPABILITIES
    ):
        return ()
    return CAPABILITIES[observation.tool_name].successors


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
