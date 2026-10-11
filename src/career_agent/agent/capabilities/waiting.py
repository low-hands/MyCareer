"""Tool results that suppress another offer until the next user turn."""

from collections.abc import Iterable

from career_agent.agent.contracts.observations import DecisionObservation
from career_agent.agent.presentation.delivery_policy import DELIVERY_POLICIES, is_waiting


# These results need a fresh user instruction for selection, although the
# presenter does not end the turn on them.
SELECTION_ONLY_WAITING_STATES = frozenset({
    "working_notes_derived_argument",
    "job_intent_proposed",
})

WAITING_FOR_USER_STATES = frozenset(
    state for state in DELIVERY_POLICIES if is_waiting(state)
) | SELECTION_ONLY_WAITING_STATES


def waiting_tool_names(observations: Iterable[DecisionObservation]) -> frozenset[str]:
    """Names with a waiting result in the supplied observation window."""
    return frozenset(
        item.tool_name for item in observations
        if item.state in WAITING_FOR_USER_STATES
    )
