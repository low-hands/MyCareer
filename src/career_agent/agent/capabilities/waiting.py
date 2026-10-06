"""Tool results that suppress another offer until the next user turn."""

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
