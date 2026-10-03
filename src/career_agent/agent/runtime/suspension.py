from __future__ import annotations

from typing import cast

from langgraph.types import interrupt

from career_agent.agent.runtime.state import MainAgentState


def suspend_for_interaction(state: MainAgentState) -> MainAgentState:
    """Pause after the interaction has been projected into checkpointed state.

    Questionnaire continuation resumes with a complete fresh-turn state. Other
    interaction kinds deliberately start a new turn through their existing
    durable owner instead of resuming this checkpoint.
    """

    decision = state.get("decision")
    resumed = interrupt(
        {
            "kind": "main_agent_interaction",
            "action": decision.action if decision is not None else "unknown",
        }
    )
    if not isinstance(resumed, dict) or "context" not in resumed:
        raise ValueError("main agent resume requires a complete graph state")
    return cast(MainAgentState, resumed)
