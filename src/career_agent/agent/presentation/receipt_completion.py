"""Terminal-tool delivery, gated by receipt integrity."""
from __future__ import annotations

from typing import TYPE_CHECKING

from career_agent.agent.presentation.delivery_policy import policy_for

if TYPE_CHECKING:
    from career_agent.agent.runtime.state import MainAgentState

_DIRECT_TOOLS = frozenset({"export_resume_artifact"})
_DISCOVERY = frozenset({"search_capabilities", "load_skill"})


def completes_tool_delivery(state: MainAgentState) -> bool:
    pending = state["pending"]
    result = pending["result"]
    policy = policy_for(result.state)
    if (pending.get("name") not in _DIRECT_TOOLS or result.disposition != "completed"
            or pending.get("synthetic_kind") is not None or result.next_action):
        return False
    if result.state != "resume_artifact_ready" or not result.payload.get("artifact_id"):
        return False
    if policy.body_delivery == "resource_card" and result.resource_ref is None:
        return False
    if pending.get("effect") == "WRITE" and result.execution_outcome != "committed":
        return False
    context = state.get("context")
    if context is None or context.turn_continuation_capability is not None:
        return False
    decision = state.get("decision")
    call = decision.tool_call if decision is not None and decision.action == "tool_call" else None
    # A stale decision cannot terminate a different runtime action.
    if call is None or call.name != pending.get("name"):
        return False
    results = state.get("tool_results", ())
    # The direct presenter delivers the last result. Prior business results,
    # refusals or failures require composition and must not disappear.
    business = [item for item in results if item.tool_name not in _DISCOVERY]
    if len(business) != 1 or business[0] != result:
        return False
    return all(item.disposition == "completed" for item in results)
