from __future__ import annotations

from typing import Any, Literal

from pydantic import ConfigDict, TypeAdapter, ValidationError
from typing_extensions import TypedDict

from career_agent.agent.contracts.context import MainAgentContext
from career_agent.agent.contracts.decisions import AgentDecision
from career_agent.agent.contracts.observations import ToolObservation
from career_agent.agent.capabilities.effects import ToolEffect


class PendingAction(TypedDict, total=False):
    """One capability proposal as it moves through authorize → act → observe."""

    __pydantic_config__ = ConfigDict(extra="forbid")

    name: str
    kind: Literal["atomic_tool", "workflow"]
    runtime_owned: bool
    effect: ToolEffect
    reducer_result: ToolObservation
    arguments: dict[str, Any]
    result: ToolObservation
    synthetic_kind: Literal["projection", "authorization", "confirmation"]
    # Present only after the confirmation ingress consumed a durable seal.
    owner_confirmed: bool
    confirmation_id: str
    # Runtime policy selected this action, so it does not return to the model.
    policy_owned: bool
    # This policy read opens the turn; the next edge returns to ``hydrate``.
    policy_prelude: bool


class LoopControl(TypedDict, total=False):
    """Per-turn counters and replay guards shared by graph nodes."""

    __pydantic_config__ = ConfigDict(extra="forbid")

    read_calls: int
    # Profile switches change the next offered tool set and consume no I/O slot.
    control_calls: int
    search_calls: int
    # All durable writes, including the external subset below.
    write_calls: int
    external_write_calls: int
    # One prerequisite JD-analysis write is allowed in the resume-match chain.
    job_analysis_write_used: bool
    projection_refusals: int
    authorization_refusals: int
    fingerprints: tuple[str, ...]
    retryable_fingerprints: tuple[str, ...]
    retry_counts: dict[str, int]
    # Re-entering ``decide`` must not stamp the same projected episodes twice.
    episodes_marked: bool
    # Exact schemas supplied to the most recent model decision. This is a
    # per-turn execution boundary, independent of the legacy tool profile.
    offered_tool_names: tuple[str, ...]


class MainAgentState(TypedDict):
    """Complete checkpoint schema for every main-agent graph invocation."""

    __pydantic_config__ = ConfigDict(extra="forbid")

    context: MainAgentContext
    # ``None`` explicitly clears a prior checkpoint when a new turn starts.
    decision: AgentDecision | None
    pending: PendingAction
    authorization_route: Literal["act", "observe", "present", "interrupt"] | None
    tool_results: tuple[ToolObservation, ...]
    control: LoopControl
    artifact_ids: tuple[str, ...]
    assistant_message: str
    model_message: str
    # Scopes shown before a later write reloads and invalidates the projection.
    career_memory_scope_keys: tuple[str, ...]


_MAIN_AGENT_STATE = TypeAdapter(MainAgentState)


class MainAgentStateValidationError(ValueError):
    """A graph boundary received an incomplete or undeclared state channel."""


def validate_main_agent_state(
    state: object,
    *,
    boundary: Literal["invoke", "resume", "result"],
) -> MainAgentState:
    """Validate and normalize a complete state at a LangGraph boundary."""

    candidate = state
    if boundary == "result" and isinstance(state, dict) and "__interrupt__" in state:
        # LangGraph adds this reserved envelope field to an interrupted invoke
        # result. It is execution metadata, not a persisted application channel.
        candidate = {key: value for key, value in state.items() if key != "__interrupt__"}
    try:
        return _MAIN_AGENT_STATE.validate_python(candidate)
    except ValidationError as error:
        raise MainAgentStateValidationError(
            f"invalid main agent state at {boundary} boundary"
        ) from error
