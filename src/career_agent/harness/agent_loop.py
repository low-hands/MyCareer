from __future__ import annotations

from collections.abc import Callable
from hashlib import sha256
from typing import Any, Protocol

from langgraph.types import Command

from career_agent.agent.contracts.context import MainAgentContext
from career_agent.agent.contracts.decisions import AgentDecision
from career_agent.agent.runtime.state import (
    MainAgentState,
    PendingAction,
    validate_main_agent_state,
)
from career_agent.agent.contracts.turn import (
    MainAgentTurnResult,
    ModelDecision,
    TurnOrigin,
)
from career_agent.domain.resume import ResumeArtifactDelivery


class GraphInvoker(Protocol):
    """The narrow part of a compiled LangGraph used by the turn harness."""

    def invoke(
        self,
        input: MainAgentState | Command[Any],
        config: dict[str, object] | None = None,
    ) -> MainAgentState: ...


def main_graph_thread_id(*, user_id: str, conversation_id: str) -> str:
    """Return a stable, opaque checkpoint namespace for one conversation."""

    identity = f"{len(user_id)}:{user_id}{len(conversation_id)}:{conversation_id}"
    return f"main-agent:{sha256(identity.encode('utf-8')).hexdigest()}"


class ResumeArtifactDeliverer(Protocol):
    def __call__(
        self, *, user_id: str, artifact_id: str
    ) -> ResumeArtifactDelivery: ...


class AgentLoop:
    """Run one main-agent graph turn and translate graph state to its envelope.

    LangGraph still owns the model/tool loop. This adapter owns only the
    repeated invocation boundary around it: initial channels, invocation, and
    conversion of the terminal state into ``MainAgentTurnResult``.
    """

    def __init__(
        self,
        *,
        graph: GraphInvoker,
        memory_scope_keys: Callable[[MainAgentContext], tuple[str, ...]],
        deliver_resume_artifact: ResumeArtifactDeliverer,
    ) -> None:
        self._graph = graph
        self._memory_scope_keys = memory_scope_keys
        self._deliver_resume_artifact = deliver_resume_artifact

    def initial_state(
        self,
        context: MainAgentContext,
        *,
        decision: AgentDecision | None = None,
        pending: PendingAction | None = None,
        prelude: MainAgentState | None = None,
    ) -> MainAgentState:
        """Build the common graph channels without sharing mutable defaults."""

        state: MainAgentState = {
            "context": context,
            # A new input on an existing LangGraph thread merges with its last
            # checkpoint. Reset every per-turn channel explicitly so only the
            # durable checkpoint history carries across turns, never stale
            # execution state.
            "decision": None,
            "pending": {},
            "authorization_route": None,
            "career_memory_scope_keys": self._memory_scope_keys(context),
            "artifact_ids": (),
            "tool_results": (),
            "assistant_message": "",
            "model_message": "",
            "control": {
                "read_calls": 0,
                "write_calls": 0,
                "projection_refusals": 0,
                "authorization_refusals": 0,
                "fingerprints": (),
                "retryable_fingerprints": (),
                "retry_counts": {},
                "offered_tool_names": (),
            },
        }
        if prelude:
            state.update(prelude)
        if decision is not None:
            state["decision"] = decision
        if pending is not None:
            state["pending"] = pending
        return state

    def invoke(
        self,
        context: MainAgentContext,
        *,
        decision: AgentDecision | None = None,
        pending: PendingAction | None = None,
        prelude: MainAgentState | None = None,
    ) -> MainAgentState:
        initial = validate_main_agent_state(
            self.initial_state(
                context,
                decision=decision,
                pending=pending,
                prelude=prelude,
            ),
            boundary="invoke",
        )
        return validate_main_agent_state(
            self._graph.invoke(
                initial,
                config=self._config(context),
            ),
            boundary="result",
        )

    def resume_model(self, context: MainAgentContext) -> MainAgentTurnResult:
        """Resume a questionnaire suspension with validated fresh-turn state."""

        resumed = validate_main_agent_state(
            self.initial_state(context),
            boundary="resume",
        )
        state = validate_main_agent_state(
            self._graph.invoke(
                Command(resume=resumed),
                config=self._config(context),
            ),
            boundary="result",
        )
        decision = state.get("decision")
        if decision is None:
            raise RuntimeError("resumed main agent graph completed without a decision")
        artifacts = tuple(
            self._deliver_resume_artifact(
                user_id=context.profile.user_id,
                artifact_id=artifact_id,
            )
            for artifact_id in state.get("artifact_ids", ())
        )
        return self.result_from_state(
            state,
            origin=ModelDecision(decision),
            artifacts=artifacts,
        )

    @staticmethod
    def _config(context: MainAgentContext) -> dict[str, object]:
        return {
            "configurable": {
                "thread_id": main_graph_thread_id(
                    user_id=context.profile.user_id,
                    conversation_id=context.conversation_id,
                )
            }
        }

    def run_model(
        self,
        context: MainAgentContext,
        *,
        prelude: MainAgentState | None = None,
    ) -> MainAgentTurnResult:
        state = self.invoke(context, prelude=prelude)
        decision = state.get("decision")
        if decision is None:
            raise RuntimeError("main agent graph completed without a decision")
        artifacts = tuple(
            self._deliver_resume_artifact(
                user_id=context.profile.user_id,
                artifact_id=artifact_id,
            )
            for artifact_id in state.get("artifact_ids", ())
        )
        return self.result_from_state(
            state,
            origin=ModelDecision(decision),
            artifacts=artifacts,
        )

    def run_decided(
        self,
        context: MainAgentContext,
        *,
        decision: AgentDecision,
        pending: PendingAction,
        origin: TurnOrigin,
    ) -> MainAgentTurnResult:
        state = self.invoke(context, decision=decision, pending=pending)
        return self.result_from_state(state, origin=origin)

    @staticmethod
    def last_result(state: MainAgentState):
        results = state.get("tool_results", ())
        return results[-1] if results else None

    @classmethod
    def result_from_state(
        cls,
        state: MainAgentState,
        *,
        origin: TurnOrigin,
        artifacts: tuple[ResumeArtifactDelivery, ...] = (),
    ) -> MainAgentTurnResult:
        control = state.get("control", {})
        return MainAgentTurnResult(
            origin=origin,
            context=state["context"],
            assistant_message=state["assistant_message"],
            tool_result=cls.last_result(state),
            tool_results=state.get("tool_results", ()),
            artifacts=artifacts,
            content_streamed=False,
            model_message=state.get("model_message", ""),
            delegated_read_count=control.get("read_calls", 0),
            delegated_write_count=control.get("write_calls", 0),
            career_memory_scope_keys=state.get("career_memory_scope_keys", ()),
        )
