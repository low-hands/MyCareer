from __future__ import annotations

from typing import Literal, Protocol

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from career_agent.agent.main_state import MainAgentState


class MainGraphHost(Protocol):
    """Node implementations and routers bound into the fixed main graph."""

    def _hydrate_career_context(self, state: MainAgentState) -> MainAgentState: ...

    def _decide(self, state: MainAgentState) -> MainAgentState: ...

    def _authorize(self, state: MainAgentState) -> MainAgentState: ...

    def _act(self, state: MainAgentState) -> MainAgentState: ...

    def _observe(self, state: MainAgentState) -> MainAgentState: ...

    def _present(self, state: MainAgentState) -> MainAgentState: ...

    def _interrupt(self, state: MainAgentState) -> MainAgentState: ...

    def _route_entry(
        self, state: MainAgentState
    ) -> Literal["hydrate", "authorize"]: ...

    def _route_decision(
        self, state: MainAgentState
    ) -> Literal["authorize", "present", "interrupt"]: ...

    def _after_authorize(
        self, state: MainAgentState
    ) -> Literal["act", "observe", "present", "interrupt"]: ...

    def _after_observe(
        self, state: MainAgentState
    ) -> Literal["hydrate", "decide", "present", "interrupt"]: ...


def build_main_graph(host: MainGraphHost) -> CompiledStateGraph:
    """Compile the main-agent topology without owning any node behavior."""

    graph = StateGraph(MainAgentState)
    graph.add_node("hydrate", host._hydrate_career_context)
    graph.add_node("decide", host._decide)
    graph.add_node("authorize", host._authorize)
    graph.add_node("act", host._act)
    graph.add_node("observe", host._observe)
    graph.add_node("present", host._present)
    graph.add_node("interrupt", host._interrupt)
    graph.add_conditional_edges(
        START,
        host._route_entry,
        {
            "hydrate": "hydrate",
            "authorize": "authorize",
        },
    )
    graph.add_edge("hydrate", "decide")
    graph.add_conditional_edges(
        "decide",
        host._route_decision,
        {
            "authorize": "authorize",
            "present": "present",
            "interrupt": "interrupt",
        },
    )
    graph.add_conditional_edges(
        "authorize",
        host._after_authorize,
        {
            "act": "act",
            "observe": "observe",
            "present": "present",
            "interrupt": "interrupt",
        },
    )
    graph.add_edge("act", "observe")
    graph.add_conditional_edges(
        "observe",
        host._after_observe,
        {
            "hydrate": "hydrate",
            "decide": "decide",
            "present": "present",
            "interrupt": "interrupt",
        },
    )
    graph.add_edge("present", END)
    graph.add_edge("interrupt", END)
    return graph.compile()
