from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from career_agent.agent.runtime.state import MainAgentState


@dataclass(frozen=True)
class MainGraphNodes:
    """Explicit node and routing functions bound into the fixed main graph."""

    hydrate: Callable[[MainAgentState], MainAgentState]
    decide: Callable[[MainAgentState], MainAgentState]
    authorize: Callable[[MainAgentState], MainAgentState]
    act: Callable[[MainAgentState], MainAgentState]
    observe: Callable[[MainAgentState], MainAgentState]
    present: Callable[[MainAgentState], MainAgentState]
    interrupt: Callable[[MainAgentState], MainAgentState]
    route_entry: Callable[
        [MainAgentState], Literal["hydrate", "authorize"]
    ]
    route_decision: Callable[
        [MainAgentState], Literal["authorize", "present", "interrupt"]
    ]
    after_authorize: Callable[
        [MainAgentState], Literal["act", "observe", "present", "interrupt"]
    ]
    after_observe: Callable[
        [MainAgentState], Literal["hydrate", "decide", "present", "interrupt"]
    ]

    @classmethod
    def from_host(cls, host: object) -> MainGraphNodes:
        """Adapt the former host API to the graph's explicit callable surface."""

        legacy: Any = host
        return cls(
            hydrate=legacy._hydrate_career_context,
            decide=legacy._decide,
            authorize=legacy._authorize,
            act=legacy._act,
            observe=legacy._observe,
            present=legacy._present,
            interrupt=legacy._interrupt,
            route_entry=legacy._route_entry,
            route_decision=legacy._route_decision,
            after_authorize=legacy._after_authorize,
            after_observe=legacy._after_observe,
        )


def build_main_graph(
    nodes: MainGraphNodes | object,
) -> CompiledStateGraph:
    """Compile the main-agent topology without owning any node behavior."""

    if not isinstance(nodes, MainGraphNodes):
        nodes = MainGraphNodes.from_host(nodes)

    graph = StateGraph(MainAgentState)
    graph.add_node("hydrate", nodes.hydrate)
    graph.add_node("decide", nodes.decide)
    graph.add_node("authorize", nodes.authorize)
    graph.add_node("act", nodes.act)
    graph.add_node("observe", nodes.observe)
    graph.add_node("present", nodes.present)
    graph.add_node("interrupt", nodes.interrupt)
    graph.add_conditional_edges(
        START,
        nodes.route_entry,
        {
            "hydrate": "hydrate",
            "authorize": "authorize",
        },
    )
    graph.add_edge("hydrate", "decide")
    graph.add_conditional_edges(
        "decide",
        nodes.route_decision,
        {
            "authorize": "authorize",
            "present": "present",
            "interrupt": "interrupt",
        },
    )
    graph.add_conditional_edges(
        "authorize",
        nodes.after_authorize,
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
        nodes.after_observe,
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
