from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import sqlite3
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import RetryPolicy

from career_agent.agent.runtime.state import MainAgentState


def retry_transient_hydration(error: Exception) -> bool:
    """Retry only failures safe to repeat before the model/tool loop starts."""

    if isinstance(error, (ConnectionError, TimeoutError)):
        return True
    if isinstance(error, sqlite3.OperationalError):
        detail = str(error).lower()
        return "locked" in detail or "busy" in detail
    return False


HYDRATION_RETRY_POLICY = RetryPolicy(
    initial_interval=0.25,
    backoff_factor=2.0,
    max_interval=1.0,
    max_attempts=2,
    jitter=True,
    retry_on=retry_transient_hydration,
)


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
    suspend: Callable[[MainAgentState], MainAgentState]
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

def build_main_graph(
    nodes: MainGraphNodes,
    *,
    checkpointer: Any | None = None,
) -> CompiledStateGraph:
    """Compile the main-agent topology without owning any node behavior."""

    graph = StateGraph(MainAgentState)
    # Hydration is a read-only projection and is safe to repeat for narrowly
    # classified transient failures. Decision providers already own their
    # bounded retry loop, while act may cross an external side-effect boundary;
    # neither is retried again by LangGraph.
    graph.add_node(
        "hydrate",
        nodes.hydrate,
        retry_policy=HYDRATION_RETRY_POLICY,
    )
    graph.add_node("decide", nodes.decide)
    graph.add_node("authorize", nodes.authorize)
    graph.add_node("act", nodes.act)
    graph.add_node("observe", nodes.observe)
    graph.add_node("present", nodes.present)
    graph.add_node("interrupt", nodes.interrupt)
    graph.add_node("suspend", nodes.suspend)
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
    graph.add_edge("interrupt", "suspend")
    graph.add_edge("suspend", "hydrate")
    return graph.compile(checkpointer=checkpointer)
