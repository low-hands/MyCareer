from typing import Literal

from langgraph.checkpoint.memory import InMemorySaver

from career_agent.agent.runtime.graph import MainGraphNodes, build_main_graph
from career_agent.agent.runtime.state import MainAgentState


class GraphHost:
    @staticmethod
    def _node(state: MainAgentState) -> MainAgentState:
        return {}

    _hydrate_career_context = _node
    _decide = _node
    _authorize = _node
    _act = _node
    _observe = _node
    _present = _node
    _interrupt = _node

    @staticmethod
    def _route_entry(
        state: MainAgentState,
    ) -> Literal["hydrate", "authorize"]:
        return "hydrate"

    @staticmethod
    def _route_decision(
        state: MainAgentState,
    ) -> Literal["authorize", "present", "interrupt"]:
        return "present"

    @staticmethod
    def _after_authorize(
        state: MainAgentState,
    ) -> Literal["act", "observe", "present", "interrupt"]:
        return "act"

    @staticmethod
    def _after_observe(
        state: MainAgentState,
    ) -> Literal["hydrate", "decide", "present", "interrupt"]:
        return "present"


def test_main_graph_topology_is_exact() -> None:
    host = GraphHost()
    graph = build_main_graph(
        MainGraphNodes(
            hydrate=host._hydrate_career_context,
            decide=host._decide,
            authorize=host._authorize,
            act=host._act,
            observe=host._observe,
            present=host._present,
            interrupt=host._interrupt,
            suspend=host._interrupt,
            route_entry=host._route_entry,
            route_decision=host._route_decision,
            after_authorize=host._after_authorize,
            after_observe=host._after_observe,
        )
    ).get_graph()

    assert {
        (edge.source, edge.target, edge.conditional) for edge in graph.edges
    } == {
        ("__start__", "authorize", True),
        ("__start__", "hydrate", True),
        ("hydrate", "decide", False),
        ("decide", "authorize", True),
        ("decide", "present", True),
        ("decide", "interrupt", True),
        ("authorize", "act", True),
        ("authorize", "observe", True),
        ("authorize", "present", True),
        ("authorize", "interrupt", True),
        ("act", "observe", False),
        ("observe", "hydrate", True),
        ("observe", "decide", True),
        ("observe", "present", True),
        ("observe", "interrupt", True),
        ("present", "__end__", False),
        ("interrupt", "suspend", False),
        ("suspend", "hydrate", False),
    }


def test_main_graph_compiles_with_the_supplied_checkpointer() -> None:
    host = GraphHost()
    checkpointer = InMemorySaver()

    graph = build_main_graph(
        MainGraphNodes(
            hydrate=host._hydrate_career_context,
            decide=host._decide,
            authorize=host._authorize,
            act=host._act,
            observe=host._observe,
            present=host._present,
            interrupt=host._interrupt,
            suspend=host._interrupt,
            route_entry=host._route_entry,
            route_decision=host._route_decision,
            after_authorize=host._after_authorize,
            after_observe=host._after_observe,
        ),
        checkpointer=checkpointer,
    )

    assert graph.checkpointer is checkpointer
