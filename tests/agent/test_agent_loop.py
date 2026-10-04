from career_agent.agent.contracts.context import MainAgentContext
from career_agent.agent.contracts.decisions import (
    AgentDecision,
    ToolCall,
)
from career_agent.agent.contracts.observations import ToolObservation
from career_agent.agent.contracts.profile import CareerProfileContext
from career_agent.agent.contracts.turn import ModelDecision, RuntimePolicyAction
from career_agent.harness.agent_loop import AgentLoop, main_graph_thread_id


class RecordingGraph:
    def __init__(self, *, artifact_ids=()) -> None:
        self.input = None
        self.artifact_ids = artifact_ids
        self.config = None

    def invoke(self, state, config=None):
        self.input = state
        self.config = config
        terminal = dict(state)
        terminal.update(
            {
                "decision": state.get("decision")
                or AgentDecision(action="final", message="完成。"),
                "assistant_message": "完成。",
                "model_message": "模型正文",
                "artifact_ids": self.artifact_ids,
                "tool_results": (
                    ToolObservation(
                        tool_name="example",
                        state="completed",
                        message="完成。",
                    ),
                ),
                "control": {"read_calls": 2, "write_calls": 1},
            }
        )
        return terminal


def _context():
    return MainAgentContext(
        conversation_id="c1",
        profile=CareerProfileContext(user_id="u1"),
        user_message="继续",
    )


def test_agent_loop_owns_common_initial_state_and_model_result_conversion() -> None:
    graph = RecordingGraph(artifact_ids=("artifact-1",))
    deliveries = []
    loop = AgentLoop(
        graph=graph,
        memory_scope_keys=lambda context: ("scope-1",),
        deliver_resume_artifact=lambda **kwargs: deliveries.append(kwargs) or "file",
    )
    prelude_decision = AgentDecision(
        action="tool_call",
        tool_call=ToolCall(name="read_conversation_span", arguments={}),
    )

    turn = loop.run_model(
        _context(),
        prelude={
            "decision": prelude_decision,
            "pending": {"name": "read_conversation_span", "policy_prelude": True},
        },
    )

    assert graph.input["career_memory_scope_keys"] == ("scope-1",)
    assert graph.input["artifact_ids"] == ()
    assert graph.input["tool_results"] == ()
    assert graph.input["decision"] is prelude_decision
    assert graph.input["pending"]["policy_prelude"] is True
    assert graph.input["authorization_route"] is None
    assert graph.input["assistant_message"] == ""
    assert graph.input["model_message"] == ""
    assert graph.input["control"] == {
        "read_calls": 0,
        "write_calls": 0,
        "projection_refusals": 0,
        "authorization_refusals": 0,
        "fingerprints": (),
        "retryable_fingerprints": (),
        "retry_counts": {},
    }
    assert graph.config == {
        "configurable": {
            "thread_id": main_graph_thread_id(user_id="u1", conversation_id="c1")
        }
    }
    assert isinstance(turn.origin, ModelDecision)
    assert turn.artifacts == ("file",)
    assert turn.delegated_read_count == 2
    assert turn.delegated_write_count == 1
    assert deliveries == [{"user_id": "u1", "artifact_id": "artifact-1"}]


def test_agent_loop_preserves_runtime_owned_origin() -> None:
    graph = RecordingGraph()
    loop = AgentLoop(
        graph=graph,
        memory_scope_keys=lambda context: (),
        deliver_resume_artifact=lambda **kwargs: None,
    )
    decision = AgentDecision(
        action="tool_call",
        tool_call=ToolCall(name="confirm_career_fact", arguments={}),
    )
    origin = RuntimePolicyAction(policy="career_fact_confirmation")

    turn = loop.run_decided(
        _context(),
        decision=decision,
        pending={"name": "confirm_career_fact", "policy_owned": True},
        origin=origin,
    )

    assert graph.input["decision"] is decision
    assert graph.input["pending"]["policy_owned"] is True
    assert turn.origin is origin
    assert turn.model_decision is None


def test_main_graph_thread_id_is_stable_scoped_and_opaque() -> None:
    first = main_graph_thread_id(user_id="u1", conversation_id="conversation-a")

    assert first == main_graph_thread_id(
        user_id="u1", conversation_id="conversation-a"
    )
    assert first != main_graph_thread_id(
        user_id="u2", conversation_id="conversation-a"
    )
    assert first != main_graph_thread_id(
        user_id="u1", conversation_id="conversation-b"
    )
    assert "u1" not in first
    assert "conversation-a" not in first
