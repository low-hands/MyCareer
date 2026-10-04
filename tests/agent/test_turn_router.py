from __future__ import annotations

from typing import Any

import pytest

from career_agent.agent.contracts.context import MainAgentContext
from career_agent.agent.contracts.profile import CareerProfileContext
from career_agent.agent.contracts.task_state import ConversationTaskState
from career_agent.agent.contracts.turn import RuntimeAction
from career_agent.harness.turn_router import TurnRouter


class RecordingAgentLoop:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.result = object()

    def run_decided(self, context, **kwargs):
        self.calls.append({"context": context, **kwargs})
        return self.result


@pytest.mark.parametrize(
    ("task", "expected"),
    [
        (ConversationTaskState(), False),
        (
            ConversationTaskState(
                active_workflow="job_discovery",
                run_id="job-run",
                phase="running",
            ),
            False,
        ),
        (
            ConversationTaskState(
                active_workflow="mock_interview",
                run_id="mock-run",
                phase="mock_interview_running",
            ),
            True,
        ),
        (
            ConversationTaskState(
                active_workflow="mock_interview",
                run_id="mock-run",
                phase="mock_interview_checkpoint_missing",
            ),
            False,
        ),
        (
            ConversationTaskState(
                active_workflow="mock_interview",
                run_id="mock-run",
                phase="mock_interview_graph_incompatible",
            ),
            False,
        ),
    ],
)
def test_owns_next_turn_only_for_a_resumable_mock_interview(
    task: ConversationTaskState,
    expected: bool,
) -> None:
    assert TurnRouter.owns_next_turn(task) is expected


@pytest.mark.parametrize(
    ("phase", "expected_tool", "expected_arguments"),
    [
        ("mock_interview_running", "handle_mock_interview_input", {"message": "继续"}),
        ("failed", "retry_mock_interview", {}),
    ],
)
def test_owned_workflow_turn_builds_runtime_owned_graph_input(
    phase: str,
    expected_tool: str,
    expected_arguments: dict[str, str],
) -> None:
    loop = RecordingAgentLoop()
    router = TurnRouter(
        context_manager=object(),  # type: ignore[arg-type]
        confirmation_store=None,
        agent_loop=loop,  # type: ignore[arg-type]
    )
    context = MainAgentContext(
        conversation_id="c1",
        profile=CareerProfileContext(user_id="u1"),
        task=ConversationTaskState(
            active_workflow="mock_interview",
            run_id="session-1",
            phase=phase,
        ),
        user_message="[workflow-owned input withheld]",
    )

    result = router.run_owned_workflow_turn(context=context, user_message="继续")

    assert result is loop.result
    assert len(loop.calls) == 1
    call = loop.calls[0]
    assert call["context"] is context
    assert call["decision"].tool_call.name == expected_tool
    assert call["decision"].tool_call.arguments == {}
    assert call["pending"] == {
        "name": expected_tool,
        "runtime_owned": True,
        "arguments": expected_arguments,
    }
    assert call["origin"] == RuntimeAction(workflow="mock_interview")


def test_owned_workflow_turn_requires_a_resumable_session() -> None:
    router = TurnRouter(
        context_manager=object(),  # type: ignore[arg-type]
        confirmation_store=None,
        agent_loop=RecordingAgentLoop(),  # type: ignore[arg-type]
    )
    context = MainAgentContext(
        conversation_id="c1",
        profile=CareerProfileContext(user_id="u1"),
        user_message="继续",
    )

    with pytest.raises(ValueError, match="no resumable session"):
        router.run_owned_workflow_turn(context=context, user_message="继续")
