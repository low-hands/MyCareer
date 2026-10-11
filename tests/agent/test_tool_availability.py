from career_agent.agent.capabilities.registry import MainAgentToolRegistry
from types import SimpleNamespace
from career_agent.agent.contracts.task_state import ConversationTaskState
from career_agent.agent.middleware.authorization import AuthorizationMiddleware
from career_agent.agent.middleware.contracts import AuthorizationRefusal
from career_agent.agent.middleware.tool_availability import ToolAvailabilityMiddleware


def _resolve(name: str, *, waiting: tuple[str, ...] = ()) -> AuthorizationRefusal:
    result = ToolAvailabilityMiddleware(tools=MainAgentToolRegistry()).resolve(
        name=name,
        task=ConversationTaskState(),
        offered_tool_names=("search_capabilities",),
        waiting_tool_names=waiting,
        runtime_owned=False,
        owner_confirmed=False,
        policy_owned=False,
    )
    assert isinstance(result, AuthorizationRefusal)
    return result


def test_waiting_tool_gets_an_observation_reason_without_loading() -> None:
    refusal = _resolve("list_action_items", waiting=("list_action_items",))
    assert refusal.kind == "waiting_for_user"
    assert "等待用户" in refusal.reason
    state_update = AuthorizationMiddleware(
        tracing=SimpleNamespace(authorization_refused=lambda *args, **kwargs: None),
        max_refusals=3,
    ).refuse({"control": {}}, name="list_action_items", refusal=refusal)
    assert state_update["authorization_route"] == "observe"
    assert state_update["pending"]["result"].message == refusal.reason


def test_unreachable_tool_gets_its_missing_precondition() -> None:
    refusal = _resolve("match_resume_to_job")
    assert refusal.kind == "precondition"
    assert refusal.next_action
    assert "match_resume_to_job" in refusal.reason
