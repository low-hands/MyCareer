from typing import Any, cast

import pytest

from career_agent.agent.main_agent_contracts import (
    AgentDecision,
    MainAgentContext,
    ToolCall,
    ToolObservation,
)
from career_agent.agent.main_state import MainAgentState
from career_agent.agent.middleware.argument_projection import (
    ArgumentProjectionMiddleware,
    ProjectedArguments,
    ProjectionRefusal,
)


class ProjectionHost:
    def __init__(self, *, failure: ValueError | None = None) -> None:
        self.failure = failure
        self.calls: list[str] = []

    def _project_runtime_workflow_arguments(
        self, state: MainAgentState, name: str
    ) -> dict[str, Any]:
        self.calls.append("runtime")
        return {"bound": "runtime"}

    def _project_atomic_tool_arguments(
        self, context: MainAgentContext, name: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        self.calls.append("atomic")
        if self.failure is not None:
            raise self.failure
        return {"bound": arguments["value"]}

    def _project_workflow_arguments(
        self, context: MainAgentContext, name: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        self.calls.append("workflow")
        return {"bound": "workflow"}

    @staticmethod
    def _reraise_security_refusal(error: ValueError) -> None:
        if str(error).startswith("Unknown "):
            raise error

    @staticmethod
    def _rejection_observation(
        name: str, error: ValueError
    ) -> ToolObservation:
        return ToolObservation(
            tool_name=name,
            state="invalid_input",
            message=str(error),
        )


def state(*, projection_refusals: int = 0) -> MainAgentState:
    return {
        "context": cast(MainAgentContext, object()),
        "decision": AgentDecision(
            action="tool_call",
            tool_call=ToolCall(name="example", arguments={"value": "model"}),
        ),
        "control": {"projection_refusals": projection_refusals},
    }


def test_projects_the_selected_capability_kind() -> None:
    host = ProjectionHost()
    middleware = ArgumentProjectionMiddleware(
        host=host, max_projection_refusals=2
    )

    outcome = middleware.project(
        state(),
        name="example",
        kind="atomic_tool",
        runtime_owned=False,
        owner_confirmed=False,
        policy_owned=False,
        policy_prelude=False,
    )

    assert outcome == ProjectedArguments(arguments={"bound": "model"})
    assert host.calls == ["atomic"]


def test_soft_projection_error_becomes_an_observation_until_the_cap() -> None:
    middleware = ArgumentProjectionMiddleware(
        host=ProjectionHost(failure=ValueError("selection is unavailable")),
        max_projection_refusals=2,
    )

    retry = middleware.project(
        state(projection_refusals=1),
        name="example",
        kind="atomic_tool",
        runtime_owned=False,
        owner_confirmed=False,
        policy_owned=True,
        policy_prelude=True,
    )
    capped = middleware.project(
        state(projection_refusals=2),
        name="example",
        kind="atomic_tool",
        runtime_owned=False,
        owner_confirmed=False,
        policy_owned=False,
        policy_prelude=False,
    )

    assert isinstance(retry, ProjectionRefusal)
    assert retry.state_update["authorization_route"] == "observe"
    assert retry.state_update["pending"]["synthetic_kind"] == "projection"
    assert retry.state_update["pending"]["policy_owned"] is True
    assert isinstance(capped, ProjectionRefusal)
    assert capped.state_update == {"authorization_route": "present"}


def test_security_projection_error_remains_a_hard_failure() -> None:
    middleware = ArgumentProjectionMiddleware(
        host=ProjectionHost(failure=ValueError("Unknown protected tool")),
        max_projection_refusals=2,
    )

    with pytest.raises(ValueError, match="Unknown protected tool"):
        middleware.project(
            state(),
            name="example",
            kind="atomic_tool",
            runtime_owned=False,
            owner_confirmed=False,
            policy_owned=False,
            policy_prelude=False,
        )
