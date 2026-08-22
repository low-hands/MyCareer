from __future__ import annotations

from collections.abc import Callable
from typing import Any

from career_agent.agent.job_discovery_gateway import JobDiscoveryGateway, JobDiscoveryGatewayResult
from career_agent.agent.main_agent_contracts import JobDiscoveryToolArguments, JobDiscoveryWorkflowInput


class MainAgentToolRegistry:
    def __init__(self, gateway: JobDiscoveryGateway) -> None:
        self._handlers: dict[str, Callable[[dict[str, Any]], JobDiscoveryGatewayResult]] = {
            "job_discovery": self._job_discovery,
        }
        self._gateway = gateway

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._handlers)

    def schemas(self) -> tuple[dict[str, Any], ...]:
        return (
            {
                "type": "function",
                "function": {
                    "name": "job_discovery",
                    "description": "Enter or continue the user's read-only job discovery workflow. The workflow decides whether to search, wait for selection, fetch one JD, or accept user-provided JD based on its persisted state.",
                    "parameters": JobDiscoveryToolArguments.model_json_schema(),
                },
            },
        )

    def invoke(self, name: str, arguments: dict[str, Any]) -> JobDiscoveryGatewayResult:
        handler = self._handlers.get(name)
        if handler is None:
            raise ValueError(f"Unknown main-agent tool: {name}")
        return handler(arguments)

    def _job_discovery(self, arguments: dict[str, Any]) -> JobDiscoveryGatewayResult:
        workflow_input = JobDiscoveryWorkflowInput.model_validate(arguments)
        return self._gateway.advance(
            user_id=workflow_input.user_id,
            conversation_id=workflow_input.conversation_id,
            task=workflow_input.task,
            user_message=workflow_input.user_message,
            selection_indices=workflow_input.selection_indices,
            selection_index=workflow_input.selection_index,
            jd_selection_index=workflow_input.jd_selection_index,
            research_request=workflow_input.research_request,
        )
