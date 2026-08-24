from __future__ import annotations

from collections.abc import Callable
from typing import Any

from career_agent.agent.job_discovery_gateway import JobDiscoveryGateway, JobDiscoveryGatewayResult
from career_agent.agent.main_agent_contracts import (
    FindSavedJobsToolArguments,
    GetSavedJobToolArguments,
    JobDiscoveryToolArguments,
    JobDiscoveryWorkflowInput,
    ToolObservation,
)
from career_agent.storage.jobs import JobPostingRepository


MainAgentToolOutput = JobDiscoveryGatewayResult | ToolObservation


class MainAgentToolRegistry:
    def __init__(self, gateway: JobDiscoveryGateway, *, job_repository: JobPostingRepository | None = None) -> None:
        self._handlers: dict[str, Callable[[dict[str, Any]], MainAgentToolOutput]] = {
            "job_discovery": self._job_discovery,
        }
        self._gateway = gateway
        self._job_repository = job_repository
        if job_repository is not None:
            self._handlers.update(
                {
                    "find_saved_jobs": self._find_saved_jobs,
                    "get_saved_job": self._get_saved_job,
                }
            )

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._handlers)

    def schemas(self) -> tuple[dict[str, Any], ...]:
        schemas = [
            {
                "type": "function",
                "function": {
                    "name": "job_discovery",
                    "description": "Enter or continue the user's read-only job discovery workflow. The workflow decides whether to search, wait for selection, fetch one JD, or accept user-provided JD based on its persisted state.",
                    "parameters": JobDiscoveryToolArguments.model_json_schema(),
                },
            },
        ]
        if self._job_repository is not None:
            schemas.extend(
                [
                    {
                        "type": "function",
                        "function": {
                            "name": "find_saved_jobs",
                            "description": "Search only the current user's previously saved or viewed jobs. Use this for historical recall, not for discovering new online jobs. Returns summaries and job_posting_id values, never complete JD text.",
                            "parameters": FindSavedJobsToolArguments.model_json_schema(),
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "get_saved_job",
                            "description": "Read one previously saved job and its complete JD by job_posting_id. Use only when the user asks to inspect a specific saved result or complete JD.",
                            "parameters": GetSavedJobToolArguments.model_json_schema(),
                        },
                    },
                ]
            )
        return tuple(schemas)

    def invoke(self, name: str, arguments: dict[str, Any]) -> MainAgentToolOutput:
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

    def _find_saved_jobs(self, arguments: dict[str, Any]) -> ToolObservation:
        if self._job_repository is None:
            raise ValueError("Saved-job repository is not configured")
        user_id = str(arguments["user_id"])
        model_arguments = FindSavedJobsToolArguments.model_validate(
            {key: value for key, value in arguments.items() if key != "user_id"}
        )
        items = self._job_repository.search_saved_jobs(
            user_id=user_id,
            query=model_arguments.query,
            limit=model_arguments.limit,
        )
        payload = {
            "items": [item.model_dump(mode="json") for item in items],
            "query": model_arguments.query,
        }
        return ToolObservation(
            tool_name="find_saved_jobs",
            state="saved_jobs_found" if items else "no_saved_jobs_found",
            message=f"找到 {len(items)} 个已保存职位。" if items else "没有找到匹配的已保存职位。",
            payload=payload,
        )

    def _get_saved_job(self, arguments: dict[str, Any]) -> ToolObservation:
        if self._job_repository is None:
            raise ValueError("Saved-job repository is not configured")
        user_id = str(arguments["user_id"])
        model_arguments = GetSavedJobToolArguments.model_validate(
            {key: value for key, value in arguments.items() if key != "user_id"}
        )
        record = self._job_repository.get_job(user_id=user_id, job_posting_id=model_arguments.job_posting_id)
        if record is None:
            return ToolObservation(
                tool_name="get_saved_job",
                state="saved_job_not_found",
                message="没有找到这个已保存职位，或它不属于当前用户。",
                payload={"job_posting_id": model_arguments.job_posting_id},
            )
        payload = {
            "job": {
                "job_posting_id": record.posting.id,
                "title": record.posting.title,
                "company_name": record.posting.company_name,
                "city": record.city,
                "salary": record.salary,
                "source_name": record.posting.source_name,
                "source_url": record.posting.source_url,
                "availability_status": record.availability_status,
            },
            "jd_snapshot": {
                "version": record.snapshot.version,
                "content": record.snapshot.content,
                "captured_at": record.snapshot.captured_at.isoformat(),
                "provenance": record.snapshot.provenance.model_dump(mode="json"),
            },
            "analysis": record.analysis.analysis.model_dump(mode="json") if record.analysis else None,
        }
        return ToolObservation(
            tool_name="get_saved_job",
            state="saved_job_ready",
            message=f"已读取 {record.posting.title}（{record.posting.company_name}）的完整 JD。",
            payload=payload,
        )
