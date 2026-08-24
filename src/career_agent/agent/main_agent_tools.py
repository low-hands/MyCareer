from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

from career_agent.agent.job_discovery_gateway import JobDiscoveryGateway, JobDiscoveryGatewayResult
from career_agent.agent.main_agent_contracts import (
    FindSavedJobsToolArguments,
    GetResumeMetadataToolArguments,
    GetSavedJobToolArguments,
    JobDiscoveryToolArguments,
    JobDiscoveryWorkflowInput,
    ListResumesToolArguments,
    ListTargetRolesToolArguments,
    ToolObservation,
)
from career_agent.storage.jobs import JobPostingRepository
from career_agent.storage.resumes import ResumeStore


MainAgentToolOutput = JobDiscoveryGatewayResult | ToolObservation
CapabilityKind = Literal["atomic_tool", "workflow"]


class MainAgentToolRegistry:
    def __init__(self, gateway: JobDiscoveryGateway, *, job_repository: JobPostingRepository | None = None, resume_store: ResumeStore | None = None) -> None:
        self._workflow_handlers: dict[str, Callable[[dict[str, Any]], JobDiscoveryGatewayResult]] = {
            "job_discovery": self._job_discovery,
        }
        self._atomic_handlers: dict[str, Callable[[dict[str, Any]], ToolObservation]] = {}
        self._gateway = gateway
        self._job_repository = job_repository
        self._resume_store = resume_store
        if job_repository is not None:
            self._atomic_handlers.update(
                {
                    "find_saved_jobs": self._find_saved_jobs,
                    "get_saved_job": self._get_saved_job,
                }
            )
        if resume_store is not None:
            self._atomic_handlers.update(
                {
                    "list_target_roles": self._list_target_roles,
                    "list_resumes": self._list_resumes,
                    "get_resume_metadata": self._get_resume_metadata,
                }
            )

    @property
    def names(self) -> tuple[str, ...]:
        return (*self.workflow_names, *self.atomic_tool_names)

    @property
    def workflow_names(self) -> tuple[str, ...]:
        return tuple(self._workflow_handlers)

    @property
    def atomic_tool_names(self) -> tuple[str, ...]:
        return tuple(self._atomic_handlers)

    def capability_kind(self, name: str) -> CapabilityKind:
        if name in self._workflow_handlers:
            return "workflow"
        if name in self._atomic_handlers:
            return "atomic_tool"
        raise ValueError(f"Unknown main-agent capability: {name}")

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
        if self._resume_store is not None:
            schemas.extend(
                [
                    {
                        "type": "function",
                        "function": {
                            "name": "list_target_roles",
                            "description": "List the current user's resume target-role categories. Returns safe metadata and target_role_id values; never returns resume document content.",
                            "parameters": ListTargetRolesToolArguments.model_json_schema(),
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "list_resumes",
                            "description": "List the current user's resume families, optionally filtered by a target_role_id returned by list_target_roles. Returns safe metadata and IDs only; never returns resume document content.",
                            "parameters": ListResumesToolArguments.model_json_schema(),
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "get_resume_metadata",
                            "description": "Read one of the current user's resume families and its immutable version metadata by resume_id. Never returns PDF, text, Markdown, extracted content, or file paths.",
                            "parameters": GetResumeMetadataToolArguments.model_json_schema(),
                        },
                    },
                ]
            )
        return tuple(schemas)

    def invoke_workflow(self, name: str, arguments: dict[str, Any]) -> JobDiscoveryGatewayResult:
        handler = self._workflow_handlers.get(name)
        if handler is None:
            raise ValueError(f"Unknown main-agent workflow: {name}")
        return handler(arguments)

    def invoke_atomic_tool(self, name: str, arguments: dict[str, Any]) -> ToolObservation:
        handler = self._atomic_handlers.get(name)
        if handler is None:
            raise ValueError(f"Unknown main-agent atomic tool: {name}")
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

    def _list_target_roles(self, arguments: dict[str, Any]) -> ToolObservation:
        if self._resume_store is None:
            raise ValueError("Resume store is not configured")
        user_id = str(arguments["user_id"])
        ListTargetRolesToolArguments.model_validate(
            {key: value for key, value in arguments.items() if key != "user_id"}
        )
        roles = self._resume_store.list_target_roles(user_id=user_id)
        return ToolObservation(
            tool_name="list_target_roles",
            state="target_roles_found" if roles else "no_target_roles_found",
            message=f"找到 {len(roles)} 个简历目标岗位分类。" if roles else "当前还没有简历目标岗位分类。",
            payload={
                "items": [
                    {
                        "selection_index": index,
                        "target_role_id": role.id,
                        "title": role.title,
                        "priority": role.priority,
                        "status": role.status,
                    }
                    for index, role in enumerate(roles, start=1)
                ]
            },
        )

    def _list_resumes(self, arguments: dict[str, Any]) -> ToolObservation:
        if self._resume_store is None:
            raise ValueError("Resume store is not configured")
        user_id = str(arguments["user_id"])
        model_arguments = ListResumesToolArguments.model_validate(
            {key: value for key, value in arguments.items() if key != "user_id"}
        )
        resumes = self._resume_store.list_resumes(
            user_id=user_id,
            target_role_id=model_arguments.target_role_id,
        )
        return ToolObservation(
            tool_name="list_resumes",
            state="resumes_found" if resumes else "no_resumes_found",
            message=f"找到 {len(resumes)} 份简历。" if resumes else "没有找到匹配的简历。",
            payload={
                "target_role_id": model_arguments.target_role_id,
                "items": [
                    {
                        "selection_index": index,
                        "resume_id": resume.id,
                        "target_role_id": resume.target_role_id,
                        "name": resume.name,
                        "status": resume.status,
                        "latest_version_id": resume.latest_version_id,
                        "updated_at": resume.updated_at.isoformat(),
                    }
                    for index, resume in enumerate(resumes, start=1)
                ],
            },
        )

    def _get_resume_metadata(self, arguments: dict[str, Any]) -> ToolObservation:
        if self._resume_store is None:
            raise ValueError("Resume store is not configured")
        user_id = str(arguments["user_id"])
        model_arguments = GetResumeMetadataToolArguments.model_validate(
            {key: value for key, value in arguments.items() if key != "user_id"}
        )
        resume = self._resume_store.get_resume(user_id=user_id, resume_id=model_arguments.resume_id)
        if resume is None:
            return ToolObservation(
                tool_name="get_resume_metadata",
                state="resume_not_found",
                message="没有找到这份简历，或它不属于当前用户。",
                payload={"resume_id": model_arguments.resume_id},
            )
        versions = self._resume_store.list_versions(user_id=user_id, resume_id=resume.id)
        return ToolObservation(
            tool_name="get_resume_metadata",
            state="resume_metadata_ready",
            message=f"已读取简历“{resume.name}”及其 {len(versions)} 个版本的元数据。",
            payload={
                "resume": {
                    "resume_id": resume.id,
                    "target_role_id": resume.target_role_id,
                    "name": resume.name,
                    "status": resume.status,
                    "latest_version_id": resume.latest_version_id,
                    "created_at": resume.created_at.isoformat(),
                    "updated_at": resume.updated_at.isoformat(),
                },
                "versions": [
                    {
                        "resume_version_id": version.id,
                        "version_number": version.version_number,
                        "source_type": version.source_type,
                        "document_format": version.document_format,
                        "byte_size": version.byte_size,
                        "created_at": version.created_at.isoformat(),
                    }
                    for version in versions
                ],
            },
        )
