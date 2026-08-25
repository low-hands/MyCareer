from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

from career_agent.agent.job_discovery_gateway import JobDiscoveryGateway, JobDiscoveryGatewayResult
from career_agent.agent.main_agent_contracts import (
    AnalyzeResumeToolArguments,
    ConfirmResumeAnalysisToolArguments,
    DraftResumeTailoringToolArguments,
    FindSavedJobsToolArguments,
    GetResumeMetadataToolArguments,
    GetResumeAnalysisToolArguments,
    GetResumeJobMatchToolArguments,
    GetResumeTailoringDraftToolArguments,
    GetSavedJobToolArguments,
    JobDiscoveryToolArguments,
    JobDiscoveryWorkflowInput,
    ListResumesToolArguments,
    ListTargetRolesToolArguments,
    MatchResumeToJobToolArguments,
    ToolObservation,
)
from career_agent.agent.openai_compatible_client import AgentWorkerError
from career_agent.services.resume_analysis import (
    ResumeAnalysisNotFoundError,
    ResumeAnalysisService,
    ResumeVersionNotFoundError,
)
from career_agent.services.resume_job_match import (
    ResumeJobMatchInputNotFoundError,
    ResumeJobMatchService,
)
from career_agent.services.resume_tailoring import (
    ResumeTailoringDraftNotFoundError,
    ResumeTailoringService,
)
from career_agent.storage.jobs import JobPostingRepository
from career_agent.storage.resumes import ResumeStore
from career_agent.storage.resume_tailoring import StoredResumeTailoringDraft


MainAgentToolOutput = JobDiscoveryGatewayResult | ToolObservation
CapabilityKind = Literal["atomic_tool", "workflow"]


class MainAgentToolRegistry:
    def __init__(
        self,
        gateway: JobDiscoveryGateway,
        *,
        job_repository: JobPostingRepository | None = None,
        resume_store: ResumeStore | None = None,
        resume_analysis_service: ResumeAnalysisService | None = None,
        resume_job_match_service: ResumeJobMatchService | None = None,
        resume_tailoring_service: ResumeTailoringService | None = None,
    ) -> None:
        self._workflow_handlers: dict[str, Callable[[dict[str, Any]], JobDiscoveryGatewayResult]] = {
            "job_discovery": self._job_discovery,
        }
        self._atomic_handlers: dict[str, Callable[[dict[str, Any]], ToolObservation]] = {}
        self._gateway = gateway
        self._job_repository = job_repository
        self._resume_store = resume_store
        self._resume_analysis_service = resume_analysis_service
        self._resume_job_match_service = resume_job_match_service
        self._resume_tailoring_service = resume_tailoring_service
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
        if resume_analysis_service is not None:
            self._atomic_handlers.update(
                {
                    "analyze_resume": self._analyze_resume,
                    "get_resume_analysis": self._get_resume_analysis,
                    "confirm_resume_analysis": self._confirm_resume_analysis,
                }
            )
        if resume_job_match_service is not None:
            self._atomic_handlers.update(
                {
                    "match_resume_to_job": self._match_resume_to_job,
                    "get_resume_job_match": self._get_resume_job_match,
                }
            )
        if resume_tailoring_service is not None:
            self._atomic_handlers.update(
                {
                    "draft_resume_tailoring": self._draft_resume_tailoring,
                    "get_resume_tailoring_draft": self._get_resume_tailoring_draft,
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
        if self._resume_analysis_service is not None:
            schemas.extend(
                [
                    {
                        "type": "function",
                        "function": {
                            "name": "analyze_resume",
                            "description": "Analyze one current-user resume version by resume_version_id. Use when the user asks to read, extract, review, or analyze resume content. Returns a pending analysis_id plus structured candidate career records, grounded evidence quotes, clarification questions, and warnings; never returns the original file. Candidates are not career facts until the user explicitly confirms them.",
                            "parameters": AnalyzeResumeToolArguments.model_json_schema(),
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "get_resume_analysis",
                            "description": "Retrieve one current user's unexpired resume analysis draft by analysis_id so its candidates can be reviewed before confirmation. Never returns the original resume file.",
                            "parameters": GetResumeAnalysisToolArguments.model_json_schema(),
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "confirm_resume_analysis",
                            "description": "Confirm all candidates in one resume analysis by analysis_id and persist them as CareerRecord and confirmed CareerEvidence. Call only after the user explicitly confirms that specific analysis; never infer confirmation.",
                            "parameters": ConfirmResumeAnalysisToolArguments.model_json_schema(),
                        },
                    },
                ]
            )
        if self._resume_job_match_service is not None:
            schemas.extend(
                [
                    {
                        "type": "function",
                        "function": {
                            "name": "match_resume_to_job",
                            "description": "Compare one exact current-user resume version with one exact saved job's complete JD. Use resume_version_id and job_posting_id returned by the resume and saved-job tools. Returns a persisted match_id and grounded requirement-by-requirement assessment; does not search online and never returns either original document.",
                            "parameters": MatchResumeToJobToolArguments.model_json_schema(),
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "get_resume_job_match",
                            "description": "Retrieve a previously persisted resume-job match by match_id. When omitted, the current conversation's active match is used. Returns only the structured assessment, never the original resume or complete JD.",
                            "parameters": GetResumeJobMatchToolArguments.model_json_schema(),
                        },
                    },
                ]
            )
        if self._resume_tailoring_service is not None:
            schemas.extend(
                [
                    {
                        "type": "function",
                        "function": {
                            "name": "draft_resume_tailoring",
                            "description": "Create a reviewable tailoring draft from a persisted resume-job match. Uses the active match when match_id is omitted. May accept a user tailoring goal. Returns grounded proposed changes and a draft_id; it does not alter or create a resume version.",
                            "parameters": DraftResumeTailoringToolArguments.model_json_schema(),
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "get_resume_tailoring_draft",
                            "description": "Retrieve an unexpired tailoring draft by draft_id, or use the active draft when omitted. Returns proposed changes for review; it does not apply them.",
                            "parameters": GetResumeTailoringDraftToolArguments.model_json_schema(),
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

    def _analyze_resume(self, arguments: dict[str, Any]) -> ToolObservation:
        if self._resume_analysis_service is None:
            raise ValueError("Resume analysis service is not configured")
        user_id = str(arguments["user_id"])
        model_arguments = AnalyzeResumeToolArguments.model_validate(
            {key: value for key, value in arguments.items() if key != "user_id"}
        )
        try:
            draft = self._resume_analysis_service.analyze_version(
                user_id=user_id,
                resume_version_id=model_arguments.resume_version_id,
            )
        except ResumeVersionNotFoundError:
            return ToolObservation(
                tool_name="analyze_resume",
                state="resume_version_not_found",
                message="没有找到这个简历版本，或它不属于当前用户。",
                payload={"resume_version_id": model_arguments.resume_version_id},
            )
        except AgentWorkerError as error:
            return ToolObservation(
                tool_name="analyze_resume",
                state="failed",
                message="简历分析暂时失败，请稍后重试。" if error.retryable else "简历分析失败。",
                payload={
                    "resume_version_id": model_arguments.resume_version_id,
                    "error_code": error.code,
                    "retryable": error.retryable,
                },
            )
        return ToolObservation(
            tool_name="analyze_resume",
            state="resume_analysis_ready",
            message=f"已分析该简历版本，提取出 {len(draft.result.records)} 段候选经历。",
            next_action="review_and_confirm_extracted_career_facts",
            payload={
                "analysis_id": draft.id,
                "resume_version_id": model_arguments.resume_version_id,
                "expires_at": draft.expires_at.isoformat(),
                "records": [record.model_dump(mode="json") for record in draft.result.records],
                "clarification_questions": draft.result.clarification_questions,
                "warnings": draft.result.warnings,
            },
        )

    def _get_resume_analysis(self, arguments: dict[str, Any]) -> ToolObservation:
        if self._resume_analysis_service is None:
            raise ValueError("Resume analysis service is not configured")
        user_id = str(arguments["user_id"])
        model_arguments = GetResumeAnalysisToolArguments.model_validate(
            {key: value for key, value in arguments.items() if key != "user_id"}
        )
        if model_arguments.analysis_id is None:
            raise ValueError("get_resume_analysis requires analysis_id")
        try:
            draft = self._resume_analysis_service.get_analysis(
                user_id=user_id,
                analysis_id=model_arguments.analysis_id,
            )
        except ResumeAnalysisNotFoundError:
            return ToolObservation(
                tool_name="get_resume_analysis",
                state="resume_analysis_not_found",
                message="没有找到这次简历分析，或它已经过期。",
                payload={"analysis_id": model_arguments.analysis_id},
            )
        return ToolObservation(
            tool_name="get_resume_analysis",
            state="resume_analysis_ready",
            message=f"已读取这次简历分析，其中有 {len(draft.result.records)} 段候选经历。",
            next_action=(
                "review_and_confirm_extracted_career_facts"
                if draft.status == "pending"
                else None
            ),
            payload={
                "analysis_id": draft.id,
                "resume_version_id": draft.resume_version_id,
                "status": draft.status,
                "expires_at": draft.expires_at.isoformat(),
                "records": [record.model_dump(mode="json") for record in draft.result.records],
                "clarification_questions": draft.result.clarification_questions,
                "warnings": draft.result.warnings,
            },
        )

    def _confirm_resume_analysis(self, arguments: dict[str, Any]) -> ToolObservation:
        if self._resume_analysis_service is None:
            raise ValueError("Resume analysis service is not configured")
        user_id = str(arguments["user_id"])
        model_arguments = ConfirmResumeAnalysisToolArguments.model_validate(
            {key: value for key, value in arguments.items() if key != "user_id"}
        )
        if model_arguments.analysis_id is None:
            raise ValueError("confirm_resume_analysis requires analysis_id")
        try:
            imported = self._resume_analysis_service.confirm_analysis(
                user_id=user_id,
                analysis_id=model_arguments.analysis_id,
            )
        except ResumeAnalysisNotFoundError:
            return ToolObservation(
                tool_name="confirm_resume_analysis",
                state="resume_analysis_not_found",
                message="没有找到这次简历分析，或它已经过期，无法确认。",
                payload={"analysis_id": model_arguments.analysis_id},
            )
        return ToolObservation(
            tool_name="confirm_resume_analysis",
            state="resume_analysis_confirmed",
            message=(
                f"已确认并保存 {len(imported.records)} 段职业经历和 "
                f"{len(imported.evidence)} 条事实证据。"
            ),
            payload={
                "analysis_id": model_arguments.analysis_id,
                "career_record_ids": [record.id for record in imported.records],
                "career_evidence_ids": [evidence.id for evidence in imported.evidence],
            },
        )

    def _match_resume_to_job(self, arguments: dict[str, Any]) -> ToolObservation:
        if self._resume_job_match_service is None:
            raise ValueError("Resume-job match service is not configured")
        user_id = str(arguments["user_id"])
        model_arguments = MatchResumeToJobToolArguments.model_validate(
            {key: value for key, value in arguments.items() if key != "user_id"}
        )
        try:
            stored = self._resume_job_match_service.match(
                user_id=user_id,
                resume_version_id=model_arguments.resume_version_id,
                job_posting_id=model_arguments.job_posting_id,
            )
        except ResumeJobMatchInputNotFoundError as error:
            return ToolObservation(
                tool_name="match_resume_to_job",
                state="match_input_not_found",
                message=(
                    "没有找到这个简历版本，或它不属于当前用户。"
                    if error.input_kind == "resume_version"
                    else "没有找到这个已保存职位，或它不属于当前用户。"
                ),
                payload={
                    "missing_input": error.input_kind,
                    "resume_version_id": model_arguments.resume_version_id,
                    "job_posting_id": model_arguments.job_posting_id,
                },
            )
        except AgentWorkerError as error:
            return ToolObservation(
                tool_name="match_resume_to_job",
                state="failed",
                message="简历与岗位匹配暂时失败，请稍后重试。" if error.retryable else "简历与岗位匹配失败。",
                payload={
                    "resume_version_id": model_arguments.resume_version_id,
                    "job_posting_id": model_arguments.job_posting_id,
                    "error_code": error.code,
                    "retryable": error.retryable,
                },
            )
        return ToolObservation(
            tool_name="match_resume_to_job",
            state="resume_job_match_ready",
            message=f"已完成逐项匹配，整体匹配度为 {stored.result.overall_fit}。",
            next_action="explain_match_or_offer_resume_tailoring",
            payload={
                "match_id": stored.id,
                "resume_version_id": model_arguments.resume_version_id,
                "job_posting_id": model_arguments.job_posting_id,
                "created_at": stored.created_at.isoformat(),
                **stored.result.model_dump(mode="json"),
            },
        )

    def _get_resume_job_match(self, arguments: dict[str, Any]) -> ToolObservation:
        if self._resume_job_match_service is None:
            raise ValueError("Resume-job match service is not configured")
        user_id = str(arguments["user_id"])
        model_arguments = GetResumeJobMatchToolArguments.model_validate(
            {key: value for key, value in arguments.items() if key != "user_id"}
        )
        if model_arguments.match_id is None:
            raise ValueError("get_resume_job_match requires match_id")
        try:
            stored = self._resume_job_match_service.get_match(
                user_id=user_id,
                match_id=model_arguments.match_id,
            )
        except ResumeJobMatchInputNotFoundError:
            return ToolObservation(
                tool_name="get_resume_job_match",
                state="resume_job_match_not_found",
                message="没有找到这次简历岗位匹配，或它不属于当前用户。",
                payload={"match_id": model_arguments.match_id},
            )
        return ToolObservation(
            tool_name="get_resume_job_match",
            state="resume_job_match_ready",
            message=f"已读取匹配结果，整体匹配度为 {stored.result.overall_fit}。",
            next_action="explain_match_or_offer_resume_tailoring",
            payload={
                "match_id": stored.id,
                "resume_version_id": stored.resume_version_id,
                "job_posting_id": stored.job_posting_id,
                "created_at": stored.created_at.isoformat(),
                **stored.result.model_dump(mode="json"),
            },
        )

    def _draft_resume_tailoring(self, arguments: dict[str, Any]) -> ToolObservation:
        if self._resume_tailoring_service is None:
            raise ValueError("Resume tailoring service is not configured")
        user_id = str(arguments["user_id"])
        model_arguments = DraftResumeTailoringToolArguments.model_validate(
            {key: value for key, value in arguments.items() if key != "user_id"}
        )
        if model_arguments.match_id is None:
            raise ValueError("draft_resume_tailoring requires match_id")
        try:
            draft = self._resume_tailoring_service.create_draft(
                user_id=user_id,
                match_id=model_arguments.match_id,
                tailoring_goal=model_arguments.tailoring_goal,
            )
        except ResumeJobMatchInputNotFoundError:
            return ToolObservation(
                tool_name="draft_resume_tailoring",
                state="resume_job_match_not_found",
                message="没有找到可用于定制的匹配结果，或它不属于当前用户。",
                payload={"match_id": model_arguments.match_id},
            )
        except AgentWorkerError as error:
            return ToolObservation(
                tool_name="draft_resume_tailoring",
                state="failed",
                message="简历定制暂时失败，请稍后重试。" if error.retryable else "简历定制失败。",
                payload={"error_code": error.code, "retryable": error.retryable},
            )
        return self._tailoring_observation(
            tool_name="draft_resume_tailoring",
            draft=draft,
            message=f"已生成 {len(draft.result.changes)} 条待审阅的简历修改建议。",
        )

    def _get_resume_tailoring_draft(self, arguments: dict[str, Any]) -> ToolObservation:
        if self._resume_tailoring_service is None:
            raise ValueError("Resume tailoring service is not configured")
        user_id = str(arguments["user_id"])
        model_arguments = GetResumeTailoringDraftToolArguments.model_validate(
            {key: value for key, value in arguments.items() if key != "user_id"}
        )
        if model_arguments.draft_id is None:
            raise ValueError("get_resume_tailoring_draft requires draft_id")
        try:
            draft = self._resume_tailoring_service.get_draft(
                user_id=user_id,
                draft_id=model_arguments.draft_id,
            )
        except ResumeTailoringDraftNotFoundError:
            return ToolObservation(
                tool_name="get_resume_tailoring_draft",
                state="resume_tailoring_draft_not_found",
                message="没有找到这份简历定制草稿，或它已经过期。",
                payload={"draft_id": model_arguments.draft_id},
            )
        return self._tailoring_observation(
            tool_name="get_resume_tailoring_draft",
            draft=draft,
            message=f"已读取包含 {len(draft.result.changes)} 条修改建议的草稿。",
        )

    @staticmethod
    def _tailoring_observation(
        *,
        tool_name: str,
        draft: StoredResumeTailoringDraft,
        message: str,
    ) -> ToolObservation:
        return ToolObservation(
            tool_name=tool_name,
            state="resume_tailoring_draft_ready",
            message=message,
            next_action="review_tailoring_changes",
            payload={
                "draft_id": draft.id,
                "match_id": draft.match_id,
                "status": draft.status,
                "tailoring_goal": draft.tailoring_goal,
                "expires_at": draft.expires_at.isoformat(),
                **draft.result.model_dump(mode="json"),
            },
        )
