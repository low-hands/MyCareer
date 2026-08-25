from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

from career_agent.agent.job_discovery_gateway import JobDiscoveryGateway, JobDiscoveryGatewayResult
from career_agent.agent.main_agent_contracts import (
    AnalyzeResumeToolArguments,
    ConfirmResumeAnalysisToolArguments,
    CompleteInterviewToolArguments,
    CreateInterviewToolArguments,
    CreateApplicationToolArguments,
    DraftResumeTailoringToolArguments,
    ExportResumeArtifactToolArguments,
    FindSavedJobsToolArguments,
    FinalizeResumeTailoringToolArguments,
    GetApplicationToolArguments,
    GetInterviewToolArguments,
    GetResumeMetadataToolArguments,
    GetResumeAnalysisToolArguments,
    GetResumeJobMatchToolArguments,
    GetResumeTailoringDraftToolArguments,
    GetSavedJobToolArguments,
    JobDiscoveryToolArguments,
    JobDiscoveryWorkflowInput,
    ListResumesToolArguments,
    ListApplicationsToolArguments,
    ListTargetRolesToolArguments,
    ListEmailEventsToolArguments,
    ListInterviewsToolArguments,
    MatchResumeToJobToolArguments,
    ReviewResumeTailoringToolArguments,
    UpdateApplicationStatusToolArguments,
    ResolveEmailEventToolArguments,
    SyncApplicationEmailsToolArguments,
    UpdateInterviewToolArguments,
    ToolObservation,
)
from career_agent.agent.openai_compatible_client import AgentWorkerError
from career_agent.connectors.email_accounts import EmailCredentialError
from career_agent.connectors.gmail_readonly import GmailAPIError
from career_agent.services.resume_analysis import (
    ResumeAnalysisNotFoundError,
    ResumeAnalysisService,
    ResumeVersionNotFoundError,
)
from career_agent.services.applications import (
    ApplicationInputNotFoundError,
    ApplicationService,
    ConcurrentApplicationUpdateError,
    InvalidApplicationTransitionError,
)
from career_agent.services.email_tracking import (
    EmailAccountNotFoundError,
    EmailEventNotFoundError,
    EmailEventResolutionError,
    EmailTrackingService,
)
from career_agent.services.interviews import (
    InterviewApplicationConflictError,
    InterviewNotFoundError,
    InterviewService,
)
from career_agent.services.resume_job_match import (
    ResumeJobMatchInputNotFoundError,
    ResumeJobMatchService,
)
from career_agent.services.resume_export import (
    ResumeExportNotFoundError,
    ResumeExportService,
)
from career_agent.domain.resume import ResumeArtifactDelivery
from career_agent.services.resume_tailoring import (
    ResumeTailoringAlreadyFinalizedError,
    ResumeTailoringDraftNotFoundError,
    ResumeTailoringNotReadyError,
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
        resume_export_service: ResumeExportService | None = None,
        application_service: ApplicationService | None = None,
        email_tracking_service: EmailTrackingService | None = None,
        interview_service: InterviewService | None = None,
    ) -> None:
        self._workflow_handlers: dict[str, Callable[[dict[str, Any]], MainAgentToolOutput]] = {
            "job_discovery": self._job_discovery,
        }
        self._atomic_handlers: dict[str, Callable[[dict[str, Any]], ToolObservation]] = {}
        self._gateway = gateway
        self._job_repository = job_repository
        self._resume_store = resume_store
        self._resume_analysis_service = resume_analysis_service
        self._resume_job_match_service = resume_job_match_service
        self._resume_tailoring_service = resume_tailoring_service
        self._resume_export_service = resume_export_service
        self._application_service = application_service
        self._email_tracking_service = email_tracking_service
        self._interview_service = interview_service
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
        if resume_export_service is not None:
            self._atomic_handlers["export_resume_artifact"] = (
                self._export_resume_artifact
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
                    "review_resume_tailoring": self._review_resume_tailoring,
                    "finalize_resume_tailoring": self._finalize_resume_tailoring,
                }
            )
        if application_service is not None:
            self._atomic_handlers.update(
                {
                    "create_application": self._create_application,
                    "update_application_status": self._update_application_status,
                    "list_applications": self._list_applications,
                    "get_application": self._get_application,
                }
            )
        if email_tracking_service is not None:
            self._workflow_handlers["sync_application_emails"] = (
                self._sync_application_emails
            )
            self._atomic_handlers.update(
                {
                    "list_email_events": self._list_email_events,
                    "resolve_email_event": self._resolve_email_event,
                }
            )
        if interview_service is not None:
            self._atomic_handlers.update(
                {
                    "list_interviews": self._list_interviews,
                    "get_interview": self._get_interview,
                    "create_interview": self._create_interview,
                    "update_interview": self._update_interview,
                    "complete_interview": self._complete_interview,
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
                    {
                        "type": "function",
                        "function": {
                            "name": "review_resume_tailoring",
                            "description": "Accept or reject specific 1-based change indices in an active tailoring draft. Use only decisions the user explicitly made; never infer acceptance from vague approval. Decisions are persisted and may be completed across turns. This does not create a new resume version.",
                            "parameters": ReviewResumeTailoringToolArguments.model_json_schema(),
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "finalize_resume_tailoring",
                            "description": "Create one new immutable Markdown ResumeVersion from the explicitly accepted changes in a fully reviewed tailoring draft. Call only when the user explicitly asks to generate/save the new version after reviewing every change. Repeated calls are idempotent. Never use vague approval as authorization.",
                            "parameters": FinalizeResumeTailoringToolArguments.model_json_schema(),
                        },
                    },
                ]
            )
        if self._resume_export_service is not None:
            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": "export_resume_artifact",
                        "description": "Prepare an owned immutable resume version for download and return only an opaque artifact reference plus safe file metadata. Use the active version when resume_version_id is omitted. Call only when the user asks to download, export, or receive the resume file. Never place file content or a local path in the conversation.",
                        "parameters": ExportResumeArtifactToolArguments.model_json_schema(),
                    },
                }
            )
        if self._application_service is not None:
            schemas.extend(
                [
                    {
                        "type": "function",
                        "function": {
                            "name": "create_application",
                            "description": "Track a real externally submitted application using one exact owned resume version and the saved job's current immutable JD snapshot. Use the active job and resume version when IDs are omitted. Call only after the user explicitly reports that they actually applied; planning or preparing is not sufficient. Repeated calls for the same job return the original application.",
                            "parameters": CreateApplicationToolArguments.model_json_schema(),
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "update_application_status",
                            "description": "Update the active or specified application to a valid pipeline status, or add a note by supplying the unchanged status with a note. Use only status changes or facts explicitly supplied by the user.",
                            "parameters": UpdateApplicationStatusToolArguments.model_json_schema(),
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "list_applications",
                            "description": "List the current user's tracked applications, optionally filtered by pipeline statuses. Returns safe job and application metadata, not resume or JD contents.",
                            "parameters": ListApplicationsToolArguments.model_json_schema(),
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "get_application",
                            "description": "Read one tracked application and its append-only event timeline. Uses the active application when application_id is omitted.",
                            "parameters": GetApplicationToolArguments.model_json_schema(),
                        },
                    },
                ]
            )
        if self._email_tracking_service is not None:
            schemas.extend(
                [
                    {
                        "type": "function",
                        "function": {
                            "name": "sync_application_emails",
                            "description": "Run the read-only Gmail/QQ recruiting-email synchronization workflow. It fetches metadata first, reads only candidate bodies in an isolated worker, links events to tracked applications, and returns safe summaries. Use when the user asks to check or refresh employer email progress.",
                            "parameters": SyncApplicationEmailsToolArguments.model_json_schema(),
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "list_email_events",
                            "description": "List safe structured recruiting-email events, optionally only those awaiting confirmation. Never returns email bodies or credentials.",
                            "parameters": ListEmailEventsToolArguments.model_json_schema(),
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "resolve_email_event",
                            "description": "Approve or dismiss one pending email event. Approval may explicitly correct its application link and updates the application only when the transition is valid. Use only after clear user confirmation.",
                            "parameters": ResolveEmailEventToolArguments.model_json_schema(),
                        },
                    },
                ]
            )
        if self._interview_service is not None:
            schemas.extend(
                [
                    {
                        "type": "function",
                        "function": {
                            "name": "list_interviews",
                            "description": "List real interview appointments, optionally for one application or by status. sequence_number is only the system's chronological appointment number, not an employer-confirmed round label.",
                            "parameters": ListInterviewsToolArguments.model_json_schema(),
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "get_interview",
                            "description": "Read one interview appointment and its append-only invitation, reschedule, detail-update, cancellation, and completion history.",
                            "parameters": GetInterviewToolArguments.model_json_schema(),
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "create_interview",
                            "description": "Create a user-reported real interview appointment for an application. employer_label may be provided only when the employer explicitly used that label; never infer 一面/二面 from sequence.",
                            "parameters": CreateInterviewToolArguments.model_json_schema(),
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "update_interview",
                            "description": "Apply an explicitly user-reported reschedule, added detail, or cancellation to one existing interview. A reschedule updates the same appointment rather than creating another one.",
                            "parameters": UpdateInterviewToolArguments.model_json_schema(),
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "complete_interview",
                            "description": "Mark one real interview completed only after the user explicitly confirms they attended it. Time passing alone is never confirmation.",
                            "parameters": CompleteInterviewToolArguments.model_json_schema(),
                        },
                    },
                ]
            )
        return tuple(schemas)

    def invoke_workflow(self, name: str, arguments: dict[str, Any]) -> MainAgentToolOutput:
        handler = self._workflow_handlers.get(name)
        if handler is None:
            raise ValueError(f"Unknown main-agent workflow: {name}")
        return handler(arguments)

    def _sync_application_emails(self, arguments: dict[str, Any]) -> ToolObservation:
        if self._email_tracking_service is None:
            raise ValueError("Email tracking service is not configured")
        user_id = str(arguments["user_id"])
        model_arguments = SyncApplicationEmailsToolArguments.model_validate(
            {key: value for key, value in arguments.items() if key != "user_id"}
        )
        try:
            result = self._email_tracking_service.sync(
                user_id=user_id,
                account_id=model_arguments.account_id,
            )
        except EmailAccountNotFoundError:
            return ToolObservation(
                tool_name="sync_application_emails",
                state="email_account_not_found",
                message="没有找到可用的邮箱账号，请先连接 Gmail 或 QQ 邮箱。",
            )
        except (EmailCredentialError, GmailAPIError, RuntimeError, AgentWorkerError) as error:
            return ToolObservation(
                tool_name="sync_application_emails",
                state="failed",
                message="邮箱同步暂时失败；邮箱内容、凭证和错误响应均未进入 Main Agent 上下文。",
                payload={
                    "error_code": (
                        error.code
                        if isinstance(error, AgentWorkerError)
                        else type(error).__name__
                    ),
                    "retryable": isinstance(error, (GmailAPIError, AgentWorkerError)),
                },
            )
        pending = [event for event in result.events_created if event.status == "pending_confirmation"]
        return ToolObservation(
            tool_name="sync_application_emails",
            state="email_events_pending" if pending else "email_sync_complete",
            message=(
                f"已同步 {result.accounts_synced} 个邮箱，检查 {result.messages_seen} 封新邮件，"
                f"识别 {len(result.events_created)} 个求职事件，其中 {len(pending)} 个需要确认。"
            ),
            next_action="confirm_email_events" if pending else "track_application_progress",
            payload={
                "accounts_synced": result.accounts_synced,
                "messages_seen": result.messages_seen,
                "candidate_messages": result.candidate_messages,
                "events": [self._email_event_payload(event) for event in result.events_created],
            },
        )

    def _list_email_events(self, arguments: dict[str, Any]) -> ToolObservation:
        if self._email_tracking_service is None:
            raise ValueError("Email tracking service is not configured")
        user_id = str(arguments["user_id"])
        model_arguments = ListEmailEventsToolArguments.model_validate(
            {key: value for key, value in arguments.items() if key != "user_id"}
        )
        events = self._email_tracking_service.list_events(
            user_id=user_id,
            status=model_arguments.status,
            limit=model_arguments.limit,
        )
        return ToolObservation(
            tool_name="list_email_events",
            state="email_events_found" if events else "no_email_events_found",
            message=f"找到 {len(events)} 个邮件事件。" if events else "没有找到匹配的邮件事件。",
            payload={"items": [self._email_event_payload(event) for event in events]},
        )

    def _resolve_email_event(self, arguments: dict[str, Any]) -> ToolObservation:
        if self._email_tracking_service is None:
            raise ValueError("Email tracking service is not configured")
        user_id = str(arguments["user_id"])
        model_arguments = ResolveEmailEventToolArguments.model_validate(
            {key: value for key, value in arguments.items() if key != "user_id"}
        )
        try:
            event = self._email_tracking_service.resolve_event(
                user_id=user_id,
                event_id=model_arguments.event_id,
                approve=model_arguments.approve,
                application_id=model_arguments.application_id,
                interview_round_id=model_arguments.interview_round_id,
            )
        except EmailEventNotFoundError:
            return ToolObservation(
                tool_name="resolve_email_event",
                state="email_event_not_found",
                message="没有找到这个邮件事件，或它不属于当前用户。",
            )
        except EmailEventResolutionError as error:
            return ToolObservation(
                tool_name="resolve_email_event",
                state="email_event_resolution_conflict",
                message="该邮件事件暂时不能应用到投递记录。",
                payload={"reason": str(error)},
            )
        return ToolObservation(
            tool_name="resolve_email_event",
            state="email_event_resolved",
            message="邮件事件已应用。" if event.status == "applied" else "邮件事件已忽略。",
            payload=self._email_event_payload(event),
        )

    def _list_interviews(self, arguments: dict[str, Any]) -> ToolObservation:
        if self._interview_service is None:
            raise ValueError("Interview service is not configured")
        user_id = str(arguments["user_id"])
        model_arguments = ListInterviewsToolArguments.model_validate(
            {key: value for key, value in arguments.items() if key != "user_id"}
        )
        interviews = self._interview_service.list_interviews(
            user_id=user_id,
            application_id=model_arguments.application_id,
            statuses=model_arguments.statuses,
            limit=model_arguments.limit,
        )
        return ToolObservation(
            tool_name="list_interviews",
            state="interviews_found" if interviews else "no_interviews_found",
            message=f"找到 {len(interviews)} 场面试。" if interviews else "没有找到匹配的面试安排。",
            payload={
                "items": [
                    {
                        "selection_index": index,
                        **self._interview_payload(interview),
                    }
                    for index, interview in enumerate(interviews, start=1)
                ]
            },
        )

    def _get_interview(self, arguments: dict[str, Any]) -> ToolObservation:
        if self._interview_service is None:
            raise ValueError("Interview service is not configured")
        user_id = str(arguments["user_id"])
        model_arguments = GetInterviewToolArguments.model_validate(
            {key: value for key, value in arguments.items() if key != "user_id"}
        )
        if model_arguments.interview_round_id is None:
            raise ValueError("get_interview requires interview_round_id")
        try:
            detail = self._interview_service.get_interview(
                user_id=user_id,
                interview_round_id=model_arguments.interview_round_id,
            )
        except InterviewNotFoundError:
            return ToolObservation(
                tool_name="get_interview",
                state="interview_not_found",
                message="没有找到这场面试，或它不属于当前用户。",
            )
        return ToolObservation(
            tool_name="get_interview",
            state="interview_ready",
            message=f"已读取系统中的第 {detail.interview.sequence_number} 场面试。",
            payload={
                **self._interview_payload(detail.interview),
                "events": [
                    {
                        "event_type": event.event_type,
                        "source": event.source,
                        "email_event_id": event.email_event_id,
                        "details": event.details.model_dump(mode="json"),
                        "occurred_at": event.occurred_at.isoformat(),
                    }
                    for event in detail.events
                ],
            },
        )

    def _create_interview(self, arguments: dict[str, Any]) -> ToolObservation:
        if self._interview_service is None:
            raise ValueError("Interview service is not configured")
        user_id = str(arguments["user_id"])
        model_arguments = CreateInterviewToolArguments.model_validate(
            {key: value for key, value in arguments.items() if key != "user_id"}
        )
        if model_arguments.application_id is None:
            raise ValueError("create_interview requires application_id")
        try:
            interview = self._interview_service.create_manual(
                user_id=user_id,
                application_id=model_arguments.application_id,
                details=model_arguments.details,
            )
        except InterviewApplicationConflictError as error:
            return ToolObservation(
                tool_name="create_interview",
                state="interview_application_conflict",
                message="无法为这条投递创建面试安排。",
                payload={"reason": str(error)},
            )
        return ToolObservation(
            tool_name="create_interview",
            state="interview_ready",
            message=f"已记录系统中的第 {interview.sequence_number} 场面试。",
            payload=self._interview_payload(interview),
        )

    def _update_interview(self, arguments: dict[str, Any]) -> ToolObservation:
        if self._interview_service is None:
            raise ValueError("Interview service is not configured")
        user_id = str(arguments["user_id"])
        model_arguments = UpdateInterviewToolArguments.model_validate(
            {key: value for key, value in arguments.items() if key != "user_id"}
        )
        if model_arguments.interview_round_id is None:
            raise ValueError("update_interview requires interview_round_id")
        try:
            interview = self._interview_service.update_manual(
                user_id=user_id,
                interview_round_id=model_arguments.interview_round_id,
                details=model_arguments.details,
            )
        except InterviewNotFoundError:
            return ToolObservation(
                tool_name="update_interview", state="interview_not_found",
                message="没有找到这场面试，或它不属于当前用户。",
            )
        except InterviewApplicationConflictError as error:
            return ToolObservation(
                tool_name="update_interview", state="interview_update_conflict",
                message="这场面试当前不能按该方式更新。",
                payload={"reason": str(error)},
            )
        return ToolObservation(
            tool_name="update_interview",
            state="interview_ready",
            message="面试安排已更新，原安排仍保留在事件历史中。",
            payload=self._interview_payload(interview),
        )

    def _complete_interview(self, arguments: dict[str, Any]) -> ToolObservation:
        if self._interview_service is None:
            raise ValueError("Interview service is not configured")
        user_id = str(arguments["user_id"])
        model_arguments = CompleteInterviewToolArguments.model_validate(
            {key: value for key, value in arguments.items() if key != "user_id"}
        )
        if model_arguments.interview_round_id is None:
            raise ValueError("complete_interview requires interview_round_id")
        try:
            interview = self._interview_service.complete_interview(
                user_id=user_id,
                interview_round_id=model_arguments.interview_round_id,
                completed_at=model_arguments.completed_at,
            )
        except InterviewNotFoundError:
            return ToolObservation(
                tool_name="complete_interview", state="interview_not_found",
                message="没有找到这场面试，或它不属于当前用户。",
            )
        except InterviewApplicationConflictError as error:
            return ToolObservation(
                tool_name="complete_interview", state="interview_completion_conflict",
                message="这场面试当前不能标记为完成。",
                payload={"reason": str(error)},
            )
        return ToolObservation(
            tool_name="complete_interview",
            state="interview_ready",
            message="已将这场面试标记为完成。",
            next_action="offer_interview_retro",
            payload=self._interview_payload(interview),
        )

    @staticmethod
    def _email_event_payload(event) -> dict[str, Any]:
        return {
            "email_event_id": event.id,
            "application_id": event.application_id,
            "event_type": event.event_type,
            "status": event.status,
            "confidence": event.confidence,
            "classifier": event.classifier,
            "summary": event.summary,
            "occurred_at": event.occurred_at.isoformat(),
            "interview_details": (
                event.interview_details.model_dump(mode="json")
                if event.interview_details is not None
                else None
            ),
        }

    @staticmethod
    def _interview_payload(interview) -> dict[str, Any]:
        return {
            "interview_round_id": interview.id,
            "application_id": interview.application_id,
            "sequence_number": interview.sequence_number,
            "employer_label": interview.employer_label,
            "status": interview.status,
            "scheduled_start": (
                interview.scheduled_start.isoformat()
                if interview.scheduled_start is not None
                else None
            ),
            "scheduled_end": (
                interview.scheduled_end.isoformat()
                if interview.scheduled_end is not None
                else None
            ),
            "timezone": interview.timezone,
            "interview_format": interview.interview_format,
            "location": interview.location,
            "meeting_url": interview.meeting_url,
            "contact_summary": interview.contact_summary,
        }

    def invoke_atomic_tool(self, name: str, arguments: dict[str, Any]) -> ToolObservation:
        handler = self._atomic_handlers.get(name)
        if handler is None:
            raise ValueError(f"Unknown main-agent atomic tool: {name}")
        return handler(arguments)

    def deliver_resume_artifact(
        self, *, user_id: str, artifact_id: str
    ) -> ResumeArtifactDelivery:
        """Resolve an attachment outside the model-visible tool/state loop."""
        if self._resume_export_service is None:
            raise ValueError("Resume export service is not configured")
        return self._resume_export_service.read_artifact(
            user_id=user_id,
            artifact_id=artifact_id,
        )

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

    def _review_resume_tailoring(self, arguments: dict[str, Any]) -> ToolObservation:
        if self._resume_tailoring_service is None:
            raise ValueError("Resume tailoring service is not configured")
        user_id = str(arguments["user_id"])
        model_arguments = ReviewResumeTailoringToolArguments.model_validate(
            {key: value for key, value in arguments.items() if key != "user_id"}
        )
        if model_arguments.draft_id is None:
            raise ValueError("review_resume_tailoring requires draft_id")
        try:
            draft = self._resume_tailoring_service.review_draft(
                user_id=user_id,
                draft_id=model_arguments.draft_id,
                accepted_change_indices=model_arguments.accepted_change_indices,
                rejected_change_indices=model_arguments.rejected_change_indices,
                feedback=model_arguments.feedback,
            )
        except ResumeTailoringDraftNotFoundError:
            return ToolObservation(
                tool_name="review_resume_tailoring",
                state="resume_tailoring_draft_not_found",
                message="没有找到这份简历定制草稿，或它已经过期。",
                payload={"draft_id": model_arguments.draft_id},
            )
        except ResumeTailoringAlreadyFinalizedError:
            return ToolObservation(
                tool_name="review_resume_tailoring",
                state="resume_tailoring_already_finalized",
                message="这份草稿已经生成了新简历版本，审阅决定不能再修改。",
                payload={"draft_id": model_arguments.draft_id},
            )
        return self._tailoring_observation(
            tool_name="review_resume_tailoring",
            draft=draft,
            message=(
                "所有简历修改建议都已完成审阅。"
                if draft.status == "reviewed"
                else f"已记录审阅决定，还有 {len(draft.pending_change_indices)} 条建议待处理。"
            ),
        )

    def _finalize_resume_tailoring(self, arguments: dict[str, Any]) -> ToolObservation:
        if self._resume_tailoring_service is None:
            raise ValueError("Resume tailoring service is not configured")
        user_id = str(arguments["user_id"])
        model_arguments = FinalizeResumeTailoringToolArguments.model_validate(
            {key: value for key, value in arguments.items() if key != "user_id"}
        )
        if model_arguments.draft_id is None:
            raise ValueError("finalize_resume_tailoring requires draft_id")
        try:
            finalized = self._resume_tailoring_service.finalize_draft(
                user_id=user_id,
                draft_id=model_arguments.draft_id,
            )
        except ResumeTailoringDraftNotFoundError:
            return ToolObservation(
                tool_name="finalize_resume_tailoring",
                state="resume_tailoring_draft_not_found",
                message="没有找到这份简历定制草稿，或它已经过期。",
                payload={"draft_id": model_arguments.draft_id},
            )
        except ResumeTailoringNotReadyError as error:
            return ToolObservation(
                tool_name="finalize_resume_tailoring",
                state="resume_tailoring_not_ready",
                message="必须先逐条审阅所有建议，并至少接受一条修改。",
                payload={
                    "draft_id": model_arguments.draft_id,
                    "reason": str(error),
                },
            )
        except AgentWorkerError as error:
            return ToolObservation(
                tool_name="finalize_resume_tailoring",
                state="failed",
                message="生成新简历版本暂时失败，请稍后重试。" if error.retryable else "生成新简历版本失败。",
                payload={"error_code": error.code, "retryable": error.retryable},
            )
        return ToolObservation(
            tool_name="finalize_resume_tailoring",
            state="resume_tailoring_finalized",
            message=(
                "已生成新的不可变 Markdown 简历版本。"
                if finalized.created
                else "这份定制草稿已经生成过简历版本，已返回原结果。"
            ),
            next_action="offer_resume_export_or_review",
            payload={
                "draft_id": finalized.draft_id,
                "resume_id": finalized.resume.id,
                "resume_version_id": finalized.resume_version.id,
                "version_number": finalized.resume_version.version_number,
                "document_format": finalized.resume_version.document_format,
                "source_type": finalized.resume_version.source_type,
                "created_at": finalized.resume_version.created_at.isoformat(),
                "applied_change_indices": finalized.applied_change_indices,
                "warnings": finalized.warnings,
                "created": finalized.created,
            },
        )

    def _export_resume_artifact(self, arguments: dict[str, Any]) -> ToolObservation:
        if self._resume_export_service is None:
            raise ValueError("Resume export service is not configured")
        user_id = str(arguments["user_id"])
        model_arguments = ExportResumeArtifactToolArguments.model_validate(
            {key: value for key, value in arguments.items() if key != "user_id"}
        )
        if model_arguments.resume_version_id is None:
            raise ValueError("export_resume_artifact requires resume_version_id")
        try:
            artifact = self._resume_export_service.prepare_export(
                user_id=user_id,
                resume_version_id=model_arguments.resume_version_id,
            )
        except ResumeExportNotFoundError:
            return ToolObservation(
                tool_name="export_resume_artifact",
                state="resume_version_not_found",
                message="没有找到可导出的简历版本，或它不属于当前用户。",
                payload={"resume_version_id": model_arguments.resume_version_id},
            )
        return ToolObservation(
            tool_name="export_resume_artifact",
            state="resume_artifact_ready",
            message=f"简历文件 {artifact.filename} 已准备好。",
            next_action="deliver_artifact",
            payload={
                "artifact_id": artifact.id,
                "resume_version_id": artifact.resume_version_id,
                "filename": artifact.filename,
                "media_type": artifact.media_type,
                "byte_size": artifact.byte_size,
                "created_at": artifact.created_at.isoformat(),
            },
        )

    def _create_application(self, arguments: dict[str, Any]) -> ToolObservation:
        if self._application_service is None:
            raise ValueError("Application service is not configured")
        user_id = str(arguments["user_id"])
        model_arguments = CreateApplicationToolArguments.model_validate(
            {key: value for key, value in arguments.items() if key != "user_id"}
        )
        if (
            model_arguments.job_posting_id is None
            or model_arguments.resume_version_id is None
        ):
            raise ValueError("create_application requires job and resume version IDs")
        try:
            result = self._application_service.create_application(
                user_id=user_id,
                job_posting_id=model_arguments.job_posting_id,
                resume_version_id=model_arguments.resume_version_id,
                submitted_at=model_arguments.submitted_at,
                note=model_arguments.note,
            )
            detail = self._application_service.get_application(
                user_id=user_id,
                application_id=result.application.id,
            )
        except ApplicationInputNotFoundError as error:
            return ToolObservation(
                tool_name="create_application",
                state="application_input_not_found",
                message="没有找到对应的已保存岗位或简历版本，或它不属于当前用户。",
                payload={"missing_input": str(error)},
            )
        return ToolObservation(
            tool_name="create_application",
            state="application_ready",
            message=(
                "已创建投递记录。"
                if result.created
                else "这个岗位已有进行中的投递记录，已返回原记录。"
            ),
            next_action="track_application_progress",
            payload={
                **self._application_payload(result.application, detail.job),
                "created": result.created,
            },
        )

    def _update_application_status(
        self, arguments: dict[str, Any]
    ) -> ToolObservation:
        if self._application_service is None:
            raise ValueError("Application service is not configured")
        user_id = str(arguments["user_id"])
        model_arguments = UpdateApplicationStatusToolArguments.model_validate(
            {key: value for key, value in arguments.items() if key != "user_id"}
        )
        if model_arguments.application_id is None:
            raise ValueError("update_application_status requires application_id")
        try:
            application = self._application_service.update_application(
                user_id=user_id,
                application_id=model_arguments.application_id,
                status=model_arguments.status,
                note=model_arguments.note,
            )
            detail = self._application_service.get_application(
                user_id=user_id,
                application_id=application.id,
            )
        except ApplicationInputNotFoundError:
            return ToolObservation(
                tool_name="update_application_status",
                state="application_not_found",
                message="没有找到这条投递记录，或它不属于当前用户。",
                payload={"application_id": model_arguments.application_id},
            )
        except InvalidApplicationTransitionError as error:
            return ToolObservation(
                tool_name="update_application_status",
                state="invalid_application_transition",
                message="这次投递状态变化不符合当前流程。",
                payload={
                    "application_id": model_arguments.application_id,
                    "reason": str(error),
                },
            )
        except ConcurrentApplicationUpdateError:
            return ToolObservation(
                tool_name="update_application_status",
                state="application_update_conflict",
                message="这条投递记录刚刚发生了变化，请重新读取后再更新。",
                payload={"application_id": model_arguments.application_id},
            )
        return ToolObservation(
            tool_name="update_application_status",
            state="application_ready",
            message=f"投递状态已更新为 {application.status}。",
            next_action="track_application_progress",
            payload=self._application_payload(application, detail.job),
        )

    def _list_applications(self, arguments: dict[str, Any]) -> ToolObservation:
        if self._application_service is None:
            raise ValueError("Application service is not configured")
        user_id = str(arguments["user_id"])
        model_arguments = ListApplicationsToolArguments.model_validate(
            {key: value for key, value in arguments.items() if key != "user_id"}
        )
        items = self._application_service.list_applications(
            user_id=user_id,
            statuses=model_arguments.statuses,
            limit=model_arguments.limit,
        )
        return ToolObservation(
            tool_name="list_applications",
            state="applications_found" if items else "no_applications_found",
            message=f"找到 {len(items)} 条投递记录。" if items else "当前没有匹配的投递记录。",
            payload={
                "statuses": model_arguments.statuses,
                "items": [
                    {
                        "selection_index": index,
                        **self._application_payload(item.application, item.job),
                    }
                    for index, item in enumerate(items, start=1)
                ],
            },
        )

    def _get_application(self, arguments: dict[str, Any]) -> ToolObservation:
        if self._application_service is None:
            raise ValueError("Application service is not configured")
        user_id = str(arguments["user_id"])
        model_arguments = GetApplicationToolArguments.model_validate(
            {key: value for key, value in arguments.items() if key != "user_id"}
        )
        if model_arguments.application_id is None:
            raise ValueError("get_application requires application_id")
        try:
            detail = self._application_service.get_application(
                user_id=user_id,
                application_id=model_arguments.application_id,
            )
        except ApplicationInputNotFoundError:
            return ToolObservation(
                tool_name="get_application",
                state="application_not_found",
                message="没有找到这条投递记录，或它不属于当前用户。",
                payload={"application_id": model_arguments.application_id},
            )
        return ToolObservation(
            tool_name="get_application",
            state="application_ready",
            message=f"已读取 {detail.job.posting.company_name} 的投递记录。",
            next_action="track_application_progress",
            payload={
                **self._application_payload(detail.application, detail.job),
                "events": [
                    {
                        "event_type": event.event_type,
                        "source": event.source,
                        "previous_status": event.previous_status,
                        "new_status": event.new_status,
                        "note": event.note,
                        "occurred_at": event.occurred_at.isoformat(),
                    }
                    for event in detail.events
                ],
            },
        )

    @staticmethod
    def _application_payload(application, job) -> dict[str, Any]:
        return {
            "application_id": application.id,
            "job_posting_id": application.job_posting_id,
            "resume_version_id": application.resume_version_id,
            "jd_snapshot_id": application.jd_snapshot_id,
            "title": job.posting.title,
            "company_name": job.posting.company_name,
            "status": application.status,
            "submitted_at": application.submitted_at.isoformat(),
            "created_at": application.created_at.isoformat(),
            "updated_at": application.updated_at.isoformat(),
        }

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
                "change_reviews": [
                    review.model_dump(mode="json") for review in draft.change_reviews
                ],
                "pending_change_indices": draft.pending_change_indices,
                "strategy_summary": draft.result.strategy_summary,
                "changes": [
                    {
                        "change_index": index,
                        **change.model_dump(mode="json"),
                    }
                    for index, change in enumerate(draft.result.changes, start=1)
                ],
                "preserved_strengths": draft.result.preserved_strengths,
                "unresolved_gaps": draft.result.unresolved_gaps,
                "clarification_questions": draft.result.clarification_questions,
                "warnings": draft.result.warnings,
            },
        )
