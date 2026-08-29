from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from typing import Any, Literal

from career_agent.agent.job_discovery_gateway import JobDiscoveryGateway, JobDiscoveryGatewayResult
from career_agent.agent.mock_interview_contracts import (
    MockInterviewGraphResult,
    MockInterviewStartRequest,
)
from career_agent.agent.mock_interview_graph import (
    MockInterviewCheckpointMissingError,
    MockInterviewGraph,
    MockInterviewGraphVersionError,
)
from career_agent.agent.main_agent_contracts import (
    AnalyzeResumeToolArguments,
    ConfirmResumeAnalysisToolArguments,
    CompleteInterviewToolArguments,
    PrepareInterviewToolArguments,
    GetInterviewPreparationToolArguments,
    CreateInterviewToolArguments,
    CreateApplicationToolArguments,
    DraftResumeTailoringToolArguments,
    ExportResumeArtifactToolArguments,
    FindSavedJobsToolArguments,
    FinalizeResumeTailoringToolArguments,
    GetApplicationToolArguments,
    GetDailyBriefToolArguments,
    GetMockInterviewResultToolArguments,
    RestartMockInterviewToolArguments,
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
    ListActionItemsToolArguments,
    ListCalendarAccountsToolArguments,
    ListCalendarLinksToolArguments,
    ListTargetRolesToolArguments,
    ListEmailEventsToolArguments,
    ListInterviewsToolArguments,
    MatchResumeToJobToolArguments,
    ReviewResumeTailoringToolArguments,
    ReviseResumeTailoringToolArguments,
    UpdateApplicationStatusToolArguments,
    ResolveEmailEventToolArguments,
    ResolveActionItemToolArguments,
    SnoozeActionItemToolArguments,
    StartMockInterviewToolArguments,
    StartMockInterviewWorkflowInput,
    PrepareInterviewCalendarSyncToolArguments,
    GetCalendarProposalToolArguments,
    ExecuteCalendarProposalToolArguments,
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
from career_agent.services.action_center import (
    ActionCenterService,
    ActionItemNotFoundError,
    InvalidActionTransitionError,
)
from career_agent.connectors.calendar import CalendarConnectorError
from career_agent.services.calendar import (
    CalendarAccountNotFoundError,
    CalendarProposalConflictError,
    CalendarProposalNotFoundError,
    CalendarService,
    CalendarSyncNotAvailableError,
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
from career_agent.services.interview_preparation import (
    InterviewPreparationInputNotFoundError,
    InterviewPreparationNotAvailableError,
    InterviewPreparationService,
)
from career_agent.services.resume_job_match import (
    ResumeJobMatchInputNotFoundError,
    ResumeJobMatchService,
)
from career_agent.services.resume_export import (
    ResumeExportNotFoundError,
    ResumeExportService,
)
from career_agent.domain.mock_interviews import (
    MockInterviewReport,
    MockInterviewSession,
    MockInterviewTurn,
)
from career_agent.domain.resume import ResumeArtifactDelivery
from career_agent.services.resume_tailoring import (
    ResumeFinalReviewBlockedError,
    ResumeTailoringAlreadyFinalizedError,
    ResumeTailoringDraftNotFoundError,
    ResumeTailoringNotReadyError,
    ResumeTailoringReviewBlockedError,
    ResumeTailoringService,
    ResumeTailoringSupersededError,
)
from career_agent.storage.jobs import JobPostingRepository
from career_agent.storage.mock_interviews import SQLiteMockInterviewStore
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
        interview_preparation_service: InterviewPreparationService | None = None,
        action_center_service: ActionCenterService | None = None,
        calendar_service: CalendarService | None = None,
        mock_interview_graph: MockInterviewGraph | None = None,
        mock_interview_store: SQLiteMockInterviewStore | None = None,
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
        self._interview_preparation_service = interview_preparation_service
        self._action_center_service = action_center_service
        self._calendar_service = calendar_service
        self._mock_interview_graph = mock_interview_graph
        self._mock_interview_store = mock_interview_store
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
                    "revise_resume_tailoring": self._revise_resume_tailoring,
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
        if interview_preparation_service is not None:
            self._atomic_handlers.update(
                {
                    "prepare_interview": self._prepare_interview,
                    "get_interview_preparation": self._get_interview_preparation,
                }
            )
        if mock_interview_graph is not None and application_service is not None:
            self._workflow_handlers["start_mock_interview"] = (
                self._start_mock_interview
            )
        if mock_interview_store is not None:
            self._atomic_handlers["get_mock_interview_result"] = (
                self._get_mock_interview_result
            )
        if mock_interview_graph is not None and mock_interview_store is not None:
            # A workflow, not an atomic tool: it starts a run that then owns
            # subsequent turns, exactly as start_mock_interview does.
            self._workflow_handlers["restart_mock_interview"] = (
                self._restart_mock_interview
            )
        if action_center_service is not None:
            self._atomic_handlers.update(
                {
                    "get_daily_brief": self._get_daily_brief,
                    "list_action_items": self._list_action_items,
                    "complete_action_item": self._complete_action_item,
                    "dismiss_action_item": self._dismiss_action_item,
                    "snooze_action_item": self._snooze_action_item,
                }
            )
        if calendar_service is not None:
            self._atomic_handlers.update(
                {
                    "list_calendar_accounts": self._list_calendar_accounts,
                    "list_calendar_links": self._list_calendar_links,
                    "prepare_interview_calendar_sync": self._prepare_interview_calendar_sync,
                    "get_calendar_proposal": self._get_calendar_proposal,
                    "execute_calendar_proposal": self._execute_calendar_proposal,
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
                            "description": "Search only the current user's previously saved or viewed jobs. Use this for historical recall, not for discovering new online jobs. Results become numbered saved-job candidates; complete JD text stays outside the decision context.",
                            "parameters": FindSavedJobsToolArguments.model_json_schema(),
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "get_saved_job",
                            "description": "Read the selected or active saved job. Pass selection_index after find_saved_jobs, or omit it to use the active job. The complete JD is delivered outside the decision context.",
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
                            "description": "List the current user's resume target-role categories as numbered candidates. Never returns resume document content.",
                            "parameters": ListTargetRolesToolArguments.model_json_schema(),
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "list_resumes",
                            "description": "List the current user's resume families, optionally filtered with target_role_selection_index from list_target_roles. Returns numbered safe metadata only; never returns resume document content.",
                            "parameters": ListResumesToolArguments.model_json_schema(),
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "get_resume_metadata",
                            "description": "Read a resume family selected by selection_index and list its immutable versions as numbered metadata. Never returns PDF, text, Markdown, extracted content, or file paths.",
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
                            "description": "Analyze a selected resume version, or the active latest version when selection_index is omitted. Use when the user asks to read, extract, review, or analyze resume content. Structured candidates are delivered outside the decision context and are not career facts until explicitly confirmed.",
                            "parameters": AnalyzeResumeToolArguments.model_json_schema(),
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "get_resume_analysis",
                            "description": "Retrieve the active unexpired resume analysis draft so its candidates can be reviewed before confirmation. Never returns the original resume file.",
                            "parameters": GetResumeAnalysisToolArguments.model_json_schema(),
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "confirm_resume_analysis",
                            "description": "Confirm all candidates in the active resume analysis and persist them as CareerRecord and confirmed CareerEvidence. Call only after explicit user confirmation; never infer confirmation.",
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
                            "description": "Compare an exact current-user resume version with one saved job's complete JD. Use resume_version_selection_index and job_selection_index to choose directly from existing candidates, or omit either selector to use its active object. Returns a grounded assessment outside the decision context; does not search online and never returns either original document.",
                            "parameters": MatchResumeToJobToolArguments.model_json_schema(),
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "get_resume_job_match",
                            "description": "Retrieve the current conversation's active persisted resume-job match. Returns only the structured assessment, never the original resume or complete JD.",
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
                            "description": "Create a reviewable tailoring draft from the active persisted resume-job match. May accept a user tailoring goal. Grounded proposed changes are delivered outside decision context; this does not alter or create a resume version.",
                            "parameters": DraftResumeTailoringToolArguments.model_json_schema(),
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "get_resume_tailoring_draft",
                            "description": "Retrieve the active unexpired tailoring draft. Returns proposed changes for review outside decision context; it does not apply them.",
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
                            "name": "revise_resume_tailoring",
                            "description": "Create a new child draft from the active tailoring draft using explicit user feedback. The new draft discards all prior accept/reject decisions, reruns the bounded Writer/Reviewer loop, and must be reviewed again. It never overwrites the parent draft or creates a ResumeVersion.",
                            "parameters": ReviseResumeTailoringToolArguments.model_json_schema(),
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
                        "description": "Prepare the active owned immutable resume version for download and return only an opaque artifact reference plus safe file metadata. Call only when the user asks to download, export, or receive the resume file. Never place file content or a local path in the conversation.",
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
                            "description": "Track a real externally submitted application using one exact owned resume version and the saved job's current immutable JD snapshot. Use job_selection_index or resume_version_selection_index to override the active objects. Call only after the user explicitly reports that they actually applied; planning or preparing is not sufficient. Repeated calls for the same job return the original application.",
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
                            "description": "Read a tracked application selected by selection_index, or use the active application, including its append-only event timeline.",
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
        if self._interview_preparation_service is not None:
            schemas.extend(
                [
                    {
                        "type": "function",
                        "function": {
                            "name": "prepare_interview",
                            "description": "Generate or reuse a grounded preparation guide for one real upcoming interview from its exact JD snapshot, submitted resume version, confirmed evidence, and logistics. This is preparation, not a mock interview and not employer inside information.",
                            "parameters": PrepareInterviewToolArguments.model_json_schema(),
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "get_interview_preparation",
                            "description": "Read one persisted interview preparation result without re-reading full source documents.",
                            "parameters": GetInterviewPreparationToolArguments.model_json_schema(),
                        },
                    },
                ]
            )
        if self._mock_interview_graph is not None and self._application_service is not None:
            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": "start_mock_interview",
                        "description": (
                            "Start one stateful mock interview for the active or numbered "
                            "application, using its exact submitted resume and immutable JD. "
                            "Optionally bind a numbered real interview appointment for context. "
                            "After the first question, user answers are routed directly to the "
                            "active mock-interview workflow; do not call this tool again to answer."
                        ),
                        "parameters": StartMockInterviewToolArguments.model_json_schema(),
                    },
                }
            )
        if self._mock_interview_graph is not None and self._mock_interview_store is not None:
            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": "restart_mock_interview",
                        "description": (
                            "Retire a mock interview that reported "
                            "mock_interview_checkpoint_missing or "
                            "mock_interview_graph_incompatible and start a replacement "
                            "against the same application, resume version, and JD. Call "
                            "this only after the user agrees to abandon the stuck run: "
                            "its answers stay readable but it can never be finished. "
                            "Takes no arguments."
                        ),
                        "parameters": RestartMockInterviewToolArguments.model_json_schema(),
                    },
                }
            )
        if self._mock_interview_store is not None:
            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": "get_mock_interview_result",
                        "description": (
                            "Read back the latest finished mock interview for the active or "
                            "numbered application. Without a question number this lists the "
                            "questions with their ratings; with one it returns that question, "
                            "the user's full answer, its evaluation, and any follow-ups. Use "
                            "this whenever the user asks about a past mock interview, since "
                            "the conversation only keeps a condensed summary of the report."
                        ),
                        "parameters": GetMockInterviewResultToolArguments.model_json_schema(),
                    },
                }
            )
        if self._action_center_service is not None:
            schemas.extend(
                [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_daily_brief",
                            "description": "Generate the current user's source-grounded daily career brief from applications, recruiting email events, and real interviews. The report is computed on demand and is not stored as stale narrative memory.",
                            "parameters": GetDailyBriefToolArguments.model_json_schema(),
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "list_action_items",
                            "description": "Refresh and list persisted career action items such as follow-ups, pending email confirmations, interview preparation, reminders, material requests, and retrospectives.",
                            "parameters": ListActionItemsToolArguments.model_json_schema(),
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "complete_action_item",
                            "description": "Mark one action item completed only after the user explicitly reports completing it.",
                            "parameters": ResolveActionItemToolArguments.model_json_schema(),
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "dismiss_action_item",
                            "description": "Dismiss one action item only after the user explicitly says it is not applicable or should be ignored.",
                            "parameters": ResolveActionItemToolArguments.model_json_schema(),
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "snooze_action_item",
                            "description": "Snooze one action item until an explicit future timestamp requested by the user.",
                            "parameters": SnoozeActionItemToolArguments.model_json_schema(),
                        },
                    },
                ]
            )
        if self._calendar_service is not None:
            schemas.extend(
                [
                    {
                        "type": "function",
                        "function": {
                            "name": "list_calendar_accounts",
                            "description": "List safe Google Calendar account metadata. Credentials are never returned.",
                            "parameters": ListCalendarAccountsToolArguments.model_json_schema(),
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "list_calendar_links",
                            "description": "List the user's interview-to-calendar synchronization links and current sync status.",
                            "parameters": ListCalendarLinksToolArguments.model_json_schema(),
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "prepare_interview_calendar_sync",
                            "description": "Prepare a fixed create, update, or cancel preview for one real InterviewRound. This does not write to an external calendar and must be shown to the user for approval.",
                            "parameters": PrepareInterviewCalendarSyncToolArguments.model_json_schema(),
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "get_calendar_proposal",
                            "description": "Read one pending or historical fixed calendar-change proposal without executing it.",
                            "parameters": GetCalendarProposalToolArguments.model_json_schema(),
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "execute_calendar_proposal",
                            "description": "Execute exactly one unchanged, unexpired calendar proposal only after the user explicitly approves that displayed proposal. This is an external write.",
                            "parameters": ExecuteCalendarProposalToolArguments.model_json_schema(),
                        },
                    },
                ]
            )
        return tuple(self._decision_tool_schema(schema) for schema in schemas)

    @staticmethod
    def _decision_tool_schema(schema: dict[str, Any]) -> dict[str, Any]:
        """Remove internal identifiers from the model-callable contract.

        Handlers still receive injected identifiers after argument projection;
        the decision model can only choose indexes already present in its task
        context or rely on the active object.
        """

        projected = deepcopy(schema)

        def scrub(node: object) -> None:
            if isinstance(node, dict):
                properties = node.get("properties")
                if isinstance(properties, dict):
                    for key in tuple(properties):
                        if key.endswith("_id"):
                            del properties[key]
                required = node.get("required")
                if isinstance(required, list):
                    node["required"] = [
                        key for key in required if not str(key).endswith("_id")
                    ]
                for value in node.values():
                    scrub(value)
            elif isinstance(node, list):
                for value in node:
                    scrub(value)

        scrub(projected["function"]["parameters"])
        return projected

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

    def _start_mock_interview(
        self, arguments: dict[str, Any]
    ) -> ToolObservation:
        if self._mock_interview_graph is None or self._application_service is None:
            raise ValueError("Mock interview workflow is not configured")
        workflow_input = StartMockInterviewWorkflowInput.model_validate(arguments)
        try:
            detail = self._application_service.get_application(
                user_id=workflow_input.user_id,
                application_id=workflow_input.application_id,
            )
            application = detail.application
            if workflow_input.interview_round_id is not None:
                if self._interview_service is None:
                    raise ValueError("Interview service is not configured")
                interview = self._interview_service.get_interview(
                    user_id=workflow_input.user_id,
                    interview_round_id=workflow_input.interview_round_id,
                ).interview
                if interview.application_id != application.id:
                    raise InterviewApplicationConflictError(
                        "mock interview appointment belongs to another application"
                    )
            result = self._mock_interview_graph.start(
                MockInterviewStartRequest(
                    user_id=workflow_input.user_id,
                    application_id=application.id,
                    interview_round_id=workflow_input.interview_round_id,
                    job_posting_id=application.job_posting_id,
                    jd_snapshot_id=application.jd_snapshot_id,
                    resume_version_id=application.resume_version_id,
                    interview_type=workflow_input.interview_type,
                    max_primary_questions=workflow_input.max_primary_questions,
                    max_follow_ups_per_question=(
                        workflow_input.max_follow_ups_per_question
                    ),
                )
            )
        except (ApplicationInputNotFoundError, AgentWorkerError, ValueError) as error:
            return ToolObservation(
                tool_name="start_mock_interview",
                state="failed",
                message="模拟面试暂时无法启动；请确认投递记录和对应材料仍然可用。",
                payload={
                    "error_code": (
                        error.code
                        if isinstance(error, AgentWorkerError)
                        else type(error).__name__
                    ),
                    "retryable": (
                        error.retryable
                        if isinstance(error, AgentWorkerError)
                        else False
                    ),
                },
            )
        return self._mock_interview_observation(result)

    def resume_mock_interview(
        self, *, user_id: str, session_id: str, answer: str
    ) -> ToolObservation:
        """Resume the active workflow through an internal, non-model-facing path."""
        if self._mock_interview_graph is None:
            raise ValueError("Mock interview workflow is not configured")
        return self._drive_mock_interview(
            session_id=session_id,
            drive=lambda graph: graph.resume(
                user_id=user_id,
                session_id=session_id,
                answer=answer,
            ),
        )

    def retry_mock_interview(
        self, *, user_id: str, session_id: str
    ) -> ToolObservation:
        """Re-drive a failed step from the answer the store already holds.

        The failure happened after the answer was persisted, so asking the
        candidate to retype it verbatim is both hostile and fragile: any
        rewording is rejected as a conflicting answer for the same turn. This
        path takes no answer argument at all, which is what makes recovery
        independent of what the candidate can remember.
        """
        if self._mock_interview_graph is None:
            raise ValueError("Mock interview workflow is not configured")
        return self._drive_mock_interview(
            session_id=session_id,
            drive=lambda graph: graph.retry(
                user_id=user_id,
                session_id=session_id,
            ),
        )

    def _drive_mock_interview(
        self,
        *,
        session_id: str,
        drive: Callable[[Any], MockInterviewGraphResult],
    ) -> ToolObservation:
        """Run one graph advance and map its failures to a closed observation."""
        try:
            result = drive(self._mock_interview_graph)
        except MockInterviewCheckpointMissingError:
            return ToolObservation(
                tool_name="start_mock_interview",
                state="mock_interview_checkpoint_missing",
                message=(
                    "模拟面试的业务记录仍在，但执行断点已经丢失，当前会话无法继续。"
                ),
                next_action="restart_mock_interview",
                payload={"session_id": session_id},
            )
        except MockInterviewGraphVersionError:
            return ToolObservation(
                tool_name="start_mock_interview",
                state="mock_interview_graph_incompatible",
                message=(
                    "这次模拟面试由不兼容的旧版流程创建，不能用当前版本安全恢复。"
                ),
                next_action="restart_mock_interview",
                payload={"session_id": session_id},
            )
        except (AgentWorkerError, ValueError) as error:
            return ToolObservation(
                tool_name="start_mock_interview",
                state="failed",
                message=(
                    "这次模拟面试回答暂时无法处理。你的回答已经保存，"
                    "下一条消息会从保存的回答继续，不需要重新输入。"
                ),
                payload={
                    "session_id": session_id,
                    "error_code": (
                        error.code
                        if isinstance(error, AgentWorkerError)
                        else type(error).__name__
                    ),
                    "retryable": (
                        error.retryable
                        if isinstance(error, AgentWorkerError)
                        else False
                    ),
                },
            )
        return self._mock_interview_observation(result)

    @staticmethod
    def _mock_interview_observation(
        result: MockInterviewGraphResult,
    ) -> ToolObservation:
        state = {
            "awaiting_answer": "mock_interview_answer_required",
            "running": "mock_interview_running",
            "completed": "mock_interview_completed",
            "cancelled": "mock_interview_cancelled",
        }[result.state]
        message = result.message
        if result.state == "awaiting_answer" and result.question is not None:
            blocks = []
            if result.evaluation is not None:
                blocks.append(
                    "上一题反馈：\n"
                    f"{result.evaluation.summary}\n"
                    f"下一步原因：{result.evaluation.next_action_reason}"
                )
            blocks.append(f"模拟面试题：\n{result.question}")
            message = "\n\n".join(blocks)
        elif result.state == "completed" and result.report is not None:
            report = result.report
            strengths = "\n".join(f"- {item}" for item in report.strengths) or "- 暂无"
            development = (
                "\n".join(f"- {item}" for item in report.development_areas)
                or "- 暂无"
            )
            actions = (
                "\n".join(f"- {item}" for item in report.practice_actions)
                or "- 暂无"
            )
            message = (
                f"模拟面试完成。\n\n总结\n{report.summary}\n\n"
                f"表现亮点\n{strengths}\n\n待提升\n{development}\n\n"
                f"练习建议\n{actions}"
            )
        return ToolObservation(
            tool_name="start_mock_interview",
            state=state,
            message=message,
            next_action=(
                "answer_mock_interview_question"
                if result.state == "awaiting_answer"
                else None
            ),
            payload=result.model_dump(mode="json"),
        )

    def _restart_mock_interview(self, arguments: dict[str, Any]) -> ToolObservation:
        """Retire a run that cannot continue and start a fresh one in its place.

        Only reachable for the two phases that hold the slot without being able
        to advance. The store allows one unfinished run per user, so without
        retiring the stuck one first a new interview cannot be created at all.
        The replacement reuses the stuck run's own application, resume version,
        and JD snapshot, so a restart cannot silently change what is being
        practised against.
        """
        if self._mock_interview_graph is None:
            raise ValueError("Mock interview workflow is not configured")
        if self._mock_interview_store is None:
            raise ValueError("Mock interview store is not configured")
        user_id = str(arguments["user_id"])
        stuck = self._mock_interview_store.find_resumable(user_id=user_id)
        if stuck is None:
            return ToolObservation(
                tool_name="restart_mock_interview",
                state="no_mock_interview_to_restart",
                message="没有卡住的模拟面试，直接开始一场新的即可。",
            )
        # Cancel through the graph, not the store: the graph also deletes the
        # checkpoint thread, and for a version-incompatible run that thread is
        # the only thing still holding the old graph's state.
        self._mock_interview_graph.cancel(user_id=user_id, session_id=stuck.id)
        try:
            result = self._mock_interview_graph.start(
                MockInterviewStartRequest(
                    user_id=user_id,
                    application_id=stuck.application_id,
                    interview_round_id=stuck.interview_round_id,
                    job_posting_id=stuck.job_posting_id,
                    jd_snapshot_id=stuck.jd_snapshot_id,
                    resume_version_id=stuck.resume_version_id,
                    interview_type=stuck.interview_type,
                    max_primary_questions=stuck.max_primary_questions,
                    max_follow_ups_per_question=stuck.max_follow_ups_per_question,
                )
            )
        except (AgentWorkerError, ValueError) as error:
            # The old run is already terminal and graph.start cancels its own
            # partially-created session on failure. There is therefore nothing
            # resumable to retain in the conversation task: this is not the same
            # failure as an evaluate/report step whose answer is durable.
            return ToolObservation(
                tool_name="restart_mock_interview",
                state="mock_interview_restart_failed",
                message=(
                    "旧的模拟面试已经安全结束，但替代面试暂时启动失败。"
                    "你可以稍后重新开始一场模拟面试。"
                ),
                next_action="start_mock_interview",
                payload={
                    "error_code": (
                        error.code
                        if isinstance(error, AgentWorkerError)
                        else type(error).__name__
                    ),
                    "retryable": (
                        error.retryable
                        if isinstance(error, AgentWorkerError)
                        else False
                    ),
                },
            )
        return self._mock_interview_observation(result)

    def _get_mock_interview_result(self, arguments: dict[str, Any]) -> ToolObservation:
        """Read back a finished run the conversation only holds in condensed form.

        The run's own exchanges never enter the conversation, so without this
        the full questions and answers are unreachable once the run ends.
        """
        if self._mock_interview_store is None:
            raise ValueError("Mock interview store is not configured")
        user_id = str(arguments["user_id"])
        application_id = str(arguments["application_id"])
        question_number = arguments.get("question_number")
        # Cancelled runs keep every turn they got through, so they are readable
        # too; only the report is missing. Runs still in progress are excluded
        # because the workflow, not this tool, owns a turn while it is driving.
        sessions = self._mock_interview_store.list_sessions(
            user_id=user_id,
            application_id=application_id,
            statuses=("completed", "cancelled"),
            limit=1,
        )
        if not sessions:
            return ToolObservation(
                tool_name="get_mock_interview_result",
                state="no_mock_interview_result_found",
                message="这个投递还没有结束过的模拟面试。",
            )
        session = sessions[0]
        report = self._mock_interview_store.get_report(
            user_id=user_id, session_id=session.id
        )
        turns = self._mock_interview_store.list_turns(
            user_id=user_id, session_id=session.id
        )
        if question_number is not None:
            return self._mock_interview_question_observation(
                turns=turns, question_number=int(question_number)
            )
        return self._mock_interview_result_observation(
            session=session, report=report, turns=turns
        )

    @staticmethod
    def _mock_interview_question_observation(
        *, turns: tuple[MockInterviewTurn, ...], question_number: int
    ) -> ToolObservation:
        """Return one exchange in full, follow-ups included."""
        matching = tuple(
            turn for turn in turns if turn.plan_item_number == question_number
        )
        if not matching:
            return ToolObservation(
                tool_name="get_mock_interview_result",
                state="no_mock_interview_result_found",
                message=f"这次模拟面试没有第 {question_number} 题。",
            )
        blocks = []
        for turn in matching:
            label = "追问" if turn.turn_type == "follow_up" else "主问题"
            lines = [f"{label}\n{turn.question}"]
            if turn.answer is not None:
                lines.append(f"你的回答\n{turn.answer}")
            if turn.evaluation is not None:
                lines.append(
                    f"评价（{turn.evaluation.rating}）\n{turn.evaluation.summary}"
                )
            blocks.append("\n\n".join(lines))
        return ToolObservation(
            tool_name="get_mock_interview_result",
            state="mock_interview_result_found",
            message=f"第 {question_number} 题：\n\n" + "\n\n---\n\n".join(blocks),
        )

    @staticmethod
    def _mock_interview_result_observation(
        *,
        session: MockInterviewSession,
        report: MockInterviewReport | None,
        turns: tuple[MockInterviewTurn, ...],
    ) -> ToolObservation:
        """List the run's questions with ratings, without their full text.

        An index rather than a transcript: the model can name a question number
        to read that exchange in full, so a long run costs one short message
        instead of every answer at once.
        """
        primary = tuple(turn for turn in turns if turn.turn_type == "primary")
        lines = []
        for turn in primary:
            if turn.evaluation is not None:
                rating = turn.evaluation.rating
            elif turn.answer is None:
                # Asked and abandoned, which is not the same as answered but
                # unscored: there is nothing here to go back and read.
                rating = "未回答"
            else:
                rating = "未评价"
            follow_ups = sum(
                1
                for candidate in turns
                if candidate.turn_type == "follow_up"
                and candidate.plan_item_number == turn.plan_item_number
            )
            suffix = f"，追问 {follow_ups} 次" if follow_ups else ""
            lines.append(
                f"{turn.plan_item_number}. [{rating}{suffix}] "
                f"{turn.question[:60]}"
            )
        # Asked and answered are different numbers once a run can stop early: a
        # question the user never answered is still a row here. Reporting only
        # the row count would present an abandoned question as an attempted one.
        answered = sum(1 for turn in primary if turn.answer is not None)
        header = f"模拟面试（{session.interview_type}，{len(primary)} 题"
        if answered < len(primary):
            header += f"，答了 {answered} 题"
        if session.status == "cancelled":
            header += "，中途取消"
        blocks = [
            header + "）",
            "题目\n" + ("\n".join(lines) if lines else "暂无"),
        ]
        if report is not None:
            blocks.append(f"总结\n{report.summary}")
        blocks.append("要看某题的完整问答，说题号。")
        return ToolObservation(
            tool_name="get_mock_interview_result",
            state="mock_interview_result_found",
            message="\n\n".join(blocks),
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

    def _prepare_interview(self, arguments: dict[str, Any]) -> ToolObservation:
        if self._interview_preparation_service is None:
            raise ValueError("Interview preparation service is not configured")
        user_id = str(arguments["user_id"])
        model_arguments = PrepareInterviewToolArguments.model_validate(
            {key: value for key, value in arguments.items() if key != "user_id"}
        )
        if model_arguments.interview_round_id is None:
            raise ValueError("prepare_interview requires interview_round_id")
        try:
            preparation = self._interview_preparation_service.prepare(
                user_id=user_id,
                interview_round_id=model_arguments.interview_round_id,
            )
        except InterviewPreparationInputNotFoundError as error:
            return ToolObservation(
                tool_name="prepare_interview",
                state="interview_preparation_input_not_found",
                message="无法读取这场面试对应的完整准备输入。",
                payload={"reason": str(error)},
            )
        except InterviewPreparationNotAvailableError as error:
            return ToolObservation(
                tool_name="prepare_interview",
                state="interview_preparation_not_available",
                message="当前面试状态不适合生成准备材料。",
                payload={"reason": str(error)},
            )
        return ToolObservation(
            tool_name="prepare_interview",
            state="interview_preparation_ready",
            message="面试准备材料已生成。",
            next_action="review_interview_preparation",
            payload=self._interview_preparation_payload(preparation),
        )

    def _get_interview_preparation(
        self, arguments: dict[str, Any]
    ) -> ToolObservation:
        if self._interview_preparation_service is None:
            raise ValueError("Interview preparation service is not configured")
        user_id = str(arguments["user_id"])
        model_arguments = GetInterviewPreparationToolArguments.model_validate(
            {key: value for key, value in arguments.items() if key != "user_id"}
        )
        if model_arguments.preparation_id is None:
            raise ValueError("get_interview_preparation requires preparation_id")
        try:
            preparation = self._interview_preparation_service.get(
                user_id=user_id,
                preparation_id=model_arguments.preparation_id,
            )
        except InterviewPreparationInputNotFoundError:
            return ToolObservation(
                tool_name="get_interview_preparation",
                state="interview_preparation_not_found",
                message="没有找到该面试准备结果，或它不属于当前用户。",
            )
        return ToolObservation(
            tool_name="get_interview_preparation",
            state="interview_preparation_ready",
            message="已读取面试准备材料。",
            payload=self._interview_preparation_payload(preparation),
        )

    def _get_daily_brief(self, arguments: dict[str, Any]) -> ToolObservation:
        if self._action_center_service is None:
            raise ValueError("Action Center service is not configured")
        user_id = str(arguments["user_id"])
        model_arguments = GetDailyBriefToolArguments.model_validate(
            {key: value for key, value in arguments.items() if key != "user_id"}
        )
        try:
            brief = self._action_center_service.daily_brief(
                user_id=user_id,
                timezone_name=model_arguments.timezone,
            )
        except ValueError:
            return ToolObservation(
                tool_name="get_daily_brief",
                state="invalid_timezone",
                message="无法识别该时区，请提供 IANA 时区名称，例如 Asia/Shanghai。",
            )
        sections = {
            "overdue": brief.overdue,
            "due_today": brief.due_today,
            "upcoming": brief.upcoming,
            "no_due_date": brief.no_due_date,
        }
        count = sum(len(items) for items in sections.values())
        return ToolObservation(
            tool_name="get_daily_brief",
            state="daily_brief_ready",
            message=f"今日职业简报包含 {count} 个待办事项。" if count else "今日没有待办事项。",
            payload={
                "timezone": brief.timezone,
                "generated_at": brief.generated_at.isoformat(),
                **{
                    name: [self._action_payload(item) for item in items]
                    for name, items in sections.items()
                },
            },
        )

    def _list_action_items(self, arguments: dict[str, Any]) -> ToolObservation:
        if self._action_center_service is None:
            raise ValueError("Action Center service is not configured")
        user_id = str(arguments["user_id"])
        model_arguments = ListActionItemsToolArguments.model_validate(
            {key: value for key, value in arguments.items() if key != "user_id"}
        )
        try:
            items = self._action_center_service.list_actions(
                user_id=user_id,
                statuses=model_arguments.statuses,
                limit=model_arguments.limit,
                timezone_name=model_arguments.timezone,
            )
        except ValueError:
            return ToolObservation(
                tool_name="list_action_items",
                state="invalid_timezone",
                message="无法识别该时区，请提供 IANA 时区名称，例如 Asia/Shanghai。",
            )
        return ToolObservation(
            tool_name="list_action_items",
            state="action_items_found" if items else "no_action_items_found",
            message=f"找到 {len(items)} 个行动事项。" if items else "当前没有匹配的行动事项。",
            payload={
                "items": [
                    {"selection_index": index, **self._action_payload(item)}
                    for index, item in enumerate(items, start=1)
                ]
            },
        )

    def _complete_action_item(self, arguments: dict[str, Any]) -> ToolObservation:
        return self._resolve_action_item(arguments, action="complete")

    def _dismiss_action_item(self, arguments: dict[str, Any]) -> ToolObservation:
        return self._resolve_action_item(arguments, action="dismiss")

    def _resolve_action_item(
        self, arguments: dict[str, Any], *, action: Literal["complete", "dismiss"]
    ) -> ToolObservation:
        if self._action_center_service is None:
            raise ValueError("Action Center service is not configured")
        user_id = str(arguments["user_id"])
        model_arguments = ResolveActionItemToolArguments.model_validate(
            {key: value for key, value in arguments.items() if key != "user_id"}
        )
        if model_arguments.action_item_id is None:
            raise ValueError(f"{action}_action_item requires action_item_id")
        try:
            item = (
                self._action_center_service.complete_action(
                    user_id=user_id,
                    action_item_id=model_arguments.action_item_id,
                )
                if action == "complete"
                else self._action_center_service.dismiss_action(
                    user_id=user_id,
                    action_item_id=model_arguments.action_item_id,
                )
            )
        except ActionItemNotFoundError:
            return ToolObservation(
                tool_name=f"{action}_action_item",
                state="action_item_not_found",
                message="没有找到这个行动事项，或它不属于当前用户。",
            )
        return ToolObservation(
            tool_name=f"{action}_action_item",
            state="action_item_resolved",
            message="行动事项已完成。" if action == "complete" else "行动事项已忽略。",
            payload=self._action_payload(item),
        )

    def _snooze_action_item(self, arguments: dict[str, Any]) -> ToolObservation:
        if self._action_center_service is None:
            raise ValueError("Action Center service is not configured")
        user_id = str(arguments["user_id"])
        model_arguments = SnoozeActionItemToolArguments.model_validate(
            {key: value for key, value in arguments.items() if key != "user_id"}
        )
        if model_arguments.action_item_id is None:
            raise ValueError("snooze_action_item requires action_item_id")
        try:
            item = self._action_center_service.snooze_action(
                user_id=user_id,
                action_item_id=model_arguments.action_item_id,
                snoozed_until=model_arguments.snoozed_until,
            )
        except ActionItemNotFoundError:
            return ToolObservation(
                tool_name="snooze_action_item",
                state="action_item_not_found",
                message="没有找到这个行动事项，或它不属于当前用户。",
            )
        except InvalidActionTransitionError as error:
            return ToolObservation(
                tool_name="snooze_action_item",
                state="invalid_action_transition",
                message="无法将该行动事项稍后提醒。",
                payload={"reason": str(error)},
            )
        return ToolObservation(
            tool_name="snooze_action_item",
            state="action_item_snoozed",
            message="行动事项已设置为稍后提醒。",
            payload=self._action_payload(item),
        )

    def _list_calendar_accounts(self, arguments: dict[str, Any]) -> ToolObservation:
        if self._calendar_service is None:
            raise ValueError("Calendar service is not configured")
        user_id = str(arguments["user_id"])
        ListCalendarAccountsToolArguments.model_validate(
            {key: value for key, value in arguments.items() if key != "user_id"}
        )
        accounts = self._calendar_service.list_accounts(user_id=user_id)
        return ToolObservation(
            tool_name="list_calendar_accounts",
            state="calendar_accounts_found" if accounts else "no_calendar_accounts",
            message=(
                f"找到 {len(accounts)} 个 Calendar 账户。"
                if accounts else "尚未配置 Calendar 账户。"
            ),
            payload={
                "items": [
                    {
                        "selection_index": index,
                        "calendar_account_id": account.id,
                        "provider": account.provider,
                        "email_address": account.email_address,
                        "calendar_id": account.calendar_id,
                        "status": account.status,
                    }
                    for index, account in enumerate(accounts, start=1)
                ]
            },
        )

    def _list_calendar_links(self, arguments: dict[str, Any]) -> ToolObservation:
        if self._calendar_service is None:
            raise ValueError("Calendar service is not configured")
        user_id = str(arguments["user_id"])
        ListCalendarLinksToolArguments.model_validate(
            {key: value for key, value in arguments.items() if key != "user_id"}
        )
        links = self._calendar_service.list_links(user_id=user_id)
        return ToolObservation(
            tool_name="list_calendar_links",
            state="calendar_links_found" if links else "no_calendar_links",
            message=(
                f"找到 {len(links)} 条面试 Calendar 同步记录。"
                if links else "当前没有面试 Calendar 同步记录。"
            ),
            payload={
                "items": [
                    {
                        "calendar_link_id": link.id,
                        "calendar_account_id": link.calendar_account_id,
                        "interview_round_id": link.interview_round_id,
                        "status": link.status,
                        "external_html_link": link.external_html_link,
                        "updated_at": link.updated_at.isoformat(),
                    }
                    for link in links
                ]
            },
        )

    def _prepare_interview_calendar_sync(
        self, arguments: dict[str, Any]
    ) -> ToolObservation:
        if self._calendar_service is None:
            raise ValueError("Calendar service is not configured")
        user_id = str(arguments["user_id"])
        model_arguments = PrepareInterviewCalendarSyncToolArguments.model_validate(
            {key: value for key, value in arguments.items() if key != "user_id"}
        )
        if model_arguments.interview_round_id is None:
            raise ValueError("prepare_interview_calendar_sync requires interview_round_id")
        try:
            proposal = self._calendar_service.prepare_interview_sync(
                user_id=user_id,
                interview_round_id=model_arguments.interview_round_id,
                calendar_account_id=model_arguments.calendar_account_id,
            )
        except CalendarAccountNotFoundError as error:
            return ToolObservation(
                tool_name="prepare_interview_calendar_sync",
                state="calendar_account_required",
                message="需要先配置或选择一个 Calendar 账户。",
                payload={"reason": str(error)},
            )
        except CalendarSyncNotAvailableError as error:
            return ToolObservation(
                tool_name="prepare_interview_calendar_sync",
                state="calendar_sync_not_available",
                message="当前面试没有需要执行的 Calendar 变更。",
                payload={"reason": str(error)},
            )
        return ToolObservation(
            tool_name="prepare_interview_calendar_sync",
            state="calendar_approval_required",
            message="Calendar 变更预览已生成；执行前需要用户明确确认。",
            next_action="ask_calendar_approval",
            payload=self._calendar_proposal_payload(proposal),
        )

    def _get_calendar_proposal(self, arguments: dict[str, Any]) -> ToolObservation:
        if self._calendar_service is None:
            raise ValueError("Calendar service is not configured")
        user_id = str(arguments["user_id"])
        model_arguments = GetCalendarProposalToolArguments.model_validate(
            {key: value for key, value in arguments.items() if key != "user_id"}
        )
        if model_arguments.proposal_id is None:
            raise ValueError("get_calendar_proposal requires proposal_id")
        try:
            proposal = self._calendar_service.get_proposal(
                user_id=user_id, proposal_id=model_arguments.proposal_id
            )
        except CalendarProposalNotFoundError:
            return ToolObservation(
                tool_name="get_calendar_proposal",
                state="calendar_proposal_not_found",
                message="没有找到该 Calendar 变更预览，或它不属于当前用户。",
            )
        return ToolObservation(
            tool_name="get_calendar_proposal",
            state="calendar_proposal_ready",
            message="已读取 Calendar 变更预览。",
            payload=self._calendar_proposal_payload(proposal),
        )

    def _execute_calendar_proposal(self, arguments: dict[str, Any]) -> ToolObservation:
        if self._calendar_service is None:
            raise ValueError("Calendar service is not configured")
        user_id = str(arguments["user_id"])
        model_arguments = ExecuteCalendarProposalToolArguments.model_validate(
            {key: value for key, value in arguments.items() if key != "user_id"}
        )
        if model_arguments.proposal_id is None:
            raise ValueError("execute_calendar_proposal requires proposal_id")
        try:
            execution = self._calendar_service.execute_proposal(
                user_id=user_id, proposal_id=model_arguments.proposal_id
            )
        except CalendarProposalNotFoundError:
            return ToolObservation(
                tool_name="execute_calendar_proposal",
                state="calendar_proposal_not_found",
                message="没有找到该 Calendar 变更预览，或它不属于当前用户。",
            )
        except CalendarProposalConflictError as error:
            return ToolObservation(
                tool_name="execute_calendar_proposal",
                state="calendar_approval_invalid",
                message="该 Calendar 批准已失效，没有执行外部写入。",
                payload={"reason": str(error)},
            )
        except CalendarConnectorError as error:
            return ToolObservation(
                tool_name="execute_calendar_proposal",
                state="calendar_write_failed",
                message="Calendar 外部写入没有获得成功确认。",
                payload={"error_code": error.code, "error_detail": str(error)},
            )
        return ToolObservation(
            tool_name="execute_calendar_proposal",
            state="calendar_sync_complete",
            message="Calendar 变更已执行并获得成功确认。",
            payload={
                **self._calendar_proposal_payload(execution.proposal),
                "calendar_link_id": execution.link.id,
                "calendar_link_status": execution.link.status,
                "external_html_link": execution.link.external_html_link,
            },
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

    @staticmethod
    def _action_payload(item) -> dict[str, Any]:
        return {
            "action_item_id": item.id,
            "action_type": item.action_type,
            "source_type": item.source_type,
            "source_id": item.source_id,
            "application_id": item.application_id,
            "title": item.title,
            "summary": item.summary,
            "due_at": item.due_at.isoformat() if item.due_at is not None else None,
            "status": item.status,
            "snoozed_until": (
                item.snoozed_until.isoformat()
                if item.snoozed_until is not None
                else None
            ),
        }

    @staticmethod
    def _calendar_proposal_payload(proposal) -> dict[str, Any]:
        return {
            "proposal_id": proposal.id,
            "calendar_account_id": proposal.calendar_account_id,
            "interview_round_id": proposal.interview_round_id,
            "operation": proposal.operation,
            "status": proposal.status,
            "payload_hash": proposal.payload_hash,
            "payload": (
                proposal.payload.model_dump(mode="json")
                if proposal.payload is not None
                else None
            ),
            "created_at": proposal.created_at.isoformat(),
            "expires_at": proposal.expires_at.isoformat(),
            "executed_at": (
                proposal.executed_at.isoformat()
                if proposal.executed_at is not None
                else None
            ),
        }

    @staticmethod
    def _interview_preparation_payload(preparation) -> dict[str, Any]:
        return {
            "preparation_id": preparation.id,
            "interview_round_id": preparation.interview_round_id,
            "application_id": preparation.application_id,
            "job_posting_id": preparation.job_posting_id,
            "jd_snapshot_id": preparation.jd_snapshot_id,
            "resume_version_id": preparation.resume_version_id,
            "created_at": preparation.created_at.isoformat(),
            "preparation": preparation.result.model_dump(mode="json"),
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
        except ResumeTailoringReviewBlockedError as error:
            return ToolObservation(
                tool_name="draft_resume_tailoring",
                state="resume_tailoring_review_blocked",
                message="自动审核未能产出安全的简历修改草稿，需要调整目标或人工确认。",
                payload={"reason": str(error)},
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
        except ResumeTailoringSupersededError:
            return ToolObservation(
                tool_name="review_resume_tailoring",
                state="resume_tailoring_superseded",
                message="该草稿已有更新版本，请审阅当前最新草稿。",
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

    def _revise_resume_tailoring(self, arguments: dict[str, Any]) -> ToolObservation:
        if self._resume_tailoring_service is None:
            raise ValueError("Resume tailoring service is not configured")
        user_id = str(arguments["user_id"])
        model_arguments = ReviseResumeTailoringToolArguments.model_validate(
            {key: value for key, value in arguments.items() if key != "user_id"}
        )
        if model_arguments.draft_id is None:
            raise ValueError("revise_resume_tailoring requires draft_id")
        try:
            draft = self._resume_tailoring_service.revise_draft(
                user_id=user_id,
                draft_id=model_arguments.draft_id,
                feedback=model_arguments.feedback,
            )
        except ResumeTailoringDraftNotFoundError:
            return ToolObservation(
                tool_name="revise_resume_tailoring",
                state="resume_tailoring_draft_not_found",
                message="没有找到要修改的简历草稿，或它已经过期。",
                payload={"draft_id": model_arguments.draft_id},
            )
        except ResumeTailoringAlreadyFinalizedError:
            return ToolObservation(
                tool_name="revise_resume_tailoring",
                state="resume_tailoring_already_finalized",
                message="该草稿已经生成简历版本；如需继续修改，应基于新版本重新匹配和定制。",
                payload={"draft_id": model_arguments.draft_id},
            )
        except ResumeTailoringSupersededError:
            return ToolObservation(
                tool_name="revise_resume_tailoring",
                state="resume_tailoring_superseded",
                message="该草稿已有更新版本，请基于当前最新草稿继续反馈。",
                payload={"draft_id": model_arguments.draft_id},
            )
        except ResumeTailoringReviewBlockedError as error:
            return ToolObservation(
                tool_name="revise_resume_tailoring",
                state="resume_tailoring_review_blocked",
                message="根据用户反馈生成的新草稿未通过自动审核，原草稿保持不变。",
                payload={"reason": str(error)},
            )
        except AgentWorkerError as error:
            return ToolObservation(
                tool_name="revise_resume_tailoring",
                state="failed",
                message="重新生成简历草稿暂时失败，请稍后重试。" if error.retryable else "重新生成简历草稿失败。",
                payload={"error_code": error.code, "retryable": error.retryable},
            )
        return self._tailoring_observation(
            tool_name="revise_resume_tailoring",
            draft=draft,
            message=(
                f"已根据反馈生成第 {draft.revision_number} 版草稿；"
                "旧审批决定未继承，请重新逐条审阅。"
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
        except ResumeFinalReviewBlockedError as error:
            return ToolObservation(
                tool_name="finalize_resume_tailoring",
                state="resume_final_review_blocked",
                message="最终审核发现未获批准或不安全的实质变化，因此没有创建新简历版本。",
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
                "parent_draft_id": draft.parent_draft_id,
                "revision_number": draft.revision_number,
                "status": draft.status,
                "tailoring_goal": draft.tailoring_goal,
                "expires_at": draft.expires_at.isoformat(),
                "automated_review": (
                    {
                        "status": draft.automated_review.status,
                        "attempt_count": len(draft.automated_review.attempts),
                        "warning_categories": [
                            issue.category
                            for issue in draft.automated_review.attempts[-1].result.issues
                            if issue.severity == "warning"
                        ],
                    }
                    if draft.automated_review is not None
                    else None
                ),
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
