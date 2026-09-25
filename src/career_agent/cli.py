from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence, TextIO

from career_agent.agent.context_deployment_config import (
    ContextDeploymentConfig,
    ConversationSummaryAgentConfig,
    validate_model_window,
)
from career_agent.agent.context_manager import ContextManager
from career_agent.agent.career_context import CareerContextProjector
from career_agent.agent.semantic_career_retrieval import optional_semantic_retriever
from career_agent.agent.main_agent_contracts import (
    ToolObservation,
    canonical_confirm_before,
)
from career_agent.agent.main_agent_runtime import MainAgentRuntime, ReplayedTurn
from career_agent.agent.main_agent_tools import MainAgentToolRegistry
from career_agent.harness.streaming import (
    InteractionRequiredEvent,
    InteractionResponse,
    capability_confirmation_event,
    resume_analysis_confirmation_event,
    scoped_interaction_message,
)
from career_agent.agent.mock_interview_graph import (
    MockInterviewGraph,
    StoredMockInterviewSourceProvider,
)
from career_agent.agent.mock_interview_skill_loader import MockInterviewSkillLoader
from career_agent.agent.openai_compatible_client import AgentConfigurationError, AgentWorkerError, OpenAICompatibleAgentConfig
from career_agent.agent.job_research_config import job_research_config_from_env
from career_agent.agent.openai_compatible_main_agent import (
    OpenAICompatibleMainAgentDecisionMaker,
    max_output_tokens_from_env,
)
from career_agent.agent.openai_conversation_summary_worker import OpenAIConversationSummaryWorker
from career_agent.agent.openai_resume_analysis_worker import OpenAIResumeAnalysisWorker
from career_agent.agent.openai_job_analysis_worker import OpenAIJobAnalysisWorker
from career_agent.agent.openai_resume_job_match_worker import OpenAIResumeJobMatchWorker
from career_agent.agent.openai_resume_tailoring_reviewer import (
    OpenAIResumeTailoringReviewer,
)
from career_agent.agent.openai_interview_preparation_worker import OpenAIInterviewPreparationWorker
from career_agent.agent.openai_mock_interview_worker import OpenAIMockInterviewWorker
from career_agent.agent.openai_email_tracking_worker import OpenAIEmailTrackingWorker
from career_agent.agent.deepagent_resume_tailoring_worker import (
    DeepAgentResumeFinalizationWorker,
    DeepAgentResumeTailoringWorker,
)
from career_agent.agent.deepagent_job_research_worker import (
    DeepAgentJobResearchWorker,
)
from career_agent.connectors.email_accounts import EnvironmentEmailConnectorResolver
from career_agent.connectors.calendar import EnvironmentCalendarConnectorResolver
from career_agent.services.applications import ApplicationService
from career_agent.services.action_center import ActionCenterService
from career_agent.services.calendar import CalendarService
from career_agent.services.email_tracking import EmailTrackingService
from career_agent.services.interviews import InterviewService
from career_agent.services.interview_preparation import InterviewPreparationService
from career_agent.services.interview_context import InterviewPreparationContextFactory
from career_agent.services.job_research import JobResearchService
from career_agent.services.memory_report import build_memory_report
from career_agent.services.memory_review import MemoryReviewService
from career_agent.services.resume_analysis import ResumeAnalysisService
from career_agent.services.resume_export import ResumeExportService
from career_agent.services.job_analysis import JobAnalysisService
from career_agent.services.resume_job_match import ResumeJobMatchService
from career_agent.services.resume_import import (
    MAX_RESUME_IMPORT_BYTES,
    validate_resume_document,
)
from career_agent.services.resume_tailoring import ResumeTailoringService
from career_agent.services.episode_reconciliation import EpisodeReconciler
from career_agent.storage.api_keys import (
    DEFAULT_EXPIRY_DAYS,
    KNOWN_SCOPES,
    SQLiteApiKeyStore,
)
from career_agent.storage.context import CareerContextStore
from career_agent.storage.connector_secrets import KeyringConnectorSecretStore
from career_agent.storage.checkpoints import SQLiteCheckpointOwner
from career_agent.storage.applications import SQLiteApplicationStore
from career_agent.storage.action_center import SQLiteActionItemStore
from career_agent.storage.calendar import SQLiteCalendarStore
from career_agent.storage.email_tracking import SQLiteEmailTrackingStore
from career_agent.storage.episodes import DecayPolicy, SQLiteCareerEpisodeStore
from career_agent.storage.interviews import SQLiteInterviewStore
from career_agent.storage.interview_preparations import SQLiteInterviewPreparationStore
from career_agent.storage.job_research import SQLiteJobResearchStore
from career_agent.storage.mock_interviews import SQLiteMockInterviewStore
from career_agent.storage.career_history import CareerHistoryStore
from career_agent.storage.job_captures import SQLiteJobCaptureStore
from career_agent.storage.jobs import SQLiteJobPostingRepository, StoredJobRecord, StoredJobSummary
from career_agent.storage.resumes import ResumeStore
from career_agent.storage.run_events import SQLiteTraceRecorder
from career_agent.storage.capability_confirmations import (
    CapabilityConfirmationSettledError,
    SQLiteCapabilityConfirmationStore,
)
from career_agent.storage.turn_receipts import (
    SQLiteTurnReceiptStore,
)
from career_agent.storage.action_executions import (
    RESULT_STATE_RECEIPT_KEY,
    SQLiteActionExecutionStore,
)
from career_agent.storage.resume_analysis import SQLiteResumeAnalysisDraftStore
from career_agent.storage.resume_artifacts import SQLiteResumeArtifactStore
from career_agent.services.job_comparison import JobComparisonService
from career_agent.storage.resume_job_matches import SQLiteResumeJobMatchStore
from career_agent.storage.resume_tailoring import SQLiteResumeTailoringDraftStore
from career_agent.storage.working_notes import WorkingNotesStore
from career_agent.storage.backup import (
    INCONSISTENT_SNAPSHOT_WARNING,
    BackupError,
    BackupPlan,
    create_backup,
    restore_backup,
    verify_backup,
)
from career_agent.api.single_worker import SingleWorkerError, SingleWorkerLock, lock_path_for


EXIT_OK = 0
EXIT_ARGUMENT_ERROR = 2
EXIT_CONFIGURATION_ERROR = 3
EXIT_WORKFLOW_ERROR = 5
EXIT_UNKNOWN_ERROR = 6


def build_main_agent_runtime(args: argparse.Namespace) -> MainAgentRuntime:
    """Wire the production runtime. Called only by the workspace lock's holder.

    Both callers (``chat`` and the API lifespan) take ``api-server.lock`` before
    calling this, so at this point no other process can be executing a turn.
    That is what lets the turn receipts left ``RUNNING`` by a dead process be
    settled here: they would otherwise answer ``TURN_IN_PROGRESS`` forever.
    """

    main_config = replace(OpenAICompatibleAgentConfig.from_env(prefix="MAIN_AGENT"), timeout_seconds=args.main_agent_timeout_seconds)
    context_config = ContextDeploymentConfig.from_env()
    main_output_tokens = max_output_tokens_from_env()
    validate_model_window(
        input_tokens=main_config.max_input_tokens,
        output_tokens=main_output_tokens,
        context_window_tokens=context_config.main_context_window_tokens,
        prefix="MAIN_AGENT",
    )
    summary_config = ConversationSummaryAgentConfig.from_env(
        main_config=main_config,
        main_context_window_tokens=context_config.main_context_window_tokens,
    )
    context_store = CareerContextStore(Path(args.context_store).expanduser())
    turn_receipt_store = SQLiteTurnReceiptStore(Path(args.context_store).expanduser())
    turn_receipt_store.fail_orphaned_running()
    episode_store = SQLiteCareerEpisodeStore(
        Path(args.context_store).expanduser()
    )
    resume_store = ResumeStore(Path(args.resume_store).expanduser())
    working_notes_store = WorkingNotesStore(
        Path(args.context_store).expanduser().with_name("working-notes")
    )
    context_manager = ContextManager(
        context_store,
        summary_worker=OpenAIConversationSummaryWorker(
            summary_config.provider,
            max_output_tokens=summary_config.max_output_tokens,
            disable_thinking=summary_config.disable_thinking,
        ),
        recent_message_limit=context_config.recent_message_limit,
        summary_batch_size=context_config.summary_batch_size,
        compact_occupancy_threshold=context_config.compact_occupancy_threshold,
        target_role_source=resume_store,
        episode_store=episode_store,
        working_notes_store=working_notes_store,
    )
    resume_analysis_config = replace(
        OpenAICompatibleAgentConfig.from_env(prefix="RESUME_ANALYSIS_AGENT"),
        timeout_seconds=args.agent_timeout_seconds,
    )
    career_history_store = CareerHistoryStore(Path(args.resume_store).expanduser())
    try:
        semantic_retriever = optional_semantic_retriever(
            career_history=career_history_store,
            cache_path=Path(args.resume_store)
            .expanduser()
            .with_name("career_embeddings.sqlite3"),
        )
    except ValueError as error:
        raise AgentConfigurationError(
            "AGENT_CONFIGURATION_INVALID",
            str(error),
        ) from error
    job_repository = SQLiteJobPostingRepository(Path(args.job_store).expanduser())
    match_store = SQLiteResumeJobMatchStore(Path(args.resume_store).expanduser())
    application_store = SQLiteApplicationStore(
        Path(args.application_store).expanduser()
    )
    application_service = ApplicationService(
        application_store,
        job_repository,
        resume_store,
    )
    interview_store = SQLiteInterviewStore(
        Path(args.application_store).expanduser()
    )
    interview_service = InterviewService(
        interview_store,
        application_service,
    )
    connector_secrets = KeyringConnectorSecretStore()
    email_tracking_service = EmailTrackingService(
        SQLiteEmailTrackingStore(Path(args.email_store).expanduser()),
        application_service,
        EnvironmentEmailConnectorResolver(secret_store=connector_secrets),
        OpenAIEmailTrackingWorker(resume_analysis_config),
        interview_service,
    )
    action_center_service = ActionCenterService(
        SQLiteActionItemStore(Path(args.action_store).expanduser()),
        application_service,
        email_tracking_service,
        interview_service,
        job_repository=job_repository,
        resume_tailoring_drafts=SQLiteResumeTailoringDraftStore(
            Path(args.resume_store).expanduser()
        ),
    )
    calendar_service = CalendarService(
        SQLiteCalendarStore(Path(args.calendar_store).expanduser()),
        interview_service,
        application_service,
        EnvironmentCalendarConnectorResolver(secret_store=connector_secrets),
    )
    interview_context_factory = InterviewPreparationContextFactory(
        interviews=interview_service,
        applications=application_service,
        resumes=resume_store,
        career_history=career_history_store,
    )
    interview_preparation_service = InterviewPreparationService(
        interview_service,
        application_service,
        resume_store,
        career_history_store,
        OpenAIInterviewPreparationWorker(resume_analysis_config),
        SQLiteInterviewPreparationStore(Path(args.resume_store).expanduser()),
        context_factory=interview_context_factory,
    )
    job_research_checkpoint_owner = SQLiteCheckpointOwner(
        Path(args.job_research_checkpoint_store).expanduser()
    )
    job_research_store = SQLiteJobResearchStore(
        Path(args.job_research_store).expanduser()
    )
    job_research_service = JobResearchService(
        jobs=job_repository,
        store=job_research_store,
        worker=DeepAgentJobResearchWorker(
            # JOB_RESEARCH_AGENT_* when any is set (all-or-nothing), otherwise
            # the shared specialist endpoint as before.
            job_research_config_from_env(fallback=resume_analysis_config),
            skills_root=Path(args.job_research_skills_dir),
            checkpointer=job_research_checkpoint_owner.saver,
        ),
    )
    mock_checkpoint_owner = SQLiteCheckpointOwner(
        Path(args.mock_interview_checkpoint_store).expanduser()
    )
    # Shared with the read-back tool: one instance so the tool reads the same
    # file the workflow writes, and one place to change the path.
    mock_interview_store = SQLiteMockInterviewStore(
        Path(args.mock_interview_store).expanduser()
    )
    mock_interview_graph = MockInterviewGraph(
        store=mock_interview_store,
        worker=OpenAIMockInterviewWorker(
            resume_analysis_config,
            skill_loader=MockInterviewSkillLoader(
                Path(args.mock_interview_skills_dir)
            ),
        ),
        sources=StoredMockInterviewSourceProvider(
            context_factory=interview_context_factory,
            jobs=job_repository,
            research=job_research_store,
        ),
        checkpointer=mock_checkpoint_owner.saver,
    )
    return MainAgentRuntime(
        context_manager=context_manager,
        episode_reconciler=EpisodeReconciler(
            episodes=SQLiteCareerEpisodeStore(
                Path(args.context_store).expanduser()
            ),
            applications=application_store,
            interviews=interview_store,
            mock_interviews=mock_interview_store,
            job_research=job_research_store,
        ),
        decision_maker=OpenAICompatibleMainAgentDecisionMaker(
            main_config, max_output_tokens=main_output_tokens,
        ),
        career_context_projector=CareerContextProjector(
            career_history_store,
            semantic_retriever=semantic_retriever,
        ),
        trace_recorder=SQLiteTraceRecorder(Path(args.run_events_store).expanduser()),
        action_execution_store=SQLiteActionExecutionStore(
            Path(args.context_store).expanduser()
        ),
        # Same file as the context it gates: a sealed action and the rule that
        # sealed it have no reason to live in different databases, and a
        # deployment that loses one while keeping the other would enforce a rule
        # it cannot let the owner satisfy.
        capability_confirmation_store=SQLiteCapabilityConfirmationStore(
            Path(args.context_store).expanduser()
        ),
        turn_receipt_store=turn_receipt_store,
        owned_resources=(mock_checkpoint_owner, job_research_checkpoint_owner),
        tools=MainAgentToolRegistry(
            job_repository=job_repository,
            job_capture_store=SQLiteJobCaptureStore(Path(args.job_store).expanduser()),
            job_research_service=job_research_service,
            resume_store=resume_store,
            career_history_store=career_history_store,
            episode_store=episode_store,
            resume_export_service=ResumeExportService(
                resume_store,
                SQLiteResumeArtifactStore(Path(args.resume_store).expanduser()),
            ),
            application_service=application_service,
            interview_service=interview_service,
            interview_preparation_service=interview_preparation_service,
            email_tracking_service=email_tracking_service,
            action_center_service=action_center_service,
            calendar_service=calendar_service,
            mock_interview_graph=mock_interview_graph,
            mock_interview_store=mock_interview_store,
            resume_analysis_service=ResumeAnalysisService(
                resume_store,
                OpenAIResumeAnalysisWorker.from_env(
                    timeout_seconds=args.agent_timeout_seconds,
                ),
                SQLiteResumeAnalysisDraftStore(Path(args.resume_store).expanduser()),
                career_history_store,
            ),
            job_comparison_service=JobComparisonService(job_repository, match_store),
            job_analysis_service=JobAnalysisService(
                job_repository,
                OpenAIJobAnalysisWorker(resume_analysis_config),
            ),
            career_profile_store=context_store,
            owner_settings_store=context_store,
            conversation_store=context_store,
            resume_job_match_service=ResumeJobMatchService(
                resume_store,
                job_repository,
                career_history_store,
                OpenAIResumeJobMatchWorker(resume_analysis_config),
                match_store,
                career_profile_store=context_store,
            ),
            semantic_evidence_cache=semantic_retriever,
            working_notes_store=working_notes_store,
            resume_tailoring_service=ResumeTailoringService(
                resume_store,
                job_repository,
                career_history_store,
                match_store,
                SQLiteResumeTailoringDraftStore(Path(args.resume_store).expanduser()),
                DeepAgentResumeTailoringWorker(
                    resume_analysis_config,
                    skills_root=Path(args.resume_tailoring_skills_dir),
                ),
                DeepAgentResumeFinalizationWorker(
                    resume_analysis_config,
                    skills_root=Path(args.resume_tailoring_skills_dir),
                ),
                reviewer=OpenAIResumeTailoringReviewer(resume_analysis_config),
            ),
        ),
    )


def _read_resume_import(path: Path) -> tuple[bytes, str]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("--file must be a regular non-symlink file.")
    if path.stat().st_size > MAX_RESUME_IMPORT_BYTES:
        raise ValueError("Resume file exceeds the 5 MiB import limit.")
    return validate_resume_document(path.name, path.read_bytes())


def _resume_payload(resume, versions=()) -> dict[str, object]:
    payload = {"resume": resume.model_dump(mode="json")}
    if versions:
        payload["versions"] = [version.model_dump(mode="json") for version in versions]
    return payload


def _add_runtime_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--agent-timeout-seconds", type=float, default=300.0, help="Model call timeout (default: 300).")
    parser.add_argument(
        "--run-events-store",
        default="~/.career-agent/run-events.sqlite3",
        help="Local best-effort telemetry store path, redacted and independent of business stores.",
    )
    parser.add_argument("--job-store", default="~/.career-agent/jobs.sqlite3", help="Local durable job and JD snapshot store path.")
    parser.add_argument(
        "--job-research-store",
        default="~/.career-agent/job-research.sqlite3",
        help="Local durable job-research run, source, and report store path.",
    )
    parser.add_argument(
        "--job-research-checkpoint-store",
        default="~/.career-agent/job-research-checkpoints.sqlite3",
        help="Local durable DeepAgent checkpoint store for job research.",
    )
    parser.add_argument("--resume-store", default="~/.career-agent/resumes.sqlite3", help="Local resume metadata and artifact store path.")
    parser.add_argument("--application-store", default="~/.career-agent/applications.sqlite3", help="Local application tracking and event store path.")
    parser.add_argument("--email-store", default="~/.career-agent/email.sqlite3", help="Local email-account metadata, cursor, and event store path.")
    parser.add_argument("--action-store", default="~/.career-agent/actions.sqlite3", help="Local generated career action-item and lifecycle store path.")
    parser.add_argument("--calendar-store", default="~/.career-agent/calendar.sqlite3", help="Local Calendar account, approval proposal, event-link, and audit store path.")
    parser.add_argument(
        "--mock-interview-store",
        default="~/.career-agent/mock-interviews.sqlite3",
        help="Local mock-interview session, turn, and report store path.",
    )
    parser.add_argument(
        "--mock-interview-checkpoint-store",
        default="~/.career-agent/mock-interview-checkpoints.sqlite3",
        help="Local durable LangGraph checkpoint store for mock interviews.",
    )
    parser.add_argument(
        "--resume-tailoring-skills-dir",
        default=os.environ.get("RESUME_TAILORING_SKILLS_DIR", "skills"),
        help="Local skill source directory containing resume-tailoring/SKILL.md (default: RESUME_TAILORING_SKILLS_DIR or skills).",
    )
    parser.add_argument(
        "--mock-interview-skills-dir",
        default=os.environ.get("MOCK_INTERVIEW_SKILLS_DIR", "skills"),
        help="Local skill source directory containing mock-interview/SKILL.md (default: MOCK_INTERVIEW_SKILLS_DIR or skills).",
    )
    parser.add_argument(
        "--job-research-skills-dir",
        default=os.environ.get("JOB_RESEARCH_SKILLS_DIR", "skills"),
        help="Local skill source directory containing job-research/SKILL.md (default: JOB_RESEARCH_SKILLS_DIR or skills).",
    )
    parser.add_argument("--json", action="store_true", help="Emit one machine-readable JSON object.")
    parser.add_argument("--show-trace", action="store_true", help="Include the complete safe run trace in output.")
    parser.add_argument("--non-interactive", action="store_true", help="Never prompt for input.")


def _add_lock_anchor(parser: argparse.ArgumentParser) -> None:
    """A write command that does not open the context store still takes the
    workspace lock beside it, so it and the API/backup contend for one file."""

    parser.add_argument(
        "--context-store",
        default="~/.career-agent/context.sqlite3",
        help=(
            "Context store of the workspace this write belongs to. Only its "
            "directory is used here, to take the workspace's api-server.lock."
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="career-agent", description="Run the local Career Agent application.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    chat = subparsers.add_parser("chat", help="Send one natural-language turn to the Main Agent.", description="Run one non-interactive Main Agent turn. The agent decides whether to enter a registered workflow.", epilog="Example: career-agent chat --user-id u1 --session-id s1 --message 'Help me find AI Engineer jobs'")
    chat.add_argument("--user-id", required=True, help="Stable user identifier.")
    chat.add_argument("--session-id", required=True, help="Conversation session identifier.")
    chat.add_argument(
        "--message",
        help=(
            "Current user message. Required for a normal turn; not allowed when "
            "answering a pending interaction, where the transcript records the "
            "label of the option chosen, exactly as the web client sends it."
        ),
    )
    interaction = chat.add_mutually_exclusive_group()
    interaction.add_argument(
        "--confirm-interaction",
        metavar="INTERACTION_ID",
        help=(
            "Answer the pending interaction printed by the previous turn as "
            "'confirm'. The sealed action runs exactly as it was shown; the "
            "model is not consulted again."
        ),
    )
    interaction.add_argument(
        "--cancel-interaction",
        metavar="INTERACTION_ID",
        help="Answer the pending interaction printed by the previous turn as 'cancel'.",
    )
    chat.add_argument(
        "--interaction-scope",
        choices=("capability_confirmation", "resume_analysis_confirmation"),
        help=(
            "Scope of the interaction being answered, as printed in "
            "pending_interaction (default: capability_confirmation, the owner's "
            "gate on external writes). Only valid with --confirm-interaction or "
            "--cancel-interaction."
        ),
    )
    chat.add_argument(
        "--request-id",
        help=(
            "Stable caller request identity for safe cross-process write replay. "
            "Reuse it only when retrying the same logical turn."
        ),
    )
    chat.add_argument("--context-store", default="~/.career-agent/context.sqlite3", help="Local session and context store path.")
    chat.add_argument("--main-agent-timeout-seconds", type=float, default=60.0, help="Main Agent model timeout (default: 60).")
    _add_runtime_options(chat)

    actions = subparsers.add_parser(
        "actions",
        help="Inspect durable write actions that require reconciliation.",
    )
    action_subparsers = actions.add_subparsers(
        dest="actions_command", required=True
    )
    reconcile = action_subparsers.add_parser(
        "reconcile",
        help="List uncertain actions without automatically replaying them.",
    )
    reconcile.add_argument("--user-id", required=True)
    reconcile.add_argument(
        "--context-store",
        default="~/.career-agent/context.sqlite3",
        help="Local context and action-execution store path.",
    )
    settle = action_subparsers.add_parser(
        "settle",
        help="Record what reconciliation established about one pending action.",
        description=(
            "Separate from 'reconcile' because the finding comes from outside "
            "this system — a calendar checked, a record looked up, a provider's "
            "own log read. Nothing here re-runs the action: an external write "
            "cannot be rolled back or safely repeated from a CLI, so the only "
            "honest operations are to look and to write down what was seen."
        ),
    )
    settle.add_argument("--action-id", required=True)
    settle_outcome = settle.add_mutually_exclusive_group(required=True)
    settle_outcome.add_argument(
        "--executed",
        action="store_true",
        help=(
            "The effect did happen. Policy is not re-checked: the effect "
            "already exists, and refusing to record it would leave the row "
            "pending forever."
        ),
    )
    settle_outcome.add_argument(
        "--not-executed",
        action="store_true",
        help=(
            "The effect never happened. The row becomes terminal, and a fresh "
            "attempt needs a NEW request id: this identity now names a "
            "finished fact."
        ),
    )
    settle.add_argument(
        "--output",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help=(
            "Identifiers the action produced. Required only when the capability "
            "declares a reducer receipt; otherwise they remain audit metadata. "
            "Scalars only — report bodies stay in their own stores."
        ),
    )
    settle.add_argument(
        "--reason",
        default=None,
        help="Required with --not-executed. What the investigation established.",
    )
    settle.add_argument(
        "--context-store",
        default="~/.career-agent/context.sqlite3",
        help="Local context and action-execution store path.",
    )

    target_role = subparsers.add_parser("target-role", help="Manage resume target-role categories.", description="Create and list user-scoped resume categories.")
    target_role_subparsers = target_role.add_subparsers(dest="target_role_command", required=True)
    target_role_create = target_role_subparsers.add_parser("create", help="Create a resume target-role category.")
    target_role_create.add_argument("--user-id", required=True, help="Target-role owner identifier.")
    target_role_create.add_argument("--title", required=True, help="Target role title, such as AI Engineer.")
    target_role_create.add_argument("--priority", type=int, default=0, help="Lower values appear first (default: 0).")
    target_role_create.add_argument("--resume-store", default="~/.career-agent/resumes.sqlite3", help="Local resume store path.")
    _add_lock_anchor(target_role_create)
    target_role_list = target_role_subparsers.add_parser("list", help="List target-role categories.")
    target_role_list.add_argument("--user-id", required=True, help="Target-role owner identifier.")
    target_role_list.add_argument("--resume-store", default="~/.career-agent/resumes.sqlite3", help="Local resume store path.")

    resume = subparsers.add_parser("resume", help="Import and manage local resume versions.", description="Store user-imported PDF or UTF-8 text/Markdown resumes locally without sending content to a model.")
    resume_subparsers = resume.add_subparsers(dest="resume_command", required=True)
    resume_import = resume_subparsers.add_parser("import", help="Create a resume or append an immutable version.")
    resume_import.add_argument("--user-id", required=True, help="Resume owner identifier.")
    name_or_id = resume_import.add_mutually_exclusive_group(required=True)
    name_or_id.add_argument("--name", help="Display name for a new resume.")
    name_or_id.add_argument("--resume-id", help="Existing resume ID to receive a new version.")
    resume_import.add_argument("--target-role-id", help="Required target-role category for a new resume family.")
    resume_import.add_argument("--file", type=Path, required=True, help="PDF or UTF-8 .txt/.md/.markdown resume file.")
    resume_import.add_argument("--resume-store", default="~/.career-agent/resumes.sqlite3", help="Local resume store path.")
    _add_lock_anchor(resume_import)
    resume_list = resume_subparsers.add_parser("list", help="List safe resume metadata.")
    resume_list.add_argument("--user-id", required=True, help="Resume owner identifier.")
    resume_list.add_argument("--target-role-id", help="Filter to one target-role category.")
    resume_list.add_argument("--resume-store", default="~/.career-agent/resumes.sqlite3", help="Local resume store path.")
    resume_show = resume_subparsers.add_parser("show", help="Show safe resume and version metadata.")
    resume_show.add_argument("--user-id", required=True, help="Resume owner identifier.")
    resume_show.add_argument("--resume-id", required=True, help="Resume identifier.")
    resume_show.add_argument("--resume-store", default="~/.career-agent/resumes.sqlite3", help="Local resume store path.")

    job = subparsers.add_parser("job", help="List, find, and read saved jobs and complete JD snapshots.")
    job_subparsers = job.add_subparsers(dest="job_command", required=True)
    job_list = job_subparsers.add_parser("list", help="List recently persisted jobs.")
    job_list.add_argument("--user-id", required=True, help="Job owner identifier.")
    job_list.add_argument("--limit", type=int, default=20, help="Maximum results, from 1 to 100 (default: 20).")
    job_list.add_argument("--job-store", default="~/.career-agent/jobs.sqlite3", help="Local durable job and JD snapshot store path.")
    job_find = job_subparsers.add_parser("find", help="Find saved jobs by metadata and complete JD text.")
    job_find.add_argument("--user-id", required=True, help="Job owner identifier.")
    job_find.add_argument("--query", required=True, help="Company, title, city, skill, or JD text to find.")
    job_find.add_argument("--limit", type=int, default=20, help="Maximum results, from 1 to 100 (default: 20).")
    job_find.add_argument("--job-store", default="~/.career-agent/jobs.sqlite3", help="Local durable job and JD snapshot store path.")
    job_show = job_subparsers.add_parser("show", help="Show one complete persisted JD with provenance.")
    job_show.add_argument("--user-id", required=True, help="Job owner identifier.")
    job_show.add_argument("--job-posting-id", help="Posting ID returned by job list or find.")
    job_show.add_argument("--run-id", help="Job Discovery run that produced the result.")
    job_show.add_argument("--selection-index", type=int, help="One-based result index within --run-id.")
    job_show.add_argument("--job-store", default="~/.career-agent/jobs.sqlite3", help="Local durable job and JD snapshot store path.")

    email_command = subparsers.add_parser(
        "email", help="Connect Gmail or QQ accounts without storing mailbox passwords."
    )
    email_subparsers = email_command.add_subparsers(dest="email_command", required=True)
    email_add = email_subparsers.add_parser("add-account", help="Register an email account and an env-based secret reference.")
    email_add.add_argument("--user-id", required=True)
    email_add.add_argument("--provider", choices=("gmail", "qq"), required=True)
    email_add.add_argument("--address", required=True)
    email_add.add_argument("--credential-env", required=True, help="Environment variable containing Gmail OAuth JSON or a QQ authorization code.")
    email_add.add_argument("--email-store", default="~/.career-agent/email.sqlite3")
    _add_lock_anchor(email_add)
    email_list = email_subparsers.add_parser("list-accounts", help="List safe email account metadata.")
    email_list.add_argument("--user-id", required=True)
    email_list.add_argument("--email-store", default="~/.career-agent/email.sqlite3")

    calendar_command = subparsers.add_parser(
        "calendar", help="Connect Google Calendar without storing OAuth secrets."
    )
    calendar_subparsers = calendar_command.add_subparsers(
        dest="calendar_command", required=True
    )
    calendar_add = calendar_subparsers.add_parser(
        "add-account", help="Register Google Calendar and an env-based OAuth reference."
    )
    calendar_add.add_argument("--user-id", required=True)
    calendar_add.add_argument("--address", required=True)
    calendar_add.add_argument(
        "--calendar-id", default="primary",
        help="Google Calendar identifier (default: primary).",
    )
    calendar_add.add_argument(
        "--credential-env", required=True,
        help="Environment variable containing Google OAuth JSON with Calendar scope.",
    )
    calendar_add.add_argument(
        "--calendar-store", default="~/.career-agent/calendar.sqlite3"
    )
    _add_lock_anchor(calendar_add)
    calendar_list = calendar_subparsers.add_parser(
        "list-accounts", help="List safe Calendar account metadata."
    )
    calendar_list.add_argument("--user-id", required=True)
    calendar_list.add_argument(
        "--calendar-store", default="~/.career-agent/calendar.sqlite3"
    )

    eval_command = subparsers.add_parser(
        "eval",
        help="Run or re-record the Main Agent trajectory evaluation.",
        description=(
            "Trajectory scenarios pin what the Main Agent decides, which is the "
            "one thing the ordinary test suite cannot see: every other test "
            "scripts the decision and checks the runtime. Replay runs offline "
            "and checks this project against decisions the model already made; "
            "only --record evaluates the model itself."
        ),
    )
    eval_subparsers = eval_command.add_subparsers(dest="eval_command", required=True)
    eval_trajectories = eval_subparsers.add_parser(
        "trajectories", help="Replay the scenario catalogue, or re-record it."
    )
    eval_trajectories.add_argument(
        "--record",
        action="store_true",
        help=(
            "Ask the live model and overwrite missing or stale cassettes. "
            "Needs MAIN_AGENT_* configured. Independent samples run concurrently "
            "up to --jobs; steps inside one sample stay sequential. Pass --force "
            "to recut cassettes that are still current."
        ),
    )
    eval_trajectories.add_argument(
        "--jobs",
        type=int,
        default=8,
        help=(
            "Max concurrent live model calls while recording (default: 8). "
            "Use 1 to restore serial recuts."
        ),
    )
    eval_trajectories.add_argument(
        "--force",
        action="store_true",
        help="Recut cassettes even when prompt and context fingerprints still match.",
    )
    eval_trajectories.add_argument(
        "--main-agent-timeout-seconds",
        type=float,
        default=60.0,
        help="Main Agent model timeout while recording (default: 60).",
    )
    eval_trajectories.add_argument(
        "--scenario",
        action="append",
        default=None,
        help="Limit to named scenarios. Repeatable.",
    )
    eval_trajectories.add_argument(
        "--samples",
        type=int,
        default=None,
        help=(
            "Set the live recording sample count for every selected scenario "
            "(1-5). It may increase, but not lower, a scenario's requirement."
        ),
    )
    eval_rederivation = eval_subparsers.add_parser(
        "rederivation",
        help="Count repeated tool calls across production compaction points.",
    )
    eval_rederivation.add_argument("--user-id", required=True)
    eval_rederivation.add_argument("--session-id", required=True)
    eval_rederivation.add_argument(
        "--run-events-store",
        default="~/.career-agent/run-events.sqlite3",
        help="Local best-effort telemetry store path.",
    )
    eval_memory_exposure = eval_subparsers.add_parser(
        "memory-exposure",
        help=(
            "Measure P1 staleness, supersedence, and post-tombstone zombie exposure."
        ),
    )
    eval_memory_exposure.add_argument("--user-id", required=True)
    eval_memory_exposure.add_argument("--session-id", required=True)
    eval_memory_exposure.add_argument(
        "--run-events-store",
        default="~/.career-agent/run-events.sqlite3",
        help="Local best-effort telemetry store path.",
    )
    eval_memory_exposure.add_argument(
        "--max-events",
        type=int,
        default=10_000,
        help="Bound the newest memory observations read from telemetry.",
    )
    eval_match = eval_subparsers.add_parser(
        "resume-job-match",
        help="Run repeated live-model samples for the 096 match rubric (no cassette).",
    )
    eval_match.add_argument(
        "--samples", type=int, default=6,
        help="Independent samples (1-20; default: 6).",
    )
    eval_match.add_argument(
        "--suite", choices=("smoke", "boundaries"), default="boundaries",
        help="Use the full 096 boundary fixture suite (default) or one smoke fixture.",
    )
    eval_match.add_argument(
        "--case",
        action="append",
        default=None,
        help="Run only this named boundary case. Repeatable; valid only for boundaries.",
    )
    eval_match.add_argument("--timeout-seconds", type=float, default=60.0)

    memory = subparsers.add_parser(
        "memory",
        help="Inspect memory freshness without changing stored memory.",
    )
    memory_subparsers = memory.add_subparsers(
        dest="memory_command", required=True
    )
    memory_report = memory_subparsers.add_parser(
        "report",
        help="Report episode decay and preference-maintenance counts.",
    )
    memory_report.add_argument("--user-id", required=True)
    memory_report.add_argument(
        "--context-store",
        default="~/.career-agent/context.sqlite3",
        help="Existing local context store path (opened read-only).",
    )
    memory_report.add_argument(
        "--half-life-days",
        type=float,
        default=DecayPolicy().half_life_days,
    )
    memory_report.add_argument(
        "--access-boost",
        type=float,
        default=DecayPolicy().access_boost,
    )
    memory_report.add_argument(
        "--projection-threshold",
        type=float,
        default=DecayPolicy().projection_threshold,
    )
    memory_report.add_argument(
        "--quarantine-stale-days",
        type=int,
        default=14,
        help="Age after which an unconfirmed quarantine candidate is stale.",
    )
    memory_export = memory_subparsers.add_parser(
        "export",
        help="Write a complete MEMORY.md directly to a local file.",
    )
    memory_review = memory_subparsers.add_parser(
        "review",
        help="Show the complete diff for an edited MEMORY.md without a model.",
    )
    memory_apply = memory_subparsers.add_parser(
        "apply",
        help="Apply one previously reviewed MEMORY.md diff as a batch.",
    )
    for command in (memory_export, memory_review, memory_apply):
        command.add_argument("--user-id", required=True)
        command.add_argument(
            "--context-store",
            default="~/.career-agent/context.sqlite3",
        )
        command.add_argument(
            "--resume-store",
            default="~/.career-agent/resumes.sqlite3",
        )
    memory_export.add_argument("--output", type=Path, required=True)
    memory_export.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing output file.",
    )
    for command in (memory_review, memory_apply):
        command.add_argument("--file", type=Path, required=True)
    memory_apply.add_argument("--confirmation-digest", required=True)
    memory_apply.add_argument(
        "--confirm",
        action="store_true",
        help="Explicitly confirm the entire displayed diff.",
    )
    memory_apply.add_argument(
        "--working-notes-dir",
        type=Path,
        default=None,
        help="Defaults to the working-notes directory beside the context store.",
    )

    keys_command = subparsers.add_parser(
        "api-keys",
        help="Issue, list and revoke the credentials the API authenticates with.",
        description=(
            "Every API route derives its user from a key rather than from a "
            "field the caller sends, so a key is the only way in. Issue one per "
            "client and give it only the scopes that client needs: the browser "
            "extension cannot hold a secret safely, so its key should carry "
            "capture:write and nothing else."
        ),
    )
    backup_command = subparsers.add_parser(
        "backup",
        help="Back up, verify and restore the whole local workspace.",
        description=(
            "The workspace is every SQLite store plus the working-notes directory. "
            "A backup copies all of them at once into one directory with a "
            "checksummed manifest; restore refuses to touch the live workspace "
            "unless that whole set verifies, and keeps a safety copy of what it "
            "replaces."
        ),
    )
    backup_subparsers = backup_command.add_subparsers(dest="backup_command", required=True)
    backup_create = backup_subparsers.add_parser(
        "create", help="Copy every store into a new directory and write manifest.json."
    )
    backup_create.add_argument(
        "--dest",
        default=None,
        help=(
            "Directory to create. Defaults to ~/.career-agent-backups/<UTC timestamp>. "
            "Must not exist or must be empty."
        ),
    )
    backup_create.add_argument(
        "--allow-running-api",
        action="store_true",
        help=(
            "Copy even while the API is running. Each database is then consistent "
            "with itself but not necessarily with the others; the manifest records "
            "consistent_snapshot=false and verify/restore warn about it."
        ),
    )
    backup_verify = backup_subparsers.add_parser(
        "verify", help="Check a backup's files against its manifest and SQLite integrity_check."
    )
    backup_verify.add_argument("--source", required=True, help="Backup directory.")
    backup_restore = backup_subparsers.add_parser(
        "restore", help="Replace the current workspace with a verified backup."
    )
    backup_restore.add_argument("--source", required=True, help="Backup directory.")
    backup_restore.add_argument(
        "--yes",
        action="store_true",
        help="Required. Restore overwrites the current stores (after a safety copy).",
    )
    backup_restore.add_argument(
        "--no-safety-copy",
        action="store_true",
        help="Do not back up the current workspace before overwriting it.",
    )
    for backup_parser in (backup_create, backup_verify, backup_restore):
        backup_parser.add_argument(
            "--context-store",
            default="~/.career-agent/context.sqlite3",
            help="Local session and context store path.",
        )
        backup_parser.add_argument(
            "--api-key-store",
            default=None,
            help="API key store path. Defaults to $CAREER_AGENT_DATA_DIR/api_keys.sqlite3.",
        )
        _add_runtime_options(backup_parser)

    keys_subparsers = keys_command.add_subparsers(dest="keys_command", required=True)
    issue_keys = keys_subparsers.add_parser(
        "issue", help="Mint a key. Its secret is printed once and never stored."
    )
    issue_keys.add_argument("--user-id", required=True)
    issue_keys.add_argument(
        "--name", required=True, help="What holds this key, so a human can revoke it."
    )
    issue_keys.add_argument(
        "--scope",
        action="append",
        required=True,
        choices=sorted(KNOWN_SCOPES),
        help="Repeatable. Grant the narrowest set that client needs.",
    )
    issue_keys.add_argument(
        "--expires-in-days",
        type=int,
        default=DEFAULT_EXPIRY_DAYS,
        help=(
            f"Default {DEFAULT_EXPIRY_DAYS}. A key with no end date is a key "
            "nobody rotates, so a permanent one takes --no-expiry."
        ),
    )
    issue_keys.add_argument(
        "--no-expiry",
        action="store_true",
        help="Issue a key that never expires. Deliberate, not the default.",
    )
    issue_keys.add_argument("--api-key-store", default="data/api_keys.sqlite3")
    _add_lock_anchor(issue_keys)
    list_keys = keys_subparsers.add_parser("list", help="Show keys without secrets.")
    list_keys.add_argument("--user-id", default=None)
    list_keys.add_argument("--api-key-store", default="data/api_keys.sqlite3")
    revoke_keys = keys_subparsers.add_parser("revoke", help="Retire one key.")
    revoke_keys.add_argument("--key-id", required=True)
    revoke_keys.add_argument("--api-key-store", default="data/api_keys.sqlite3")
    _add_lock_anchor(revoke_keys)

    settings = subparsers.add_parser(
        "settings",
        help="Read and change the owner rules the runtime enforces.",
        description=(
            "These rules gate what the agent may do before it acts. They are "
            "changed directly here, or through an agent proposal that remains "
            "sealed until the owner confirms it. Conversation text alone cannot "
            "relax the rule that constrains the agent."
        ),
    )
    settings_subparsers = settings.add_subparsers(
        dest="settings_command", required=True
    )
    settings_show = settings_subparsers.add_parser(
        "show", help="Print the rules currently in force."
    )
    settings_set = settings_subparsers.add_parser(
        "set", help="Change one rule. Takes effect on the next turn."
    )
    settings_history = settings_subparsers.add_parser(
        "history", help="Show the append-only owner-settings change history."
    )
    settings_set.add_argument(
        "--application-confirmation",
        choices=("always_ask", "on_user_report"),
        help=(
            "always_ask holds every create_application for your confirmation "
            "before it runs; on_user_report records it directly."
        ),
    )
    settings_set.add_argument(
        "--boss-search",
        choices=("explicit_request_only", "allowed"),
        help="Whether job search may run without being explicitly asked for.",
    )
    settings_set.add_argument(
        "--confirm-before",
        help=(
            "Comma-separated WRITE capability names to approve one by one "
            "before they run; replaces the current list. Pass '' to clear."
        ),
    )
    for sub in (settings_show, settings_set, settings_history):
        sub.add_argument("--user-id", required=True, help="Whose rules to act on.")
        sub.add_argument(
            "--context-store",
            default="~/.career-agent/context.sqlite3",
            help="Local context store path.",
        )

    return parser


def _stored_job_summary_payload(item: StoredJobSummary) -> dict[str, object]:
    return item.model_dump(mode="json")


def _stored_job_payload(record: StoredJobRecord) -> dict[str, object]:
    return {
        "job": {
            **record.posting.model_dump(mode="json"),
            "city": record.city,
            "salary": record.salary,
            "availability_status": record.availability_status,
            "last_checked_at": record.last_checked_at.isoformat(),
            "closed_at": record.closed_at.isoformat() if record.closed_at else None,
        },
        "jd_snapshot": record.snapshot.model_dump(mode="json"),
        "analysis": record.analysis.analysis.model_dump(mode="json") if record.analysis else None,
    }


def _chat_tool_result_payload(result: ToolObservation) -> dict[str, object]:
    return result.model_dump(mode="json")


def _chat_interaction_response(args: argparse.Namespace) -> InteractionResponse | None:
    if args.confirm_interaction is None and args.cancel_interaction is None:
        if args.interaction_scope is not None:
            raise ValueError(
                "--interaction-scope only applies together with "
                "--confirm-interaction or --cancel-interaction"
            )
        return None
    if args.message is not None:
        raise ValueError(
            "--message cannot be combined with --confirm-interaction or "
            "--cancel-interaction: answering an interaction records the label "
            "of the option chosen, as the web client does"
        )
    scope = args.interaction_scope or "capability_confirmation"
    if args.confirm_interaction is not None:
        return InteractionResponse(
            interaction_id=args.confirm_interaction, scope=scope, action="confirm"
        )
    return InteractionResponse(
        interaction_id=args.cancel_interaction, scope=scope, action="cancel"
    )


def _chat_user_message(
    args: argparse.Namespace, interaction_response: InteractionResponse | None
) -> str:
    """What this turn records as the user's words.

    Answering an interaction is a button press, not prose: the transcript gets
    the same label the web client would have sent, and nothing else.
    """

    if interaction_response is None:
        if args.message is None:
            raise ValueError(
                "--message is required unless --confirm-interaction or "
                "--cancel-interaction answers a pending interaction"
            )
        return args.message
    return scoped_interaction_message(
        interaction_response.scope, interaction_response.action
    )


def _chat_pending_interaction_event(
    turn, *, session_id: str
) -> InteractionRequiredEvent | None:
    """The scoped gates the CLI can answer, rebuilt from durable state.

    Only interactions with a scope are listed: those are the ones the runtime
    routes on ``InteractionResponse`` rather than on a natural-language reply,
    so a CLI user needs their id. The same builders serve the SSE stream and
    the transcript reload, so all three clients see one id.
    """

    tool_result = turn.tool_result
    if tool_result is None:
        return None
    if tool_result.state == "capability_confirmation_required":
        confirmation_id = tool_result.payload.get("confirmation_id")
        if not isinstance(confirmation_id, str):
            return None
        return capability_confirmation_event(
            conversation_id=session_id,
            confirmation_id=confirmation_id,
            prompt=tool_result.message,
        )
    task = getattr(turn.context, "task", None)
    if (
        tool_result.state == "resume_analysis_ready"
        and task is not None
        and task.resume_analysis_status == "pending"
        and task.active_resume_analysis_id is not None
    ):
        return resume_analysis_confirmation_event(
            conversation_id=session_id,
            analysis_id=task.active_resume_analysis_id,
        )
    return None


def _chat_replayed_interaction_event(
    turn: ReplayedTurn,
) -> InteractionRequiredEvent | None:
    """The gate the original run published, if it stopped at one.

    A replay executes nothing, so the pending interaction is whatever the
    receipt recorded; the runtime answers it by id exactly as it would have
    answered the first time.
    """

    return next(
        (
            event
            for event in reversed(turn.events)
            if isinstance(event, InteractionRequiredEvent) and event.scope is not None
        ),
        None,
    )


def _chat_pending_interaction_payload(
    event: InteractionRequiredEvent | None,
) -> dict[str, object] | None:
    """A pending gate, addressed to whoever runs the CLI.

    A sealed confirmation can only be answered with its interaction id; a
    fresh natural-language turn saying "yes" would make the model propose the
    write again and seal another request. The payload carries the id and the
    exact flags that answer it.
    """

    if event is None or event.scope is None:
        return None
    scope_flag = (
        "" if event.scope == "capability_confirmation" else f" --interaction-scope {event.scope}"
    )
    return {
        "interaction_id": event.interaction_id,
        "scope": event.scope,
        "kind": event.kind,
        "options": [option.model_dump(mode="json") for option in event.options],
        "confirm_with": f"--confirm-interaction {event.interaction_id}{scope_flag}",
        "cancel_with": f"--cancel-interaction {event.interaction_id}{scope_flag}",
    }


def _write_chat_payload(
    turn,
    *,
    user_id: str,
    session_id: str,
    output: TextIO,
    notice: str | None = None,
) -> int:
    tool_result = turn.tool_result
    tool_results = turn.tool_results or ((tool_result,) if tool_result else ())
    failed_result = next(
        (result for result in reversed(tool_results) if result.state == "failed"),
        None,
    )
    model_decision = turn.model_decision
    payload = {
        "state": "failed" if failed_result else "completed",
        "user_id": user_id,
        "session_id": session_id,
        "assistant_message": turn.assistant_message,
        # A turn the model never decided reports no decision. This used to be a
        # flag check against a field that always held an ``AgentDecision``, so
        # publishing an invented action and tool name was one forgotten
        # condition away — and that is precisely the incident that happened. The
        # decision is now absent rather than guarded: ``model_decision`` is
        # ``None`` for both runtime-owned ingresses, and there is nothing to
        # remember. What actually executed is in ``tool_results``.
        "decision": {
            "origin": turn.origin.label,
            "requested_by": turn.requested_by,
            "action": model_decision.action if model_decision else None,
            "tool_name": (
                model_decision.tool_call.name
                if model_decision and model_decision.tool_call
                else None
            ),
        },
        "tool_result": _chat_tool_result_payload(tool_result) if tool_result else None,
        "tool_results": [
            _chat_tool_result_payload(result) for result in tool_results
        ],
        "artifacts": [
            artifact.reference.model_dump(mode="json")
            for artifact in turn.artifacts
        ],
        "pending_interaction": _chat_pending_interaction_payload(
            _chat_pending_interaction_event(turn, session_id=session_id)
        ),
        # Addressed to whoever runs the CLI, not to the agent: it never entered
        # the model's context, so the model cannot act on it.
        "maintenance_notice": notice,
    }
    json.dump(payload, output, ensure_ascii=False, separators=(",", ":"))
    output.write("\n")
    return _failure_exit_code(failed_result) if failed_result else EXIT_OK


def _write_chat_replay(
    turn: ReplayedTurn,
    *,
    user_id: str,
    session_id: str,
    output: TextIO,
) -> int:
    payload = {
        "state": "replayed",
        "user_id": user_id,
        "session_id": session_id,
        "request_id": turn.request_id,
        "turn_id": turn.turn_id,
        "assistant_message": turn.assistant_message,
        "pending_interaction": _chat_pending_interaction_payload(
            _chat_replayed_interaction_event(turn)
        ),
    }
    json.dump(payload, output, ensure_ascii=False, separators=(",", ":"))
    output.write("\n")
    return EXIT_OK


def _write_chat_error(error: Exception, output: TextIO, *, code: int, next_action: str | None = None) -> int:
    payload = {"state": "failed", "error_code": getattr(error, "code", "CHAT_ERROR"), "error_detail": str(error), "next_action": next_action}
    json.dump(payload, output, ensure_ascii=False, separators=(",", ":"))
    output.write("\n")
    return code


def _failure_exit_code(result: ToolObservation) -> int:
    if result.state != "failed":
        return EXIT_OK
    return EXIT_WORKFLOW_ERROR


def _trajectory_tool_specs():
    """Every tool the registry can offer, wired with placeholder services.

    Scenarios never execute a tool, so the services only have to exist.
    The trajectory harness filters this registry by each step's active profile.
    """
    import inspect

    from career_agent.agent.main_agent_tools import MainAgentToolRegistry

    parameters = tuple(
        name
        for name in inspect.signature(MainAgentToolRegistry.__init__).parameters
        if name != "self"
    )
    return MainAgentToolRegistry(**{name: object() for name in parameters}).schemas()


def _run_action_settle(args, stdout) -> int:
    """Write down what an investigation found about one pending action.

    Deliberately not "reconcile it for me": what happened lives in an external
    system, and the four outcomes the ledger recognises are decided out there.
    This only records the conclusion so the row stops being pending and task
    state can be repaired from it.
    """

    def refuse(message: str) -> int:
        stdout.write(json.dumps({"error": message}, ensure_ascii=False))
        stdout.write("\n")
        return EXIT_ARGUMENT_ERROR

    store = SQLiteActionExecutionStore(Path(args.context_store).expanduser())
    try:
        if args.executed:
            execution = store.get(action_id=args.action_id)
            if execution is None:
                return refuse("action execution not found")
            if execution.status != "PENDING":
                return refuse("action execution is no longer pending")
            settlement_specs = {
                "create_application": (
                    "application_ready",
                    frozenset(
                        {
                            "application_id",
                            "job_posting_id",
                            "resume_version_id",
                            "status",
                        }
                    ),
                )
            }
            spec = settlement_specs.get(execution.tool_name)
            if spec is not None and not args.output:
                return refuse(
                    "--executed needs the required --output KEY=VALUE fields "
                    "for this capability"
                )
            output: dict[str, str | int | float | bool | None] = {}
            for pair in args.output or ():
                key, separator, value = pair.partition("=")
                if not separator or not key:
                    return refuse(f"--output expects KEY=VALUE, got {pair!r}")
                if key in output:
                    return refuse(f"duplicate --output field: {key}")
                if key == RESULT_STATE_RECEIPT_KEY:
                    return refuse(
                        f"{RESULT_STATE_RECEIPT_KEY} is reserved for the runtime"
                    )
                output[key] = value
            if spec is None:
                # The investigation may close any pending action, but only a
                # declared capability receipt may drive a reducer. Arbitrary
                # operator keys remain audit metadata under a no-op state.
                result_state = "action_reconciled"
            else:
                result_state, required = spec
                unknown = set(output) - required
                missing = required - set(output)
                if unknown:
                    return refuse(
                        "unsupported --output fields for "
                        f"{execution.tool_name}: {', '.join(sorted(unknown))}"
                    )
                if missing:
                    return refuse(
                        "missing --output fields for "
                        f"{execution.tool_name}: {', '.join(sorted(missing))}"
                    )
            output[RESULT_STATE_RECEIPT_KEY] = result_state
            settled = store.succeed(action_id=args.action_id, output=output)
        else:
            if not args.reason:
                return refuse(
                    "--not-executed needs --reason: the row becomes terminal, and "
                    "whoever retries it later has only this line to learn why the "
                    "same request id now fails"
                )
            settled = store.fail(
                action_id=args.action_id,
                error_code="RECONCILED_NOT_EXECUTED",
                error_detail=(
                    f"{args.reason} — a fresh attempt needs a new request id; "
                    "this identity now names a finished fact."
                ),
            )
        confirmation_prefix = "confirmation:"
        if settled.anchor.startswith(confirmation_prefix):
            confirmation_id = settled.anchor[len(confirmation_prefix):]
            SQLiteCapabilityConfirmationStore(
                Path(args.context_store).expanduser()
            ).reconcile(
                confirmation_id=confirmation_id,
                user_id=settled.user_id,
                status="EXECUTED" if args.executed else "FAILED",
            )
    except (ValueError, CapabilityConfirmationSettledError) as error:
        return refuse(str(error))

    json.dump(
        {
            "action_id": settled.action_id,
            "status": settled.status,
            "output": settled.output,
            "error_code": settled.error_code,
            "settled_at": (
                settled.settled_at.isoformat() if settled.settled_at else None
            ),
        },
        stdout,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    stdout.write("\n")
    return EXIT_OK


def _backup_plan(args: argparse.Namespace) -> BackupPlan:
    """Every path the runtime opens from these arguments. Kept in one place so
    a new store cannot be added to the runtime without deciding whether it is
    part of the backup."""

    context_store = Path(args.context_store).expanduser()
    api_key_store = (
        Path(args.api_key_store).expanduser()
        if args.api_key_store
        else Path(os.environ.get("CAREER_AGENT_DATA_DIR", "data")) / "api_keys.sqlite3"
    )
    databases = (
        context_store,
        Path(args.resume_store),
        Path(args.application_store),
        Path(args.job_store),
        Path(args.job_research_store),
        Path(args.job_research_checkpoint_store),
        Path(args.email_store),
        Path(args.action_store),
        Path(args.calendar_store),
        Path(args.mock_interview_store),
        Path(args.mock_interview_checkpoint_store),
        Path(args.run_events_store),
        api_key_store,
    )
    return BackupPlan(
        databases=tuple(path.expanduser() for path in databases),
        directories=(context_store.with_name("working-notes"),),
    )


def _run_backup(args, stdout) -> int:
    plan = _backup_plan(args)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    def emit(payload: dict) -> None:
        stdout.write(json.dumps(payload, ensure_ascii=False, indent=2))
        stdout.write("\n")

    # The API holds this lock for its whole life. Holding it here is what
    # makes a backup one point in time across all the databases (they are
    # copied one after another), and what keeps a restore from swapping
    # files under open connections.
    lock = build_workspace_lock(args)

    try:
        if args.backup_command == "create":
            destination = (
                Path(args.dest).expanduser()
                if args.dest
                else Path("~/.career-agent-backups").expanduser() / stamp
            )
            try:
                lock.acquire()
                holding_lock = True
            except SingleWorkerError as error:
                if not args.allow_running_api:
                    emit(
                        {
                            "error": (
                                "API 正在运行，请先停止后再备份，否则各库可能不是同一时刻的快照。"
                                f"如接受该风险可加 --allow-running-api。{error}"
                            )
                        }
                    )
                    return EXIT_WORKFLOW_ERROR
                holding_lock = False
            try:
                manifest = create_backup(
                    plan, destination, consistent_snapshot=holding_lock
                )
            finally:
                if holding_lock:
                    lock.release()
            emit(
                {
                    "backup": str(destination),
                    "created_at": manifest.created_at.isoformat(),
                    "consistent_snapshot": manifest.consistent_snapshot,
                    "warnings": (
                        [] if manifest.consistent_snapshot else [INCONSISTENT_SNAPSHOT_WARNING]
                    ),
                    "files": len(manifest.entries),
                    "bytes": sum(entry.size for entry in manifest.entries),
                    "entries": [entry.name for entry in manifest.entries],
                }
            )
            return EXIT_OK
        if args.backup_command == "verify":
            report = verify_backup(Path(args.source))
            emit(
                {
                    "backup": str(report.directory),
                    "ok": report.ok,
                    "created_at": (
                        report.manifest.created_at.isoformat() if report.manifest else None
                    ),
                    "consistent_snapshot": (
                        report.manifest.consistent_snapshot if report.manifest else None
                    ),
                    "files": len(report.manifest.entries) if report.manifest else 0,
                    "problems": list(report.problems),
                    "warnings": list(report.warnings),
                }
            )
            return EXIT_OK if report.ok else EXIT_WORKFLOW_ERROR
        if not args.yes:
            emit(
                {
                    "error": "restore overwrites the current workspace; re-run with --yes",
                }
            )
            return EXIT_ARGUMENT_ERROR
        try:
            lock.acquire()
        except SingleWorkerError as error:
            emit({"error": f"API 正在运行，请先停止后再恢复。{error}"})
            return EXIT_WORKFLOW_ERROR
        try:
            safety_copy_dir = (
                None
                if args.no_safety_copy
                else Path("~/.career-agent-backups").expanduser() / f"pre-restore-{stamp}"
            )
            report = restore_backup(
                Path(args.source), plan, safety_copy_dir=safety_copy_dir
            )
        finally:
            lock.release()
        emit(
            {
                "restored": list(report.restored),
                "skipped_no_target": list(report.skipped_missing_target),
                "safety_copy": str(report.safety_copy) if report.safety_copy else None,
                "warnings": list(report.warnings),
            }
        )
        return EXIT_OK
    except BackupError as error:
        emit({"error": str(error)})
        return EXIT_WORKFLOW_ERROR


def _run_api_keys(args, stdout) -> int:
    store = SQLiteApiKeyStore(Path(args.api_key_store).expanduser())
    if args.keys_command == "issue":
        try:
            issued = store.issue(
                user_id=args.user_id,
                name=args.name,
                scopes=frozenset(args.scope),
                expires_in_days=None if args.no_expiry else args.expires_in_days,
            )
        except ValueError as error:
            stdout.write(json.dumps({"error": str(error)}, ensure_ascii=False))
            stdout.write("\n")
            return EXIT_ARGUMENT_ERROR
        # The only moment the secret exists outside the client. The store keeps
        # a digest, so a lost key is reissued rather than recovered.
        stdout.write(
            json.dumps(
                {
                    "key_id": issued.key_id,
                    "user_id": issued.user_id,
                    "name": issued.name,
                    "scopes": sorted(issued.scopes),
                    "expires_at": (
                        issued.expires_at.isoformat() if issued.expires_at else None
                    ),
                    "secret": issued.secret,
                    "note": "Store this now; it cannot be shown again.",
                },
                ensure_ascii=False,
            )
        )
        stdout.write("\n")
        return EXIT_OK
    if args.keys_command == "revoke":
        removed = store.revoke(key_id=args.key_id)
        stdout.write(
            json.dumps({"key_id": args.key_id, "revoked": removed}, ensure_ascii=False)
        )
        stdout.write("\n")
        return EXIT_OK if removed else EXIT_ARGUMENT_ERROR
    stdout.write(
        json.dumps(
            [
                {
                    "key_id": record.key_id,
                    "user_id": record.user_id,
                    "name": record.name,
                    "scopes": sorted(record.scopes),
                    "created_at": record.created_at.isoformat(),
                    "expires_at": (
                        record.expires_at.isoformat() if record.expires_at else None
                    ),
                    "last_used_at": (
                        record.last_used_at.isoformat()
                        if record.last_used_at
                        else None
                    ),
                    "revoked_at": (
                        record.revoked_at.isoformat() if record.revoked_at else None
                    ),
                }
                for record in store.list_keys(user_id=args.user_id)
            ],
            ensure_ascii=False,
        )
    )
    stdout.write("\n")
    return EXIT_OK


def _run_rederivation_evaluation(args, stdout) -> int:
    from career_agent.evaluation.rederivation import summarize_rederivations

    try:
        events = SQLiteTraceRecorder(
            Path(args.run_events_store).expanduser()
        ).list_conversation_events(
            user_id=args.user_id,
            conversation_id=args.session_id,
        )
        summary = summarize_rederivations(events)
        if summary.compaction_count == 0:
            reason = "no_compaction_observed"
        elif summary.tool_call_count == 0:
            reason = "no_tool_calls_observed"
        elif summary.post_compaction_tool_call_count == 0:
            reason = "no_post_compaction_tool_calls"
        elif not summary.measurable:
            reason = "no_pre_compaction_tool_calls"
        else:
            reason = None
        payload = {
            "state": (
                "rederivation_measured"
                if summary.measurable
                else "insufficient_rederivation_trace"
            ),
            "user_id": args.user_id,
            "session_id": args.session_id,
            "compaction_count": summary.compaction_count,
            "tool_call_count": summary.tool_call_count,
            "post_compaction_tool_call_count": (
                summary.post_compaction_tool_call_count
            ),
            "rederivation_count": (
                summary.rederivation_count if summary.measurable else None
            ),
            "reason": reason,
        }
        json.dump(payload, stdout, ensure_ascii=False, separators=(",", ":"))
        stdout.write("\n")
        return EXIT_OK
    except (OSError, sqlite3.Error, ValueError) as error:
        return _write_chat_error(
            error,
            stdout,
            code=EXIT_ARGUMENT_ERROR,
            next_action="Check the run-events store path and conversation identity.",
        )


def _run_memory_exposure_evaluation(args, stdout) -> int:
    from dataclasses import asdict

    from career_agent.evaluation.memory_metrics import summarize_memory_metrics

    try:
        events = SQLiteTraceRecorder(
            Path(args.run_events_store).expanduser()
        ).list_memory_events(
            user_id=args.user_id,
            conversation_id=args.session_id,
            max_events=args.max_events,
        )
        summary = summarize_memory_metrics(events)
        zombie_value = summary.zombie_exposure.value
        zombie_detected = (
            summary.zombie_exposure.measurable
            and isinstance(zombie_value, (int, float))
            and zombie_value > 0
        )
        exposure_measurable = (
            summary.supersedence_exposure.measurable
            or summary.zombie_exposure.measurable
        )
        json.dump(
            {
                "state": (
                    "zombie_exposure_detected"
                    if zombie_detected
                    else (
                        "memory_exposure_measured"
                        if exposure_measurable
                        else "insufficient_memory_version_binding"
                    )
                ),
                "zombie_status": (
                    "detected"
                    if zombie_detected
                    else (
                        "clear"
                        if summary.zombie_exposure.measurable
                        else "not_observed"
                    )
                ),
                "user_id": args.user_id,
                "session_id": args.session_id,
                **asdict(summary),
            },
            stdout,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        stdout.write("\n")
        return EXIT_OK
    except (OSError, sqlite3.Error, ValueError) as error:
        return _write_chat_error(
            error,
            stdout,
            code=EXIT_ARGUMENT_ERROR,
            next_action="Check the run-events store path and conversation identity.",
        )


def _run_memory_report(args, stdout) -> int:
    try:
        report = build_memory_report(
            Path(args.context_store),
            user_id=args.user_id,
            decay_policy=DecayPolicy(
                half_life_days=args.half_life_days,
                access_boost=args.access_boost,
                projection_threshold=args.projection_threshold,
            ),
            quarantine_stale_days=args.quarantine_stale_days,
        )
        json.dump(
            report.as_payload(),
            stdout,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        stdout.write("\n")
        return EXIT_OK
    except (OSError, sqlite3.Error, ValueError) as error:
        return _write_chat_error(
            error,
            stdout,
            code=EXIT_ARGUMENT_ERROR,
            next_action=(
                "Check the context-store path, user identity, and decay policy."
            ),
        )


def _run_memory_file_command(args, stdout) -> int:
    try:
        context_path = Path(args.context_store).expanduser()
        context = CareerContextStore(context_path)
        history = CareerHistoryStore(Path(args.resume_store).expanduser())
        service = MemoryReviewService(
            context_store=context,
            career_history_store=history,
        )
        if args.memory_command == "export":
            output = args.output.expanduser()
            if output.exists() and not args.force:
                raise ValueError(
                    "output file exists; pass --force to replace it"
                )
            if not output.parent.exists():
                raise ValueError("output directory does not exist")
            markdown, export_id, item_count = service.export(
                user_id=args.user_id
            )
            temporary = output.with_name(f".{output.name}.tmp")
            temporary.write_text(markdown, encoding="utf-8")
            os.chmod(temporary, 0o600)
            temporary.replace(output)
            payload = {
                "state": "memory_review_exported",
                "export_id": export_id,
                "item_count": item_count,
                "path": str(output.resolve()),
            }
        else:
            markdown = args.file.expanduser().read_text(encoding="utf-8")
            if not markdown or len(markdown) > 200_000:
                raise ValueError(
                    "MEMORY.md must contain between 1 and 200000 characters"
                )
            prepared = service.prepare(
                user_id=args.user_id,
                markdown=markdown,
            )
            if args.memory_command == "review":
                payload = {
                    "state": "memory_review_ready",
                    "export_id": prepared.analysis.export_id,
                    "proposal_count": prepared.analysis.proposal_count,
                    "confirmation_digest": prepared.confirmation_digest,
                    "changes": [
                        item.model_dump(mode="json")
                        for item in prepared.changes
                    ],
                    "warnings": prepared.analysis.warnings,
                    "conflicts": prepared.analysis.conflicts,
                    "next_action": (
                        "Run memory apply with --confirm and this "
                        "confirmation_digest after reviewing every change."
                    ),
                }
            else:
                if not args.confirm:
                    raise ValueError(
                        "memory apply requires --confirm after reviewing the full diff"
                    )
                notes_root = args.working_notes_dir or context_path.with_name(
                    "working-notes"
                )
                applied = service.apply(
                    user_id=args.user_id,
                    markdown=markdown,
                    confirmation_digest=args.confirmation_digest,
                    working_notes=WorkingNotesStore(notes_root),
                )
                payload = {
                    "state": (
                        "memory_review_applied_cleanup_incomplete"
                        if applied.cleanup_incomplete
                        else "memory_review_applied"
                    ),
                    **applied.model_dump(mode="json"),
                }
        json.dump(payload, stdout, ensure_ascii=False, separators=(",", ":"))
        stdout.write("\n")
        return EXIT_OK
    except (OSError, sqlite3.Error, ValueError) as error:
        return _write_chat_error(
            error,
            stdout,
            code=EXIT_ARGUMENT_ERROR,
            next_action="Re-export MEMORY.md, review the complete diff, then retry.",
        )


def _run_trajectory_evaluation(args, stdout) -> int:
    """Replay the scenario catalogue, or re-cut it against the live model.

    Reports each scenario's contract result separately from its replay result,
    because the two mean different things and collapsing them would let a green
    run be read as "the model behaves" when no cassette exists.
    """
    from dataclasses import replace as _replace

    from career_agent.evaluation.main_agent_scenarios import SCENARIOS
    from career_agent.evaluation.trajectory import (
        cassette_staleness,
        check_contract,
        load_cassette,
        record_catalogue,
        replay_cassette,
        replay_quality,
        known_gap_reproduction,
        minimum_detectable_regression,
        quality_shortfall,
    )

    try:
        schemas = _trajectory_tool_specs()
        selected = (
            tuple(item for item in SCENARIOS if item.name in set(args.scenario))
            if args.scenario
            else SCENARIOS
        )
        if args.scenario and len(selected) != len(set(args.scenario)):
            known = ", ".join(item.name for item in SCENARIOS)
            raise ValueError(f"unknown scenario; available: {known}")
        if args.record and args.samples is not None:
            required = max(
                (scenario.recording_samples for scenario in selected),
                default=1,
            )
            if args.samples < required:
                raise ValueError(
                    f"selected scenarios require at least {required} sample(s)"
                )
            mismatched_quality = tuple(
                scenario.name
                for scenario in selected
                if scenario.has_quality_assertions
                and args.samples != scenario.recording_samples
            )
            if mismatched_quality:
                raise ValueError(
                    "quality scenarios pin their sample denominator; omit "
                    "--samples or use their declared count: "
                    + ", ".join(mismatched_quality)
                )

        config = None
        recording_error: str | None = None
        if args.record:
            config = _replace(
                OpenAICompatibleAgentConfig.from_env(prefix="MAIN_AGENT"),
                timeout_seconds=args.main_agent_timeout_seconds,
            )
            try:
                record_catalogue(
                    selected,
                    tool_specs=schemas,
                    config=config,
                    sample_count=args.samples,
                    jobs=args.jobs,
                    force=args.force,
                )
            except AgentWorkerError as error:
                # The catalogue already wrote every scenario that finished.
                # Report the run instead of dying before the report: the
                # scenario that failed shows up below as stale or unrecorded,
                # and the exit status stays non-zero.
                recording_error = f"{error.code}: {error}"
        from dotenv import load_dotenv

        load_dotenv()
        expected_model = config.model if config is not None else os.environ.get("MAIN_AGENT_MODEL", "").strip()

        results = []
        for scenario in selected:
            contract = check_contract(scenario, tool_specs=schemas)
            cassette = load_cassette(scenario.name)
            stale = (
                cassette_staleness(
                    cassette,
                    scenario=scenario,
                    tool_specs=schemas,
                    expected_model=expected_model,
                )
                if cassette is not None
                else None
            )
            sample_failures = (
                replay_cassette(
                    scenario,
                    tool_specs=schemas,
                    cassette=cassette,
                )
                if cassette is not None and stale is None and not contract
                else ()
            )
            behaviour = tuple(
                f"{failure} (sample {sample_index}/{len(sample_failures)})"
                for sample_index, failures in enumerate(sample_failures, start=1)
                for failure in failures
            )
            graded = (
                replay_quality(
                    scenario,
                    tool_specs=schemas,
                    cassette=cassette,
                )
                if cassette is not None and stale is None and not contract
                else ()
            )
            replayable = cassette is not None and stale is None and not contract
            shortfall = (
                quality_shortfall(scenario, graded)
                if replayable and scenario.has_quality_assertions
                else None
            )
            quality_passing = (
                sum(not item for item in graded)
                if replayable and scenario.has_quality_assertions
                else None
            )
            quality_total = (
                len(graded)
                if replayable and scenario.has_quality_assertions
                else None
            )
            quality_rate = (
                quality_passing / quality_total
                if quality_passing is not None and quality_total
                else None
            )
            if not scenario.has_quality_assertions:
                quality_status = "not_applicable"
            elif cassette is None:
                quality_status = "unrecorded"
            elif stale is not None:
                quality_status = "stale"
            elif contract:
                quality_status = "blocked"
            else:
                quality_status = "failed" if shortfall is not None else "passed"
            failures = list(contract)
            if stale is not None:
                failures.append(f"{scenario.name}: {stale}")
            if shortfall is not None:
                failures.append(shortfall)
            results.append(
                {
                    "scenario": scenario.name,
                    "policy": scenario.policy,
                    "contract": "passed" if not contract else "failed",
                    "behaviour": (
                        "unrecorded"
                        if cassette is None
                        else (
                            "stale"
                            if stale is not None
                            else ("passed" if not behaviour else "failed")
                        )
                    ),
                    "sample_count": cassette.sample_count if cassette else 0,
                    "quality_status": quality_status,
                    "quality_min_pass_rate": scenario.quality_min_pass_rate,
                    "quality_min_detectable_regression": (
                        minimum_detectable_regression(
                            quality_total,
                            scenario.quality_min_pass_rate,
                        )
                        if quality_total and scenario.quality_min_pass_rate
                        else None
                    ),
                    "quality_samples_passed": quality_passing,
                    "quality_sample_count": quality_total,
                    "quality_pass_rate": quality_rate,
                    "samples_passed": (
                        sum(not failures for failures in sample_failures)
                        if stale is None and not contract
                        else 0
                    ),
                    "known_gap_status": (
                        known_gap_reproduction(sample_failures)
                        if scenario.known_gap is not None and replayable
                        else None
                    ),
                    "failures": failures + list(behaviour),
                }
            )

        unrecorded = sum(1 for item in results if item["behaviour"] == "unrecorded")
        failed = [item for item in results if item["failures"]]
        payload = {
            "scenarios": len(results),
            "contract_failed": sum(
                1 for item in results if item["contract"] == "failed"
            ),
            "behaviour_failed": sum(
                1 for item in results if item["behaviour"] == "failed"
            ),
            "quality_failed": sum(
                1 for item in results if item["quality_status"] == "failed"
            ),
            "stale": sum(1 for item in results if item["behaviour"] == "stale"),
            "unrecorded": unrecorded,
            **({"recording_error": recording_error} if recording_error else {}),
            # Said outright rather than left to be inferred from the counts: a
            # run with no cassettes is green and proves nothing about the model.
            "note": (
                "contract results say the scenario is well posed; hard behaviour "
                "uses pass^k; quality uses the declared observed-rate floor "
                "and reports the largest perfect-run drop that floor still "
                "cannot see (quality_min_detectable_regression)"
            ),
            "results": results,
        }
        json.dump(payload, stdout, ensure_ascii=False, separators=(",", ":"))
        stdout.write("\n")
        return EXIT_OK if not failed and recording_error is None else EXIT_ARGUMENT_ERROR
    except AgentConfigurationError as error:
        json.dump({"state": "failed", "error_code": error.code, "error_detail": str(error)}, stdout, ensure_ascii=False, separators=(",", ":"))
        stdout.write("\n")
        return EXIT_CONFIGURATION_ERROR
    except (OSError, ValueError) as error:
        json.dump({"state": "failed", "error_code": "EVAL_INPUT_ERROR", "error_detail": str(error)}, stdout, ensure_ascii=False, separators=(",", ":"))
        stdout.write("\n")
        return EXIT_ARGUMENT_ERROR


def _run_resume_job_match_evaluation(args, stdout) -> int:
    """Run the 096 boundary fixture against the configured live model."""
    try:
        from career_agent.evaluation.resume_job_match import run_boundary_suite, run_samples

        config = replace(
            OpenAICompatibleAgentConfig.from_env(prefix="RESUME_ANALYSIS_AGENT"),
            timeout_seconds=args.timeout_seconds,
        )
        if args.suite == "boundaries":
            from career_agent.evaluation.resume_job_match import BOUNDARY_CASES

            suite = run_boundary_suite(
                config,
                sample_count=args.samples,
                case_names=tuple(args.case) if args.case else None,
            )
            cases_by_name = {case.name: case for case in BOUNDARY_CASES}
            payload = {
                "state": "completed",
                "suite": "096_boundaries",
                "case_count": len(suite),
                "sample_count": args.samples,
                "cases_passed": sum(
                    all(item.error is None for item in samples)
                    for samples in suite.values()
                ),
                "cases": {
                    name: {
                        "expected_overall_fit": cases_by_name[name].expected_fit,
                        "expected_statuses": [
                            list(status) if isinstance(status, tuple) else status
                            for status in cases_by_name[name].expected_statuses
                        ],
                        "samples_passed": sum(item.error is None for item in samples),
                        "samples": [
                            {
                                "sample": item.index,
                                "passed": item.error is None,
                                "overall_fit": (
                                    item.result.overall_fit
                                    if item.result is not None else None
                                ),
                                "requirement_statuses": (
                                    [
                                        assessment.status
                                        for assessment in item.result.requirements
                                    ]
                                    if item.result is not None else []
                                ),
                                "summary": (
                                    item.result.summary
                                    if item.result is not None else None
                                ),
                                "error": item.error,
                            }
                            for item in samples
                        ],
                        "overall_fit_values": [
                            item.result.overall_fit
                            for item in samples if item.result is not None
                        ],
                    }
                    for name, samples in suite.items()
                },
            }
            success = payload["cases_passed"] == payload["case_count"]
        else:
            if args.case:
                raise ValueError("--case is only valid with --suite boundaries")
            samples = run_samples(config, sample_count=args.samples)
            payload = {
                "state": "completed",
                "suite": "smoke",
                "sample_count": len(samples),
                "samples_passed": sum(item.result is not None for item in samples),
                "overall_fit_values": [
                    item.result.overall_fit for item in samples if item.result is not None
                ],
                "errors": [
                    {"sample": item.index, "error": item.error}
                    for item in samples if item.error is not None
                ],
            }
            success = payload["samples_passed"] == payload["sample_count"]
        json.dump(payload, stdout, ensure_ascii=False, separators=(",", ":"))
        stdout.write("\n")
        return EXIT_OK if success else EXIT_WORKFLOW_ERROR
    except AgentConfigurationError as error:
        json.dump({"state": "failed", "error_code": error.code}, stdout, ensure_ascii=False, separators=(",", ":"))
        stdout.write("\n")
        return EXIT_CONFIGURATION_ERROR
    except (OSError, ValueError) as error:
        json.dump({"state": "failed", "error_code": "EVAL_INPUT_ERROR", "error_detail": str(error)}, stdout, ensure_ascii=False, separators=(",", ":"))
        stdout.write("\n")
        return EXIT_ARGUMENT_ERROR


def build_workspace_lock(args: argparse.Namespace) -> SingleWorkerLock:
    """The one lock the API, ``backup`` and every write command contend for.

    It sits beside the context store whatever other store paths a command was
    given, so databases spread over several directories still share it. Tests
    replace this to keep the CLI out of the developer's ``~/.career-agent``.
    """

    workspace_dir = Path(args.context_store).expanduser().parent
    return SingleWorkerLock(lock_path_for(workspace_dir))


_WRITE_SUBCOMMANDS: dict[str, tuple[str, frozenset[str]]] = {
    "actions": ("actions_command", frozenset({"settle"})),
    "memory": ("memory_command", frozenset({"apply"})),
    "settings": ("settings_command", frozenset({"set"})),
    "target-role": ("target_role_command", frozenset({"create"})),
    "resume": ("resume_command", frozenset({"import"})),
    "email": ("email_command", frozenset({"add-account"})),
    "calendar": ("calendar_command", frozenset({"add-account"})),
    "api-keys": ("keys_command", frozenset({"issue", "revoke"})),
}


def _writes_workspace(args: argparse.Namespace) -> bool:
    """Whether a command must hold the workspace lock. ``backup`` takes it
    itself; read-only commands and ``memory export`` (a file outside the
    workspace) need none."""

    if args.command == "chat":
        return True
    entry = _WRITE_SUBCOMMANDS.get(args.command)
    if entry is None:
        return False
    attribute, subcommands = entry
    return getattr(args, attribute) in subcommands


def main(
    argv: Sequence[str] | None = None,
    *,
    runtime_factory: Callable[[argparse.Namespace], MainAgentRuntime] | None = None,
    resume_store_factory: Callable[[argparse.Namespace], ResumeStore] | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    stdout = stdout or sys.stdout
    stderr = stderr or sys.stderr
    parser = build_parser()
    args = parser.parse_args(argv)
    if not _writes_workspace(args):
        return _dispatch(
            args,
            parser=parser,
            runtime_factory=runtime_factory,
            resume_store_factory=resume_store_factory,
            stdout=stdout,
            stderr=stderr,
        )
    lock = build_workspace_lock(args)
    try:
        lock.acquire()
    except SingleWorkerError as error:
        stdout.write(
            json.dumps(
                {
                    "error": (
                        "API 或另一个写命令正在使用该工作区，请先停止后再执行。"
                        f"{error}"
                    )
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        return EXIT_WORKFLOW_ERROR
    except OSError as error:
        stdout.write(
            json.dumps(
                {"error": f"无法创建或打开工作区锁 {lock.path}：{error}"},
                ensure_ascii=False,
            )
            + "\n"
        )
        return EXIT_CONFIGURATION_ERROR
    try:
        return _dispatch(
            args,
            parser=parser,
            runtime_factory=runtime_factory,
            resume_store_factory=resume_store_factory,
            stdout=stdout,
            stderr=stderr,
        )
    finally:
        lock.release()


def _dispatch(
    args: argparse.Namespace,
    *,
    parser: argparse.ArgumentParser,
    runtime_factory: Callable[[argparse.Namespace], MainAgentRuntime] | None,
    resume_store_factory: Callable[[argparse.Namespace], ResumeStore] | None,
    stdout: TextIO,
    stderr: TextIO,
) -> int:
    if args.command == "target-role":
        try:
            store = resume_store_factory(args) if resume_store_factory else ResumeStore(Path(args.resume_store).expanduser())
            if args.target_role_command == "create":
                payload = {"target_role": store.create_target_role(user_id=args.user_id, title=args.title, priority=args.priority).model_dump(mode="json")}
            else:
                payload = {"target_roles": [role.model_dump(mode="json") for role in store.list_target_roles(user_id=args.user_id)]}
            json.dump(payload, stdout, ensure_ascii=False, separators=(",", ":"))
            stdout.write("\n")
            return EXIT_OK
        except (OSError, ValueError) as error:
            json.dump({"state": "failed", "error_code": "TARGET_ROLE_INPUT_ERROR", "error_detail": str(error)}, stdout, ensure_ascii=False, separators=(",", ":"))
            stdout.write("\n")
            return EXIT_ARGUMENT_ERROR
        except Exception as error:
            json.dump({"state": "failed", "error_code": "TARGET_ROLE_STORE_ERROR", "error_detail": f"{type(error).__name__}: {error}"}, stdout, ensure_ascii=False, separators=(",", ":"))
            stdout.write("\n")
            return EXIT_UNKNOWN_ERROR
    if args.command == "resume":
        try:
            store = resume_store_factory(args) if resume_store_factory else ResumeStore(Path(args.resume_store).expanduser())
            if args.resume_command == "import":
                content, document_format = _read_resume_import(args.file)
                resume, version = store.import_document(user_id=args.user_id, name=args.name, resume_id=args.resume_id, target_role_id=args.target_role_id, content=content, document_format=document_format)
                payload = _resume_payload(resume, (version,))
            elif args.resume_command == "list":
                roles = store.list_target_roles(user_id=args.user_id)
                selected_roles = tuple(role for role in roles if not args.target_role_id or role.id == args.target_role_id)
                if args.target_role_id and not selected_roles:
                    raise ValueError("Target role not found.")
                payload = {"target_roles": [
                    {**role.model_dump(mode="json"), "resumes": [resume.model_dump(mode="json") for resume in store.list_resumes(user_id=args.user_id, target_role_id=role.id)]}
                    for role in selected_roles
                ]}
            else:
                resume = store.get_resume(user_id=args.user_id, resume_id=args.resume_id)
                if resume is None:
                    raise ValueError("Resume not found.")
                payload = _resume_payload(resume, store.list_versions(user_id=args.user_id, resume_id=args.resume_id))
            json.dump(payload, stdout, ensure_ascii=False, separators=(",", ":"))
            stdout.write("\n")
            return EXIT_OK
        except (OSError, ValueError) as error:
            json.dump({"state": "failed", "error_code": "RESUME_INPUT_ERROR", "error_detail": str(error)}, stdout, ensure_ascii=False, separators=(",", ":"))
            stdout.write("\n")
            return EXIT_ARGUMENT_ERROR
        except Exception as error:
            json.dump({"state": "failed", "error_code": "RESUME_STORE_ERROR", "error_detail": f"{type(error).__name__}: {error}"}, stdout, ensure_ascii=False, separators=(",", ":"))
            stdout.write("\n")
            return EXIT_UNKNOWN_ERROR
    if args.command == "job":
        try:
            repository = SQLiteJobPostingRepository(Path(args.job_store).expanduser())
            if args.job_command == "list":
                payload = {"jobs": [_stored_job_summary_payload(item) for item in repository.list_jobs(user_id=args.user_id, limit=args.limit, include_dismissed=False)]}
            elif args.job_command == "find":
                payload = {"jobs": [_stored_job_summary_payload(item) for item in repository.search_saved_jobs(user_id=args.user_id, query=args.query, limit=args.limit, include_dismissed=False)]}
            else:
                by_posting = bool(args.job_posting_id)
                by_run = bool(args.run_id or args.selection_index is not None)
                if by_posting == by_run:
                    raise ValueError("Use either --job-posting-id or both --run-id and --selection-index.")
                if by_run and (not args.run_id or args.selection_index is None or args.selection_index < 1):
                    raise ValueError("--run-id requires a positive --selection-index.")
                record = (
                    repository.get_job(user_id=args.user_id, job_posting_id=args.job_posting_id)
                    if by_posting
                    else repository.get_for_run(user_id=args.user_id, run_id=args.run_id, selection_index=args.selection_index)
                )
                if record is None:
                    raise ValueError("Persisted job not found for this user.")
                payload = _stored_job_payload(record)
            json.dump(payload, stdout, ensure_ascii=False, separators=(",", ":"))
            stdout.write("\n")
            return EXIT_OK
        except (OSError, ValueError) as error:
            json.dump({"state": "failed", "error_code": "JOB_STORE_INPUT_ERROR", "error_detail": str(error)}, stdout, ensure_ascii=False, separators=(",", ":"))
            stdout.write("\n")
            return EXIT_ARGUMENT_ERROR
        except Exception as error:
            json.dump({"state": "failed", "error_code": "JOB_STORE_ERROR", "error_detail": f"{type(error).__name__}: {error}"}, stdout, ensure_ascii=False, separators=(",", ":"))
            stdout.write("\n")
            return EXIT_UNKNOWN_ERROR
    if args.command == "email":
        try:
            store = SQLiteEmailTrackingStore(Path(args.email_store).expanduser())
            if args.email_command == "add-account":
                if not args.credential_env.replace("_", "").isalnum():
                    raise ValueError("--credential-env must be an environment variable name")
                account = store.add_account(
                    user_id=args.user_id,
                    provider=args.provider,
                    email_address=args.address,
                    credential_ref=f"env:{args.credential_env}",
                )
                payload = {"account": {
                    "email_account_id": account.id,
                    "provider": account.provider,
                    "email_address": account.email_address,
                    "status": account.status,
                }}
            else:
                payload = {"accounts": [
                    {
                        "email_account_id": account.id,
                        "provider": account.provider,
                        "email_address": account.email_address,
                        "status": account.status,
                    }
                    for account in store.list_accounts(user_id=args.user_id)
                ]}
            json.dump(payload, stdout, ensure_ascii=False, separators=(",", ":"))
            stdout.write("\n")
            return EXIT_OK
        except (OSError, ValueError) as error:
            json.dump({"state": "failed", "error_code": "EMAIL_ACCOUNT_INPUT_ERROR", "error_detail": str(error)}, stdout, ensure_ascii=False, separators=(",", ":"))
            stdout.write("\n")
            return EXIT_ARGUMENT_ERROR
    if args.command == "calendar":
        try:
            store = SQLiteCalendarStore(Path(args.calendar_store).expanduser())
            if args.calendar_command == "add-account":
                if not args.credential_env.replace("_", "").isalnum():
                    raise ValueError(
                        "--credential-env must be an environment variable name"
                    )
                account = store.add_account(
                    user_id=args.user_id,
                    email_address=args.address,
                    calendar_id=args.calendar_id,
                    credential_ref=f"env:{args.credential_env}",
                )
                payload = {
                    "account": {
                        "calendar_account_id": account.id,
                        "provider": account.provider,
                        "email_address": account.email_address,
                        "calendar_id": account.calendar_id,
                        "status": account.status,
                    }
                }
            else:
                payload = {
                    "accounts": [
                        {
                            "calendar_account_id": account.id,
                            "provider": account.provider,
                            "email_address": account.email_address,
                            "calendar_id": account.calendar_id,
                            "status": account.status,
                        }
                        for account in store.list_accounts(user_id=args.user_id)
                    ]
                }
            json.dump(
                payload, stdout, ensure_ascii=False, separators=(",", ":")
            )
            stdout.write("\n")
            return EXIT_OK
        except (OSError, ValueError) as error:
            json.dump(
                {
                    "state": "failed",
                    "error_code": "CALENDAR_ACCOUNT_INPUT_ERROR",
                    "error_detail": str(error),
                },
                stdout,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            stdout.write("\n")
            return EXIT_ARGUMENT_ERROR
    if args.command == "api-keys":
        return _run_api_keys(args, stdout)
    if args.command == "backup":
        return _run_backup(args, stdout)
    if args.command == "actions" and args.actions_command == "settle":
        return _run_action_settle(args, stdout)
    if args.command == "actions":
        try:
            pending = SQLiteActionExecutionStore(
                Path(args.context_store).expanduser()
            ).list_pending(user_id=args.user_id)
            json.dump(
                {
                    "state": "action_reconciliation_required"
                    if pending
                    else "no_action_reconciliation_required",
                    "items": [
                        {
                            "action_id": item.action_id,
                            "conversation_id": item.conversation_id,
                            "tool": item.tool_name,
                            "status": item.status,
                            "retry_safe": item.retry_safe,
                            "started_at": item.started_at.isoformat(),
                        }
                        for item in pending
                    ],
                },
                stdout,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            stdout.write("\n")
            return EXIT_OK
        except (OSError, ValueError) as error:
            return _write_chat_error(
                error,
                stdout,
                code=EXIT_ARGUMENT_ERROR,
                next_action="Check the action store path and user identity.",
            )
    if args.command == "eval":
        if args.eval_command == "rederivation":
            return _run_rederivation_evaluation(args, stdout)
        if args.eval_command == "memory-exposure":
            return _run_memory_exposure_evaluation(args, stdout)
        if args.eval_command == "resume-job-match":
            return _run_resume_job_match_evaluation(args, stdout)
        return _run_trajectory_evaluation(args, stdout)
    if args.command == "memory":
        if args.memory_command == "report":
            return _run_memory_report(args, stdout)
        return _run_memory_file_command(args, stdout)
    if args.command == "settings":
        context_store = CareerContextStore(Path(args.context_store).expanduser())
        manager = ContextManager(context_store)
        current = manager.preferences(user_id=args.user_id)
        if args.settings_command == "history":
            stdout.write(
                json.dumps(
                    {
                        "user_id": args.user_id,
                        "events": [
                            event.model_dump(mode="json")
                            for event in context_store.list_owner_settings_events(
                                user_id=args.user_id
                            )
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            return EXIT_OK
        if args.settings_command == "set":
            if (
                args.application_confirmation is None
                and args.boss_search is None
                and args.confirm_before is None
            ):
                stderr.write("settings set needs at least one rule to change\n")
                return EXIT_ARGUMENT_ERROR
            confirm_before = current.behavior_policy.confirm_before
            if args.confirm_before is not None:
                try:
                    confirm_before = canonical_confirm_before(
                        tuple(
                            item.strip()
                            for item in args.confirm_before.split(",")
                            if item.strip()
                        )
                    )
                except ValueError as error:
                    stderr.write(f"{error}\n")
                    return EXIT_ARGUMENT_ERROR
            desired = current.model_copy(
                update={
                    "preferences": current.preferences.model_copy(
                        update={
                            "boss_search": args.boss_search
                            or current.preferences.boss_search
                        }
                    ),
                    "behavior_policy": current.behavior_policy.model_copy(
                        update={
                            "application_confirmation": (
                                args.application_confirmation
                                or current.behavior_policy.application_confirmation
                            ),
                            "confirm_before": confirm_before,
                        }
                    ),
                }
            )
            current = manager.update_owner_settings(
                user_id=args.user_id,
                desired=desired,
                expected_revision=current.revision,
                actor_type="cli",
                actor_id="local-cli",
            )
        stdout.write(
            json.dumps(
                {"user_id": args.user_id, "owner_settings": current.model_dump()},
                ensure_ascii=False,
            )
            + "\n"
        )
        return EXIT_OK
    if args.command == "chat":
        try:
            interaction_response = _chat_interaction_response(args)
            user_message = _chat_user_message(args, interaction_response)
        except ValueError as error:
            parser.error(str(error))
        runtime = None
        try:
            runtime = runtime_factory(args) if runtime_factory else build_main_agent_runtime(args)
            turn = runtime.run_turn(
                user_id=args.user_id,
                conversation_id=args.session_id,
                user_message=user_message,
                request_id=args.request_id,
                **(
                    {"interaction_response": interaction_response}
                    if interaction_response is not None
                    else {}
                ),
            )
            if isinstance(turn, ReplayedTurn):
                return _write_chat_replay(
                    turn,
                    user_id=args.user_id,
                    session_id=args.session_id,
                    output=stdout,
                )
            return _write_chat_payload(
                turn,
                user_id=args.user_id,
                session_id=args.session_id,
                output=stdout,
            )
        except AgentConfigurationError as error:
            return _write_chat_error(error, stdout, code=EXIT_CONFIGURATION_ERROR, next_action="Set MAIN_AGENT_* and the configured specialist-agent environment variables.")
        except AgentWorkerError as error:
            return _write_chat_error(error, stdout, code=EXIT_WORKFLOW_ERROR, next_action="Retry later or inspect the model configuration.")
        except (OSError, ValueError) as error:
            return _write_chat_error(error, stdout, code=EXIT_ARGUMENT_ERROR, next_action="Check the command help and session parameters.")
        except Exception as error:
            return _write_chat_error(error, stdout, code=EXIT_UNKNOWN_ERROR)
        finally:
            close = getattr(runtime, "close", None)
            if close is not None:
                close()
    parser.error(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
