from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from dataclasses import replace
from pathlib import Path
from typing import Callable, Sequence, TextIO

from career_agent.agent.context_manager import ContextManager
from career_agent.agent.career_context import CareerContextProjector
from career_agent.agent.semantic_career_retrieval import optional_semantic_retriever
from career_agent.agent.main_agent_contracts import (
    ToolObservation,
)
from career_agent.agent.main_agent_runtime import MainAgentRuntime
from career_agent.agent.main_agent_tools import MainAgentToolRegistry
from career_agent.agent.mock_interview_graph import (
    MockInterviewGraph,
    StoredMockInterviewSourceProvider,
)
from career_agent.agent.mock_interview_skill_loader import MockInterviewSkillLoader
from career_agent.agent.openai_compatible_client import AgentConfigurationError, AgentWorkerError, OpenAICompatibleAgentConfig
from career_agent.agent.openai_compatible_main_agent import OpenAICompatibleMainAgentDecisionMaker
from career_agent.agent.openai_conversation_summary_worker import OpenAIConversationSummaryWorker
from career_agent.agent.openai_resume_analysis_worker import OpenAIResumeAnalysisWorker
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
from career_agent.services.resume_analysis import ResumeAnalysisService
from career_agent.services.resume_export import ResumeExportService
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
from career_agent.storage.episodes import SQLiteCareerEpisodeStore
from career_agent.storage.interviews import SQLiteInterviewStore
from career_agent.storage.interview_preparations import SQLiteInterviewPreparationStore
from career_agent.storage.job_research import SQLiteJobResearchStore
from career_agent.storage.mock_interviews import SQLiteMockInterviewStore
from career_agent.storage.career_history import CareerHistoryStore
from career_agent.storage.jobs import SQLiteJobPostingRepository, StoredJobRecord, StoredJobSummary
from career_agent.storage.resumes import ResumeStore
from career_agent.storage.run_events import SQLiteTraceRecorder
from career_agent.storage.capability_confirmations import (
    CapabilityConfirmationSettledError,
    SQLiteCapabilityConfirmationStore,
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


EXIT_OK = 0
EXIT_ARGUMENT_ERROR = 2
EXIT_CONFIGURATION_ERROR = 3
EXIT_WORKFLOW_ERROR = 5
EXIT_UNKNOWN_ERROR = 6


def build_main_agent_runtime(args: argparse.Namespace) -> MainAgentRuntime:
    main_config = replace(OpenAICompatibleAgentConfig.from_env(prefix="MAIN_AGENT"), timeout_seconds=args.main_agent_timeout_seconds)
    context_store = CareerContextStore(Path(args.context_store).expanduser())
    resume_store = ResumeStore(Path(args.resume_store).expanduser())
    context_manager = ContextManager(
        context_store,
        summary_worker=OpenAIConversationSummaryWorker(main_config),
        target_role_source=resume_store,
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
            resume_analysis_config,
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
        decision_maker=OpenAICompatibleMainAgentDecisionMaker(main_config),
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
        owned_resources=(mock_checkpoint_owner, job_research_checkpoint_owner),
        tools=MainAgentToolRegistry(
            job_repository=job_repository,
            job_research_service=job_research_service,
            resume_store=resume_store,
            career_history_store=career_history_store,
            episode_store=SQLiteCareerEpisodeStore(
                Path(args.context_store).expanduser()
            ),
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
                OpenAIResumeAnalysisWorker(resume_analysis_config),
                SQLiteResumeAnalysisDraftStore(Path(args.resume_store).expanduser()),
                career_history_store,
            ),
            job_comparison_service=JobComparisonService(job_repository, match_store),
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="career-agent", description="Run the local Career Agent application.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    chat = subparsers.add_parser("chat", help="Send one natural-language turn to the Main Agent.", description="Run one non-interactive Main Agent turn. The agent decides whether to enter a registered workflow.", epilog="Example: career-agent chat --user-id u1 --session-id s1 --message 'Help me find AI Engineer jobs'")
    chat.add_argument("--user-id", required=True, help="Stable user identifier.")
    chat.add_argument("--session-id", required=True, help="Conversation session identifier.")
    chat.add_argument("--message", required=True, help="Current user message.")
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
    list_keys = keys_subparsers.add_parser("list", help="Show keys without secrets.")
    list_keys.add_argument("--user-id", default=None)
    list_keys.add_argument("--api-key-store", default="data/api_keys.sqlite3")
    revoke_keys = keys_subparsers.add_parser("revoke", help="Retire one key.")
    revoke_keys.add_argument("--key-id", required=True)
    revoke_keys.add_argument("--api-key-store", default="data/api_keys.sqlite3")

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
        # Addressed to whoever runs the CLI, not to the agent: it never entered
        # the model's context, so the model cannot act on it.
        "maintenance_notice": notice,
    }
    json.dump(payload, output, ensure_ascii=False, separators=(",", ":"))
    output.write("\n")
    return _failure_exit_code(failed_result) if failed_result else EXIT_OK


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

    Scenarios never execute a tool, so the services only have to exist. What
    has to match production is the fixed schema universe offered on every
    decision. Runtime handler preconditions still apply when a tool is actually
    executed; there is no task-scoped model-window filter to reproduce here.
    """
    from career_agent.agent.main_agent_tools import MainAgentToolRegistry

    parameters = (
        "job_repository", "job_comparison_service", "career_profile_store",
        "resume_store", "resume_analysis_service", "resume_job_match_service",
        "resume_tailoring_service", "resume_export_service",
        "application_service", "email_tracking_service", "interview_service",
        "interview_preparation_service", "action_center_service",
        "calendar_service", "mock_interview_graph", "mock_interview_store",
        "job_research_service", "conversation_store", "episode_store",
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
        if args.record:
            config = _replace(
                OpenAICompatibleAgentConfig.from_env(prefix="MAIN_AGENT"),
                timeout_seconds=args.main_agent_timeout_seconds,
            )
            record_catalogue(
                selected,
                tool_specs=schemas,
                config=config,
                sample_count=args.samples,
                jobs=args.jobs,
                force=args.force,
            )

        results = []
        for scenario in selected:
            contract = check_contract(scenario, tool_specs=schemas)
            cassette = load_cassette(scenario.name)
            stale = (
                cassette_staleness(
                    cassette,
                    scenario=scenario,
                    tool_specs=schemas,
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
        return EXIT_OK if not failed else EXIT_ARGUMENT_ERROR
    except AgentConfigurationError as error:
        json.dump({"state": "failed", "error_code": error.code, "error_detail": str(error)}, stdout, ensure_ascii=False, separators=(",", ":"))
        stdout.write("\n")
        return EXIT_CONFIGURATION_ERROR
    except (OSError, ValueError) as error:
        json.dump({"state": "failed", "error_code": "EVAL_INPUT_ERROR", "error_detail": str(error)}, stdout, ensure_ascii=False, separators=(",", ":"))
        stdout.write("\n")
        return EXIT_ARGUMENT_ERROR


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
        return _run_trajectory_evaluation(args, stdout)
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
            if args.application_confirmation is None and args.boss_search is None:
                stderr.write("settings set needs at least one rule to change\n")
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
                            )
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
        runtime = None
        try:
            runtime = runtime_factory(args) if runtime_factory else build_main_agent_runtime(args)
            turn = runtime.run_turn(
                user_id=args.user_id,
                conversation_id=args.session_id,
                user_message=args.message,
                request_id=args.request_id,
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
