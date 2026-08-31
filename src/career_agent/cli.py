from __future__ import annotations

import argparse
import json
import os
import stat
import sys
from dataclasses import replace
from pathlib import Path
from typing import Callable, Sequence, TextIO

from career_agent.agent.context_manager import ContextManager
from career_agent.agent.answer_writer import OpenAIStreamingAnswerWriter
from career_agent.agent.career_context import CareerContextProjector
from career_agent.agent.main_agent_contracts import ToolObservation
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
from career_agent.services.resume_tailoring import ResumeTailoringService
from career_agent.storage.context import CareerContextStore
from career_agent.storage.checkpoints import SQLiteCheckpointOwner
from career_agent.storage.applications import SQLiteApplicationStore
from career_agent.storage.action_center import SQLiteActionItemStore
from career_agent.storage.calendar import SQLiteCalendarStore
from career_agent.storage.email_tracking import SQLiteEmailTrackingStore
from career_agent.storage.interviews import SQLiteInterviewStore
from career_agent.storage.interview_preparations import SQLiteInterviewPreparationStore
from career_agent.storage.job_research import SQLiteJobResearchStore
from career_agent.storage.mock_interviews import SQLiteMockInterviewStore
from career_agent.storage.career_history import CareerHistoryStore
from career_agent.storage.jobs import SQLiteJobPostingRepository, StoredJobRecord, StoredJobSummary
from career_agent.storage.resumes import ResumeStore
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
MAX_RESUME_IMPORT_BYTES = 1_048_576


def build_main_agent_runtime(args: argparse.Namespace) -> MainAgentRuntime:
    main_config = replace(OpenAICompatibleAgentConfig.from_env(prefix="MAIN_AGENT"), timeout_seconds=args.main_agent_timeout_seconds)
    context_store = CareerContextStore(Path(args.context_store).expanduser())
    context_manager = ContextManager(
        context_store,
        summary_worker=OpenAIConversationSummaryWorker(main_config),
        compacted_message_warning_threshold=args.compacted_message_warning,
    )
    resume_analysis_config = replace(
        OpenAICompatibleAgentConfig.from_env(prefix="RESUME_ANALYSIS_AGENT"),
        timeout_seconds=args.agent_timeout_seconds,
    )
    resume_store = ResumeStore(Path(args.resume_store).expanduser())
    career_history_store = CareerHistoryStore(Path(args.resume_store).expanduser())
    job_repository = SQLiteJobPostingRepository(Path(args.job_store).expanduser())
    match_store = SQLiteResumeJobMatchStore(Path(args.resume_store).expanduser())
    application_service = ApplicationService(
        SQLiteApplicationStore(Path(args.application_store).expanduser()),
        job_repository,
        resume_store,
    )
    interview_service = InterviewService(
        SQLiteInterviewStore(Path(args.application_store).expanduser()),
        application_service,
    )
    email_tracking_service = EmailTrackingService(
        SQLiteEmailTrackingStore(Path(args.email_store).expanduser()),
        application_service,
        EnvironmentEmailConnectorResolver(),
        OpenAIEmailTrackingWorker(resume_analysis_config),
        interview_service,
    )
    action_center_service = ActionCenterService(
        SQLiteActionItemStore(Path(args.action_store).expanduser()),
        application_service,
        email_tracking_service,
        interview_service,
    )
    calendar_service = CalendarService(
        SQLiteCalendarStore(Path(args.calendar_store).expanduser()),
        interview_service,
        application_service,
        EnvironmentCalendarConnectorResolver(),
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
    job_research_service = JobResearchService(
        jobs=job_repository,
        store=SQLiteJobResearchStore(Path(args.job_research_store).expanduser()),
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
        decision_maker=OpenAICompatibleMainAgentDecisionMaker(main_config),
        answer_writer=OpenAIStreamingAnswerWriter(main_config),
        career_context_projector=CareerContextProjector(career_history_store),
        owned_resources=(mock_checkpoint_owner, job_research_checkpoint_owner),
        tools=MainAgentToolRegistry(
            job_repository=job_repository,
            job_research_service=job_research_service,
            resume_store=resume_store,
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
            resume_job_match_service=ResumeJobMatchService(
                resume_store,
                job_repository,
                career_history_store,
                OpenAIResumeJobMatchWorker(resume_analysis_config),
                match_store,
            ),
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
    if path.is_symlink() or not path.is_file() or not stat.S_ISREG(path.stat().st_mode):
        raise ValueError("--file must be a regular non-symlink file.")
    suffix = path.suffix.casefold()
    document_format = "pdf" if suffix == ".pdf" else "text" if suffix == ".txt" else "markdown" if suffix in {".md", ".markdown"} else None
    if document_format is None:
        raise ValueError("Resume import supports only .pdf, .txt, .md, and .markdown files.")
    if path.stat().st_size > 5 * MAX_RESUME_IMPORT_BYTES:
        raise ValueError("Resume file exceeds the 5 MiB import limit.")
    content = path.read_bytes()
    if not content:
        raise ValueError("Resume file must not be empty.")
    if document_format == "pdf":
        if not content.startswith(b"%PDF-"):
            raise ValueError("Resume PDF has an invalid header.")
        try:
            from pypdf import PdfReader
            from io import BytesIO

            reader = PdfReader(BytesIO(content), strict=False)
            if reader.is_encrypted:
                raise ValueError("Encrypted resume PDFs are not supported.")
            if not reader.pages:
                raise ValueError("Resume PDF contains no pages.")
        except ValueError:
            raise
        except Exception as error:
            raise ValueError("Resume PDF is malformed or unreadable.") from error
        return content, document_format
    if b"\x00" in content:
        raise ValueError("Resume text must not contain NUL bytes.")
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("Resume text must be UTF-8 encoded.") from error
    if not text.strip():
        raise ValueError("Resume text must contain non-whitespace content.")
    return content, document_format


def _resume_payload(resume, versions=()) -> dict[str, object]:
    payload = {"resume": resume.model_dump(mode="json")}
    if versions:
        payload["versions"] = [version.model_dump(mode="json") for version in versions]
    return payload


def _add_runtime_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--agent-timeout-seconds", type=float, default=300.0, help="Model call timeout (default: 300).")
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
    chat.add_argument("--context-store", default="~/.career-agent/context.sqlite3", help="Local session and context store path.")
    chat.add_argument("--compacted-message-warning", type=int, default=200, help="Warn once this many summarised originals are still stored. They are never deleted automatically; use 'context prune'.")
    chat.add_argument("--main-agent-timeout-seconds", type=float, default=60.0, help="Main Agent model timeout (default: 60).")
    _add_runtime_options(chat)

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
            "Ask the live model and overwrite the cassettes. Needs MAIN_AGENT_* "
            "configured, and costs one model call per scenario step."
        ),
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

    context_command = subparsers.add_parser(
        "context",
        help="Inspect and reclaim summarised conversation history.",
        description=(
            "Summarising a conversation keeps the original messages. They are no "
            "longer read, but they remain the only way to check a summary that "
            "looks wrong. Deleting them is therefore never automatic."
        ),
    )
    context_subparsers = context_command.add_subparsers(
        dest="context_command", required=True
    )
    context_stat = context_subparsers.add_parser(
        "stat", help="Report how much summarised history is still stored."
    )
    context_prune = context_subparsers.add_parser(
        "prune",
        help="Delete summarised originals for this user, permanently.",
        description=(
            "Deletes only messages a stored summary already covers. Irreversible: "
            "after this the summary is the only record of those turns."
        ),
    )
    context_prune.add_argument(
        "--yes",
        action="store_true",
        help="Required. Confirms the deletion cannot be undone.",
    )
    for sub in (context_stat, context_prune):
        sub.add_argument("--user-id", required=True, help="User whose history to act on.")
        sub.add_argument(
            "--session-id",
            help="Limit to one conversation. Omit to cover every conversation.",
        )
        sub.add_argument(
            "--context-store",
            default="~/.career-agent/context.sqlite3",
            help="Local session and context store path.",
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
    payload = {
        "state": "failed" if failed_result else "completed",
        "user_id": user_id,
        "session_id": session_id,
        "assistant_message": turn.assistant_message,
        "decision": {
            "action": turn.decision.action,
            "tool_name": turn.decision.tool_call.name if turn.decision.tool_call else None,
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
    has to match production is the schema list: a scenario that forbids a tool
    the model was never offered proves nothing.
    """
    from career_agent.agent.main_agent_tools import MainAgentToolRegistry

    parameters = (
        "job_repository", "job_comparison_service", "career_profile_store",
        "resume_store", "resume_analysis_service", "resume_job_match_service",
        "resume_tailoring_service", "resume_export_service",
        "application_service", "email_tracking_service", "interview_service",
        "interview_preparation_service", "action_center_service",
        "calendar_service", "mock_interview_graph", "mock_interview_store",
        "job_research_service",
    )
    return MainAgentToolRegistry(**{name: object() for name in parameters}).schemas()


def _run_trajectory_evaluation(args, stdout) -> int:
    """Replay the scenario catalogue, or re-cut it against the live model.

    Reports each scenario's contract result separately from its replay result,
    because the two mean different things and collapsing them would let a green
    run be read as "the model behaves" when no cassette exists.
    """
    from dataclasses import replace as _replace

    from career_agent.evaluation.main_agent_scenarios import SCENARIOS
    from career_agent.evaluation.trajectory import (
        check_contract,
        load_cassette,
        record,
        replay,
    )

    try:
        schemas = _trajectory_tool_specs()
        offered = frozenset(spec["function"]["name"] for spec in schemas)
        selected = (
            tuple(item for item in SCENARIOS if item.name in set(args.scenario))
            if args.scenario
            else SCENARIOS
        )
        if args.scenario and len(selected) != len(set(args.scenario)):
            known = ", ".join(item.name for item in SCENARIOS)
            raise ValueError(f"unknown scenario; available: {known}")

        config = None
        if args.record:
            config = _replace(
                OpenAICompatibleAgentConfig.from_env(prefix="MAIN_AGENT"),
                timeout_seconds=args.main_agent_timeout_seconds,
            )

        results = []
        for scenario in selected:
            contract = check_contract(scenario, offered_tools=offered)
            if args.record and not contract:
                record(scenario, tool_specs=schemas, config=config)
            responses = load_cassette(scenario.name)
            behaviour = (
                replay(scenario, tool_specs=schemas, responses=responses)
                if responses is not None and not contract
                else ()
            )
            results.append(
                {
                    "scenario": scenario.name,
                    "policy": scenario.policy,
                    "contract": "passed" if not contract else "failed",
                    "behaviour": (
                        "unrecorded"
                        if responses is None
                        else ("passed" if not behaviour else "failed")
                    ),
                    "failures": list(contract) + list(behaviour),
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
            "unrecorded": unrecorded,
            # Said outright rather than left to be inferred from the counts: a
            # run with no cassettes is green and proves nothing about the model.
            "note": (
                "contract results say the scenario is well posed; only recorded "
                "behaviour evaluates the model"
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
                payload = {"jobs": [_stored_job_summary_payload(item) for item in repository.list_jobs(user_id=args.user_id, limit=args.limit)]}
            elif args.job_command == "find":
                payload = {"jobs": [_stored_job_summary_payload(item) for item in repository.search_saved_jobs(user_id=args.user_id, query=args.query, limit=args.limit)]}
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
    if args.command == "eval":
        return _run_trajectory_evaluation(args, stdout)
    if args.command == "context":
        try:
            store = CareerContextStore(Path(args.context_store).expanduser())
            count, byte_size = store.count_compacted_messages(
                user_id=args.user_id, conversation_id=args.session_id
            )
            if args.context_command == "stat":
                payload = {
                    "compacted_messages": count,
                    "compacted_bytes": byte_size,
                    "reclaimable": count > 0,
                }
            elif not args.yes:
                # Refuse rather than prompt: this path has to work the same way
                # when it is driven by a script as when a person runs it.
                raise ValueError(
                    "context prune permanently deletes summarised messages; "
                    "pass --yes to confirm."
                )
            else:
                deleted = store.prune_compacted_messages(
                    user_id=args.user_id, conversation_id=args.session_id
                )
                payload = {"deleted_messages": deleted, "reclaimed_bytes": byte_size}
            json.dump(payload, stdout, ensure_ascii=False, separators=(",", ":"))
            stdout.write("\n")
            return EXIT_OK
        except (OSError, ValueError) as error:
            json.dump({"state": "failed", "error_code": "CONTEXT_STORE_INPUT_ERROR", "error_detail": str(error)}, stdout, ensure_ascii=False, separators=(",", ":"))
            stdout.write("\n")
            return EXIT_ARGUMENT_ERROR
        except Exception as error:
            json.dump({"state": "failed", "error_code": "CONTEXT_STORE_ERROR", "error_detail": f"{type(error).__name__}: {error}"}, stdout, ensure_ascii=False, separators=(",", ":"))
            stdout.write("\n")
            return EXIT_UNKNOWN_ERROR
    if args.command == "chat":
        runtime = None
        try:
            runtime = runtime_factory(args) if runtime_factory else build_main_agent_runtime(args)
            turn = runtime.run_turn(user_id=args.user_id, conversation_id=args.session_id, user_message=args.message)
            manager = getattr(runtime, "context_manager", None)
            notice = (
                manager.compacted_message_notice(user_id=args.user_id)
                if manager is not None
                else None
            )
            return _write_chat_payload(turn, user_id=args.user_id, session_id=args.session_id, output=stdout, notice=notice)
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
