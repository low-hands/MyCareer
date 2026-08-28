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
from career_agent.agent.career_context import CareerContextProjector
from career_agent.agent.job_discovery_contracts import JobDiscoveryRequest
from career_agent.agent.job_discovery_gateway import JobDiscoveryGateway, JobDiscoveryGatewayResult
from career_agent.agent.main_agent_contracts import ToolObservation
from career_agent.agent.main_agent_runtime import MainAgentRuntime
from career_agent.agent.main_agent_tools import MainAgentToolRegistry
from career_agent.agent.mock_interview_graph import (
    MockInterviewGraph,
    StoredMockInterviewSourceProvider,
)
from career_agent.agent.mock_interview_skill_loader import MockInterviewSkillLoader
from career_agent.agent.openai_compatible_agent_worker import OpenAICompatibleAgentWorker
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
from career_agent.connectors.boss_readonly import BossReadOnlyAdapter, SubprocessBossTransport
from career_agent.connectors.email_accounts import EnvironmentEmailConnectorResolver
from career_agent.connectors.calendar import EnvironmentCalendarConnectorResolver
from career_agent.services.job_discovery import JobDiscoveryService
from career_agent.services.applications import ApplicationService
from career_agent.services.action_center import ActionCenterService
from career_agent.services.calendar import CalendarService
from career_agent.services.email_tracking import EmailTrackingService
from career_agent.services.interviews import InterviewService
from career_agent.services.interview_preparation import InterviewPreparationService
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
from career_agent.storage.mock_interviews import SQLiteMockInterviewStore
from career_agent.storage.career_history import CareerHistoryStore
from career_agent.storage.jobs import SQLiteJobPostingRepository, StoredJobRecord, StoredJobSummary
from career_agent.storage.memory import InMemoryJobRepository
from career_agent.storage.resumes import ResumeStore
from career_agent.storage.resume_analysis import SQLiteResumeAnalysisDraftStore
from career_agent.storage.resume_artifacts import SQLiteResumeArtifactStore
from career_agent.storage.resume_job_matches import SQLiteResumeJobMatchStore
from career_agent.storage.resume_tailoring import SQLiteResumeTailoringDraftStore
from career_agent.storage.runs import JobDiscoveryRunStore


EXIT_OK = 0
EXIT_ARGUMENT_ERROR = 2
EXIT_CONFIGURATION_ERROR = 3
EXIT_CONNECTOR_ERROR = 4
EXIT_WORKFLOW_ERROR = 5
EXIT_UNKNOWN_ERROR = 6
MAX_RESUME_IMPORT_BYTES = 1_048_576


def build_gateway(args: argparse.Namespace) -> JobDiscoveryGateway:
    config = replace(OpenAICompatibleAgentConfig.from_env(), timeout_seconds=args.agent_timeout_seconds)
    worker = OpenAICompatibleAgentWorker(config)
    transport = SubprocessBossTransport(
        Path(args.boss_data_dir).expanduser(),
        executable=args.boss_executable,
        timeout_seconds=args.boss_timeout_seconds,
    )
    adapter = BossReadOnlyAdapter(transport)
    run_store = JobDiscoveryRunStore(Path(args.run_store).expanduser())
    return JobDiscoveryGateway(
        adapter,
        worker,
        JobDiscoveryService(InMemoryJobRepository()),
        run_store=run_store,
        job_repository=SQLiteJobPostingRepository(Path(args.job_store).expanduser()),
    )


def build_analysis_gateway(args: argparse.Namespace) -> JobDiscoveryGateway:
    config = replace(OpenAICompatibleAgentConfig.from_env(), timeout_seconds=args.agent_timeout_seconds)
    return JobDiscoveryGateway(
        None,
        OpenAICompatibleAgentWorker(config),
        JobDiscoveryService(InMemoryJobRepository()),
        run_store=JobDiscoveryRunStore(Path(args.run_store).expanduser()),
        job_repository=SQLiteJobPostingRepository(Path(args.job_store).expanduser()),
    )


def build_main_agent_runtime(args: argparse.Namespace) -> MainAgentRuntime:
    main_config = replace(OpenAICompatibleAgentConfig.from_env(prefix="MAIN_AGENT"), timeout_seconds=args.main_agent_timeout_seconds)
    context_manager = ContextManager(
        CareerContextStore(Path(args.context_store).expanduser()),
        summary_worker=OpenAIConversationSummaryWorker(main_config),
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
    interview_preparation_service = InterviewPreparationService(
        interview_service,
        application_service,
        resume_store,
        career_history_store,
        OpenAIInterviewPreparationWorker(resume_analysis_config),
        SQLiteInterviewPreparationStore(Path(args.resume_store).expanduser()),
    )
    checkpoint_owner = SQLiteCheckpointOwner(
        Path(args.mock_interview_checkpoint_store).expanduser()
    )
    mock_interview_graph = MockInterviewGraph(
        store=SQLiteMockInterviewStore(
            Path(args.mock_interview_store).expanduser()
        ),
        worker=OpenAIMockInterviewWorker(
            resume_analysis_config,
            skill_loader=MockInterviewSkillLoader(
                Path(args.mock_interview_skills_dir)
            ),
        ),
        sources=StoredMockInterviewSourceProvider(
            resumes=resume_store,
            jobs=job_repository,
            career_history=career_history_store,
        ),
        checkpointer=checkpoint_owner.saver,
    )
    return MainAgentRuntime(
        context_manager=context_manager,
        decision_maker=OpenAICompatibleMainAgentDecisionMaker(main_config),
        career_context_projector=CareerContextProjector(career_history_store),
        owned_resources=(checkpoint_owner,),
        tools=MainAgentToolRegistry(
            build_gateway(args),
            job_repository=job_repository,
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
            resume_analysis_service=ResumeAnalysisService(
                resume_store,
                OpenAIResumeAnalysisWorker(resume_analysis_config),
                SQLiteResumeAnalysisDraftStore(Path(args.resume_store).expanduser()),
                career_history_store,
            ),
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
    parser.add_argument("--boss-data-dir", required=True, help="BOSS CLI data directory.")
    parser.add_argument("--boss-executable", default="boss", help="BOSS executable name or path (default: boss).")
    parser.add_argument("--boss-timeout-seconds", type=float, default=300.0, help="Read-only BOSS call timeout (default: 300).")
    parser.add_argument("--agent-timeout-seconds", type=float, default=300.0, help="Model call timeout (default: 300).")
    parser.add_argument("--run-store", default="~/.career-agent/runs.sqlite3", help="Local durable run store path.")
    parser.add_argument("--job-store", default="~/.career-agent/jobs.sqlite3", help="Local durable job and JD snapshot store path.")
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
    parser.add_argument("--json", action="store_true", help="Emit one machine-readable JSON object.")
    parser.add_argument("--show-trace", action="store_true", help="Include the complete safe run trace in output.")
    parser.add_argument("--non-interactive", action="store_true", help="Never prompt for input.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="career-agent", description="Run a read-only career job discovery workflow.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    chat = subparsers.add_parser("chat", help="Send one natural-language turn to the Main Agent.", description="Run one non-interactive Main Agent turn. The agent decides whether to enter a registered workflow.", epilog="Example: career-agent chat --user-id u1 --session-id s1 --message 'Help me find AI Engineer jobs' --boss-data-dir ~/.boss-agent")
    chat.add_argument("--user-id", required=True, help="Stable user identifier.")
    chat.add_argument("--session-id", required=True, help="Conversation session identifier.")
    chat.add_argument("--message", required=True, help="Current user message.")
    chat.add_argument("--context-store", default="~/.career-agent/context.sqlite3", help="Local session and context store path.")
    chat.add_argument("--main-agent-timeout-seconds", type=float, default=60.0, help="Main Agent model timeout (default: 60).")
    _add_runtime_options(chat)

    discover = subparsers.add_parser("discover", help="Search jobs and return candidates for user selection.", description="Search BOSS in read-only mode and return up to 15 candidates.", epilog="Example: career-agent discover --user-id u1 --target-role 'AI Engineer' --boss-data-dir ~/.boss-agent --json")
    discover.add_argument("--user-id", required=True, help="Stable user identifier.")
    discover.add_argument("--target-role", required=True, help="Target role to search for.")
    discover.add_argument("--conversation-id", help="Conversation correlation identifier; generated when omitted.")
    discover.add_argument("--resume-file", type=Path, help="Read resume text from a local file; it is not persisted in the run store.")
    discover.add_argument("--city", help="Optional city filter.")
    discover.add_argument("--salary", help="Optional salary filter.")
    discover.add_argument("--experience", help="Optional experience filter.")
    discover.add_argument("--education", help="Optional education filter.")
    _add_runtime_options(discover)

    select = subparsers.add_parser("select", help="Fetch and analyze one candidate from a durable run.", description="Select one result_ref from a previous discover run and fetch its JD.", epilog="Example: career-agent select --user-id u1 --run-id RUN_ID --result-ref RESULT_REF --boss-data-dir ~/.boss-agent --json")
    select.add_argument("--user-id", required=True, help="User identifier that created the run.")
    select.add_argument("--run-id", required=True, help="Run ID returned by discover.")
    select.add_argument("--result-ref", required=True, help="Opaque result_ref returned by discover.")
    _add_runtime_options(select)

    analyze = subparsers.add_parser("analyze-jd", help="Analyze JD text the user copied after BOSS detail failed.", description="Analyze user-provided JD text without calling BOSS.")
    analyze.add_argument("--user-id", required=True, help="User identifier that created the run.")
    analyze.add_argument("--run-id", required=True, help="Run ID returned by discover.")
    analyze.add_argument("--result-ref", required=True, help="Selected result_ref whose detail is unavailable.")
    source = analyze.add_mutually_exclusive_group(required=True)
    source.add_argument("--jd-file", type=Path, help="UTF-8 text file containing the JD.")
    source.add_argument("--jd-stdin", action="store_true", help="Read JD text once from standard input.")
    analyze.add_argument("--agent-timeout-seconds", type=float, default=300.0, help="Model call timeout (default: 300).")
    analyze.add_argument("--run-store", default="~/.career-agent/runs.sqlite3", help="Local durable run store path.")
    analyze.add_argument("--job-store", default="~/.career-agent/jobs.sqlite3", help="Local durable job and JD snapshot store path.")
    analyze.add_argument("--json", action="store_true", help="Emit one machine-readable JSON object.")
    analyze.add_argument("--show-trace", action="store_true", help="Include the complete safe run trace in output.")
    analyze.add_argument("--non-interactive", action="store_true", help="Never prompt for input.")

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

    status = subparsers.add_parser("status", help="Read a durable run status and safe trace summary.", description="Read a persisted job discovery run without calling BOSS.")
    status.add_argument("--run-id", required=True, help="Run ID returned by discover.")
    _add_runtime_options(status)
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


def _chat_tool_result_payload(result: JobDiscoveryGatewayResult | ToolObservation) -> dict[str, object]:
    if isinstance(result, ToolObservation):
        return result.model_dump(mode="json")
    payload: dict[str, object] = {
        "state": result.state,
        "message": result.message,
        "next_action": result.next_action,
        "recovery_action": result.recovery_action,
        "error_code": result.error_code,
        "error_stage": result.error_stage,
        "error_detail": result.error_detail,
        "fallback_url": result.fallback_url,
        "manual_search_query": result.manual_search_query,
        "items": [
            {
                "selection_index": index,
                "title": item.title,
                "company_name": item.company_name,
                "city": item.city,
                "salary": item.salary,
                "rationale": item.rationale,
                "cautions": item.cautions,
            }
            for index, item in enumerate(result.items, start=1)
        ],
    }
    analyses = result.analysis_items or ((result.analysis,) if result.analysis else ())
    if analyses:
        rendered = [
            {
                "selection_index": index,
                "job_summary": analysis.job_summary,
                "responsibilities": analysis.responsibilities,
                "required_skills": analysis.required_skills,
                "preferred_qualifications": analysis.preferred_qualifications,
                "clarification_questions": analysis.clarification_questions,
            }
            for index, analysis in enumerate(analyses, start=1)
        ]
        payload["analyses"] = rendered
        if len(rendered) == 1:
            payload["analysis"] = {key: value for key, value in rendered[0].items() if key != "selection_index"}
    if result.comparison:
        payload["comparison"] = result.comparison.model_dump(mode="json")
    return payload


def _write_chat_payload(turn, *, user_id: str, session_id: str, output: TextIO) -> int:
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
    }
    json.dump(payload, output, ensure_ascii=False, separators=(",", ":"))
    output.write("\n")
    return _failure_exit_code(failed_result) if failed_result else EXIT_OK


def _write_chat_error(error: Exception, output: TextIO, *, code: int, next_action: str | None = None) -> int:
    payload = {"state": "failed", "error_code": getattr(error, "code", "CHAT_ERROR"), "error_detail": str(error), "next_action": next_action}
    json.dump(payload, output, ensure_ascii=False, separators=(",", ":"))
    output.write("\n")
    return code


def _result_payload(result: JobDiscoveryGatewayResult, *, show_trace: bool) -> dict[str, object]:
    payload = result.model_dump(mode="json", exclude_none=False)
    trace = payload.get("trace")
    if isinstance(trace, dict) and not show_trace:
        events = trace.get("events")
        payload["trace"] = {"run_id": trace.get("run_id"), "event_count": len(events) if isinstance(events, list) else 0, "last_event": events[-1] if isinstance(events, list) and events else None}
    return payload


def _write_json(result: JobDiscoveryGatewayResult, output: TextIO, *, show_trace: bool) -> None:
    json.dump(_result_payload(result, show_trace=show_trace), output, ensure_ascii=False, separators=(",", ":"))
    output.write("\n")


def _write_human(result: JobDiscoveryGatewayResult, output: TextIO, *, show_trace: bool) -> None:
    output.write(f"State: {result.state}\nRun ID: {result.run_id}\n{result.message}\n")
    for index, item in enumerate(result.items, start=1):
        salary = f" | {item.salary}" if item.salary else ""
        city = f" | {item.city}" if item.city else ""
        output.write(f"{index}. {item.title} — {item.company_name}{city}{salary}\n")
        if item.rationale:
            output.write(f"   Why: {item.rationale}\n")
    if result.detail:
        output.write(f"JD: {result.detail.title} at {result.detail.company_name}\n")
    if result.analysis:
        output.write(f"岗位摘要: {result.analysis.job_summary}\n")
    if result.error_code:
        output.write(f"Error: {result.error_code} at {result.error_stage or 'unknown stage'}\n")
        if result.error_detail:
            output.write(f"Detail: {result.error_detail}\n")
    if result.fallback_url:
        output.write(f"Open manually: {result.fallback_url}\n")
    if result.manual_search_query:
        output.write(f"Search BOSS manually: {result.manual_search_query}\n")
        output.write(f"Then: career-agent analyze-jd --user-id <user-id> --run-id {result.run_id} --result-ref {result.selected_result_ref} --jd-stdin --json\n")
    if result.next_action:
        output.write(f"Next action: {result.next_action}\n")
    if show_trace and result.trace:
        output.write("Trace:\n")
        for event in result.trace.events:
            output.write(f"  [{event.sequence}] {event.event_type} {event.stage} outcome={event.outcome}\n")


def _failure_exit_code(result: JobDiscoveryGatewayResult | ToolObservation) -> int:
    if result.state != "failed":
        return EXIT_OK
    if isinstance(result, ToolObservation):
        return EXIT_WORKFLOW_ERROR
    code = result.error_code or ""
    if code.startswith("AGENT_"):
        return EXIT_WORKFLOW_ERROR
    if code.startswith(("AUTH_", "BOSS_", "NETWORK_", "RATE_LIMITED", "TIMEOUT", "CLI_")):
        return EXIT_CONNECTOR_ERROR
    return EXIT_WORKFLOW_ERROR


def _write_exception(error: Exception, stdout: TextIO, stderr: TextIO, *, machine_output: bool) -> None:
    if machine_output:
        json.dump({"state": "failed", "error_code": "CLI_INPUT_ERROR", "error_detail": str(error), "next_action": "Check the command help and supplied run/user identifiers."}, stdout, ensure_ascii=False, separators=(",", ":"))
        stdout.write("\n")
    else:
        stderr.write(f"Input/configuration error: {error}\n")


def _read_provided_jd(args: argparse.Namespace, stdin: TextIO) -> str:
    if args.jd_file:
        if not args.jd_file.is_file():
            raise ValueError("--jd-file must be a regular file.")
        return args.jd_file.read_text(encoding="utf-8")
    value = stdin.read()
    if not value:
        raise ValueError("--jd-stdin received no JD text.")
    return value


def main(
    argv: Sequence[str] | None = None,
    *,
    gateway_factory: Callable[[argparse.Namespace], JobDiscoveryGateway] | None = None,
    runtime_factory: Callable[[argparse.Namespace], MainAgentRuntime] | None = None,
    resume_store_factory: Callable[[argparse.Namespace], ResumeStore] | None = None,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    stdin = stdin or sys.stdin
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
    machine_output = args.json or not stdout.isatty()
    if args.command == "chat":
        runtime = None
        try:
            runtime = runtime_factory(args) if runtime_factory else build_main_agent_runtime(args)
            turn = runtime.run_turn(user_id=args.user_id, conversation_id=args.session_id, user_message=args.message)
            return _write_chat_payload(turn, user_id=args.user_id, session_id=args.session_id, output=stdout)
        except AgentConfigurationError as error:
            return _write_chat_error(error, stdout, code=EXIT_CONFIGURATION_ERROR, next_action="Set MAIN_AGENT_*, JOB_DISCOVERY_AGENT_*, and RESUME_ANALYSIS_AGENT_* configuration.")
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
    try:
        gateway = gateway_factory(args) if gateway_factory else (build_analysis_gateway(args) if args.command == "analyze-jd" else build_gateway(args))
        if args.command == "discover":
            resume_text = args.resume_file.read_text(encoding="utf-8") if args.resume_file else None
            request = JobDiscoveryRequest(
                user_id=args.user_id,
                conversation_id=args.conversation_id or f"cli-{args.user_id}-{args.target_role.casefold().replace(' ', '-')}",
                target_role=args.target_role,
                resume_text=resume_text,
                city=args.city,
                salary=args.salary,
                experience=args.experience,
                education=args.education,
            )
            result = (gateway.research(request) if hasattr(gateway, "research") else gateway.start(request))
        elif args.command == "select":
            result = gateway.select(run_id=args.run_id, result_ref=args.result_ref, user_id=args.user_id)
        elif args.command == "analyze-jd":
            result = gateway.analyze_provided_jd(run_id=args.run_id, result_ref=args.result_ref, jd_text=_read_provided_jd(args, stdin), user_id=args.user_id)
        else:
            result = gateway.status(run_id=args.run_id)
    except AgentConfigurationError as error:
        payload = {"state": "failed", "error_code": error.code, "error_detail": str(error), "next_action": "Set JOB_DISCOVERY_AGENT_BASE_URL, JOB_DISCOVERY_AGENT_API_KEY, and JOB_DISCOVERY_AGENT_MODEL."}
        json.dump(payload, stdout, ensure_ascii=False, separators=(",", ":"))
        stdout.write("\n")
        return EXIT_CONFIGURATION_ERROR
    except (OSError, ValueError) as error:
        _write_exception(error, stdout, stderr, machine_output=machine_output)
        return EXIT_ARGUMENT_ERROR
    except Exception as error:
        if machine_output:
            json.dump({"state": "failed", "error_code": "CLI_UNKNOWN_ERROR", "error_detail": f"{type(error).__name__}: {error}"}, stdout, ensure_ascii=False, separators=(",", ":"))
            stdout.write("\n")
        else:
            stderr.write(f"Unexpected error: {type(error).__name__}: {error}\n")
        return EXIT_UNKNOWN_ERROR

    if machine_output:
        _write_json(result, stdout, show_trace=args.show_trace)
    else:
        _write_human(result, stdout, show_trace=args.show_trace)
    return _failure_exit_code(result)


if __name__ == "__main__":
    raise SystemExit(main())
