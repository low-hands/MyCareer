from __future__ import annotations

from datetime import datetime, timezone

from career_agent.agent.context_manager import ContextManager
from career_agent.agent.main_agent_contracts import (
    AgentDecision,
    CareerProfileContext,
    ConversationTaskState,
    MainAgentContext,
    ResumeCandidateContextItem,
    ToolCall,
    ToolResult,
)
from career_agent.agent.main_agent_runtime import MainAgentRuntime
from career_agent.agent.main_agent_tools import MainAgentToolRegistry
from career_agent.domain.job_discovery import JobDetail, Provenance
from career_agent.services.applications import ApplicationService
from career_agent.storage.applications import SQLiteApplicationStore
from career_agent.storage.context import CareerContextStore
from career_agent.storage.jobs import SQLiteJobPostingRepository
from career_agent.storage.resumes import ResumeStore


class SequenceDecisionMaker:
    def __init__(self, *decisions: AgentDecision) -> None:
        self.decisions = list(decisions)
        self.contexts = []

    def decide(self, context, tool_specs):
        self.contexts.append(context)
        return self.decisions.pop(0)


def build_application_agent(tmp_path, decisions):
    resumes = ResumeStore(tmp_path / "resumes.sqlite3")
    role = resumes.create_target_role(user_id="u1", title="AI Engineer", priority=1)
    resume, version = resumes.import_document(
        user_id="u1",
        target_role_id=role.id,
        name="AI Resume",
        content=b"PRIVATE RESUME",
        document_format="text",
    )
    jobs = SQLiteJobPostingRepository(tmp_path / "jobs.sqlite3")
    captured_at = datetime(2026, 8, 25, tzinfo=timezone.utc)
    job = jobs.save_detail(
        user_id="u1",
        run_id="run-1",
        result_ref="ref-1",
        selection_index=1,
        detail=JobDetail(
            source_name="test",
            source_job_id="job-1",
            title="RAG Engineer",
            company_name="Acme",
            description="PRIVATE JD",
            captured_at=captured_at,
            provenance=Provenance(
                source_name="test",
                source_job_id="job-1",
                captured_at=captured_at,
                operation="detail",
                adapter_version="test-v1",
            ),
        ),
    )
    service = ApplicationService(
        SQLiteApplicationStore(tmp_path / "applications.sqlite3"),
        jobs,
        resumes,
    )
    manager = ContextManager(CareerContextStore(tmp_path / "context.sqlite3"))
    manager.upsert_profile(CareerProfileContext(user_id="u1"))
    tools = MainAgentToolRegistry(
        job_repository=jobs,
        resume_store=resumes,
        application_service=service,
    )
    runtime = MainAgentRuntime(
        context_manager=manager,
        decision_maker=decisions,
        tools=tools,
        max_read_calls=5,
        max_write_calls=2,
    )
    return runtime, manager, tools, job, resume, version


def test_main_agent_tracks_active_job_and_resume_then_updates_across_turns(
    tmp_path,
) -> None:
    decisions = SequenceDecisionMaker()
    runtime, manager, tools, job, resume, version = build_application_agent(
        tmp_path, decisions
    )
    decisions.decisions.extend(
        [
            AgentDecision(
                action="tool_call",
                tool_call=ToolCall(
                    name="find_saved_jobs",
                    arguments={"query": job.posting.title},
                ),
            ),
            AgentDecision(
                action="tool_call",
                tool_call=ToolCall(
                    name="get_saved_job",
                    arguments={"selection_index": 1},
                ),
            ),
            AgentDecision(
                action="tool_call",
                tool_call=ToolCall(name="list_resumes", arguments={}),
            ),
            AgentDecision(
                action="tool_call",
                tool_call=ToolCall(
                    name="get_resume_metadata",
                    arguments={"selection_index": 1},
                ),
            ),
            AgentDecision(
                action="tool_call",
                tool_call=ToolCall(
                    name="create_application",
                    arguments={
                        "resume_version_selection_index": 1,
                        "note": "Applied on the company site.",
                    },
                ),
            ),
            AgentDecision(action="final", message="已记录这次投递。"),
        ]
    )

    created_result = runtime.run_turn(
        user_id="u1",
        conversation_id="c1",
        user_message="我用这份简历投了刚才那个岗位，帮我记录",
    )

    created = created_result.tool_results[-1]
    assert created.state == "application_ready"
    assert created.payload["job_posting_id"] == job.posting.id
    assert created.payload["resume_version_id"] == version.id
    assert created_result.context.task.active_application_id == created.payload[
        "application_id"
    ]
    assert created_result.context.task.active_application_status == "submitted"
    assert "PRIVATE RESUME" not in created.model_dump_json()
    assert "PRIVATE JD" not in created.model_dump_json()
    assert all(
        "user_id" not in spec["function"]["parameters"].get("properties", {})
        for spec in tools.schemas()
    )

    update_decisions = SequenceDecisionMaker(
        AgentDecision(
            action="tool_call",
            tool_call=ToolCall(
                name="update_application_status",
                arguments={"status": "interviewing", "note": "进入一面"},
            ),
        ),
        AgentDecision(action="final", message="已更新为面试阶段。"),
    )
    updated_result = MainAgentRuntime(
        context_manager=manager,
        decision_maker=update_decisions,
        tools=tools,
    ).run_turn(
        user_id="u1",
        conversation_id="c1",
        user_message="这个岗位进一面了",
    )
    assert updated_result.context.task.active_application_status == "interviewing"

    list_decisions = SequenceDecisionMaker(
        AgentDecision(
            action="tool_call",
            tool_call=ToolCall(
                name="list_applications",
                arguments={"statuses": ["interviewing"]},
            ),
        ),
        AgentDecision(action="final", message="找到一条面试中的投递。"),
    )
    listed_result = MainAgentRuntime(
        context_manager=manager,
        decision_maker=list_decisions,
        tools=tools,
    ).run_turn(
        user_id="u1",
        conversation_id="c1",
        user_message="列出面试中的投递",
    )
    assert len(listed_result.context.task.application_candidates) == 1

    get_decisions = SequenceDecisionMaker(
        AgentDecision(
            action="tool_call",
            tool_call=ToolCall(
                name="get_application", arguments={"selection_index": 1}
            ),
        ),
        AgentDecision(action="final", message="这是这次投递的进展。"),
    )
    detail_result = MainAgentRuntime(
        context_manager=manager,
        decision_maker=get_decisions,
        tools=tools,
    ).run_turn(
        user_id="u1",
        conversation_id="c1",
        user_message="看看刚才那个投递的记录",
    )
    detail = detail_result.tool_result
    assert [event["event_type"] for event in detail.payload["events"]] == [
        "created",
        "status_changed",
    ]
    assert all("user_id" not in event for event in detail.payload["events"])


def test_reading_other_resume_metadata_does_not_replace_application_version(
    tmp_path,
) -> None:
    decisions = SequenceDecisionMaker()
    runtime, manager, tools, job, resume, tailored_version = build_application_agent(
        tmp_path, decisions
    )
    resumes = ResumeStore(tmp_path / "resumes.sqlite3")
    other_resume, other_version = resumes.import_document(
        user_id="u1",
        target_role_id=resume.target_role_id,
        name="Other Resume",
        content=b"OTHER RESUME V9",
        document_format="text",
    )
    seeded = manager.load_for_turn(
        user_id="u1", conversation_id="c1", user_message="seed active v10"
    )
    manager.commit_turn(
        context=seeded,
        task=seeded.task.model_copy(
            update={
                "active_job_posting_id": job.posting.id,
                "active_resume_version_id": tailored_version.id,
                "resume_candidates": (
                    ResumeCandidateContextItem(
                        resume_id=other_resume.id,
                        target_role_id=other_resume.target_role_id,
                        name=other_resume.name,
                        status=other_resume.status,
                        latest_version_id=other_version.id,
                    ),
                ),
            }
        ),
        assistant_message="seeded",
    )
    decisions.decisions.extend(
        [
            AgentDecision(
                action="tool_call",
                tool_call=ToolCall(
                    name="get_resume_metadata", arguments={"selection_index": 1}
                ),
            ),
            AgentDecision(
                action="tool_call",
                tool_call=ToolCall(name="create_application", arguments={}),
            ),
            AgentDecision(action="final", message="已记录这次投递。"),
        ]
    )

    result = runtime.run_turn(
        user_id="u1",
        conversation_id="c1",
        user_message="我看完另一份简历了，记一下刚才用定制版完成的投递",
    )

    assert result.tool_results[0].payload["resume"]["latest_version_id"] == (
        other_version.id
    )
    assert result.tool_results[1].payload["resume_version_id"] == (
        tailored_version.id
    )
    schema = next(
        item
        for item in tools.schemas()
        if item["function"]["name"] == "create_application"
    )["function"]["parameters"]["properties"]
    assert "resume_version_selection_index" in schema
    assert "resume_version_id" not in schema


def test_get_application_selects_application_without_replacing_job_or_resume() -> None:
    context = MainAgentContext(
        conversation_id="c1",
        profile=CareerProfileContext(user_id="u1"),
        task=ConversationTaskState(
            active_job_posting_id="current-job",
            active_resume_version_id="current-resume-v10",
        ),
        user_message="查看历史投递",
    )
    historical = ToolResult(
        tool_name="get_application",
        state="application_ready",
        message="已读取历史投递。",
        payload={
            "application_id": "old-application",
            "status": "rejected",
            "job_posting_id": "old-job",
            "resume_version_id": "old-resume-v3",
        },
    )

    updated = MainAgentRuntime._update_atomic_task(context, historical)

    assert updated.task.active_application_id == "old-application"
    assert updated.task.active_application_status == "rejected"
    assert updated.task.active_job_posting_id == "current-job"
    assert updated.task.active_resume_version_id == "current-resume-v10"
