from __future__ import annotations

from datetime import datetime, timezone

from career_agent.agent.context_manager import ContextManager
from career_agent.agent.main_agent_contracts import (
    AgentDecision,
    CareerProfileContext,
    ToolCall,
)
from career_agent.agent.main_agent_runtime import MainAgentRuntime
from career_agent.agent.main_agent_tools import MainAgentToolRegistry
from career_agent.domain.job_discovery import JobDetail, Provenance
from career_agent.services.applications import ApplicationService
from career_agent.storage.applications import SQLiteApplicationStore
from career_agent.storage.context import CareerContextStore
from career_agent.storage.jobs import SQLiteJobPostingRepository
from career_agent.storage.resumes import ResumeStore


class UnusedGateway:
    def advance(self, **kwargs):
        raise AssertionError("Application tools must not enter Job Discovery")


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
        UnusedGateway(),
        job_repository=jobs,
        resume_store=resumes,
        application_service=service,
    )
    runtime = MainAgentRuntime(
        context_manager=manager,
        decision_maker=decisions,
        tools=tools,
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
                    name="get_saved_job",
                    arguments={"job_posting_id": job.posting.id},
                ),
            ),
            AgentDecision(
                action="tool_call",
                tool_call=ToolCall(
                    name="get_resume_metadata",
                    arguments={"resume_id": resume.id},
                ),
            ),
            AgentDecision(
                action="tool_call",
                tool_call=ToolCall(
                    name="create_application",
                    arguments={"note": "Applied on the company site."},
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

    created = decisions.contexts[3].tool_observations[-1]
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
    MainAgentRuntime(
        context_manager=manager,
        decision_maker=get_decisions,
        tools=tools,
    ).run_turn(
        user_id="u1",
        conversation_id="c1",
        user_message="看看刚才那个投递的记录",
    )
    detail = get_decisions.contexts[1].tool_observations[-1]
    assert [event["event_type"] for event in detail.payload["events"]] == [
        "created",
        "status_changed",
    ]
    assert all("user_id" not in event for event in detail.payload["events"])
