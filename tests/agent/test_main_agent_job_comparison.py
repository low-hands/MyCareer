from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from career_agent.agent.context_manager import ContextManager
from career_agent.agent.main_agent_contracts import (
    AgentDecision,
    CareerProfileContext,
    ToolCall,
)
from career_agent.agent.main_agent_runtime import MainAgentRuntime
from career_agent.agent.main_agent_tools import MainAgentToolRegistry
from career_agent.agent.resume_job_match_contracts import ResumeJobMatchResult
from career_agent.services.job_comparison import JobComparisonService
from career_agent.storage.context import CareerContextStore
from career_agent.storage.jobs import StoredJobSummary
from career_agent.storage.resume_job_matches import (
    SQLiteResumeJobMatchStore,
    StoredResumeJobMatch,
)


NOW = datetime(2026, 8, 31, tzinfo=timezone.utc)


class Jobs:
    def search_saved_jobs(self, *, user_id, query, limit=20):
        return tuple(
            StoredJobSummary(
                job_posting_id=job_posting_id,
                title=f"AI Engineer {job_posting_id}",
                company_name="Acme",
                city="上海",
                salary=None,
                source_name="boss",
                source_url=None,
                availability_status="active",
                captured_at=NOW,
                last_checked_at=NOW,
            )
            for job_posting_id in ("job-1", "job-2")
        )

    def get_job(self, *, user_id, job_posting_id):
        if job_posting_id not in {"job-1", "job-2"}:
            return None
        return SimpleNamespace(
            posting=SimpleNamespace(
                id=job_posting_id,
                title=f"AI Engineer {job_posting_id}",
                company_name="Acme",
            ),
            city="上海",
            salary=None if job_posting_id == "job-2" else "30-50K",
            availability_status="active",
        )


class NoMatches:
    def find_latest_for_job(self, *, user_id, job_posting_id):
        return None


class SequenceDecisionMaker:
    def __init__(self, *decisions: AgentDecision) -> None:
        self.decisions = list(decisions)
        self.contexts = []

    def decide(self, context, tool_specs):
        self.contexts.append(context)
        return self.decisions.pop(0)


def build_runtime(tmp_path, decisions):
    manager = ContextManager(CareerContextStore(tmp_path / "context.sqlite3"))
    manager.upsert_profile(
        CareerProfileContext(user_id="u1", default_city="上海")
    )
    tools = MainAgentToolRegistry(
        job_repository=Jobs(),
        job_comparison_service=JobComparisonService(Jobs(), NoMatches()),
    )
    return (
        MainAgentRuntime(
            context_manager=manager,
            decision_maker=decisions,
            tools=tools,
        ),
        tools,
        manager,
    )


def test_the_tool_appears_only_when_the_comparison_service_is_configured(
    tmp_path,
) -> None:
    without = MainAgentToolRegistry(job_repository=Jobs())
    with_service = MainAgentToolRegistry(
        job_repository=Jobs(),
        job_comparison_service=JobComparisonService(Jobs(), NoMatches()),
    )

    assert "compare_saved_jobs" not in without.atomic_tool_names
    assert "compare_saved_jobs" in with_service.atomic_tool_names
    assert with_service.capability_kind("compare_saved_jobs") == "atomic_tool"


def test_the_model_chooses_jobs_by_index_and_never_sees_an_internal_id(
    tmp_path,
) -> None:
    """The model addresses saved jobs positionally, as with every other tool."""
    decisions = SequenceDecisionMaker(
        AgentDecision(
            action="tool_call",
            tool_call=ToolCall(name="find_saved_jobs", arguments={"query": "AI"}),
        ),
        AgentDecision(
            action="tool_call",
            tool_call=ToolCall(
                name="compare_saved_jobs",
                arguments={"job_selection_indices": [1, 2]},
            ),
        ),
        AgentDecision(action="final", message=""),
    )
    runtime, tools, _ = build_runtime(tmp_path, decisions)

    schema = next(
        spec
        for spec in tools.schemas()
        if spec["function"]["name"] == "compare_saved_jobs"
    )
    projected = MainAgentToolRegistry._decision_tool_schema(schema)

    assert "job_posting_id" not in str(projected)
    assert set(
        projected["function"]["parameters"]["properties"]
    ) == {"job_selection_indices"}


def test_an_out_of_range_index_is_rejected(tmp_path) -> None:
    """Empty context: nothing to re-select, so the turn ends on the refusal."""
    decisions = SequenceDecisionMaker(
        AgentDecision(
            action="tool_call",
            tool_call=ToolCall(
                name="compare_saved_jobs",
                arguments={"job_selection_indices": [1, 2]},
            ),
        ),
    )
    runtime, _, _ = build_runtime(tmp_path, decisions)

    result = runtime.run_turn(user_id="u1", conversation_id="c1", user_message="对比一下")

    assert result.tool_result is not None
    assert result.tool_result.state == "invalid_input"
    assert "selection index is out of range" in result.assistant_message
    assert len(decisions.contexts) == 1


def test_an_out_of_range_index_with_listed_jobs_is_rerouted_to_decide(
    tmp_path,
) -> None:
    """Candidates exist, so the refusal loops back and the model re-selects."""
    decisions = SequenceDecisionMaker(
        AgentDecision(
            action="tool_call",
            tool_call=ToolCall(name="find_saved_jobs", arguments={"query": "AI"}),
        ),
        AgentDecision(
            action="tool_call",
            tool_call=ToolCall(
                name="compare_saved_jobs",
                arguments={"job_selection_indices": [1, 3]},
            ),
        ),
        AgentDecision(
            action="tool_call",
            tool_call=ToolCall(
                name="compare_saved_jobs",
                arguments={"job_selection_indices": [1, 2]},
            ),
        ),
        AgentDecision(action="final", message=""),
    )
    runtime, _, _ = build_runtime(tmp_path, decisions)

    result = runtime.run_turn(user_id="u1", conversation_id="c1", user_message="对比一下")

    assert result.tool_result is not None
    assert result.tool_result.state == "saved_jobs_compared"
    # First compare refused, looped back, model re-selected against the listed
    # jobs, and the comparison went through without bouncing the user.
    assert len(decisions.contexts) == 4
    assert any(
        "selection index is out of range" in context.model_dump_json()
        for context in decisions.contexts
    )


def test_the_comparison_is_rendered_for_the_user_but_stays_out_of_the_model_context(
    tmp_path,
) -> None:
    """The matrix stays internal; the model gets only its bounded receipt."""
    decisions = SequenceDecisionMaker(
        AgentDecision(
            action="tool_call",
            tool_call=ToolCall(name="find_saved_jobs", arguments={"query": "AI"}),
        ),
        AgentDecision(
            action="tool_call",
            tool_call=ToolCall(
                name="compare_saved_jobs",
                arguments={"job_selection_indices": [1, 2]},
            ),
        ),
        AgentDecision(action="final", message=""),
    )
    runtime, _, _ = build_runtime(tmp_path, decisions)

    result = runtime.run_turn(
        user_id="u1", conversation_id="c1", user_message="这两个岗位怎么选"
    )

    assert "岗位横向对比" in result.assistant_message
    assert "不是排名" in result.assistant_message
    observation = decisions.contexts[-1].model_context()["tool_observations"][-1]
    assert observation == {
        "tool_name": "compare_saved_jobs",
        "state": "saved_jobs_compared",
        "message": "已对比 2 个已保存岗位。",
        "facts": {},
        "next_action": "discuss_comparison_or_match_missing_jobs",
    }
    assert "job-1" not in str(observation)


def test_the_store_returns_the_most_recent_match_for_a_job(tmp_path) -> None:
    store = SQLiteResumeJobMatchStore(tmp_path / "resumes.sqlite3")
    for index, resume_version_id in enumerate(("rv1", "rv2")):
        store.save(
            user_id="u1",
            resume_version_id=resume_version_id,
            job_posting_id="job-1",
            jd_snapshot_id=f"snap-{index}",
            matcher_version="resume-job-match-v1",
            evidence_fingerprint=f"fp-{index}",
            result=ResumeJobMatchResult(
                overall_fit="weak" if index == 0 else "strong",
                summary=f"summary {index}",
            ),
        )

    latest = store.find_latest_for_job(user_id="u1", job_posting_id="job-1")
    foreign = store.find_latest_for_job(user_id="u2", job_posting_id="job-1")

    assert latest is not None
    assert latest.result.overall_fit == "strong"
    assert foreign is None
