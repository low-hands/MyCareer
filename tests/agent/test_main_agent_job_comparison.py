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
                source_name="boss",
                source_url=None,
            ),
            city="上海",
            salary=None if job_posting_id == "job-2" else "30-50K",
            availability_status="active",
            snapshot=SimpleNamespace(
                version=1,
                content="负责 AI 平台研发与评估。",
                captured_at=NOW,
                provenance=SimpleNamespace(
                    model_dump=lambda **_: {
                        "source_name": "boss",
                        "source_url": None,
                    }
                ),
            ),
            analysis=None,
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


def test_find_get_compare_read_chain_finishes_in_one_turn(tmp_path) -> None:
    """Acceptance trajectory: three delegated reads, then a grounded answer."""
    decisions = SequenceDecisionMaker(
        AgentDecision(
            action="tool_call",
            tool_call=ToolCall(name="find_saved_jobs", arguments={"query": "AI"}),
        ),
        AgentDecision(
            action="tool_call",
            tool_call=ToolCall(
                name="get_saved_job", arguments={"selection_index": 1}
            ),
        ),
        AgentDecision(
            action="tool_call",
            tool_call=ToolCall(
                name="compare_saved_jobs",
                arguments={"job_selection_indices": [1, 2]},
            ),
        ),
        AgentDecision(action="final", message="已完成比较。"),
    )
    runtime, _, _ = build_runtime(tmp_path, decisions)

    result = runtime.run_turn(
        user_id="u1",
        conversation_id="c1",
        user_message="先找出 AI 岗位，看第一份详情，再比较这两份。",
    )

    assert [item.state for item in result.tool_results] == [
        "saved_jobs_found",
        "saved_job_ready",
        "saved_jobs_compared",
    ]
    # F: the model closes the composed turn in one sentence covering all three
    # steps, and that sentence is what the transcript keeps. The comparison has
    # no card, so the table it reasoned over is delivered beneath the reply
    # rather than replaced by it.
    assert result.model_message == "已完成比较。"
    assert result.assistant_message.startswith("已完成比较。")
    assert "岗位横向对比" in result.assistant_message
    assert "岗位横向对比" in decisions.contexts[-1].tool_observations[-1].body
    assert len(decisions.contexts) == 4
    assert result.context.tool_observations[-1].state == "saved_jobs_compared"


def test_repeated_projection_refusals_end_the_turn_on_the_counter(tmp_path) -> None:
    """The bound that replaced the reroute table.

    Sending every refusal back to the model is only safe because
    ``max_projection_refusals`` exits on its own. That bound is not incidental:
    every ``invalid_input`` comes from ``_rejection_observation``, which runs
    only on the synthetic projection path, so each one increments the counter —
    a tool returning ``invalid_input`` from a real execution would not be
    bounded here, and none does.
    """
    refusal = AgentDecision(
        action="tool_call",
        tool_call=ToolCall(
            name="compare_saved_jobs", arguments={"job_selection_indices": [1, 2]}
        ),
    )
    second = AgentDecision(
        action="tool_call",
        tool_call=ToolCall(
            name="compare_saved_jobs", arguments={"job_selection_indices": [4, 5]}
        ),
    )
    third = AgentDecision(
        action="tool_call",
        tool_call=ToolCall(
            name="compare_saved_jobs", arguments={"job_selection_indices": [7, 8]}
        ),
    )
    decisions = SequenceDecisionMaker(refusal, second, third)
    runtime, _, _ = build_runtime(tmp_path, decisions)

    result = runtime.run_turn(user_id="u1", conversation_id="c1", user_message="对比一下")

    # Two refusals reach the model; the third attempt is cut by the limit
    # rather than by a table, and the turn still ends.
    assert len(decisions.contexts) == 3
    assert result.assistant_message


def test_an_unrepairable_refusal_is_still_the_model_s_to_answer(tmp_path) -> None:
    """Nothing in task state could repair this — and the model still gets to say so.

    A ``REROUTE_FIELDS`` table used to end the turn here, on the grounds that no
    candidate list existed for the model to re-select from. The premise was
    right and the conclusion was not: "can this call succeed" is the harness's
    question, but "retry, ask, or explain" is the agent's, and a canned
    presenter line quoting ``selection index is out of range`` is a worse answer
    than the one the model writes with the refusal in front of it.

    So the reason now reaches the model rather than the user, and the reply is
    the model's. The bound is ``max_projection_refusals``, not a table.
    """
    decisions = SequenceDecisionMaker(
        AgentDecision(
            action="tool_call",
            tool_call=ToolCall(
                name="compare_saved_jobs",
                arguments={"job_selection_indices": [1, 2]},
            ),
        ),
        AgentDecision(action="final", message="请先选择要比较的岗位。"),
    )
    runtime, _, _ = build_runtime(tmp_path, decisions)

    result = runtime.run_turn(user_id="u1", conversation_id="c1", user_message="对比一下")

    assert result.tool_result is None
    # The refusal is an observation, not a delivery: the model saw the reason…
    assert len(decisions.contexts) == 2
    assert result.context.tool_observations[-1].state == "invalid_input"
    assert "out of range" in decisions.contexts[-1].tool_observations[-1].message
    # …and the user reads the model's sentence instead of the raw error.
    assert result.assistant_message == "请先选择要比较的岗位。"


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


def test_the_comparison_reaches_the_model_as_a_body_and_keeps_ids_out(
    tmp_path,
) -> None:
    """H/F: the matrix the reader gets is the matrix the model reasoned over.

    It used to be withheld, which was coherent only while the presenter also
    wrote the reply. Once the model narrates, withholding the table would mean
    nobody delivers it: the state has no card to fall back on. It is therefore
    registered as condensed, so the bounded rendering reaches the model as an
    observation body — while internal ids stay out of both, as always.
    """
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

    observation = decisions.contexts[-1].model_context()["tool_observations"][-1]
    assert observation["tool_name"] == "compare_saved_jobs"
    assert observation["state"] == "saved_jobs_compared"
    assert observation["message"] == "已对比 2 个已保存岗位。"
    assert observation["facts"] == {}
    # No hint: "saved_jobs_compared" already says the comparison is ready, and a
    # token restating it only lengthened every decision prompt that carried it.
    assert "next_action" not in observation
    assert "岗位横向对比" in observation["body"]
    assert "不是排名" in observation["body"]
    # The fixture's own titles embed "job-1", so an internal-id leak cannot be
    # distinguished here; that guarantee is held exhaustively, with a real
    # 32-hex id, by test_real_condensed_presenters_do_not_render_internal_identifiers.
    # The model answered with an empty message here, so the presenter still
    # delivers — that fallback is what keeps a silent model from losing the
    # table entirely.
    assert "岗位横向对比" in result.assistant_message


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
