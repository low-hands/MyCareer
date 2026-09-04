"""Restarting a mock interview that cannot be continued.

Two failures leave a run holding the workflow slot with no way to advance: a
deleted checkpoint thread and a run created by an incompatible graph version.
Both told the candidate to restart long before a restart existed, and because the
store allows one unfinished run per user, the stuck run had to be retired before
any new interview could be created at all.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from career_agent.agent.context_manager import ContextManager
from career_agent.agent.main_agent_contracts import (
    AgentDecision,
    CareerProfileContext,
    ConversationTaskState,
    MainAgentContext,
    ToolCall,
    project_restart_mock_interview_arguments,
)
from career_agent.agent.main_agent_runtime import MainAgentRuntime
from career_agent.agent.main_agent_tools import MainAgentToolRegistry
from career_agent.agent.mock_interview_contracts import MockInterviewStartRequest
from career_agent.agent.mock_interview_graph import MockInterviewGraph
from career_agent.agent.openai_compatible_client import AgentWorkerError
from career_agent.storage.context import CareerContextStore
from career_agent.storage.mock_interviews import SQLiteMockInterviewStore
from conftest import FixedSources, OneQuestionWorker


class _UnusedGateway:
    """The restart path never reaches job discovery."""


def _stuck_run(tmp_path: Path, *, break_checkpoint: bool = True):
    """A run awaiting an answer whose checkpoint thread is then deleted."""
    store = SQLiteMockInterviewStore(tmp_path / "mock.sqlite3")
    graph = MockInterviewGraph(
        store=store, worker=OneQuestionWorker(), sources=FixedSources()
    )
    started = graph.start(
        MockInterviewStartRequest(
            user_id="u1",
            application_id="app-1",
            job_posting_id="job-1",
            jd_snapshot_id="jd-1",
            resume_version_id="rv-1",
            interview_type="technical",
            max_primary_questions=1,
            max_follow_ups_per_question=0,
        )
    )
    if break_checkpoint:
        # Drop the execution thread while leaving the business session active.
        # This is the exact split the checkpoint_missing observation reports, and
        # it is reachable in production through checkpoint retention.
        graph.checkpointer.delete_thread(started.session_id)
    return store, graph, started.session_id


def _registry(store, graph) -> MainAgentToolRegistry:
    return MainAgentToolRegistry(
        mock_interview_graph=graph, mock_interview_store=store,
    )


def test_a_stuck_run_is_retired_and_replaced_by_a_fresh_one(tmp_path) -> None:
    """The stuck run must be cancelled first or the new one cannot be created.

    One unfinished run per user is a store invariant, so a restart that only
    called start would be rejected by the same rule that made the run stuck.
    """
    store, graph, stuck_id = _stuck_run(tmp_path)
    with pytest.raises(Exception):
        graph.resume(user_id="u1", session_id=stuck_id, answer="答不进去")

    result = _registry(store, graph).invoke_workflow(
        "restart_mock_interview", {"user_id": "u1"}
    )

    assert result.state == "mock_interview_answer_required"
    new_id = result.payload["session_id"]
    assert new_id != stuck_id
    assert store.get_session(user_id="u1", session_id=stuck_id).status == "cancelled"
    assert store.find_resumable(user_id="u1").id == new_id


def test_the_replacement_practises_against_the_same_material(tmp_path) -> None:
    """A restart copies the retired run's own targets, not the model's guess.

    The tool takes no arguments for exactly this reason: if the replacement could
    name its own application or resume version, a restart could quietly move the
    practice to a different job than the one the candidate was preparing for.
    """
    store, graph, stuck_id = _stuck_run(tmp_path)
    stuck = store.get_session(user_id="u1", session_id=stuck_id)

    result = _registry(store, graph).invoke_workflow(
        "restart_mock_interview", {"user_id": "u1"}
    )

    fresh = store.get_session(
        user_id="u1", session_id=result.payload["session_id"]
    )
    assert fresh.application_id == stuck.application_id
    assert fresh.job_posting_id == stuck.job_posting_id
    assert fresh.jd_snapshot_id == stuck.jd_snapshot_id
    assert fresh.resume_version_id == stuck.resume_version_id
    assert fresh.interview_type == stuck.interview_type
    assert fresh.max_primary_questions == stuck.max_primary_questions
    assert fresh.max_follow_ups_per_question == stuck.max_follow_ups_per_question


def test_nothing_stuck_says_so_instead_of_starting_a_second_run(tmp_path) -> None:
    """Restart is not a second way to start, so with nothing stuck it declines."""
    store, graph, stuck_id = _stuck_run(tmp_path, break_checkpoint=False)
    graph.cancel(user_id="u1", session_id=stuck_id)

    result = _registry(store, graph).invoke_workflow(
        "restart_mock_interview", {"user_id": "u1"}
    )

    assert result.state == "no_mock_interview_to_restart"
    assert store.find_resumable(user_id="u1") is None


def test_a_failed_replacement_releases_the_retired_workflow_slot(tmp_path) -> None:
    """A failed replacement has no durable answer or run that can be retried."""

    class FailReplacementPlan(OneQuestionWorker):
        fail = False

        def plan(self, **kwargs):
            if self.fail:
                raise AgentWorkerError(
                    "PLAN_FAILED", "replacement plan failed", retryable=True
                )
            return super().plan(**kwargs)

    store = SQLiteMockInterviewStore(tmp_path / "mock.sqlite3")
    worker = FailReplacementPlan()
    graph = MockInterviewGraph(store=store, worker=worker, sources=FixedSources())
    started = graph.start(
        MockInterviewStartRequest(
            user_id="u1",
            application_id="app-1",
            job_posting_id="job-1",
            jd_snapshot_id="jd-1",
            resume_version_id="rv-1",
            interview_type="technical",
            max_primary_questions=1,
            max_follow_ups_per_question=0,
        )
    )
    graph.checkpointer.delete_thread(started.session_id)
    worker.fail = True
    result = _registry(store, graph).invoke_workflow(
        "restart_mock_interview", {"user_id": "u1"}
    )
    context = MainAgentContext(
        conversation_id="c1",
        profile=CareerProfileContext(user_id="u1"),
        task=ConversationTaskState(
            active_workflow="mock_interview",
            run_id=started.session_id,
            phase="mock_interview_checkpoint_missing",
        ),
        user_message="重新开始",
    )

    updated = MainAgentRuntime._update_mock_interview_task(context, result)

    assert result.state == "mock_interview_restart_failed"
    # Prose, because the one thing worth saying here is what NOT to do.
    assert "重新开一场" in result.next_action
    assert result.payload["error_code"] == "PLAN_FAILED"
    assert store.find_resumable(user_id="u1") is None
    assert updated.task.active_workflow == "none"
    assert updated.task.run_id is None


def test_no_stuck_business_run_clears_a_stale_mock_workflow_task(tmp_path) -> None:
    store, graph, stuck_id = _stuck_run(tmp_path, break_checkpoint=False)
    graph.cancel(user_id="u1", session_id=stuck_id)
    result = _registry(store, graph).invoke_workflow(
        "restart_mock_interview", {"user_id": "u1"}
    )
    context = MainAgentContext(
        conversation_id="c1",
        profile=CareerProfileContext(user_id="u1"),
        task=ConversationTaskState(
            active_workflow="mock_interview",
            run_id=stuck_id,
            phase="mock_interview_checkpoint_missing",
        ),
        user_message="重新开始",
    )

    updated = MainAgentRuntime._update_mock_interview_task(context, result)

    assert result.state == "no_mock_interview_to_restart"
    assert updated.task.active_workflow == "none"


def test_the_retired_run_stays_readable_after_being_replaced(tmp_path) -> None:
    """Restarting abandons the run, not what it recorded.

    The candidate is told the old answers remain readable, so the question it got
    through has to survive the cancellation that clears the way for the new run.
    """
    store, graph, stuck_id = _stuck_run(tmp_path)
    registry = _registry(store, graph)
    registry.invoke_workflow("restart_mock_interview", {"user_id": "u1"})

    turns = store.list_turns(user_id="u1", session_id=stuck_id)

    assert [turn.question for turn in turns] == ["介绍一个你负责的检索改进。"]


def test_the_tool_needs_both_the_graph_and_the_store(tmp_path) -> None:
    """Retiring reads the stuck run's settings, so a graph alone is not enough."""
    store, graph, _ = _stuck_run(tmp_path)

    graph_only = MainAgentToolRegistry(mock_interview_graph=graph)
    store_only = MainAgentToolRegistry(mock_interview_store=store)

    for registry in (graph_only, store_only):
        names = {
            schema["function"]["name"] for schema in registry.schemas()
        }
        assert "restart_mock_interview" not in names
    assert "restart_mock_interview" in {
        schema["function"]["name"] for schema in _registry(store, graph).schemas()
    }


def test_the_projection_passes_only_the_user_and_rejects_internal_ids() -> None:
    """No arguments reach the graph, so no argument can redirect the restart."""
    context = MainAgentContext(
        conversation_id="c1",
        user_message="重新开始",
        profile=CareerProfileContext(user_id="u1"),
        task=ConversationTaskState(active_application_id="app-1"),
    )

    assert project_restart_mock_interview_arguments(context, {}) == {"user_id": "u1"}
    with pytest.raises(ValueError):
        project_restart_mock_interview_arguments(context, {"session_id": "s-1"})


class _AlwaysRestart:
    def __init__(self) -> None:
        self.calls = 0

    def decide(self, context, tool_specs):
        self.calls += 1
        return AgentDecision(
            action="tool_call",
            tool_call=ToolCall(name="restart_mock_interview", arguments={}),
        )


def test_the_new_question_reaches_the_candidate_and_claims_the_next_turn(
    tmp_path,
) -> None:
    """A restart ends the turn on its first question, the same as a start does.

    Both enter a run, so both hand the conversation to the workflow. Routing the
    restart back to the decision model instead would leave the new question behind
    another tool call, and the candidate would be answering into a run whose
    question they were never shown.
    """
    store, graph, stuck_id = _stuck_run(tmp_path)
    manager = ContextManager(CareerContextStore(tmp_path / "context.sqlite3"))
    manager.upsert_profile(CareerProfileContext(user_id="u1"))
    seed = manager.load_for_turn(
        user_id="u1", conversation_id="c1", user_message="我的回答"
    )
    manager.commit_turn(
        context=seed,
        task=ConversationTaskState(
            active_application_id="app-1",
            active_workflow="mock_interview",
            phase="mock_interview_checkpoint_missing",
            run_id=stuck_id,
        ),
        assistant_message="执行断点已经丢失。",
    )
    decision_maker = _AlwaysRestart()
    runtime = MainAgentRuntime(
        context_manager=manager,
        decision_maker=decision_maker,
        tools=_registry(store, graph),
    )

    result = runtime.run_turn(
        user_id="u1", conversation_id="c1", user_message="重新开始一场"
    )

    assert "介绍一个你负责的检索改进。" in result.assistant_message
    assert result.context.task.phase == "mock_interview_answer_required"
    assert result.context.task.run_id != stuck_id
    # One decision, then the workflow owns the turn: the loop did not come back
    # for a second tool call after the question was asked.
    assert decision_maker.calls == 1


def test_a_failed_replacement_does_not_claim_the_next_runtime_turn(tmp_path) -> None:
    """The full observe/reducer path must release the retired run as well."""

    class FailReplacementPlan(OneQuestionWorker):
        fail = False

        def plan(self, **kwargs):
            if self.fail:
                raise AgentWorkerError(
                    "PLAN_FAILED", "replacement plan failed", retryable=True
                )
            return super().plan(**kwargs)

    store = SQLiteMockInterviewStore(tmp_path / "mock.sqlite3")
    worker = FailReplacementPlan()
    graph = MockInterviewGraph(store=store, worker=worker, sources=FixedSources())
    started = graph.start(
        MockInterviewStartRequest(
            user_id="u1",
            application_id="app-1",
            job_posting_id="job-1",
            jd_snapshot_id="jd-1",
            resume_version_id="rv-1",
            interview_type="technical",
            max_primary_questions=1,
            max_follow_ups_per_question=0,
        )
    )
    graph.checkpointer.delete_thread(started.session_id)
    worker.fail = True
    manager = ContextManager(CareerContextStore(tmp_path / "context.sqlite3"))
    seed = manager.load_for_turn(
        user_id="u1", conversation_id="c1", user_message="我的回答"
    )
    manager.commit_turn(
        context=seed,
        task=ConversationTaskState(
            active_application_id="app-1",
            active_workflow="mock_interview",
            phase="mock_interview_checkpoint_missing",
            run_id=started.session_id,
        ),
        assistant_message="执行断点已经丢失。",
    )
    runtime = MainAgentRuntime(
        context_manager=manager,
        decision_maker=_AlwaysRestart(),
        tools=_registry(store, graph),
    )

    result = runtime.run_turn(
        user_id="u1", conversation_id="c1", user_message="重新开始一场"
    )

    assert result.context.task.active_workflow == "none"
    assert result.context.task.run_id is None
    assert [item.state for item in result.tool_results] == [
        "mock_interview_restart_failed",
    ]
    assert result.context.tool_observations[-1].state == "authorization_refused"
    assert result.delegated_write_count == 1
    assert "替代面试暂时启动失败" in result.assistant_message
    persisted = manager.get_task(user_id="u1", conversation_id="c1")
    assert persisted.active_workflow == "none"
