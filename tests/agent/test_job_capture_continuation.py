"""Opening BOSS leaves a durable intent behind; the saved job comes back as
a verified ``job_posting`` attachment that starts a *new* turn in the same
conversation, with no checkpoint on the main graph."""

from __future__ import annotations

from datetime import datetime, timezone
from urllib.parse import parse_qs, urlparse

import pytest

from career_agent.agent.context_manager import ContextManager
from career_agent.agent.input_resources import InputResourceNotFoundError
from career_agent.agent.main_agent_contracts import (
    AgentDecision,
    CareerProfileContext,
    ToolCall,
)
from career_agent.agent.main_agent_runtime import MainAgentRuntime
from career_agent.agent.main_agent_tools import MainAgentToolRegistry
from career_agent.domain.job_discovery import JobDetail, Provenance
from career_agent.harness.streaming import ClientActionEvent, TurnInputResource
from career_agent.storage.context import CareerContextStore
from career_agent.storage.job_captures import SQLiteJobCaptureStore
from career_agent.storage.jobs import SQLiteJobPostingRepository


class RecordingDecisionMaker:
    def __init__(self, *decisions: AgentDecision) -> None:
        self.decisions = list(decisions)
        self.contexts = []

    def decide(self, context, tool_specs):
        self.contexts.append(context)
        return self.decisions.pop(0)


def _runtime(tmp_path, decisions, *, repository=None, capture_store=None):
    context_store = CareerContextStore(tmp_path / "context.sqlite3")
    manager = ContextManager(context_store)
    manager.upsert_profile(CareerProfileContext(user_id="u1", default_city="上海"))
    manager.upsert_profile(CareerProfileContext(user_id="u2", default_city="上海"))
    return MainAgentRuntime(
        context_manager=manager,
        decision_maker=decisions,
        tools=MainAgentToolRegistry(
            job_repository=repository, job_capture_store=capture_store
        ),
    ), context_store


def _save_job(repository, *, user_id="u1", source_job_id="boss-1"):
    captured_at = datetime(2026, 9, 12, tzinfo=timezone.utc)
    return repository.save_captured_detail(
        user_id=user_id,
        detail=JobDetail(
            source_name="boss",
            source_job_id=source_job_id,
            title="AI 产品经理",
            company_name="示例科技",
            description="负责 AI 产品规划。",
            city="上海",
            salary="25-35K",
            captured_at=captured_at,
            provenance=Provenance(
                source_name="boss",
                source_job_id=source_job_id,
                captured_at=captured_at,
                operation="browser_explicit_save",
                adapter_version="test-v1",
            ),
        ),
    )


def test_open_job_search_creates_an_intent_for_this_conversation_outside_the_url(
    tmp_path,
) -> None:
    capture_store = SQLiteJobCaptureStore(tmp_path / "jobs.sqlite3")
    decisions = RecordingDecisionMaker(
        AgentDecision(
            action="tool_call",
            tool_call=ToolCall(name="open_job_search", arguments={"keyword": "AI 产品经理"}),
        ),
        AgentDecision(action="final", message="已打开搜索页。"),
    )
    runtime, _ = _runtime(tmp_path, decisions, capture_store=capture_store)
    events = []

    result = runtime.run_turn(
        user_id="u1",
        conversation_id="c1",
        user_message="帮我找上海的 AI 产品经理岗位",
        event_sink=events.append,
    )

    action = next(event for event in events if isinstance(event, ClientActionEvent))
    assert action.capture_intent_id is not None
    intent = capture_store.get_live_intent(user_id="u1", intent_id=action.capture_intent_id)
    assert intent is not None
    assert intent.conversation_id == "c1"
    assert intent.keyword == "AI 产品经理" and intent.city == "上海"
    assert action.capture_intent_expires_at == intent.expires_at
    # The id travels through the extension bridge, never through BOSS.
    assert action.capture_intent_id not in action.url
    assert set(parse_qs(urlparse(action.url).query)) == {"query", "city"}
    payload = result.tool_results[0].payload
    assert payload["client_action"]["capture_intent_id"] == intent.id
    # The model asked for a keyword; the runtime, not the model, bound the
    # intent to the authenticated user and the current conversation.
    assert capture_store.get_live_intent(user_id="u2", intent_id=intent.id) is None


def test_open_job_search_without_a_capture_store_still_opens_the_page(tmp_path) -> None:
    decisions = RecordingDecisionMaker(
        AgentDecision(
            action="tool_call",
            tool_call=ToolCall(name="open_job_search", arguments={"keyword": "AI"}),
        ),
        AgentDecision(action="final", message=""),
    )
    events = []

    runtime, _ = _runtime(tmp_path, decisions)
    runtime.run_turn(
        user_id="u1", conversation_id="c1", user_message="找 AI 岗位", event_sink=events.append
    )

    action = next(event for event in events if isinstance(event, ClientActionEvent))
    assert action.capture_intent_id is None
    assert action.capture_intent_expires_at is None


def test_captured_job_arrives_as_a_verified_attachment_in_a_new_turn(tmp_path) -> None:
    repository = SQLiteJobPostingRepository(tmp_path / "jobs.sqlite3")
    record = _save_job(repository)
    decisions = RecordingDecisionMaker(
        AgentDecision(action="final", message="好的，先搜索。"),
        AgentDecision(action="final", message="这个岗位和你很匹配。"),
    )
    runtime, context_store = _runtime(tmp_path, decisions, repository=repository)

    first = runtime.run_turn(
        user_id="u1", conversation_id="c1", user_message="帮我找 AI 产品经理岗位"
    )
    second = runtime.run_turn(
        user_id="u1",
        conversation_id="c1",
        user_message="我刚在 BOSS 保存了「AI 产品经理 · 示例科技」，请继续分析。",
        input_resources=(TurnInputResource(kind="job_posting", id=record.posting.id),),
    )

    # Two complete turns, the second a fresh one: nothing about the first was
    # suspended or resumed, and no interaction was left open between them.
    assert first.assistant_message == "好的，先搜索。"
    assert second.assistant_message == "这个岗位和你很匹配。"
    assert first.context.task.active_job_posting_id is None
    follow_up = decisions.contexts[1]
    assert follow_up.task.active_job_posting_id == record.posting.id
    [candidate] = follow_up.task.saved_job_candidates
    assert candidate.job_posting_id == record.posting.id
    assert (candidate.title, candidate.company_name, candidate.city, candidate.salary) == (
        "AI 产品经理",
        "示例科技",
        "上海",
        "25-35K",
    )
    assert second.context.task.active_job_posting_id == record.posting.id
    assert follow_up.attached_resumes == ()
    messages = context_store.list_messages("u1", "c1", limit=10)
    assert [message.role for message in messages] == ["user", "assistant", "user", "assistant"]


def test_a_job_saved_by_someone_else_is_refused_before_anything_is_written(tmp_path) -> None:
    repository = SQLiteJobPostingRepository(tmp_path / "jobs.sqlite3")
    theirs = _save_job(repository, user_id="u2")
    decisions = RecordingDecisionMaker(AgentDecision(action="final", message="不应到达。"))
    runtime, context_store = _runtime(tmp_path, decisions, repository=repository)

    with pytest.raises(InputResourceNotFoundError):
        runtime.run_turn(
            user_id="u1",
            conversation_id="c1",
            user_message="继续分析",
            input_resources=(TurnInputResource(kind="job_posting", id=theirs.posting.id),),
        )
    with pytest.raises(InputResourceNotFoundError):
        runtime.run_turn(
            user_id="u1",
            conversation_id="c1",
            user_message="继续分析",
            input_resources=(TurnInputResource(kind="job_posting", id="job_missing"),),
        )

    assert decisions.contexts == []
    assert context_store.list_messages("u1", "c1", limit=10) == ()


def test_the_continuation_pins_the_snapshot_it_was_saved_with(tmp_path) -> None:
    """The page attaches the ``jd_snapshot`` the extension saved. A later
    re-save of the same posting makes a v2, but this conversation keeps
    reading v1 when the user says "这个岗位"."""
    repository = SQLiteJobPostingRepository(tmp_path / "jobs.sqlite3")
    saved = _save_job(repository)
    decisions = RecordingDecisionMaker(
        AgentDecision(action="final", message="已保存，我可以继续做匹配分析。"),
        AgentDecision(
            action="tool_call", tool_call=ToolCall(name="get_saved_job", arguments={})
        ),
        AgentDecision(action="final", message="匹配分析如下。"),
    )
    runtime, context_store = _runtime(tmp_path, decisions, repository=repository)

    first = runtime.run_turn(
        user_id="u1",
        conversation_id="c1",
        user_message="我已经从 BOSS 保存了岗位，请基于这份 JD 继续分析。",
        input_resources=(TurnInputResource(kind="jd_snapshot", id=saved.snapshot.id),),
    )
    assert first.context.task.active_job_posting_id == saved.posting.id
    assert first.context.task.active_jd_snapshot_id == saved.snapshot.id

    captured_at = datetime(2026, 9, 13, tzinfo=timezone.utc)
    resaved = repository.save_captured_detail(
        user_id="u1",
        detail=JobDetail(
            source_name="boss",
            source_job_id="boss-1",
            title="AI 产品经理",
            company_name="示例科技",
            description="负责 AI 产品规划，新增：负责 Agent 方向。",
            city="上海",
            salary="25-35K",
            captured_at=captured_at,
            provenance=Provenance(
                source_name="boss",
                source_job_id="boss-1",
                captured_at=captured_at,
                operation="browser_explicit_save",
                adapter_version="test-v1",
            ),
        ),
    )
    assert resaved.posting.id == saved.posting.id
    assert resaved.snapshot.version == 2

    second = runtime.run_turn(
        user_id="u1", conversation_id="c1", user_message="分析一下我和这个岗位的匹配度"
    )

    read = second.tool_results[-1]
    assert read.state == "saved_job_ready"
    assert read.payload["jd_snapshot"]["id"] == saved.snapshot.id
    assert read.payload["jd_snapshot"]["version"] == 1
    assert read.payload["jd_snapshot"]["content"] == "负责 AI 产品规划。"
    assert read.resource_ref is not None
    assert read.resource_ref.resource_id == saved.snapshot.id
    messages = context_store.list_messages("u1", "c1", limit=10)
    assert [ref.resource_id for ref in messages[-1].resource_refs] == [saved.snapshot.id]


def test_a_snapshot_saved_by_someone_else_is_refused(tmp_path) -> None:
    repository = SQLiteJobPostingRepository(tmp_path / "jobs.sqlite3")
    theirs = _save_job(repository, user_id="u2")
    decisions = RecordingDecisionMaker(AgentDecision(action="final", message="不应到达。"))
    runtime, context_store = _runtime(tmp_path, decisions, repository=repository)

    with pytest.raises(InputResourceNotFoundError):
        runtime.run_turn(
            user_id="u1",
            conversation_id="c1",
            user_message="继续分析",
            input_resources=(TurnInputResource(kind="jd_snapshot", id=theirs.snapshot.id),),
        )

    assert decisions.contexts == []
    assert context_store.list_messages("u1", "c1", limit=10) == ()
