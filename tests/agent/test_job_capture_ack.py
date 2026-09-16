from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from career_agent.agent.context_manager import ContextManager
from career_agent.agent.input_resources import InputResourceRejectedError
from career_agent.agent.main_agent_contracts import (
    AgentDecision,
    CareerProfileContext,
    ConversationTaskState,
    MainAgentContext,
)
from career_agent.agent.main_agent_runtime import (
    MainAgentRuntime,
    MainAgentTurnResult,
    ReplayedTurn,
)
from career_agent.agent.main_agent_tools import MainAgentToolRegistry
from career_agent.domain.job_discovery import JobDetail, Provenance
from career_agent.harness.streaming import PublicStreamEvent, TurnInputResource
from career_agent.storage.context import CareerContextStore
from career_agent.storage.job_captures import SQLiteJobCaptureStore
from career_agent.storage.jobs import SQLiteJobPostingRepository
from career_agent.storage.turn_receipts import SQLiteTurnReceiptStore


class DecisionMaker:
    def __init__(self) -> None:
        self.calls = 0

    def decide(self, context: MainAgentContext, tool_specs) -> AgentDecision:
        self.calls += 1
        return AgentDecision(action="final", message="已收到 JD，可以继续分析。")


@pytest.mark.parametrize("ack_on_first_success", [True, False])
def test_rejected_capture_stays_pending_until_a_committed_continuation(
    tmp_path: Path, ack_on_first_success: bool
) -> None:
    context_store = CareerContextStore(tmp_path / "context.sqlite3")
    receipts = SQLiteTurnReceiptStore(tmp_path / "context.sqlite3")
    captures = SQLiteJobCaptureStore(tmp_path / "jobs.sqlite3")
    jobs = SQLiteJobPostingRepository(tmp_path / "jobs.sqlite3")
    manager = ContextManager(context_store)
    manager.upsert_profile(CareerProfileContext(user_id="u1"))
    decisions = DecisionMaker()
    runtime = MainAgentRuntime(
        context_manager=manager,
        decision_maker=decisions,
        tools=MainAgentToolRegistry(job_repository=jobs, job_capture_store=captures),
        turn_receipt_store=receipts,
    )
    captured_at = datetime.now(timezone.utc)
    saved = jobs.save_captured_detail(
        user_id="u1",
        detail=JobDetail(
            source_name="boss",
            source_job_id="boss-1",
            title="AI 产品经理",
            company_name="示例科技",
            description="负责 AI 产品规划。",
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
    intent = captures.create_intent(
        user_id="u1", conversation_id="c1", platform="boss", keyword="AI", city=None
    )
    capture = captures.record_capture(
        intent=intent,
        job_posting_id=saved.posting.id,
        jd_snapshot_id=saved.snapshot.id,
        title=saved.posting.title,
        company_name=saved.posting.company_name,
    ).event
    manager.commit_workflow_turn(
        context=manager.load_for_workflow_turn(
            user_id="u1", conversation_id="c1", task=ConversationTaskState()
        ),
        task=ConversationTaskState(
            active_workflow="mock_interview",
            run_id="mock-1",
            phase="mock_interview_answer_required",
        ),
    )
    events: list[PublicStreamEvent] = []
    message_counts_at_commit: list[int] = []
    should_ack = ack_on_first_success

    def receive(event: PublicStreamEvent) -> None:
        events.append(event)
        if event.type in {"turn_completed", "turn_suspended"}:
            message_counts_at_commit.append(
                len(context_store.list_messages("u1", "c1", limit=10))
            )
            if should_ack:
                captures.acknowledge_event(user_id="u1", event_id=capture.id)

    def continue_capture() -> MainAgentTurnResult | ReplayedTurn:
        return runtime.run_turn(
            user_id="u1",
            conversation_id="c1",
            user_message="我已经保存了岗位，请基于这份 JD 继续分析。",
            request_id=capture.id,
            input_resources=(TurnInputResource(kind="jd_snapshot", id=saved.snapshot.id),),
            event_sink=receive,
        )

    with pytest.raises(InputResourceRejectedError):
        continue_capture()

    assert [event.type for event in events] == ["turn_started", "progress", "turn_failed"]
    assert events[-1].type == "turn_failed"
    assert events[-1].code == "INPUT_RESOURCE_REJECTED"
    assert context_store.list_messages("u1", "c1", limit=10) == ()
    assert [event.id for event in captures.list_pending_events(user_id="u1")] == [capture.id]
    assert decisions.calls == 0

    manager.commit_workflow_turn(
        context=manager.load_for_workflow_turn(
            user_id="u1",
            conversation_id="c1",
            task=manager.get_task(user_id="u1", conversation_id="c1"),
        ),
        task=ConversationTaskState(),
    )
    events.clear()
    first = continue_capture()

    assert isinstance(first, MainAgentTurnResult)
    assert message_counts_at_commit == [2]
    assert len(context_store.list_messages("u1", "c1", limit=10)) == 2
    assert len(captures.list_pending_events(user_id="u1")) == (0 if should_ack else 1)
    receipt = receipts.get(user_id="u1", conversation_id="c1", request_id=capture.id)
    assert receipt is not None and receipt.status == "COMMITTED"

    should_ack = True
    events.clear()
    replay = continue_capture()

    assert isinstance(replay, ReplayedTurn)
    assert replay.turn_id == receipt.turn_id
    assert message_counts_at_commit == [2, 2]
    assert len(context_store.list_messages("u1", "c1", limit=10)) == 2
    assert captures.list_pending_events(user_id="u1") == ()
    assert decisions.calls == 1
