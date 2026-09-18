from __future__ import annotations

from pathlib import Path

import pytest

from compaction_smoke_093 import FactSummaryWorker, MeteredSummaryWorker, run_trajectory
from career_agent.agent.context_manager import ContextManager
from career_agent.agent.main_agent_contracts import ConversationTaskState, MainAgentContext
from career_agent.agent.token_budget import message_token_count
from career_agent.harness.observability import ACTIVE_TRACE_CONTEXT, InMemoryTraceRecorder
from career_agent.storage.context import CareerContextStore


def test_thirty_synthetic_messages_reduce_five_legacy_compactions_to_one(tmp_path: Path) -> None:
    legacy = run_trajectory(tmp_path / "legacy.sqlite3", recent=8, batch=4, worker=FactSummaryWorker())
    current = run_trajectory(tmp_path / "current.sqlite3", recent=16, batch=8, worker=FactSummaryWorker())
    assert legacy.compaction_calls == 5
    assert current.compaction_calls == 1
    assert current.compaction_calls <= 2
    assert legacy.watermark == 20
    assert current.watermark == 8
    for result in (legacy, current):
        assert result.retained_known_facts == result.known_fact_count == 4
        assert result.history_reachable
        assert result.page_in_calls == 1
        assert result.duplicate_tool_calls == result.duplicate_summary_calls == 0
        assert result.passed


def test_production_defaults_keep_contiguous_history_below_complete_request_occupancy(tmp_path: Path) -> None:
    store = CareerContextStore(tmp_path / "defaults.sqlite3")
    worker = MeteredSummaryWorker(FactSummaryWorker())
    manager = ContextManager(store, summary_worker=worker)
    manager.configure_request_token_estimator(lambda context: (16000, 32000))
    recorder = InMemoryTraceRecorder()
    token = ACTIVE_TRACE_CONTEXT.set((recorder, "synthetic-defaults"))
    try:
        for index in range(15):
            context = manager.load_for_turn(user_id="u", conversation_id="c", user_message=f"synthetic-{index}")
            assert len(context.recent_messages) <= 23
            assert context.through_sequence + len(context.recent_messages) == index * 2
            if context.recent_messages:
                assert context.recent_from_sequence == context.through_sequence + 1
            manager.commit_turn(context=context, task=ConversationTaskState(), assistant_message="ok")
    finally:
        ACTIVE_TRACE_CONTEXT.reset(token)
    events = [event for event in recorder.snapshot("synthetic-defaults").events if event.event_type == "context_compacted"]
    assert len(events) == len(worker.sequences) == 1
    assert worker.sequences == [tuple(range(1, 9))]
    assert events[0].details["trigger"] == "projection_overflow"
    assert events[0].details["occupancy"] < 0.75
    assert events[0].details["input_occupancy_denominator"] == 32000
    span = store.read_conversation_span(user_id="u", conversation_id="c", from_sequence=1, through_sequence=8)
    assert [item.sequence for item in span.messages] == list(range(1, 9))
    assert span.messages[0].content == "synthetic-0"


def test_one_long_current_message_triggers_complete_request_occupancy_before_row_overflow(tmp_path: Path) -> None:
    worker = MeteredSummaryWorker(FactSummaryWorker())
    manager = ContextManager(CareerContextStore(tmp_path / "long.sqlite3"), summary_worker=worker)

    def estimate(context: MainAgentContext) -> tuple[int, int]:
        # Fixed request surface and current observation projection are included,
        # not just recent transcript characters. Only the current message varies.
        return 22000 + message_token_count(context.user_message), 32000

    manager.configure_request_token_estimator(estimate, static_input_tokens=15000, max_input_tokens=32000)
    for index in range(4):
        context = manager.load_for_turn(user_id="u", conversation_id="c", user_message=f"short-{index}")
        manager.commit_turn(context=context, task=ConversationTaskState(), assistant_message="ok")
    assert worker.sequences == []
    recorder = InMemoryTraceRecorder()
    token = ACTIVE_TRACE_CONTEXT.set((recorder, "synthetic-long"))
    try:
        current = manager.load_for_turn(user_id="u", conversation_id="c", user_message="long " * 3200)
    finally:
        ACTIVE_TRACE_CONTEXT.reset(token)
    events = [event for event in recorder.snapshot("synthetic-long").events if event.event_type == "context_compacted"]
    assert len(events) == 1
    assert events[0].details["trigger"] == "occupancy"
    assert events[0].details["projection_overflow"] is False
    assert events[0].details["occupancy"] >= 0.75
    assert worker.sequences == [tuple(range(1, 9))]
    assert current.user_message == "long " * 3200
    assert current.through_sequence == 8
    assert message_token_count(current.user_message) <= 3400


@pytest.mark.parametrize("recent,batch", [(8, 4), (16, 8), (64, 32)])
def test_larger_row_window_does_not_borrow_current_turn_or_output_tokens(tmp_path: Path, recent: int, batch: int) -> None:
    manager = ContextManager(
        CareerContextStore(tmp_path / "budget.sqlite3"),
        recent_message_limit=recent, summary_batch_size=batch,
    )
    manager.configure_request_token_estimator(
        lambda context: (16000, 32000), static_input_tokens=15000, max_input_tokens=32000,
    )
    assert manager._user_message_tokens == 3400
    assert manager._recent_context_tokens == 6800
    assert manager._recent_message_tokens == 2550
    # The same 40% dynamic reserve remains for observations and projections;
    # input itself fits beside the configured output in the declared 64k window.
    assert 32000 - 15000 - manager._user_message_tokens - manager._recent_context_tokens == 6800
    assert 32000 + 16384 <= 65536
