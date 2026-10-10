from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from threading import Event

import pytest

from career_agent.agent.context.deployment import ContextDeploymentConfig
from career_agent.agent.context.manager import ContextManager
from career_agent.agent.contracts.memory import ConversationSummaryDelta, SummaryKeyFact
from career_agent.agent.contracts.task_state import ConversationTaskState
from career_agent.agent.providers.openai_client import AgentWorkerError
from career_agent.agent.providers.token_budget import message_token_count
from career_agent.storage.context import CareerContextStore
from career_agent.harness.observability import ACTIVE_TRACE_CONTEXT, InMemoryTraceRecorder


class Worker:
    def __init__(self, *, block=False, fail_first=False):
        self.entered = Event()
        self.release = Event()
        self.calls = 0
        self.block = block
        self.fail_first = fail_first

    def extract_delta(self, *, messages, **kwargs):
        self.calls += 1
        if self.calls == 1:
            self.entered.set()
            if self.block:
                assert self.release.wait(5)
            if self.fail_first:
                raise AgentWorkerError("CONVERSATION_SUMMARY_TIMEOUT", "Timeout")
        return ConversationSummaryDelta(new_facts=tuple(
            SummaryKeyFact(fact=m.content, source_sequence=m.sequence)
            for m in messages
        ))


def setup_manager(tmp_path, worker, *, seed_turns=5, **options):
    store = CareerContextStore(tmp_path / "memory.sqlite3")
    plain = ContextManager(store)
    for i in range(seed_turns):
        context = plain.load_for_turn(user_id="u", conversation_id="c", user_message=f"fact_{i}")
        plain.commit_turn(context=context, task=ConversationTaskState(), assistant_message=f"answer_{i}")
    options.setdefault("max_recent_context_tokens", 24)
    manager = ContextManager(store, summary_worker=worker, background_summary=True, **options)
    notices = []
    manager.on_compaction(notices.append)
    return manager, store, notices


def schedule(manager, store):
    context = ContextManager(store).load_for_turn(user_id="u", conversation_id="c", user_message="next")
    manager.commit_turn(context=context, task=ConversationTaskState(), assistant_message="done")


def load(manager):
    return manager.load_for_turn(user_id="u", conversation_id="c", user_message="next")


def test_load_waits_announces_and_has_no_watermark_gap(tmp_path):
    worker = Worker(block=True)
    manager, store, notices = setup_manager(tmp_path, worker, max_recent_context_tokens=24)
    try:
        schedule(manager, store)
        assert worker.entered.wait(1)
        announced = Event()
        manager.on_compaction(lambda phase: (notices.append(phase), announced.set()))
        with ThreadPoolExecutor() as pool:
            pending = pool.submit(load, manager)
            assert announced.wait(1)
            assert not pending.done()
            worker.release.set()
            context = pending.result(timeout=2)
        assert notices == ["load"]
        assert context.through_sequence == 8
        assert [m.content for m in context.recent_messages] == ["fact_4", "answer_4", "next", "done"]
        assert {f.source_sequence for f in store.list_summary_facts(user_id="u", conversation_id="c")} == set(range(1, 9))
    finally:
        worker.release.set()
        manager.close()


@pytest.mark.parametrize("in_flight", [False, True])
def test_failed_background_job_is_retried_synchronously(tmp_path, in_flight):
    worker = Worker(block=in_flight, fail_first=True)
    manager, store, notices = setup_manager(tmp_path, worker, max_recent_context_tokens=24)
    try:
        schedule(manager, store)
        assert worker.entered.wait(1)
        worker.release.set()
        if not in_flight:
            manager._summary_pool.submit(lambda: None).result(timeout=2)
        context = load(manager)
        assert worker.calls == 2
        assert notices and set(notices) == {"load"}
        assert context.through_sequence == 8
    finally:
        worker.release.set()
        manager.close()


def test_soft_pressure_only_schedules_on_commit(tmp_path):
    worker = Worker(block=True)
    manager, store, notices = setup_manager(tmp_path, worker, max_recent_context_tokens=1000)
    manager.configure_request_token_estimator(lambda _: (650, 1000))
    try:
        context = load(manager)
        assert worker.calls == 0
        assert not notices
        manager.commit_turn(context=context, task=ConversationTaskState(), assistant_message="done")
        assert worker.entered.wait(1)
        assert not worker.release.is_set()
        assert not notices
        # A running job does not delay a load below the hard token threshold.
        loaded = load(manager)
        assert len(loaded.recent_messages) == 12
        assert loaded.through_sequence == 0
        assert worker.calls == 1
        assert not notices
    finally:
        worker.release.set()
        manager.close()


def test_wait_timeout_runs_sync_and_late_result_loses_cas(tmp_path):
    worker = Worker(block=True)
    manager, store, notices = setup_manager(tmp_path, worker, max_recent_context_tokens=24, summary_wait_timeout_seconds=0.01)
    recorder = InMemoryTraceRecorder()
    token = ACTIVE_TRACE_CONTEXT.set((recorder, "run"))
    try:
        schedule(manager, store)
        assert worker.entered.wait(1)
        context = load(manager)
        assert worker.calls == 2
        assert notices == ["load", "load"]
        events = recorder.snapshot("run").events
        waits = [e for e in events if e.event_type == "context_compaction_waited"]
        sync = [e for e in events if e.event_type == "context_compaction_sync_finished"]
        assert waits[0].details["wait_result"] == "timeout"
        assert waits[0].duration_ms is not None
        assert sync[0].details == {"phase": "load", "background": False}
        assert context.through_sequence == 8
        worker.release.set()
        manager.close()
        assert store.get_conversation_summary(user_id="u", conversation_id="c").through_sequence == 8
        assert len(store.list_summary_facts(user_id="u", conversation_id="c")) == 8
    finally:
        ACTIVE_TRACE_CONTEXT.reset(token)
        worker.release.set()
        manager.close()


def test_circuit_skips_pending_wait_and_sync(tmp_path):
    worker = Worker()
    manager, store, notices = setup_manager(tmp_path, worker, max_recent_context_tokens=24)
    pending = Future()
    manager._summary_jobs[("u", "c")] = pending
    manager._compaction_failures[("u", "c")] = (3, datetime.now(timezone.utc))
    try:
        context = load(manager)
        assert not pending.done()
        assert worker.calls == 0
        assert not notices
        assert context.through_sequence == 0
    finally:
        manager.close()


def test_job_limit_records_skipped_trace(tmp_path):
    worker = Worker(block=True)
    manager, store, notices = setup_manager(tmp_path, worker, max_recent_context_tokens=24, summary_job_limit=1)
    recorder = InMemoryTraceRecorder()
    token = ACTIVE_TRACE_CONTEXT.set((recorder, "run"))
    try:
        schedule(manager, store)
        assert worker.entered.wait(1)
        manager._schedule_summary(user_id="u", conversation_id="other", trigger="occupancy", phase="commit", pressure=manager._legacy_pressure(user_id="u", conversation_id="c", after_sequence=0))
        skipped = [event for event in recorder.snapshot("run").events if event.event_type == "context_compaction_skipped"]
        assert skipped[0].details["reason"] == "job_limit"
        assert skipped[0].details["limit"] == 1
        ACTIVE_TRACE_CONTEXT.reset(token)
    finally:
        worker.release.set()
        manager.close()


def test_restart_load_recovers_from_persisted_watermark(tmp_path):
    manager, store, _ = setup_manager(tmp_path, Worker(), max_recent_context_tokens=24)
    schedule(manager, store)
    manager.close()
    restarted = ContextManager(CareerContextStore(store.path), summary_worker=Worker(), background_summary=True, max_recent_context_tokens=24)
    try:
        assert load(restarted).through_sequence == 8
    finally:
        restarted.close()


def test_fact_migration_does_not_backfill_on_every_open(tmp_path):
    manager, store, _ = setup_manager(tmp_path, Worker(), max_recent_context_tokens=24)
    load(manager)
    manager.close()
    import sqlite3
    with sqlite3.connect(store.path) as connection:
        connection.execute("DELETE FROM conversation_fact_archive")
    assert CareerContextStore(store.path).list_summary_facts(user_id="u", conversation_id="c") == ()


def test_soft_threshold_configuration():
    assert ContextDeploymentConfig.from_env(environ={}).compact_soft_threshold == 0.6
    assert ContextDeploymentConfig.from_env(environ={"CONTEXT_COMPACT_SOFT_THRESHOLD": "0.55"}).compact_soft_threshold == 0.55
    with pytest.raises(ValueError):
        ContextDeploymentConfig(compact_soft_threshold=0.8)


def test_many_short_messages_do_not_trigger_compaction_or_leave_window_gaps(tmp_path):
    worker = Worker()
    manager, store, notices = setup_manager(
        tmp_path, worker, seed_turns=40, max_recent_context_tokens=8000
    )
    manager.configure_request_token_estimator(lambda _: (500, 1000))
    try:
        context = load(manager)
        assert context.through_sequence == 0
        assert len(context.recent_messages) == 80
        manager.commit_turn(
            context=context, task=ConversationTaskState(), assistant_message="done"
        )
        assert len(load(manager).recent_messages) == 82
        assert worker.calls == 0
        assert not manager._summary_jobs
        assert not notices
    finally:
        manager.close()


def test_pending_summary_waits_on_hard_token_pressure_then_remeasures(tmp_path):
    worker = Worker(block=True)
    manager, store, notices = setup_manager(tmp_path, worker, seed_turns=9, max_recent_context_tokens=1000)
    manager.configure_request_token_estimator(
        lambda context: ((800 if context.user_message == "large" else 650) if context.through_sequence == 0 else 500, 1000)
    )
    try:
        context = load(manager)
        manager.commit_turn(
            context=context, task=ConversationTaskState(), assistant_message="done"
        )
        assert worker.entered.wait(1)
        announced = Event()
        manager.on_compaction(lambda phase: (notices.append(phase), announced.set()))
        with ThreadPoolExecutor() as pool:
            pending = pool.submit(
                manager.load_for_turn, user_id="u", conversation_id="c", user_message="large"
            )
            assert announced.wait(1)
            assert not pending.done()
            worker.release.set()
            context = pending.result(timeout=2)
        assert context.through_sequence == 8
        assert worker.calls == 1
        assert notices == ["load"]
    finally:
        worker.release.set()
        manager.close()


def test_uncropped_recent_token_quota_triggers_even_when_request_estimate_is_low(tmp_path):
    store = CareerContextStore(tmp_path / "memory.sqlite3")
    writer = ContextManager(store)
    context = writer.load_for_turn(
        user_id="u", conversation_id="c", user_message="甲乙丙丁" * 20
    )
    writer.commit_turn(
        context=context, task=ConversationTaskState(),
        assistant_message="戊己庚辛" * 20,
    )
    worker = Worker()
    manager = ContextManager(
        store, summary_worker=worker, background_summary=True,
        max_recent_context_tokens=100, max_recent_message_tokens=100,
    )
    manager.configure_request_token_estimator(lambda _: (500, 1000))
    notices = []
    manager.on_compaction(notices.append)
    try:
        loaded = load(manager)
        assert worker.calls == 1
        assert notices == ["load"]
        assert loaded.through_sequence == 2
        assert loaded.recent_messages == ()
        assert {fact.source_sequence for fact in store.list_summary_facts(
            user_id="u", conversation_id="c"
        )} == {1, 2}
    finally:
        manager.close()


@pytest.mark.parametrize("budget", ["request", "recent"])
def test_either_budget_alone_queues_at_soft_pressure_without_load_wait(tmp_path, budget):
    worker = Worker(block=True)
    manager, store, notices = setup_manager(
        tmp_path, worker,
        max_recent_context_tokens=50 if budget == "recent" else 1000,
    )
    manager.configure_request_token_estimator(
        lambda _: (650 if budget == "request" else 100, 1000)
    )
    recorder = InMemoryTraceRecorder()
    token = ACTIVE_TRACE_CONTEXT.set((recorder, "soft"))
    try:
        context = load(manager)
        manager.commit_turn(
            context=context, task=ConversationTaskState(), assistant_message="done"
        )
        assert worker.entered.wait(1)
        loaded = load(manager)
        assert loaded.through_sequence == 0
        assert not notices
        events = recorder.snapshot("soft").events
        scheduled = [e for e in events if e.event_type == "context_compaction_scheduled"]
        assert len(scheduled) == 1
        raw_tokens = sum(
            message_token_count(record.message.content)
            for record in store.list_message_records("u", "c", limit=20)
        )
        expected_recent = raw_tokens / (50 if budget == "recent" else 1000)
        assert scheduled[0].details["recent_occupancy"] == expected_recent
        assert 0.6 <= scheduled[0].details[f"{budget}_occupancy"] < 0.75
        assert not [e for e in events if e.event_type == "context_compaction_waited"]
    finally:
        ACTIVE_TRACE_CONTEXT.reset(token)
        worker.release.set()
        manager.close()


@pytest.mark.parametrize("budget", ["request", "recent"])
@pytest.mark.parametrize("pending", [False, True])
def test_load_reuses_initial_pressure_and_remeasures_only_after_wait(
    tmp_path, monkeypatch, budget, pending
):
    worker = Worker(block=pending)
    manager, store, notices = setup_manager(
        tmp_path, worker, max_recent_context_tokens=1000
    )
    manager.configure_request_token_estimator(
        lambda context: (800 if budget == "request" and context.through_sequence == 0 else 100, 1000)
    )
    manager._recent_token_occupancy = lambda **kwargs: (
        0.8 if budget == "recent" and kwargs["after_sequence"] == 0 else 0.1
    )
    measurements = []
    original = manager._recent_pressure

    def measured(**kwargs):
        measurements.append(kwargs["after_sequence"])
        return original(**kwargs)

    monkeypatch.setattr(manager, "_recent_pressure", measured)
    announced = Event()
    manager.on_compaction(lambda phase: (notices.append(phase), announced.set()))
    try:
        if pending:
            manager._carried_request_tokens[("u", "c")] = (
                800 if budget == "request" else 100, 1000
            )
            schedule(manager, store)
            assert worker.entered.wait(1)
            with ThreadPoolExecutor() as pool:
                future = pool.submit(load, manager)
                assert announced.wait(1)
                assert not future.done()
                worker.release.set()
                context = future.result(timeout=2)
            assert measurements == [0, 8]
        else:
            context = load(manager)
            assert measurements == [0]
        assert worker.calls == 1
        assert notices == ["load"]
        assert context.through_sequence == 8
    finally:
        worker.release.set()
        manager.close()


@pytest.mark.parametrize(
    "request_tokens,recent_ratio,expected_calls",
    [(100, 0.1, 0), (100, 0.65, 1), (650, 0.1, 0)],
)
def test_workflow_exit_queues_only_when_token_budget_crosses_soft_threshold(
    tmp_path, request_tokens, recent_ratio, expected_calls
):
    worker = Worker()
    manager, store, notices = setup_manager(
        tmp_path, worker, seed_turns=0, max_recent_context_tokens=1000
    )
    manager.configure_request_token_estimator(lambda _: (request_tokens, 1000))
    manager._recent_token_occupancy = lambda **_: recent_ratio
    try:
        entry = load(manager)
        held = manager.commit_workflow_entry(
            context=entry,
            task=ConversationTaskState(
                active_workflow="mock_interview", run_id="mock-1",
                phase="mock_interview_answer_required",
            ),
        )
        context = manager.load_for_workflow_turn(
            user_id="u", conversation_id="c", task=held
        )
        # The workflow envelope invalidates the entry's request estimate;
        # only current, measurable recent pressure can queue this exit.
        assert ("u", "c") not in manager._carried_request_tokens
        manager.commit_workflow_exit(
            context=context, task=ConversationTaskState(),
            assistant_message="模拟面试完成，结果资源 mock-1。",
        )
        manager.close()
        assert worker.calls == expected_calls
        assert not notices
        records = store.list_message_records("u", "c", limit=10)
        assert len(records) == 2
    finally:
        manager.close()
