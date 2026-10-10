from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from threading import Event

import pytest

from career_agent.agent.context.deployment import ContextDeploymentConfig
from career_agent.agent.context.manager import ContextManager
from career_agent.agent.contracts.memory import ConversationSummaryDelta, SummaryKeyFact
from career_agent.agent.contracts.task_state import ConversationTaskState
from career_agent.agent.providers.openai_client import AgentWorkerError
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


def setup_manager(tmp_path, worker, **options):
    store = CareerContextStore(tmp_path / "memory.sqlite3")
    plain = ContextManager(store)
    for i in range(5):
        context = plain.load_for_turn(user_id="u", conversation_id="c", user_message=f"fact_{i}")
        plain.commit_turn(context=context, task=ConversationTaskState(), assistant_message=f"answer_{i}")
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
    manager, store, notices = setup_manager(tmp_path, worker, recent_message_limit=2)
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
    manager, store, notices = setup_manager(tmp_path, worker, recent_message_limit=2)
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
    manager, store, notices = setup_manager(tmp_path, worker)
    manager.configure_request_token_estimator(lambda _: (650, 1000))
    try:
        context = load(manager)
        assert worker.calls == 0
        assert not notices
        manager.commit_turn(context=context, task=ConversationTaskState(), assistant_message="done")
        assert worker.entered.wait(1)
        assert not worker.release.is_set()
        assert not notices
    finally:
        worker.release.set()
        manager.close()


def test_wait_timeout_runs_sync_and_late_result_loses_cas(tmp_path):
    worker = Worker(block=True)
    manager, store, notices = setup_manager(tmp_path, worker, recent_message_limit=2, summary_wait_timeout_seconds=0.01)
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
    manager, store, notices = setup_manager(tmp_path, worker, recent_message_limit=2)
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
    manager, store, notices = setup_manager(tmp_path, worker, recent_message_limit=2, summary_job_limit=1)
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
    manager, store, _ = setup_manager(tmp_path, Worker(), recent_message_limit=2)
    schedule(manager, store)
    manager.close()
    restarted = ContextManager(CareerContextStore(store.path), summary_worker=Worker(), background_summary=True, recent_message_limit=2)
    try:
        assert load(restarted).through_sequence == 8
    finally:
        restarted.close()


def test_fact_migration_does_not_backfill_on_every_open(tmp_path):
    manager, store, _ = setup_manager(tmp_path, Worker(), recent_message_limit=2)
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
