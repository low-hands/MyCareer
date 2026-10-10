from datetime import datetime, timezone
import sqlite3

import pytest

from career_agent.evaluation.latency import summarize
from career_agent.harness.observability import (
    ACTIVE_TRACE_CONTEXT, InMemoryTraceRecorder, traced_provider_call,
)
from career_agent.storage.run_events import SQLiteTraceRecorder


def test_phase_counts_failures_overlap_missing_and_zeroes(tmp_path):
    path = tmp_path / "run-events.sqlite3"
    trace = SQLiteTraceRecorder(path)
    trace.record("long", "model_attempt", "main_agent_decide", model_call_category="orchestrator_decision", details={"context_chars": 16000})
    trace.record("long", "model_retry", "main_agent_decide", model_call_category="orchestrator_decision")
    trace.record("long", "model_succeeded", "main_agent_decide", model_call_category="orchestrator_decision", duration_ms=100, outcome="succeeded")
    trace.record("long", "model_attempt", "resume_job_match", model_call_category="capability_agent")
    trace.record("long", "model_failed", "resume_job_match", model_call_category="capability_agent", duration_ms=50, outcome="failed")
    trace.record("long", "capability_executed", "act", duration_ms=80, outcome="failed", details={"tool_name": "match_resume_to_job", "private": "must not be in report"})
    trace.record("long", "provider_request", "embedding", duration_ms=0, outcome="succeeded")
    trace.record("long", "provider_request", "conversation_summary", duration_ms=20, outcome="succeeded")
    trace.record("long", "context_compacted", "conversation_summary", outcome="succeeded")
    trace.record("short", "model_attempt", "main_agent_decide", model_call_category="orchestrator_decision", details={"context_chars": 1000})
    trace.record("short", "model_failed", "main_agent_decide", model_call_category="orchestrator_decision", duration_ms=200, outcome="failed")
    trace.record("short", "context_compaction_failed", "conversation_summary", outcome="failed")
    report = summarize(path)
    decision = report["phases"]["decision"]
    assert decision["started_count"] == 2  # retry notification is not a decision
    assert decision["completed_count"] == 2
    assert decision["operation_duration_ms"]["p95"] == 200
    assert decision["failed_duration_ms"]["total"] == 200
    assert report["phases"]["summary_compaction"]["completed_count"] == 2
    assert report["phases"]["summary_compaction"]["missing_duration_count"] == 1
    assert report["phases"]["summary_compaction"]["per_run_duration_ms"]["samples"] == 1
    assert report["phases"]["embedding"]["operation_duration_ms"]["samples"] == 1
    assert report["phases"]["tool_execution"]["per_run_count"]["mean"] == .5
    assert report["context_cohorts"]["long_gte_12000"]["runs"] == 1
    assert "must not be in report" not in str(report)
    selected = summarize(path, min_context_chars=12000, require_tool_trace=True)
    assert selected["run_count"] == 1
    assert selected["phases"]["decision"]["completed_count"] == 1


def test_read_only_legacy_database_and_whole_run_date_selection(tmp_path):
    path = tmp_path / "legacy.sqlite3"
    with sqlite3.connect(path) as c:
        c.execute("CREATE TABLE run_events (run_id TEXT, sequence INTEGER, event_type TEXT, stage TEXT, occurred_at TEXT, duration_ms INTEGER, outcome TEXT, details_json TEXT)")
        c.executemany("INSERT INTO run_events VALUES (?,?,?,?,?,?,?,?)", [
            ("r", 1, "model_attempt", "main_agent_decide", "2026-10-09T00:00:00Z", None, "started", '{"context_chars":5000}'),
            ("r", 2, "model_succeeded", "main_agent_decide", "2026-10-09T00:01:00Z", 60000, "succeeded", '{}')])
    original = path.read_bytes()
    assert summarize(path, until="2026-10-09T00:00:30Z")["phases"]["decision"]["completed_count"] == 1
    assert summarize(path, since="2026-10-09T00:00:30Z")["run_count"] == 0
    assert path.read_bytes() == original
    missing = tmp_path / "missing.sqlite3"
    with pytest.raises(sqlite3.OperationalError):
        summarize(missing)
    assert not missing.exists()


def test_provider_timing_preserves_return_error_and_content_privacy():
    @traced_provider_call("embedding")
    def call(private, fail=False):
        if fail:
            raise ValueError(private)
        return private
    recorder = InMemoryTraceRecorder()
    # Active trace context is the same contract used by the turn coordinator.
    token = ACTIVE_TRACE_CONTEXT.set((recorder, "run"))
    try:
        assert call("private resume text") == "private resume text"
        with pytest.raises(ValueError, match="private resume text"):
            call("private resume text", fail=True)
    finally:
        ACTIVE_TRACE_CONTEXT.reset(token)
    events = recorder.snapshot("run").events
    assert [e.outcome for e in events] == ["succeeded", "failed"]
    assert all(e.duration_ms is not None for e in events)
    assert "private resume text" not in str(events)


def test_unfinished_model_calls_are_not_zero_latency(tmp_path):
    path = tmp_path / "trace.sqlite3"
    trace = SQLiteTraceRecorder(path)
    trace.record("partial", "model_attempt", "main_agent_decide", model_call_category="orchestrator_decision")
    trace.record("partial", "model_succeeded", "main_agent_decide", model_call_category="orchestrator_decision", duration_ms=50, outcome="succeeded")
    trace.record("partial", "model_attempt", "main_agent_decide", model_call_category="orchestrator_decision")
    report = summarize(path)
    assert report["phases"]["decision"]["unfinished_model_calls"] == 1
    assert report["phases"]["decision"]["per_run_duration_ms"]["samples"] == 0
    assert report["phases"]["embedding"]["operation_duration_ms"]["p50"] is None
    assert report["phases"]["embedding"]["per_run_duration_ms"]["mean"] is None


def test_capture_isolates_all_stores_and_retains_failed_paused_samples(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import dotenv
    import career_agent.cli as cli
    from career_agent.evaluation.latency_capture import capture

    monkeypatch.setattr(dotenv, "load_dotenv", lambda: None)
    for suffix, value in {"BASE_URL": "https://example.test/v1", "API_KEY": "test-only", "MODEL": "test-model"}.items():
        monkeypatch.setenv("MAIN_AGENT_" + suffix, value)
    monkeypatch.setenv("CONVERSATION_SUMMARY_AGENT_DISABLE_THINKING", "true")
    root = tmp_path / "live"
    calls = []
    class FakeRuntime:
        closed = False
        def run_turn(self, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                raise ValueError("private provider body")
            return SimpleNamespace(context=SimpleNamespace(task=SimpleNamespace(pending_interaction=object())))
        def close(self):
            self.closed = True
    runtime = FakeRuntime()
    def build(args):
        assert all(Path(value).parent == root for name, value in vars(args).items() if name.endswith("_store"))
        return runtime
    from pathlib import Path
    monkeypatch.setattr(cli, "build_main_agent_runtime", build)
    report = capture(root, conversations=2)
    assert [row["status"] for row in report["turns"]] == ["failed", "returned"]
    assert report["turns"][1]["interrupted"] is True
    assert runtime.closed
    assert report["summary_config"]["disable_thinking"] is True
    assert report["summary_config"]["model"] == "test-model"
    assert len(calls) == 2
    assert "private provider body" not in str(report)
    with pytest.raises(FileExistsError):
        capture(root, conversations=1)


def test_reply_fraction_separates_next_tool_and_final_and_excludes_failed_runs(tmp_path):
    from career_agent.evaluation.latency import post_tool_reply
    def event(kind, ms=None, action=None, time="2026-10-09T00:00:00Z"):
        return {"run_id": "r", "event_type": kind, "stage": "main_agent_decide", "model_call_category": "orchestrator_decision", "duration_ms": ms, "outcome": "succeeded", "occurred_at": time, "details": {"decision_action": action}}
    events = [event("model_attempt"), event("capability_executed"), event("model_succeeded", 100, "tool_call"), event("capability_executed"), event("model_succeeded", 200, "final"), event("turn_completed", time="2026-10-09T00:00:01Z")]
    report = post_tool_reply({"r": events})
    assert report["final_reply"]["completed_count"] == 1
    assert report["continued_tool_selection"]["completed_count"] == 1
    assert report["weighted_reply_share"] == .2
    assert report["per_completed_run_reply_share"]["mean"] == .2
    events[-1] = event("turn_failed", time="2026-10-09T00:00:01Z")
    failed = post_tool_reply({"r": events})
    assert failed["weighted_reply_share"] is None
    assert failed["final_reply"]["completed_count"] == 1


def test_response_view_includes_ask_user_but_never_counts_tool_selection(tmp_path):
    from pathlib import Path
    import runpy
    analyze = runpy.run_path(str(Path(__file__).resolve().parents[2] / 'evals/latency/report_post_tool_response.py'))['analyze']
    path = tmp_path / 'trace.sqlite3'
    trace = SQLiteTraceRecorder(path)
    trace.record('r', 'model_attempt', 'main_agent_decide', model_call_category='orchestrator_decision')
    trace.record('r', 'capability_executed', 'act', duration_ms=1, outcome='succeeded', details={'tool_name': 'get_daily_brief'})
    trace.record('r', 'model_succeeded', 'main_agent_decide', model_call_category='orchestrator_decision', duration_ms=0, outcome='succeeded', details={'decision_action': 'tool_call'})
    trace.record('r', 'model_succeeded', 'main_agent_decide', model_call_category='orchestrator_decision', duration_ms=0, outcome='succeeded', details={'decision_action': 'ask_user'})
    trace.record('r', 'turn_completed', 'turn', outcome='succeeded')
    full = analyze(path)
    assert full['post_tool_reply']['ask_user_reply']['completed_count'] == 1
    assert full['post_tool_reply']['other_post_tool_decision']['completed_count'] == 0
    report = full['post_tool_response_message']
    assert report['combined']['completed_count'] == 1
    assert report['by_action']['ask_user']['completed_count'] == 1
    assert report['by_action']['final']['completed_count'] == 0
    assert report['combined']['operation_duration_ms']['total'] == 0


def test_known_absence_of_reply_is_zero_but_incomplete_telemetry_is_unknown(tmp_path):
    import json
    import runpy
    from pathlib import Path
    analyze = runpy.run_path(str(Path(__file__).resolve().parents[2] / 'evals/latency/report_post_tool_response.py'))['analyze']
    path = tmp_path / 'events.sqlite3'
    with sqlite3.connect(path) as c:
        c.execute('CREATE TABLE run_events (run_id TEXT, sequence INTEGER, event_type TEXT, stage TEXT, occurred_at TEXT, duration_ms INTEGER, outcome TEXT, details_json TEXT)')
        for run in ('closed', 'unfinished'):
            entries = [('model_attempt', 0, None, {}), ('capability_executed', 100, 1, {}), ('model_succeeded', 200, 100, {'decision_action': 'tool_call'})]
            if run == 'unfinished':
                entries.append(('model_attempt', 300, None, {}))
            entries.append(('turn_completed', 1000, None, {}))
            for seq, (kind, t, duration, details) in enumerate(entries):
                timestamp = f'2026-10-09T00:00:{t // 1000:02d}.{t % 1000:03d}Z'
                c.execute('INSERT INTO run_events VALUES (?,?,?,?,?,?,?,?)', (run, seq, kind, 'main_agent_decide', timestamp, duration, 'succeeded', json.dumps(details)))
    rows = {r['run_id']: r for r in analyze(path)['post_tool_response_message']['runs']}
    assert rows['closed']['response_ms'] == 0
    assert rows['closed']['share'] == 0
    assert rows['unfinished']['response_ms'] is None
    assert rows['unfinished']['share'] is None


def test_wall_time_join_measures_real_denominator_and_rejects_ambiguous_capture(tmp_path):
    import json
    import runpy
    from pathlib import Path
    module = runpy.run_path(str(Path(__file__).resolve().parents[2] / 'evals/latency/report_post_tool_response.py'))
    path = tmp_path / 'trace.sqlite3'
    trace = SQLiteTraceRecorder(path)
    trace.record('r', 'model_attempt', 'main_agent_decide', model_call_category='orchestrator_decision', details={'conversation_id': 'latency-1'})
    trace.record('r', 'capability_executed', 'act', duration_ms=1, outcome='succeeded')
    trace.record('r', 'model_succeeded', 'main_agent_decide', model_call_category='orchestrator_decision', duration_ms=200, outcome='succeeded', details={'decision_action': 'ask_user'})
    trace.record('r', 'turn_completed', 'turn', outcome='succeeded', details={'conversation_id': 'latency-1', 'tool_call_count': 1})
    manifest = tmp_path / 'capture.json'
    turn = {'conversation': 1, 'turn': 1, 'wall_ms': 1000, 'status': 'returned'}
    manifest.write_text(json.dumps({'finished_at': '2026-10-09T00:00:00Z', 'turns': [turn]}))
    report = module['analyze'](path)
    module['attach_capture_wall'](report, path, manifest)
    assert report['capture_wall_time']['weighted_reply_share'] == .2
    assert report['capture_wall_time']['by_action']['final']['weighted_wall_share'] == 0
    assert report['capture_wall_time']['by_action']['ask_user']['weighted_wall_share'] == .2
    assert report['capture_wall_time']['by_action']['ask_user']['mean_ms_per_eligible_turn'] == 200
    manifest.write_text(json.dumps({'finished_at': '2026-10-09T00:00:00Z', 'turns': [turn, {**turn, 'turn': 2}]}))
    with pytest.raises(ValueError, match='turn counts differ'):
        module['attach_capture_wall'](report, path, manifest)


def test_provider_trace_keeps_specific_worker_error_code():
    from career_agent.agent.providers.openai_client import AgentWorkerError
    @traced_provider_call("conversation_summary")
    def fail():
        raise AgentWorkerError("CONVERSATION_SUMMARY_TIMEOUT", "private body")
    recorder = InMemoryTraceRecorder()
    token = ACTIVE_TRACE_CONTEXT.set((recorder, "run"))
    try:
        with pytest.raises(AgentWorkerError):
            fail()
    finally:
        ACTIVE_TRACE_CONTEXT.reset(token)
    event = recorder.snapshot("run").events[0]
    assert event.error_code == "CONVERSATION_SUMMARY_TIMEOUT"
    assert "private body" not in str(event)


def test_background_summary_timing_is_labelled_and_reported_separately(tmp_path):
    from career_agent.harness.observability import BACKGROUND_PROVIDER_CALL
    @traced_provider_call("conversation_summary")
    def summarize_stub():
        return "ok"
    recorder = InMemoryTraceRecorder()
    trace_token = ACTIVE_TRACE_CONTEXT.set((recorder, "background"))
    mode_token = BACKGROUND_PROVIDER_CALL.set(True)
    try:
        assert summarize_stub() == "ok"
    finally:
        BACKGROUND_PROVIDER_CALL.reset(mode_token)
        ACTIVE_TRACE_CONTEXT.reset(trace_token)
    assert recorder.snapshot("background").events[0].details == {"background": True}
    path = tmp_path / "trace.sqlite3"
    trace = SQLiteTraceRecorder(path)
    trace.record("r", "provider_request", "conversation_summary", outcome="succeeded", duration_ms=15000, details={"background": True})
    trace.record("r", "provider_request", "conversation_summary", outcome="succeeded", duration_ms=20)
    mode = summarize(path)["phases"]["summary_compaction"]["execution_mode"]
    assert mode["background"]["operation_duration_ms"]["total"] == 15000
    assert mode["foreground_or_legacy"]["operation_duration_ms"]["total"] == 20


def test_capture_reading_gaps_are_between_turns_and_excluded_from_wall_time(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import dotenv
    import career_agent.cli as cli
    import career_agent.evaluation.latency_capture as sampling

    monkeypatch.setattr(dotenv, "load_dotenv", lambda: None)
    for suffix, value in {"BASE_URL": "https://example.test/v1", "API_KEY": "test-only", "MODEL": "test-model"}.items():
        monkeypatch.setenv("MAIN_AGENT_" + suffix, value)
    clock = [0.0]
    gaps = []

    def sleep(seconds):
        gaps.append(seconds)
        clock[0] += seconds

    class Runtime:
        def run_turn(self, **kwargs):
            clock[0] += 1
            return SimpleNamespace(context=SimpleNamespace(task=SimpleNamespace(pending_interaction=None)))

        def close(self):
            pass

    monkeypatch.setattr(cli, "build_main_agent_runtime", lambda _: Runtime())
    monkeypatch.setattr(sampling, "sleep", sleep)
    monkeypatch.setattr(sampling, "perf_counter", lambda: clock[0])
    result = sampling.capture(tmp_path / "capture", conversations=1, inter_turn_delay_seconds=20)
    assert gaps == [20, 20, 20]
    assert result["inter_turn_delay_seconds"] == 20
    assert [t["wall_ms"] for t in result["turns"]] == [1000] * 4
    assert [t["inter_turn_gap_ms"] for t in result["turns"]] == [0, 20000, 20000, 20000]
