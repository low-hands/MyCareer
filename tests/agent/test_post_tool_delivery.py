from career_agent.evaluation.post_tool_delivery import summarize
from career_agent.storage.run_events import SQLiteTraceRecorder


def test_only_the_next_decision_is_counted_and_legacy_coverage_is_explicit(tmp_path):
    path = tmp_path / "events.sqlite3"
    recorder = SQLiteTraceRecorder(path)
    recorder.record("multi", "capability_executed", "act", outcome="succeeded", duration_ms=0,
                    details={"tool_name": "list_resumes", "result_state": "no_resumes_found"})
    recorder.record("multi", "model_succeeded", "main_agent_decide", model_call_category="orchestrator_decision",
                    outcome="succeeded", duration_ms=100, details={"decision_action": "tool_call"})
    recorder.record("multi", "capability_executed", "act", outcome="succeeded", duration_ms=1,
                    details={"tool_name": "list_interviews", "result_state": "no_interviews_found"})
    recorder.record("multi", "model_succeeded", "main_agent_decide", model_call_category="orchestrator_decision",
                    outcome="succeeded", duration_ms=200, details={"decision_action": "final"})
    recorder.record("direct", "capability_executed", "act", outcome="succeeded", duration_ms=0,
                    details={"tool_name": "list_interviews", "result_state": "no_interviews_found"})
    recorder.record("legacy", "turn_completed", "turn", outcome="succeeded")
    report = summarize([path])
    assert report["runs_with_execution"] == 2
    assert report["runs_without_execution_events"] == 1
    interviews, resumes = report["results"]
    assert resumes["continued_tool_work"] == 1
    assert "model_final" not in resumes
    assert interviews["model_final"] == 1
    assert interviews["no_following_model_terminal"] == 1
    assert interviews["final_decision_duration_ms"]["total"] == 200
