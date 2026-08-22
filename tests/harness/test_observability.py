from career_agent.harness.observability import InMemoryTraceRecorder


def test_in_memory_trace_is_ordered_and_keeps_only_explicit_details() -> None:
    recorder = InMemoryTraceRecorder()
    recorder.record("run-1", "run_started", "job_discovery", outcome="started", details={"candidate_count": 2})
    recorder.record("run-1", "model_attempt", "candidate_triage", attempt=1, outcome="started", details={"schema": "CandidateTriage"})
    recorder.record(
        "run-1",
        "model_failed",
        "candidate_triage",
        attempt=1,
        outcome="failed",
        error_code="AGENT_WORKER_INVALID_RESPONSE",
        error_detail='[{"loc":["selections",0,"result_ref"],"type":"missing"}]',
        details={"schema": "CandidateTriage"},
    )

    trace = recorder.snapshot("run-1")

    assert [event.sequence for event in trace.events] == [1, 2, 3]
    assert trace.events[2].error_code == "AGENT_WORKER_INVALID_RESPONSE"
    assert trace.events[2].details == {"schema": "CandidateTriage"}
    assert "resume text" not in trace.model_dump_json().lower()
    assert "job description" not in trace.model_dump_json().lower()
