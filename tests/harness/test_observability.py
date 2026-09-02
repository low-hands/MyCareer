import pytest

from career_agent.harness.observability import (
    InMemoryTraceRecorder,
    model_call_counts,
)


def test_in_memory_trace_is_ordered_and_keeps_only_explicit_details() -> None:
    recorder = InMemoryTraceRecorder()
    recorder.record("run-1", "run_started", "job_discovery", outcome="started", details={"candidate_count": 2})
    recorder.record("run-1", "model_attempt", "candidate_triage", attempt=1, outcome="started", details={"schema": "CandidateTriage"}, model_call_category="capability_agent")
    recorder.record(
        "run-1",
        "model_failed",
        "candidate_triage",
        attempt=1,
        outcome="failed",
        error_code="AGENT_WORKER_INVALID_RESPONSE",
        error_detail='[{"loc":["selections",0,"result_ref"],"type":"missing"}]',
        details={"schema": "CandidateTriage"},
        model_call_category="capability_agent",
    )

    trace = recorder.snapshot("run-1")

    assert [event.sequence for event in trace.events] == [1, 2, 3]
    assert trace.events[2].error_code == "AGENT_WORKER_INVALID_RESPONSE"
    assert trace.events[2].details == {"schema": "CandidateTriage"}
    assert "resume text" not in trace.model_dump_json().lower()
    assert "job description" not in trace.model_dump_json().lower()


def test_model_events_require_one_of_the_six_call_categories() -> None:
    recorder = InMemoryTraceRecorder()
    for category in (
        "orchestrator_decision",
        "capability_agent",
        "planner",
        "evaluator",
        "writer",
        "legacy_router",
    ):
        recorder.record(
            "run-1",
            "model_attempt",
            category,
            outcome="started",
            model_call_category=category,
        )

    counts = model_call_counts(recorder.snapshot("run-1"))
    assert counts == {
        "orchestrator_decision": 1,
        "capability_agent": 1,
        "planner": 1,
        "evaluator": 1,
        "writer": 1,
        "legacy_router": 1,
    }

    with pytest.raises(ValueError, match="model_call_category"):
        recorder.record(
            "run-2",
            "model_attempt",
            "unclassified",
            outcome="started",
        )


def test_uninstrumented_categories_are_absent_instead_of_reported_as_zero() -> None:
    recorder = InMemoryTraceRecorder()
    recorder.record(
        "run-1",
        "model_attempt",
        "main_agent_decide",
        model_call_category="orchestrator_decision",
    )

    assert model_call_counts(recorder.snapshot("run-1")) == {
        "orchestrator_decision": 1
    }
