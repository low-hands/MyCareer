"""End-to-end wiring: the runtime records the two failure events it promised.

These live here rather than beside the storage tests because they exercise the
runtime's ``_TRACE_CONTEXT`` path — the recorder is injected, a turn runs, and
the correct events land in the durable store with their payload copied out of
the tool result.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel

from career_agent.agent.context_manager import ContextManager
from career_agent.agent.main_agent_contracts import (
    AgentDecision,
    CareerProfileContext,
    DecisionObservation,
    MainAgentContext,
    ToolCall,
)
from career_agent.agent.main_agent_runtime import MainAgentRuntime, _TRACE_CONTEXT
from career_agent.agent.main_agent_tools import MainAgentToolRegistry
from career_agent.connectors.gmail_readonly import GmailAPIError
from career_agent.storage.context import CareerContextStore
from career_agent.storage.run_events import SQLiteTraceRecorder
from career_agent.harness.observability import InMemoryTraceRecorder


class FailingEmailService:
    def sync(self, **kwargs):
        raise GmailAPIError(429, "quota exceeded")


class Decisions:
    def __init__(self, *decisions):
        self.values = list(decisions)

    def decide(self, context, tool_specs):
        return self.values.pop(0)


def _runtime(tmp_path, email_service, recorder) -> MainAgentRuntime:
    manager = ContextManager(CareerContextStore(tmp_path / "context.sqlite3"))
    manager.upsert_profile(CareerProfileContext(user_id="u1"))
    return MainAgentRuntime(
        context_manager=manager,
        decision_maker=Decisions(
            AgentDecision(
                action="tool_call",
                tool_call=ToolCall(name="sync_application_emails", arguments={}),
            ),
            AgentDecision(action="final", message=""),
        ),
        tools=MainAgentToolRegistry(email_tracking_service=email_service),
        trace_recorder=recorder,
    )


def _all_events(recorder: SQLiteTraceRecorder) -> tuple:
    import sqlite3

    with recorder._connect() as connection:
        rows = connection.execute(
            "SELECT run_id, sequence FROM run_events ORDER BY sequence"
        ).fetchall()
    run_ids = {row[0] for row in rows}
    events = []
    for run_id in run_ids:
        events.extend(recorder.snapshot(run_id).events)
    return tuple(events)


def test_a_capability_failure_is_recorded_with_its_error_code(tmp_path: Path) -> None:
    recorder = SQLiteTraceRecorder(tmp_path / "run-events.sqlite3")
    runtime = _runtime(tmp_path, FailingEmailService(), recorder)

    runtime.run_turn(user_id="u1", conversation_id="c1", user_message="检查邮箱")

    events = _all_events(recorder)
    failed = [e for e in events if e.event_type == "capability_failed"]
    assert failed, "a capability failure must be traced"
    assert any(e.error_code == "GmailAPIError" for e in failed)
    assert any(
        e.event_type == "turn_completed" for e in events
    ), "a non-raising capability failure still completes the turn"
    decisions = [
        event for event in events if event.event_type == "model_attempt"
    ]
    assert len(decisions) == 2
    assert all(
        event.model_call_category == "orchestrator_decision"
        for event in decisions
    )
    assert all(event.details["context_chars"] > 0 for event in decisions)
    assert all(event.details["tool_schema_chars"] > 0 for event in decisions)
    assert decisions[1].details["observation_count"] == 1
    assert decisions[1].details["observation_chars"] > decisions[0].details[
        "observation_chars"
    ]


def test_an_escalated_turn_failure_is_traced(tmp_path: Path) -> None:
    """A decision-maker failure raises through run_turn and records turn_failed."""

    from career_agent.agent.openai_compatible_client import AgentWorkerError

    class ExplodingDecisionMaker:
        def decide(self, context, tool_specs):
            raise AgentWorkerError(
                "MAIN_AGENT_TRANSPORT_ERROR",
                "transport failed",
                retryable=True,
            )

    manager = ContextManager(CareerContextStore(tmp_path / "context.sqlite3"))
    manager.upsert_profile(CareerProfileContext(user_id="u1"))
    recorder = SQLiteTraceRecorder(tmp_path / "run-events.sqlite3")
    runtime = MainAgentRuntime(
        context_manager=manager,
        decision_maker=ExplodingDecisionMaker(),
        tools=MainAgentToolRegistry(),
        trace_recorder=recorder,
    )

    import pytest

    with pytest.raises(AgentWorkerError):
        runtime.run_turn(user_id="u1", conversation_id="c1", user_message="随便")

    events = _all_events(recorder)
    failed = [event for event in events if event.event_type == "turn_failed"]
    assert len(failed) == 1
    assert failed[0].error_code == "MAIN_AGENT_TRANSPORT_ERROR"
    assert failed[0].recoverable is True
    assert failed[0].details == {
        "conversation_id": "c1",
        "error_type": "AgentWorkerError",
    }
    model_failure = [
        event for event in events if event.event_type == "model_failed"
    ]
    assert len(model_failure) == 1
    assert model_failure[0].model_call_category == "orchestrator_decision"
    assert model_failure[0].error_code == "MAIN_AGENT_TRANSPORT_ERROR"


def test_observation_chars_measures_the_body_in_the_actual_prompt_shape(
    tmp_path: Path,
) -> None:
    manager = ContextManager(CareerContextStore(tmp_path / "context.sqlite3"))
    recorder = InMemoryTraceRecorder()
    runtime = MainAgentRuntime(
        context_manager=manager,
        decision_maker=Decisions(AgentDecision(action="final", message="done")),
        tools=MainAgentToolRegistry(),
    )
    context = MainAgentContext(
        conversation_id="c1",
        profile=CareerProfileContext(user_id="u1"),
        tool_observations=(
            DecisionObservation(
                tool_name="get_saved_job",
                state="saved_job_ready",
                message="已读取完整 JD。",
                body="J" * 6_000,
            ),
        ),
        user_message="继续。",
    )
    token = _TRACE_CONTEXT.set((recorder, "body-trace"))
    try:
        runtime._decide({"context": context})
    finally:
        _TRACE_CONTEXT.reset(token)

    attempt = recorder.snapshot("body-trace").events[0]
    projected = context.model_context()["tool_observations"]
    expected = len(json.dumps(projected, ensure_ascii=False, sort_keys=True))
    assert attempt.details["observation_chars"] == expected
    assert attempt.details["observation_chars"] > 6_000


def test_a_presenter_validation_failure_is_recorded(tmp_path: Path) -> None:
    """A payload/presenter drift must become a countable durable event."""

    class PresenterContract(BaseModel):
        required_value: str

    recorder = SQLiteTraceRecorder(tmp_path / "run-events.sqlite3")
    token = _TRACE_CONTEXT.set((recorder, "turn-presenter"))
    try:
        assert MainAgentRuntime._validated(PresenterContract, {}) is None
    finally:
        _TRACE_CONTEXT.reset(token)

    events = recorder.snapshot("turn-presenter").events
    assert len(events) == 1
    assert events[0].event_type == "presentation_degraded"
    assert events[0].stage == "PresenterContract"
    assert events[0].outcome == "failed"
    assert events[0].error_detail == "validation_error"
