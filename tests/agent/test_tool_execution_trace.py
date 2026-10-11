"""One turn's tool path is reconstructable from its trace events alone.

Each test drives a real runtime turn with an in-memory recorder and asserts the
events for one path: refusal, pause, resume, execution and unknown outcome.
The events carry names and closed-set states, never argument values.
"""

from __future__ import annotations

from career_agent.agent.capabilities.registry import MainAgentToolRegistry
from career_agent.agent.context.manager import ContextManager
from career_agent.agent.contracts.decisions import AgentDecision, ToolCall
from career_agent.agent.contracts.observations import ToolObservation
from career_agent.agent.contracts.profile import (
    AgentPreferencesContext,
    CareerProfileContext,
)
from career_agent.agent.presentation.factory import interaction_event
from career_agent.agent.runtime.main_agent_runtime import MainAgentRuntime
from career_agent.agent.runtime.ports import RuntimePorts
from career_agent.connectors.gmail_readonly import GmailAPIError
from career_agent.harness.observability import InMemoryTraceRecorder, RunEvent
from career_agent.harness.streaming import InteractionResponse
from career_agent.storage.action_executions import SQLiteActionExecutionStore
from career_agent.storage.capability_confirmations import (
    SQLiteCapabilityConfirmationStore,
)
from career_agent.storage.context import CareerContextStore
from career_agent.storage.working_notes import WorkingNotesStore
from agent_test_support import CatalogSchemaRegistry, load_capability_family


class _Decisions:
    def __init__(self, *decisions: AgentDecision) -> None:
        self.decisions = list(decisions)

    def decide(self, context, tool_specs):
        return self.decisions.pop(0)


def _call(name: str, **arguments) -> AgentDecision:
    return AgentDecision(
        action="tool_call", tool_call=ToolCall(name=name, arguments=arguments)
    )


_FINAL = AgentDecision(action="final", message="好的。")


def _runs(recorder: InMemoryTraceRecorder) -> list[list[RunEvent]]:
    """Every recorded run's events, in the order the runs started."""

    runs = [list(events) for events in recorder._events.values()]
    return sorted(runs, key=lambda events: events[0].occurred_at)


def _of(events: list[RunEvent], event_type: str) -> list[RunEvent]:
    return [event for event in events if event.event_type == event_type]


def _path(events: list[RunEvent]) -> list[str]:
    """The tool-path events of one run, without model and context telemetry."""

    return [
        event.event_type
        for event in events
        if event.event_type in {
            "authorization_refused",
            "capability_executed",
            "capability_failed",
            "write_outcome_unknown",
            "run_interrupted",
            "run_resumed",
            "turn_completed",
        }
    ]


class _SavedJobs(MainAgentToolRegistry):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[dict] = []
        self._atomic_handlers["find_saved_jobs"] = self._invoke

    def _invoke(self, arguments):
        self.calls.append(dict(arguments))
        return ToolObservation(
            tool_name="find_saved_jobs",
            state="saved_job_not_found",
            message="没有找到。",
        )


def _manager(tmp_path, **kwargs) -> ContextManager:
    manager = ContextManager(
        CareerContextStore(tmp_path / "context.sqlite3"), **kwargs
    )
    manager.upsert_profile(CareerProfileContext(user_id="u1"))
    return manager


def test_a_projection_refusal_is_traced_with_its_cap(tmp_path) -> None:
    def refuse_projection(context, name, arguments):
        raise ValueError("selection index 7 is not in the candidate list")

    recorder = InMemoryTraceRecorder()
    registry = _SavedJobs()
    runtime = MainAgentRuntime(
        context_manager=_manager(tmp_path),
        decision_maker=_Decisions(
            _call("find_saved_jobs", query="SecretQuery"),
            # A different call, so the duplicate-call gate does not answer it.
            _call("find_saved_jobs", query="OtherSecretQuery"),
        ),
        tools=registry,
        runtime_ports=RuntimePorts(project_atomic_tool_arguments=refuse_projection),
        max_projection_refusals=1,
        trace_recorder=recorder,
    )

    result = runtime.run_turn(
        user_id="u1", conversation_id="c1", user_message="找岗位"
    )

    assert registry.calls == []
    assert result.context.tool_observations[-1].state == "invalid_input"
    (run,) = _runs(recorder)
    refused = _of(run, "authorization_refused")
    assert [event.details for event in refused] == [
        {
            "tool_name": "find_saved_jobs",
            "refusal_kind": "argument_projection",
            "capped": False,
        },
        {
            "tool_name": "find_saved_jobs",
            "refusal_kind": "argument_projection",
            "capped": True,
        },
    ]
    assert [event.recoverable for event in refused] == [True, False]
    assert _of(run, "capability_executed") == []
    serialized = "".join(event.model_dump_json() for event in refused)
    assert "SecretQuery" not in serialized
    assert "selection index" not in serialized


def test_a_working_notes_refusal_is_traced_then_the_question_pauses(
    tmp_path,
) -> None:
    notes = WorkingNotesStore(tmp_path / "notes")
    notes.replace(
        user_id="u1", markdown="用户可能偏好 Rust", expected_revision="empty"
    )
    recorder = InMemoryTraceRecorder()
    registry = _SavedJobs()
    runtime = MainAgentRuntime(
        context_manager=_manager(tmp_path, working_notes_store=notes),
        decision_maker=_Decisions(
            _call("find_saved_jobs", query="Rust"),
            AgentDecision(action="ask_user", message="你确认偏好 Rust 吗？"),
        ),
        tools=registry,
        trace_recorder=recorder,
    )

    runtime.run_turn(user_id="u1", conversation_id="c1", user_message="帮我找岗位")

    assert registry.calls == []
    (run,) = _runs(recorder)
    assert _path(run) == [
        "authorization_refused",
        "run_interrupted",
        "turn_completed",
    ]
    (refused,) = _of(run, "authorization_refused")
    assert refused.details == {
        "tool_name": "find_saved_jobs",
        "refusal_kind": "working_notes",
        "capped": False,
    }
    (paused,) = _of(run, "run_interrupted")
    assert paused.outcome == "interrupted"
    assert paused.details == {"interaction_kind": "ask_user"}
    assert "rust" not in (refused.model_dump_json() + paused.model_dump_json()).lower()


def test_an_approval_pause_and_its_resume_are_traced(tmp_path) -> None:
    class Registry(CatalogSchemaRegistry):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def capability_kind(self, name):
            return "atomic_tool"

        def invoke_atomic_tool(self, name, arguments):
            self.calls += 1
            return ToolObservation(
                tool_name=name,
                state="application_ready",
                message="已创建投递记录。",
                payload={"application_id": "app-1"},
                execution_outcome="committed",
            )

    path = tmp_path / "context.sqlite3"
    manager = ContextManager(CareerContextStore(path))
    manager.upsert_profile(CareerProfileContext(user_id="u1"))
    load_capability_family(manager, "application")
    manager.upsert_preferences(
        user_id="u1",
        preferences=AgentPreferencesContext(application_confirmation="always_ask"),
    )
    recorder = InMemoryTraceRecorder()
    registry = Registry()
    runtime = MainAgentRuntime(
        context_manager=manager,
        decision_maker=_Decisions(_call("create_application", note="PrivateNote")),
        tools=registry,
        runtime_ports=RuntimePorts(
            project_atomic_tool_arguments=lambda context, name, arguments: {
                "user_id": context.profile.user_id,
                **arguments,
            }
        ),
        capability_confirmation_store=SQLiteCapabilityConfirmationStore(path),
        action_execution_store=SQLiteActionExecutionStore(path),
        trace_recorder=recorder,
    )

    held = runtime.run_turn(user_id="u1", conversation_id="c1", user_message="记录投递")
    gate = interaction_event(result=held, conversation_id="c1")
    runtime.run_turn(
        user_id="u1",
        conversation_id="c1",
        user_message="确认",
        interaction_response=InteractionResponse(
            interaction_id=gate.interaction_id,
            scope="capability_confirmation",
            action="confirm",
        ),
    )

    assert registry.calls == 1
    paused_run, resumed_run = _runs(recorder)
    assert _path(paused_run) == ["run_interrupted", "turn_completed"]
    (paused,) = _of(paused_run, "run_interrupted")
    assert paused.details == {
        "interaction_kind": "capability_confirmation_required",
        "tool_name": "create_application",
    }
    assert _path(resumed_run) == [
        "run_resumed",
        "capability_executed",
        "turn_completed",
    ]
    (resumed,) = _of(resumed_run, "run_resumed")
    assert resumed.details == {
        "interaction_kind": "capability_confirmation",
        "tool_name": "create_application",
    }
    (executed,) = _of(resumed_run, "capability_executed")
    assert executed.outcome == "succeeded"
    assert executed.duration_ms is not None and executed.duration_ms >= 0
    assert executed.details == {
        "tool_name": "create_application",
        "effect": "WRITE",
        "result_state": "application_ready",
        "disposition": "completed",
        "execution_outcome": "committed",
    }
    serialized = "".join(
        event.model_dump_json() for run in (paused_run, resumed_run) for event in run
    )
    assert "PrivateNote" not in serialized
    assert "app-1" not in serialized


def test_a_successful_read_records_one_execution_event(tmp_path) -> None:
    recorder = InMemoryTraceRecorder()
    registry = _SavedJobs()
    runtime = MainAgentRuntime(
        context_manager=_manager(tmp_path),
        decision_maker=_Decisions(_call("find_saved_jobs", query="AI"), _FINAL),
        tools=registry,
        trace_recorder=recorder,
    )

    runtime.run_turn(user_id="u1", conversation_id="c1", user_message="找岗位")

    assert len(registry.calls) == 1
    (run,) = _runs(recorder)
    assert _path(run) == ["capability_executed", "turn_completed"]
    (executed,) = _of(run, "capability_executed")
    assert executed.stage == "act"
    assert executed.outcome == "succeeded"
    assert executed.error_code is None
    assert executed.duration_ms is not None and executed.duration_ms >= 0
    assert executed.details == {
        "tool_name": "find_saved_jobs",
        "effect": "READ",
        "result_state": "saved_job_not_found",
        "disposition": "completed",
    }


def test_a_failed_tool_records_a_failed_execution_and_keeps_capability_failed(
    tmp_path,
) -> None:
    class FailingEmailService:
        def sync(self, **kwargs):
            raise GmailAPIError(429, "quota exceeded for someone@example.com")

    manager = _manager(tmp_path)
    load_capability_family(manager, "application")
    recorder = InMemoryTraceRecorder()
    runtime = MainAgentRuntime(
        context_manager=manager,
        decision_maker=_Decisions(_call("sync_application_emails"), _FINAL),
        tools=MainAgentToolRegistry(email_tracking_service=FailingEmailService()),
        trace_recorder=recorder,
    )

    runtime.run_turn(user_id="u1", conversation_id="c1", user_message="检查邮箱")

    (run,) = _runs(recorder)
    assert _path(run) == [
        "capability_executed",
        "capability_failed",
        "turn_completed",
    ]
    (executed,) = _of(run, "capability_executed")
    assert executed.outcome == "failed"
    assert executed.details["tool_name"] == "sync_application_emails"
    assert executed.details["disposition"] == "failed"
    assert executed.duration_ms is not None
    assert "someone@example.com" not in executed.model_dump_json()
    (failed,) = _of(run, "capability_failed")
    assert failed.error_code == "GmailAPIError"


def test_an_unknown_write_outcome_is_visible_in_the_trace(tmp_path) -> None:
    class Registry(CatalogSchemaRegistry):
        def capability_kind(self, name):
            return "atomic_tool"

        def invoke_atomic_tool(self, name, arguments):
            return ToolObservation(
                tool_name=name,
                state="application_write_unconfirmed",
                message="写入结果无法确认：PrivateDetail",
                execution_outcome="unknown",
            )

    path = tmp_path / "context.sqlite3"
    manager = ContextManager(CareerContextStore(path))
    manager.upsert_profile(CareerProfileContext(user_id="u1"))
    load_capability_family(manager, "application")
    recorder = InMemoryTraceRecorder()
    ledger = SQLiteActionExecutionStore(path)
    runtime = MainAgentRuntime(
        context_manager=manager,
        decision_maker=_Decisions(_call("create_application"), _FINAL),
        tools=Registry(),
        runtime_ports=RuntimePorts(
            project_atomic_tool_arguments=lambda context, name, arguments: {
                "user_id": context.profile.user_id,
            }
        ),
        action_execution_store=ledger,
        trace_recorder=recorder,
    )

    runtime.run_turn(user_id="u1", conversation_id="c1", user_message="记录投递")

    (run,) = _runs(recorder)
    assert _path(run)[:2] == ["write_outcome_unknown", "capability_executed"]
    (unknown,) = _of(run, "write_outcome_unknown")
    assert unknown.outcome == "failed"
    assert unknown.recoverable is False
    assert unknown.error_code == "APPLICATION_WRITE_UNCONFIRMED"
    assert unknown.details == {
        "tool_name": "create_application",
        "reason": "reported_unknown",
    }
    (executed,) = _of(run, "capability_executed")
    assert executed.details["execution_outcome"] == "unknown"
    assert executed.details["effect"] == "WRITE"
    assert "PrivateDetail" not in "".join(event.model_dump_json() for event in run)
