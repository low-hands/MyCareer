"""Replay model decisions through the complete Main Agent turn topology.

The trajectory catalogue verifies the prompt and each recorded decision, while
the ordinary runtime suite scripts ``DecisionMaker`` directly.  Neither proves
that a response in cassette shape survives the production model parser *and*
then follows ``run_turn -> authorize -> act -> observe -> present/interrupt``.

These scenarios close that seam without calling a live model.  They replace
only the remote response and capability result; context loading, dynamic tool
menus, argument projection, graph routing, budgets, persistence, tracing and
public stream events are all the production implementations.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from career_agent.agent.context_manager import ContextManager
from career_agent.agent.decision_messages import (
    CONTROL_CONTEXT_LABEL,
    CONTROL_REMINDER_TAG,
    TURN_OBSERVATION_LABEL,
)
from career_agent.agent.main_agent_contracts import (
    AgentDecision,
    CareerProfileContext,
    ConversationTaskState,
    ToolObservation,
)
from career_agent.agent.main_agent_runtime import MainAgentRuntime, RuntimeAction
from career_agent.agent.main_agent_tools import MainAgentToolRegistry
from career_agent.agent.mock_interview_contracts import MockInterviewGraphResult
from career_agent.agent.openai_compatible_client import OpenAICompatibleAgentConfig
from career_agent.agent.openai_compatible_main_agent import (
    OpenAICompatibleMainAgentDecisionMaker,
)
from career_agent.evaluation.trajectory import ReplayClient
from career_agent.harness.observability import InMemoryTraceRecorder
from career_agent.harness.streaming import (
    InteractionRequiredEvent,
    InteractionResponse,
    TurnCompletedEvent,
    TurnSuspendedEvent,
)
from career_agent.storage.capability_confirmations import (
    SQLiteCapabilityConfirmationStore,
)
from career_agent.storage.context import CareerContextStore


def _tool_call(name: str, **arguments: Any) -> dict[str, Any]:
    return {"tool_call": {"name": name, "arguments": arguments}}


def _final(message: str) -> dict[str, str]:
    return {
        "content": AgentDecision(action="final", message=message).model_dump_json(
            exclude_none=True
        )
    }


class _ScriptedRegistry(MainAgentToolRegistry):
    """Keep production schemas/projection and replace only capability I/O."""

    def __init__(self, results: Sequence[ToolObservation]) -> None:
        # These sentinels make both tool families available to the production
        # reachability menu.  Their methods are never called: invocation below
        # supplies the closed results declared by each loop scenario.
        super().__init__(job_repository=object(), calendar_service=object())
        self._scripted_results = list(results)
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def invoke_atomic_tool(
        self, name: str, arguments: dict[str, Any]
    ) -> ToolObservation:
        self.calls.append((name, dict(arguments)))
        if not self._scripted_results:
            raise AssertionError(f"unexpected capability call: {name}")
        result = self._scripted_results.pop(0)
        assert result.tool_name == name
        return result

    @property
    def remaining_results(self) -> int:
        return len(self._scripted_results)


def _runtime(
    tmp_path: Path,
    *,
    responses: Sequence[Mapping[str, Any]],
    results: Sequence[ToolObservation],
    task: ConversationTaskState | None = None,
    max_read_calls: int = 6,
    with_confirmations: bool = False,
) -> tuple[
    MainAgentRuntime,
    _ScriptedRegistry,
    ReplayClient,
    ContextManager,
    InMemoryTraceRecorder,
]:
    store = CareerContextStore(tmp_path / "loop-evaluation.sqlite3")
    manager = ContextManager(store)
    manager.upsert_profile(CareerProfileContext(user_id="u1"))
    if task is not None:
        store.upsert_task(user_id="u1", conversation_id="c1", task=task)

    client = ReplayClient(responses)
    maker = OpenAICompatibleMainAgentDecisionMaker(
        OpenAICompatibleAgentConfig(
            endpoint="https://loop-replay.invalid/v1/chat/completions",
            api_key="replay",
            model="loop-replay",
        ),
        client=client,
    )
    tools = _ScriptedRegistry(results)
    recorder = InMemoryTraceRecorder()
    runtime = MainAgentRuntime(
        context_manager=manager,
        decision_maker=maker,
        tools=tools,
        trace_recorder=recorder,
        max_read_calls=max_read_calls,
        capability_confirmation_store=(
            SQLiteCapabilityConfirmationStore(
                tmp_path / "loop-evaluation.sqlite3"
            )
            if with_confirmations
            else None
        ),
    )
    return runtime, tools, client, manager, recorder


def _turn_observation_payload(messages: Sequence[Mapping[str, Any]]) -> tuple:
    observations = []
    for message in messages:
        if message["role"] != "tool":
            continue
        lines = message["content"].splitlines()
        assert lines[0] == TURN_OBSERVATION_LABEL
        observations.append(json.loads("\n".join(lines[2:-1])))
    return tuple(observations)


def _request_context(client: ReplayClient, index: int) -> dict[str, Any]:
    messages = client.requests[index]["messages"]
    stable_lines = messages[1]["content"].splitlines()
    stable = json.loads("\n".join(stable_lines[2:-1]))
    control_lines = messages[2]["content"].splitlines()
    assert control_lines[0] == f"<{CONTROL_REMINDER_TAG}>"
    assert control_lines[1] == CONTROL_CONTEXT_LABEL
    assert control_lines[-1] == f"</{CONTROL_REMINDER_TAG}>"
    control = json.loads("\n".join(control_lines[2:-1]))
    lines = messages[3]["content"].splitlines()
    data = json.loads("\n".join(lines[2:-1]))
    observations = list(_turn_observation_payload(messages))
    return {
        **control,
        **stable,
        **data,
        "tool_observations": observations,
    }


def _recorded_events(recorder: InMemoryTraceRecorder) -> tuple:
    return tuple(
        event
        for events in recorder._events.values()
        for event in events
    )


def test_a_recorded_tool_choice_runs_the_whole_turn_before_model_delivery(
    tmp_path: Path,
) -> None:
    runtime, tools, client, manager, recorder = _runtime(
        tmp_path,
        responses=(
            _tool_call("find_saved_jobs", query="平台工程师"),
            _final("没有找到已保存的平台工程师岗位。"),
        ),
        results=(
            ToolObservation(
                tool_name="find_saved_jobs",
                state="no_saved_jobs_found",
                message="没有找到匹配的已保存岗位。",
                payload={"items": []},
            ),
        ),
    )
    public_events = []

    turn = runtime.run_turn(
        user_id="u1",
        conversation_id="c1",
        user_message="看看我保存过的平台工程师岗位",
        event_sink=public_events.append,
    )

    assert tools.calls == [
        (
            "find_saved_jobs",
            {"user_id": "u1", "query": "平台工程师", "limit": 10},
        )
    ]
    assert tools.remaining_results == 0
    assert len(client.requests) == 2
    observed = _request_context(client, 1)["tool_observations"][-1]
    assert observed["tool_name"] == "find_saved_jobs"
    assert observed["state"] == "no_saved_jobs_found"
    assert turn.assistant_message == "没有找到已保存的平台工程师岗位。"
    assert turn.delegated_read_count == 1
    assert isinstance(public_events[-1], TurnCompletedEvent)

    history = manager.load_for_turn(
        user_id="u1", conversation_id="c1", user_message="下一轮"
    ).recent_messages
    assert history[-1].content == turn.assistant_message
    events = _recorded_events(recorder)
    assert [event.event_type for event in events].count("model_attempt") == 2
    model_results = [
        event for event in events if event.event_type == "model_succeeded"
    ]
    assert all(
        event.details["prompt_cache_mode"] == "implicit"
        and event.details["prompt_cache_key_applied"] is True
        and event.details["prompt_cache_breakpoint_applied"] is False
        and event.details["cache_metrics_reported"] is False
        for event in model_results
    )
    assert model_results[-1].details["cache_metrics_unreported_ratio"] == 1.0
    assert events[-1].event_type == "turn_completed"


def test_a_recorded_retryable_failure_is_observed_then_recovered_in_one_turn(
    tmp_path: Path,
) -> None:
    same_call = _tool_call("find_saved_jobs", query="RAG")
    runtime, tools, client, _, recorder = _runtime(
        tmp_path,
        responses=(same_call, same_call, _final("重试后已完成查询。")),
        results=(
            ToolObservation(
                tool_name="find_saved_jobs",
                state="failed",
                message="岗位库暂时不可用。",
                payload={"error_code": "TEMPORARY", "retryable": True},
            ),
            ToolObservation(
                tool_name="find_saved_jobs",
                state="no_saved_jobs_found",
                message="没有找到匹配的已保存岗位。",
                payload={"items": []},
            ),
        ),
    )

    turn = runtime.run_turn(
        user_id="u1", conversation_id="c1", user_message="找我保存过的 RAG 岗位"
    )

    assert len(tools.calls) == 2
    assert len(client.requests) == 3
    assert _request_context(client, 1)["tool_observations"][-1]["facts"] == {
        "retryable": True
    }
    assert [item.state for item in turn.tool_results] == [
        "failed",
        "no_saved_jobs_found",
    ]
    assert turn.assistant_message == "重试后已完成查询。"
    event_types = [event.event_type for event in _recorded_events(recorder)]
    assert event_types.count("capability_failed") == 1
    assert event_types[-1] == "turn_completed"


def test_a_recorded_over_budget_call_becomes_observation_not_execution(
    tmp_path: Path,
) -> None:
    runtime, tools, client, _, _ = _runtime(
        tmp_path,
        responses=(
            _tool_call("find_saved_jobs", query="后端"),
            _tool_call("find_saved_jobs", query="算法"),
            _final("本轮读取额度已用完，算法岗位需要下一轮继续查。"),
        ),
        results=(
            ToolObservation(
                tool_name="find_saved_jobs",
                state="no_saved_jobs_found",
                message="没有找到匹配的已保存岗位。",
                payload={"items": []},
            ),
        ),
        max_read_calls=1,
    )

    turn = runtime.run_turn(
        user_id="u1", conversation_id="c1", user_message="分别查后端和算法岗位"
    )

    assert [arguments["query"] for _, arguments in tools.calls] == ["后端"]
    assert tools.remaining_results == 0
    assert len(client.requests) == 3
    refused = _request_context(client, 2)["tool_observations"][-1]
    assert refused["state"] == "authorization_refused"
    assert "READ" in refused["message"]
    assert [item.state for item in turn.tool_results] == ["no_saved_jobs_found"]
    assert turn.delegated_read_count == 1


def test_a_recorded_calendar_preview_suspends_without_an_extra_model_decision(
    tmp_path: Path,
) -> None:
    expires_at = datetime(2026, 9, 4, 18, tzinfo=timezone.utc)
    runtime, tools, client, _, _ = _runtime(
        tmp_path,
        responses=(_tool_call("prepare_interview_calendar_sync"),),
        results=(
                ToolObservation(
                    tool_name="prepare_interview_calendar_sync",
                    state="calendar_approval_required",
                    message="日历变更预览已准备好。",
                    execution_outcome="committed",
                    payload={
                    "proposal_id": "proposal-1",
                    "interview_round_id": "interview-1",
                    "operation": "create",
                    "expires_at": expires_at.isoformat(),
                    "payload": {
                        "title": "面试 · 示例科技 · 后端工程师",
                        "start_at": "2026-09-05T02:00:00+00:00",
                        "end_at": "2026-09-05T03:00:00+00:00",
                        "timezone": "Asia/Shanghai",
                        "location": "线上",
                    },
                },
            ),
        ),
        task=ConversationTaskState(active_interview_round_id="interview-1"),
    )
    public_events = []

    turn = runtime.run_turn(
        user_id="u1",
        conversation_id="c1",
        user_message="把这场面试加到日历",
        event_sink=public_events.append,
    )

    assert len(client.requests) == 1
    assert tools.calls[0][0] == "prepare_interview_calendar_sync"
    assert turn.model_decision.action == "tool_call"
    assert turn.context.task.active_calendar_proposal_id == "proposal-1"
    assert any(isinstance(event, InteractionRequiredEvent) for event in public_events)
    assert isinstance(public_events[-1], TurnSuspendedEvent)
    assert not any(isinstance(event, TurnCompletedEvent) for event in public_events)


def test_an_uncertain_calendar_write_stops_at_the_owner_and_is_not_reissued(
    tmp_path: Path,
) -> None:
    """An external write is never the model's call to make.

    The model's "execute" reaches the seal, not the calendar: the turn stops
    with the concrete event for the owner to approve. The owner's click runs
    it once without consulting the model again, and an outcome the provider
    could not confirm is recorded as unknown and never retried.
    """

    runtime, tools, client, _, recorder = _runtime(
        tmp_path,
        responses=(_tool_call("execute_calendar_proposal"),),
        results=(
            ToolObservation(
                tool_name="get_calendar_proposal",
                state="calendar_proposal_ready",
                message="预览已准备。",
                payload={
                    "operation": "create_event",
                    "payload": {
                        "title": "面试：ACME 二面",
                        "start_at": "2026-09-05T10:00:00+08:00",
                        "end_at": "2026-09-05T11:00:00+08:00",
                        "timezone": "Asia/Shanghai",
                        "location": "线上",
                    },
                    "expires_at": "2026-09-05T18:00:00+00:00",
                },
            ),
            ToolObservation(
                tool_name="execute_calendar_proposal",
                state="calendar_write_failed",
                message="Calendar 写入结果暂时无法确认。",
                execution_outcome="unknown",
                payload={
                    "error_code": "GOOGLE_CALENDAR_TRANSPORT_ERROR",
                    "retryable": False,
                    "outcome_unknown": True,
                },
            ),
        ),
        task=ConversationTaskState(
            active_calendar_proposal_id="proposal-1",
            active_calendar_proposal_expires_at=datetime(
                2026, 9, 5, 18, tzinfo=timezone.utc
            ),
            active_interview_round_id="interview-1",
        ),
        with_confirmations=True,
    )

    stopped = runtime.run_turn(
        user_id="u1", conversation_id="c1", user_message="确认执行日历变更"
    )

    assert [name for name, _ in tools.calls] == ["get_calendar_proposal"]
    assert len(client.requests) == 1
    assert stopped.tool_result.state == "capability_confirmation_required"
    assert "面试：ACME 二面" in stopped.tool_result.message
    assert "外部写入" in stopped.tool_result.message
    gate = MainAgentRuntime._interaction_event(result=stopped, conversation_id="c1")
    assert gate is not None and gate.scope == "capability_confirmation"

    turn = runtime.run_turn(
        user_id="u1",
        conversation_id="c1",
        user_message="确认",
        interaction_response=InteractionResponse(
            interaction_id=gate.interaction_id,
            scope="capability_confirmation",
            action="confirm",
        ),
    )

    assert [name for name, _ in tools.calls] == [
        "get_calendar_proposal",
        "execute_calendar_proposal",
    ]
    assert len(client.requests) == 1
    failed = turn.context.tool_observations[-1]
    assert failed.state == "calendar_write_failed"
    assert failed.facts == {"retryable": False}
    assert turn.tool_result.execution_outcome == "unknown"
    assert turn.context.task.active_calendar_proposal_id is None
    assert turn.delegated_write_count == 1
    event_types = [event.event_type for event in _recorded_events(recorder)]
    assert event_types.count("capability_failed") == 1
    assert event_types[-1] == "turn_completed"


def test_an_owned_mock_interview_turn_uses_runtime_action_without_main_model(
    tmp_path: Path,
) -> None:
    class MockInterviewGraph:
        def __init__(self) -> None:
            self.inputs: list[dict[str, str]] = []

        def handle_input(self, *, user_id: str, session_id: str, message: str):
            self.inputs.append(
                {"user_id": user_id, "session_id": session_id, "message": message}
            )
            return MockInterviewGraphResult(
                session_id=session_id,
                state="cancelled",
                message="模拟面试已取消。",
            )

    store = CareerContextStore(tmp_path / "owned-loop-evaluation.sqlite3")
    manager = ContextManager(store)
    manager.upsert_profile(CareerProfileContext(user_id="u1"))
    task = ConversationTaskState().enter_workflow(
        "mock_interview",
        run_id="mock-session-1",
        phase="mock_interview_answer_required",
        candidates=(),
    )
    store.upsert_task(user_id="u1", conversation_id="c1", task=task)
    client = ReplayClient(())
    maker = OpenAICompatibleMainAgentDecisionMaker(
        OpenAICompatibleAgentConfig(
            endpoint="https://loop-replay.invalid/v1/chat/completions",
            api_key="replay",
            model="loop-replay",
        ),
        client=client,
    )
    graph = MockInterviewGraph()
    recorder = InMemoryTraceRecorder()
    runtime = MainAgentRuntime(
        context_manager=manager,
        decision_maker=maker,
        tools=MainAgentToolRegistry(mock_interview_graph=graph),
        trace_recorder=recorder,
    )

    turn = runtime.run_turn(
        user_id="u1", conversation_id="c1", user_message="不练了，结束面试"
    )

    assert client.requests == []
    assert graph.inputs == [
        {
            "user_id": "u1",
            "session_id": "mock-session-1",
            "message": "不练了，结束面试",
        }
    ]
    # The origin says what actually happened: the user supplied the input, and
    # the runtime routed it to the workflow it owns. It used to be reported as
    # an ``AgentDecision`` naming this tool, which was a claim about a model
    # call that never occurred — ``client.requests == []`` above is the proof.
    # The business workflow, not the handler. ``handle_mock_interview_input``
    # and ``retry_mock_interview`` are internal routing; publishing either would
    # make an external surface change whenever that routing does.
    assert turn.origin == RuntimeAction(workflow="mock_interview")
    assert turn.requested_by == "user"
    assert turn.model_decision is None
    assert turn.tool_result is not None
    assert turn.tool_result.state == "mock_interview_cancelled"
    assert turn.context.task.active_workflow == "none"
    assert turn.delegated_write_count == 1
    events = _recorded_events(recorder)
    assert not any(event.event_type.startswith("model_") for event in events)
    assert events[-1].event_type == "turn_completed"
