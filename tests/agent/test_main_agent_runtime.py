from __future__ import annotations

from datetime import datetime, timezone
from io import StringIO
import hashlib
import json
import sqlite3
import time
from urllib.parse import parse_qs, urlparse

import pytest
from pydantic import ValidationError

from career_agent.agent.conversation_memory_contracts import ConversationSummaryContent
from career_agent.agent.conversation_span_presenter import render_conversation_span
from career_agent.agent.context_manager import ContextManager
from career_agent.agent.main_agent_contracts import AgentDecision, AgentPreferencesContext, CareerMemoryClaim, CareerMemoryContext, CareerMemoryRecord, CareerProfileContext, ConversationTaskState, DECISION_OBSERVATION_BODY_LIMIT, DECISION_OBSERVATION_RECEIPT_LIMIT, MAX_DECISION_OBSERVATION_BODIES, MAX_DECISION_OBSERVATION_CHARS, DecisionObservation, MainAgentContext, MAX_DECISION_OBSERVATIONS, OBSERVATION_ARGUMENTS_LIMIT, ToolCall, ToolObservation, ToolResult, append_decision_observation, decision_observation_chars, decision_observation_projection
from career_agent.agent.summary_text import DELIVERY_SUMMARY_LIMIT, MODEL_REPLY_LIMIT, clamp
from career_agent.agent.main_agent_contracts import ConversationMessageContext, ConversationResourceReference
from career_agent.agent.main_agent_runtime import _STREAM_SINK, InteractionReceipt, MainAgentTurnResult, MainAgentRuntime, ModelDecision, RuntimeAction
from career_agent.cli import main as cli_main
from career_agent.agent.main_agent_tools import MainAgentToolRegistry
from career_agent.domain.job_discovery import JobDetail, Provenance
from career_agent.storage.context import CareerContextStore
from career_agent.storage.jobs import JDAnalysisPayload, SQLiteJobPostingRepository
from career_agent.storage.capability_confirmations import SQLiteCapabilityConfirmationStore
from career_agent.storage.action_executions import (
    ActionExecutionConflictError,
    SQLiteActionExecutionStore,
)
from career_agent.harness.streaming import ClientActionEvent, InteractionRequiredEvent, InteractionResponse


class DecisionMaker:
    def __init__(self, decision: AgentDecision) -> None:
        self.decision = decision

    def decide(self, context, tool_names):
        assert tuple(spec["function"]["name"] for spec in tool_names) == ("open_job_search",)
        return self.decision


class SequenceDecisionMaker:
    def __init__(self, *decisions: AgentDecision) -> None:
        self.decisions = list(decisions)
        self.contexts = []

    def decide(self, context, tool_names):
        self.contexts.append(context)
        if not self.decisions:
            raise AssertionError("Main Agent requested more decisions than expected")
        return self.decisions.pop(0)


class StaticSummaryWorker:
    def summarize(self, *, previous, messages):
        return ConversationSummaryContent(
            user_goals=("保留会话连续性",),
            confirmed_decisions=(),
            unresolved_questions=(),
            active_constraints=(),
        )


class CountingRegistry(MainAgentToolRegistry):
    """Records atomic-tool invocations so loop and de-duplication rules stay testable."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.calls: list[tuple[str, dict]] = []

    def invoke_atomic_tool(self, name, arguments):
        self.calls.append((name, dict(arguments)))
        return super().invoke_atomic_tool(name, arguments)


def build_runtime(tmp_path, decision: AgentDecision):
    manager = ContextManager(CareerContextStore(tmp_path / "context.sqlite3"))
    manager.upsert_profile(CareerProfileContext(user_id="u1", default_city="Shanghai"))
    tools = CountingRegistry()
    return MainAgentRuntime(context_manager=manager, decision_maker=DecisionMaker(decision), tools=tools), tools, manager


def _never_called_decision_maker():
    """A decision maker whose being consulted is itself the failure."""

    class Never:
        def decide(self, context, tool_specs):
            raise AssertionError("the model must not be consulted on this ingress")

    return Never()


def test_main_graph_uses_one_authorize_act_path_for_every_capability(tmp_path) -> None:
    agent, _, _ = build_runtime(tmp_path, AgentDecision(action="final", message="done"))

    assert set(agent._graph.get_graph().nodes) == {
        "__start__",
        "hydrate",
        "decide",
        "authorize",
        "act",
        "observe",
        "present",
        "interrupt",
        "__end__",
    }


def test_runtime_does_not_rewrite_a_model_question_from_prompt_wording(tmp_path) -> None:
    manager = ContextManager(CareerContextStore(tmp_path / "context.sqlite3"))
    manager.upsert_profile(CareerProfileContext(user_id="u1"))
    context = manager.load_for_turn(
        user_id="u1",
        conversation_id="c1",
        user_message="帮我记录一次投递",
    )

    decision = AgentDecision(
        action="ask_user",
        message="请告诉我这次实际投递的是哪个已保存职位。",
    )

    runtime = MainAgentRuntime(
        context_manager=manager,
        decision_maker=SequenceDecisionMaker(decision),
        tools=MainAgentToolRegistry(),
    )
    result = runtime.run_turn(
        user_id="u1", conversation_id="c1", user_message="记录投递"
    )

    assert result.model_decision == decision
    assert result.model_decision.action == "ask_user"


def test_main_graph_has_distinct_delivery_and_suspension_exits(tmp_path) -> None:
    agent, _, _ = build_runtime(tmp_path, AgentDecision(action="final", message="done"))
    graph = agent._graph.get_graph()

    ends = {edge.source for edge in graph.edges if edge.target == "__end__"}
    assert ends == {"present", "interrupt"}


def test_create_application_reuses_a_succeeded_request_slot_without_reinvoking(
    tmp_path,
) -> None:
    class Registry(MainAgentToolRegistry):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def capability_kind(self, name):
            assert name == "create_application"
            return "atomic_tool"

        def invoke_atomic_tool(self, name, arguments):
            self.calls += 1
            return ToolObservation(
                tool_name=name,
                state="application_ready",
                message="已创建投递记录。",
                payload={
                    "application_id": "application_123456",
                    "job_posting_id": "job_123456",
                    "resume_version_id": "resume_version_123456",
                    "status": "submitted",
                    "created": True,
                },
                execution_outcome="committed",
            )

    class Runtime(MainAgentRuntime):
        @staticmethod
        def _project_atomic_tool_arguments(context, name, arguments):
            return {"user_id": context.profile.user_id, **arguments}

    manager = ContextManager(CareerContextStore(tmp_path / "context.sqlite3"))
    manager.upsert_profile(CareerProfileContext(user_id="u1"))
    registry = Registry()
    ledger = SQLiteActionExecutionStore(tmp_path / "context.sqlite3")

    def run_once(note):
        return Runtime(
            context_manager=manager,
            decision_maker=SequenceDecisionMaker(
                AgentDecision(
                    action="tool_call",
                    tool_call=ToolCall(
                        name="create_application", arguments={"note": note}
                    ),
                ),
                AgentDecision(action="final", message="完成。"),
            ),
            tools=registry,
            action_execution_store=ledger,
        ).run_turn(
            user_id="u1",
            conversation_id="c1",
            user_message="记录投递",
            request_id="request-1",
        )

    first = run_once("官网投递")
    replay = run_once("官网投递")

    assert registry.calls == 1
    assert first.context.task.active_application_id == "application_123456"
    assert replay.context.task.active_application_id == "application_123456"
    assert "此前已经完成" in replay.tool_result.message
    assert replay.tool_result.state == "action_execution_replayed"
    assert replay.tool_result.payload == {}
    assert ledger.list_pending(user_id="u1") == ()

    with pytest.raises(ActionExecutionConflictError):
        run_once("内推投递")
    assert registry.calls == 1


def test_manual_settlement_repairs_task_state_on_request_replay(tmp_path) -> None:
    """A CLI finding becomes a reducer receipt, not arbitrary audit payload."""

    class Registry(MainAgentToolRegistry):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def capability_kind(self, name):
            return "atomic_tool"

        def invoke_atomic_tool(self, name, arguments):  # pragma: no cover - guarded
            self.calls += 1
            raise AssertionError("a settled action must not be executed again")

    class Runtime(MainAgentRuntime):
        @staticmethod
        def _project_atomic_tool_arguments(context, name, arguments):
            return {"user_id": context.profile.user_id, **arguments}

    database = tmp_path / "context.sqlite3"
    manager = ContextManager(CareerContextStore(database))
    manager.upsert_profile(CareerProfileContext(user_id="u1"))
    ledger = SQLiteActionExecutionStore(database)
    projected_arguments = {"user_id": "u1"}
    fingerprint = hashlib.sha256(
        json.dumps(
            {"tool": "create_application", "arguments": projected_arguments},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode()
    ).hexdigest()
    execution, _ = ledger.prepare(
        user_id="u1",
        conversation_id="c1",
        anchor="request-1",
        request_id="request-1",
        write_slot=0,
        tool_name="create_application",
        fingerprint=fingerprint,
        policy_epoch=1,
    )
    stdout = StringIO()
    assert cli_main(
        [
            "actions", "settle",
            "--context-store", str(database),
            "--action-id", execution.action_id,
            "--executed",
            "--output", "application_id=app-1",
            "--output", "job_posting_id=job-1",
            "--output", "resume_version_id=resume-1",
            "--output", "status=submitted",
        ],
        stdout=stdout,
        stderr=StringIO(),
    ) == 0

    registry = Registry()
    turn = Runtime(
        context_manager=manager,
        decision_maker=SequenceDecisionMaker(
            AgentDecision(
                action="tool_call",
                tool_call=ToolCall(name="create_application", arguments={}),
            ),
            AgentDecision(action="final", message="已恢复投递状态。"),
        ),
        tools=registry,
        action_execution_store=ledger,
    ).run_turn(
        user_id="u1",
        conversation_id="c1",
        user_message="继续",
        request_id="request-1",
    )

    assert registry.calls == 0
    assert turn.tool_result.state == "action_execution_replayed"
    assert turn.context.task.active_application_id == "app-1"
    assert turn.context.task.active_job_posting_id == "job-1"
    assert turn.context.task.active_resume_version_id == "resume-1"
    assert turn.context.task.active_application_status == "submitted"


def test_a_replayed_interaction_write_does_not_recreate_the_old_interaction(
    tmp_path,
) -> None:
    """Reducer repair and model observation have deliberately different states."""

    class Registry(MainAgentToolRegistry):
        def capability_kind(self, name):
            return "atomic_tool"

        def invoke_atomic_tool(self, name, arguments):  # pragma: no cover - guarded
            raise AssertionError("a succeeded analysis must not run again")

    class Runtime(MainAgentRuntime):
        @staticmethod
        def _project_atomic_tool_arguments(context, name, arguments):
            return {"user_id": context.profile.user_id, **arguments}

    database = tmp_path / "context.sqlite3"
    manager = ContextManager(CareerContextStore(database))
    manager.upsert_profile(CareerProfileContext(user_id="u1"))
    ledger = SQLiteActionExecutionStore(database)
    arguments = {"user_id": "u1"}
    fingerprint = hashlib.sha256(
        json.dumps(
            {"tool": "analyze_resume", "arguments": arguments},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode()
    ).hexdigest()
    execution, _ = ledger.prepare(
        user_id="u1",
        conversation_id="c1",
        anchor="request-1",
        request_id="request-1",
        write_slot=0,
        tool_name="analyze_resume",
        fingerprint=fingerprint,
        policy_epoch=1,
    )
    ledger.succeed(
        action_id=execution.action_id,
        output={
            "__result_state__": "resume_analysis_ready",
            "analysis_id": "analysis-1",
            "resume_version_id": "resume-1",
        },
    )

    turn = Runtime(
        context_manager=manager,
        decision_maker=SequenceDecisionMaker(
            AgentDecision(
                action="tool_call",
                tool_call=ToolCall(name="analyze_resume", arguments={}),
            ),
            AgentDecision(action="final", message="分析已经做过，可以读取结果。"),
        ),
        tools=Registry(),
        action_execution_store=ledger,
    ).run_turn(
        user_id="u1",
        conversation_id="c1",
        user_message="继续分析",
        request_id="request-1",
    )

    assert turn.tool_result.state == "action_execution_replayed"
    assert turn.context.task.active_resume_analysis_id == "analysis-1"
    assert turn.context.task.active_resume_version_id == "resume-1"
    assert turn.context.task.resume_analysis_status == "pending"
    assert turn.model_decision.action == "final"


@pytest.mark.parametrize(
    ("state", "execution_outcome", "expected_status"),
    (
        # Two independent axes, so all four corners. ``state`` says whether the
        # capability and the control flow failed; ``execution_outcome`` says
        # whether the side effect committed. The ledger must record the second
        # one, and the first two rows would each be settled the opposite way by
        # reading the state.
        ("application_input_not_found", "not_committed", "FAILED"),
        ("application_input_not_found", "unknown", "PENDING"),
        # The corner the dispatch order exists for: the external write landed
        # and the local handling failed afterwards. Recording FAILED here would
        # tell a reconciler to redo a write that already committed.
        ("calendar_write_failed", "committed", "SUCCEEDED"),
    ),
)
def test_a_declared_execution_outcome_settles_the_ledger_over_the_state(
    tmp_path, state, execution_outcome, expected_status
) -> None:
    class Registry(MainAgentToolRegistry):
        def capability_kind(self, name):
            return "atomic_tool"

        def invoke_atomic_tool(self, name, arguments):
            return ToolObservation(
                tool_name=name,
                state=state,
                message="这一步的结果见下。",
                execution_outcome=execution_outcome,
            )

    class Runtime(MainAgentRuntime):
        @staticmethod
        def _project_atomic_tool_arguments(context, name, arguments):
            return {"user_id": context.profile.user_id, **arguments}

    database = tmp_path / f"{state}-{execution_outcome}.sqlite3"
    manager = ContextManager(CareerContextStore(database))
    manager.upsert_profile(CareerProfileContext(user_id="u1"))
    ledger = SQLiteActionExecutionStore(database)
    Runtime(
        context_manager=manager,
        decision_maker=SequenceDecisionMaker(
            AgentDecision(
                action="tool_call",
                tool_call=ToolCall(name="create_application", arguments={}),
            ),
            AgentDecision(action="final", message="已说明。"),
        ),
        tools=Registry(),
        action_execution_store=ledger,
    ).run_turn(
        user_id="u1",
        conversation_id="c1",
        user_message="记录投递",
        request_id="request-1",
    )

    execution = ledger.list_for_anchor(
        user_id="u1", conversation_id="c1", anchor="request-1"
    )[0]
    assert execution.status == expected_status


def test_an_undeclared_write_outcome_fails_loudly_and_stays_pending(tmp_path) -> None:
    class Registry(MainAgentToolRegistry):
        def capability_kind(self, name):
            return "atomic_tool"

        def invoke_atomic_tool(self, name, arguments):
            return ToolObservation(
                tool_name=name,
                state="application_ready",
                message="结果遗漏了执行轴。",
            )

    class Runtime(MainAgentRuntime):
        @staticmethod
        def _project_atomic_tool_arguments(context, name, arguments):
            return {"user_id": context.profile.user_id, **arguments}

    database = tmp_path / "undeclared.sqlite3"
    manager = ContextManager(CareerContextStore(database))
    manager.upsert_profile(CareerProfileContext(user_id="u1"))
    ledger = SQLiteActionExecutionStore(database)

    with pytest.raises(ValueError, match="returned without execution_outcome"):
        Runtime(
            context_manager=manager,
            decision_maker=SequenceDecisionMaker(
                AgentDecision(
                    action="tool_call",
                    tool_call=ToolCall(name="create_application", arguments={}),
                )
            ),
            tools=Registry(),
            action_execution_store=ledger,
        ).run_turn(
            user_id="u1",
            conversation_id="c1",
            user_message="记录投递",
            request_id="request-1",
        )

    assert ledger.list_for_anchor(
        user_id="u1", conversation_id="c1", anchor="request-1"
    )[0].status == "PENDING"


@pytest.mark.parametrize(
    "origin",
    (
        ModelDecision(AgentDecision(action="final", message="好的。")),
        InteractionReceipt(scope="resume_analysis_confirmation", action="confirm"),
        RuntimeAction(workflow="mock_interview"),
    ),
)
def test_accountability_is_derived_from_the_origin_and_cannot_be_stated(origin) -> None:
    """The field is a property, so no construction site can contradict its type.

    The predecessors of this design were both stated fields — ``decision_source``
    and then ``requested_by``/``authority`` — sitting beside a ``decision`` that
    was always an ``AgentDecision`` whether or not a model made one. A stated
    field can be set wrongly at a new ingress, and the whole point of the union
    is that the answer follows from the shape rather than from a value someone
    remembered to pass.
    """
    turn = MainAgentTurnResult(origin=origin, context=None, assistant_message="")

    assert turn.requested_by == origin.requested_by
    assert (turn.model_decision is not None) is isinstance(origin, ModelDecision)
    with pytest.raises(TypeError):
        MainAgentTurnResult(
            origin=origin,
            requested_by="model",
            context=None,
            assistant_message="",
        )


def test_multiple_write_budget_uses_distinct_durable_write_slots(tmp_path) -> None:
    class Registry(MainAgentToolRegistry):
        def capability_kind(self, name):
            return "atomic_tool"

        def invoke_atomic_tool(self, name, arguments):
            return ToolObservation(
                tool_name=name,
                state="application_ready",
                message="已记录。",
                payload={"application_id": arguments["note"]},
                execution_outcome="committed",
            )

    class Runtime(MainAgentRuntime):
        @staticmethod
        def _project_atomic_tool_arguments(context, name, arguments):
            return {"user_id": context.profile.user_id, **arguments}

    database = tmp_path / "context.sqlite3"
    manager = ContextManager(CareerContextStore(database))
    manager.upsert_profile(CareerProfileContext(user_id="u1"))
    ledger = SQLiteActionExecutionStore(database)
    Runtime(
        context_manager=manager,
        decision_maker=SequenceDecisionMaker(
            AgentDecision(
                action="tool_call",
                tool_call=ToolCall(name="create_application", arguments={"note": "one"}),
            ),
            AgentDecision(
                action="tool_call",
                tool_call=ToolCall(name="create_application", arguments={"note": "two"}),
            ),
            AgentDecision(action="final", message="完成。"),
        ),
        tools=Registry(),
        action_execution_store=ledger,
        max_read_calls=5,
        max_write_calls=2,
    ).run_turn(
        user_id="u1", conversation_id="c1", user_message="记录两次", request_id="request-1"
    )

    executions = ledger.list_for_anchor(
        user_id="u1", conversation_id="c1", anchor="request-1"
    )
    assert [item.write_slot for item in executions] == [0, 1]


def test_loop_state_and_budget_window_are_structurally_bounded(tmp_path) -> None:
    from career_agent.agent.main_agent_runtime import (
        DEFAULT_MAX_AUTHORIZATION_REFUSALS,
        DEFAULT_MAX_EXTERNAL_WRITE_CALLS,
        DEFAULT_MAX_PROJECTION_REFUSALS,
        DEFAULT_MAX_READ_CALLS,
        DEFAULT_MAX_WRITE_CALLS,
        MainAgentState,
    )

    # A ratchet against loop-state creep, not a derived limit: the loop should
    # not grow a new channel without someone deciding it earned one. Raised to
    # 10 for ``career_memory_scope_keys``, which records what the prompt showed
    # so a later tombstone can find the messages that saw it. It cannot ride on
    # ``career_memory`` because that projection is deliberately invalidated by
    # the write it must outlive.
    assert len(MainAgentState.__annotations__) <= 10
    assert (
        DEFAULT_MAX_READ_CALLS
        + DEFAULT_MAX_WRITE_CALLS
        + DEFAULT_MAX_EXTERNAL_WRITE_CALLS
        + DEFAULT_MAX_PROJECTION_REFUSALS
        + DEFAULT_MAX_AUTHORIZATION_REFUSALS
        <= MAX_DECISION_OBSERVATIONS
    )
    with pytest.raises(ValueError, match="must fit the observation window"):
        MainAgentRuntime(
            context_manager=ContextManager(
                CareerContextStore(tmp_path / "overflow-context.sqlite3")
            ),
            decision_maker=SequenceDecisionMaker(
                AgentDecision(action="final", message="done")
            ),
            tools=MainAgentToolRegistry(),
            max_read_calls=7,
            max_write_calls=1,
            max_external_write_calls=1,
            max_projection_refusals=2,
            max_authorization_refusals=1,
        )


def test_runtime_rejects_an_unguarded_request_token_estimator(tmp_path) -> None:
    class PartialEstimator:
        @staticmethod
        def request_token_usage(context, tool_specs):
            return 1000, 32000

        @staticmethod
        def decide(context, tool_specs):
            return AgentDecision(action="final", message="done")

    with pytest.raises(ValueError, match="static request token usage"):
        MainAgentRuntime(
            context_manager=ContextManager(
                CareerContextStore(tmp_path / "unguarded-token-estimator.sqlite3")
            ),
            decision_maker=PartialEstimator(),
            tools=MainAgentToolRegistry(),
        )


def test_a_mid_turn_reload_keeps_a_clipped_message_whole_in_storage(
    tmp_path,
) -> None:
    """A memory write reloads the context from the message as sent, not its
    prompt copy, so the turn still stores what the user typed."""
    from career_agent.agent.token_budget import message_token_count

    long_message = (
        "负责大模型推理服务的性能优化与稳定性建设，熟悉分布式训练框架，"
        "具备鑫龘饕餮等生僻字处理经验；" * 800
    )[:32_000]

    class NoteWritingRegistry(MainAgentToolRegistry):
        def capability_kind(self, name):
            return "atomic_tool"

        def invoke_atomic_tool(self, name, arguments):
            return ToolObservation(
                tool_name=name,
                state="working_notes_updated",
                message="已更新工作笔记。",
                execution_outcome="committed",
            )

    class BudgetedDecisions(SequenceDecisionMaker):
        @staticmethod
        def static_request_token_usage(tool_specs):
            return 12_066, 32_000

        @staticmethod
        def request_token_usage(context, tool_specs):
            return 20_000, 32_000

    class Runtime(MainAgentRuntime):
        @staticmethod
        def _project_atomic_tool_arguments(context, name, arguments):
            return {"user_id": context.profile.user_id, **arguments}

    manager = ContextManager(CareerContextStore(tmp_path / "context.sqlite3"))
    manager.upsert_profile(CareerProfileContext(user_id="u1"))
    decisions = BudgetedDecisions(
        AgentDecision(
            action="tool_call",
            tool_call=ToolCall(name="update_working_notes", arguments={}),
        ),
        AgentDecision(action="final", message="已记录。"),
    )

    Runtime(
        context_manager=manager,
        decision_maker=decisions,
        tools=NoteWritingRegistry(),
    ).run_turn(user_id="u1", conversation_id="c1", user_message=long_message)

    # The write reloaded the context between the two decisions; both still saw
    # the clipped copy and knew it was clipped.
    assert len(decisions.contexts) == 2
    for context in decisions.contexts:
        assert context.user_message_clipped is True
        assert message_token_count(context.user_message) <= 3_986
    stored = manager._store.list_message_records("u1", "c1", limit=10)
    assert stored[0].message.content == long_message


def test_a_turns_first_estimate_pairs_with_its_first_provider_count(
    tmp_path,
) -> None:
    """The calibration sample for the request estimator, read the way C will.

    Each load leaves its estimate as a trace. The first estimate of a run and
    the run's first model_succeeded describe the same request, before any
    observation joins it; a reload after a memory write adds a second estimate
    that pairs with nothing earlier.
    """
    from career_agent.storage.run_events import SQLiteTraceRecorder

    class NoteWritingRegistry(MainAgentToolRegistry):
        def capability_kind(self, name):
            return "atomic_tool"

        def invoke_atomic_tool(self, name, arguments):
            return ToolObservation(
                tool_name=name,
                state="working_notes_updated",
                message="已更新工作笔记。",
                execution_outcome="committed",
            )

    class MeteredDecisions(SequenceDecisionMaker):
        @staticmethod
        def static_request_token_usage(tool_specs):
            return 12_066, 32_000

        @staticmethod
        def request_token_usage(context, tool_specs):
            return 13_500, 32_000

        def consume_cache_metrics(self):
            return {"input_units": 9_100 + 100 * len(self.contexts)}

    class Runtime(MainAgentRuntime):
        @staticmethod
        def _project_atomic_tool_arguments(context, name, arguments):
            return {"user_id": context.profile.user_id, **arguments}

    recorder = SQLiteTraceRecorder(tmp_path / "run_events.sqlite3")
    manager = ContextManager(
        CareerContextStore(tmp_path / "context.sqlite3"),
        summary_worker=StaticSummaryWorker(),
    )
    manager.upsert_profile(CareerProfileContext(user_id="u1"))
    Runtime(
        context_manager=manager,
        decision_maker=MeteredDecisions(
            AgentDecision(
                action="tool_call",
                tool_call=ToolCall(name="update_working_notes", arguments={}),
            ),
            AgentDecision(action="final", message="已记录。"),
        ),
        tools=NoteWritingRegistry(),
        trace_recorder=recorder,
    ).run_turn(user_id="u1", conversation_id="c1", user_message="记一下我偏好远程")

    events = recorder.list_conversation_events(user_id="u1", conversation_id="c1")

    assert len({event.run_id for event in events}) == 1
    assert [event.event_type for event in events] == [
        "context_estimated",
        "model_succeeded",
        "context_estimated",
        "model_succeeded",
    ]
    first_estimate, first_call = events[0], events[1]
    # Stored through the real SQLite recorder, so a redacted field shows here.
    assert first_estimate.details["input_occupancy_numerator"] == 13_500
    assert first_estimate.details["input_occupancy_denominator"] == 32_000
    assert first_call.details["input_units"] == 9_200


def test_navigation_only_job_search_opens_boss_without_discovery_gateway(
    tmp_path,
) -> None:
    manager = ContextManager(CareerContextStore(tmp_path / "context.sqlite3"))
    manager.upsert_profile(
        CareerProfileContext(user_id="u1", default_city="上海")
    )
    decisions = SequenceDecisionMaker(
        AgentDecision(
            action="tool_call",
            tool_call=ToolCall(
                name="open_job_search", arguments={"keyword": "AI 产品经理"}
            ),
        ),
        AgentDecision(action="final", message=""),
    )
    tools = MainAgentToolRegistry()
    runtime = MainAgentRuntime(
        context_manager=manager,
        decision_maker=decisions,
        tools=tools,
    )
    events = []

    result = runtime.run_turn(
        user_id="u1",
        conversation_id="c1",
        user_message="帮我找上海的 AI 产品经理岗位",
        event_sink=events.append,
    )

    assert tools.workflow_names == ()
    assert tools.atomic_tool_names == ("open_job_search",)
    assert [spec["function"]["name"] for spec in tools.schemas()] == [
        "open_job_search"
    ]
    description = tools.schemas()[0]["function"]["description"]
    assert "ask for the city instead of guessing or searching nationwide" in description
    assert result.tool_results[0].state == "job_search_page_ready"
    action = next(event for event in events if isinstance(event, ClientActionEvent))
    parsed = urlparse(action.url)
    assert parsed.hostname == "www.zhipin.com"
    assert parse_qs(parsed.query) == {
        "query": ["AI 产品经理"],
        "city": ["101020100"],
    }
    assert result.context.task.active_workflow == "none"


def test_graph_hydrates_career_memory_before_first_decision(tmp_path) -> None:
    class Projector:
        def project(self, *, user_id, query):
            assert user_id == "u1"
            assert query == "帮我规划下一步"
            return CareerMemoryContext(
                records=(
                    CareerMemoryRecord(
                        record_type="work",
                        organization="Example Inc.",
                        title="Product Manager",
                        is_current=True,
                        confirmed_highlights=(
                            CareerMemoryClaim(
                                claim="Led an AI product",
                                origin="user_input",
                                recorded_at=datetime(
                                    2026, 9, 1, tzinfo=timezone.utc
                                ),
                                revision=1,
                                detail_ref="detail_" + "a" * 24,
                            ),
                        ),
                    ),
                )
            )

    manager = ContextManager(CareerContextStore(tmp_path / "context.sqlite3"))
    manager.upsert_profile(CareerProfileContext(user_id="u1"))
    decisions = SequenceDecisionMaker(AgentDecision(action="final", message="done"))
    runtime = MainAgentRuntime(
        context_manager=manager,
        decision_maker=decisions,
        tools=MainAgentToolRegistry(),
        career_context_projector=Projector(),
    )

    runtime.run_turn(
        user_id="u1",
        conversation_id="c1",
        user_message="帮我规划下一步",
    )

    assert decisions.contexts[0].career_memory.records[0].title == "Product Manager"


def test_registry_classifies_workflows_and_atomic_tools(tmp_path) -> None:
    repository = SQLiteJobPostingRepository(tmp_path / "jobs.sqlite3")
    tools = MainAgentToolRegistry(job_repository=repository)

    assert tools.workflow_names == ()
    assert tools.atomic_tool_names == ("open_job_search", "find_saved_jobs", "get_saved_job")
    assert tools.capability_kind("open_job_search") == "atomic_tool"
    assert tools.capability_kind("find_saved_jobs") == "atomic_tool"


def test_runtime_streams_real_progress_and_fake_final_content(tmp_path) -> None:
    agent, _, _ = build_runtime(
        tmp_path,
        AgentDecision(
            action="final",
            message="第一段回答。\n\n第二段回答用于验证分块。",
        ),
    )
    events = []

    result = agent.run_turn(
        user_id="u1",
        conversation_id="c1",
        user_message="请直接回答",
        event_sink=events.append,
    )

    event_types = [event.type for event in events]
    assert event_types[0] == "turn_started"
    assert "progress" in event_types
    assert event_types[-1] == "turn_completed"
    assert "".join(
        event.delta for event in events if event.type == "content_delta"
    ) == result.assistant_message


def test_the_reply_streams_before_the_turn_is_saved(tmp_path) -> None:
    agent, _, _ = build_runtime(
        tmp_path,
        AgentDecision(action="final", message="答案在保存之前就发出。"),
    )
    events = []

    agent.run_turn(
        user_id="u1",
        conversation_id="c1",
        user_message="请直接回答",
        event_sink=events.append,
    )

    kinds = [
        (event.type, getattr(event, "stage", None)) for event in events
    ]
    first_delta = kinds.index(("content_delta", None))
    saving = kinds.index(("progress", "saving"))
    assert first_delta < saving < kinds.index(("turn_completed", None))


def test_a_commit_failure_after_the_reply_names_the_save_not_the_answer(
    tmp_path, monkeypatch
) -> None:
    agent, _, manager = build_runtime(
        tmp_path,
        AgentDecision(action="final", message="这条回复已经生成。"),
    )

    def broken_commit(**kwargs):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(manager, "commit_turn", broken_commit)
    events = []

    with pytest.raises(sqlite3.OperationalError):
        agent.run_turn(
            user_id="u1",
            conversation_id="c1",
            user_message="请直接回答",
            event_sink=events.append,
        )

    assert "".join(
        event.delta for event in events if event.type == "content_delta"
    ) == "这条回复已经生成。"
    failed = events[-1]
    assert failed.type == "turn_failed"
    assert failed.code == "TURN_COMMIT_FAILED"
    assert "未能保存" in failed.message


def test_a_failure_before_any_reply_stays_a_plain_turn_failure(tmp_path) -> None:
    class Exploding:
        def decide(self, context, tool_names):
            raise RuntimeError("provider down")

    manager = ContextManager(CareerContextStore(tmp_path / "context.sqlite3"))
    manager.upsert_profile(CareerProfileContext(user_id="u1"))
    agent = MainAgentRuntime(
        context_manager=manager, decision_maker=Exploding(), tools=CountingRegistry()
    )
    events = []

    with pytest.raises(RuntimeError):
        agent.run_turn(
            user_id="u1",
            conversation_id="c1",
            user_message="请直接回答",
            event_sink=events.append,
        )

    assert not [event for event in events if event.type == "content_delta"]
    assert events[-1].type == "turn_failed"
    assert events[-1].code == "TURN_EXECUTION_FAILED"


def test_a_decision_retry_and_a_long_wait_are_announced_as_progress(
    tmp_path,
) -> None:
    from career_agent.agent.decision_attempts import (
        DecisionAttempt,
        notify_decision_attempt,
    )

    class RetryingDecisionMaker:
        def decide(self, context, tool_names):
            notify_decision_attempt(
                DecisionAttempt(attempt=1, max_attempts=2, elapsed_seconds=0.0)
            )
            notify_decision_attempt(
                DecisionAttempt(
                    attempt=2,
                    max_attempts=2,
                    elapsed_seconds=61.2,
                    previous_error_code="MAIN_AGENT_TRANSPORT_ERROR",
                )
            )
            time.sleep(0.15)
            return AgentDecision(action="final", message="完成。")

    manager = ContextManager(CareerContextStore(tmp_path / "context.sqlite3"))
    manager.upsert_profile(CareerProfileContext(user_id="u1"))
    agent = MainAgentRuntime(
        context_manager=manager,
        decision_maker=RetryingDecisionMaker(),
        tools=CountingRegistry(),
    )
    agent.DECISION_HEARTBEAT_SECONDS = 0.05
    events = []

    agent.run_turn(
        user_id="u1",
        conversation_id="c1",
        user_message="请直接回答",
        event_sink=events.append,
    )

    deciding = [
        event.message
        for event in events
        if event.type == "progress" and event.stage == "deciding"
    ]
    assert deciding[0] == "正在判断下一步操作……"
    assert "上一次请求超时或连接中断，正在重新判断（第 2/2 次，已等待 61 秒）……" in deciding
    assert any(message.startswith("仍在等待模型判断（已等待") for message in deciding)


def test_capability_steps_and_a_long_tool_call_are_announced_as_progress(
    tmp_path,
) -> None:
    from career_agent.harness.capability_steps import notify_capability_step

    class Registry(MainAgentToolRegistry):
        def capability_kind(self, name):
            return "atomic_tool"

        def invoke_atomic_tool(self, name, arguments):
            notify_capability_step("resume_analysis")
            notify_capability_step("internal_label_nobody_maps")
            notify_capability_step("resume_analysis", kind="retry")
            notify_capability_step("job_research", index=2)
            notify_capability_step("job_research.read_file", kind="tool")
            notify_capability_step("email_sync.scan", kind="io", index=3, total=12)
            time.sleep(0.15)
            return ToolObservation(
                tool_name=name,
                state="resume_analysis_ready",
                message="分析完成。",
                execution_outcome="committed",
            )

    class Runtime(MainAgentRuntime):
        @staticmethod
        def _project_atomic_tool_arguments(context, name, arguments):
            return {"user_id": context.profile.user_id, **arguments}

    manager = ContextManager(CareerContextStore(tmp_path / "context.sqlite3"))
    manager.upsert_profile(CareerProfileContext(user_id="u1"))
    agent = Runtime(
        context_manager=manager,
        decision_maker=SequenceDecisionMaker(
            AgentDecision(
                action="tool_call",
                tool_call=ToolCall(name="analyze_resume", arguments={}),
            ),
            AgentDecision(action="final", message="完成。"),
        ),
        tools=Registry(),
    )
    agent.DECISION_HEARTBEAT_SECONDS = 0.05
    events = []

    agent.run_turn(
        user_id="u1",
        conversation_id="c1",
        user_message="分析简历",
        event_sink=events.append,
    )

    running = [
        event.message
        for event in events
        if event.type == "progress" and event.stage == "running_capability"
    ]
    assert running[:5] == [
        "正在分析简历内容……",
        "正在分析简历内容时请求失败，正在重试……",
        "正在调研岗位背景（第 2 次调用模型）……",
        "正在阅读工作指南……",
        "正在扫描邮件（第 3/12 项）……",
    ]
    assert not any("internal_label" in message for message in running)
    heartbeat = next(
        event
        for event in events
        if event.type == "progress" and event.message.startswith("正在扫描邮件（已等待")
    )
    completed = next(event for event in events if event.type == "capability_completed")
    assert events.index(heartbeat) < events.index(completed)


def test_stream_observer_failure_does_not_fail_business_turn(tmp_path) -> None:
    agent, _, manager = build_runtime(
        tmp_path,
        AgentDecision(action="final", message="仍然完成。"),
    )

    def broken_sink(_event) -> None:
        raise RuntimeError("client disconnected")

    result = agent.run_turn(
        user_id="u1",
        conversation_id="c1",
        user_message="继续执行",
        event_sink=broken_sink,
    )

    assert result.assistant_message == "仍然完成。"
    history = manager.load_for_turn(
        user_id="u1", conversation_id="c1", user_message="下一轮"
    ).recent_messages
    assert history[-1].content == "仍然完成。"


def test_tool_observation_returns_to_model_before_final_answer(tmp_path) -> None:
    manager = ContextManager(CareerContextStore(tmp_path / "context.sqlite3"))
    manager.upsert_profile(CareerProfileContext(user_id="u1"))
    tools = CountingRegistry()
    decisions = SequenceDecisionMaker(
        AgentDecision(action="tool_call", tool_call=ToolCall(name="open_job_search", arguments={"keyword": "AI Engineer"})),
        AgentDecision(action="ask_user", message="搜索页已经打开，你想先看哪一个岗位？"),
    )
    agent = MainAgentRuntime(context_manager=manager, decision_maker=decisions, tools=tools)

    result = agent.run_turn(user_id="u1", conversation_id="c1", user_message="帮我找工作")

    assert result.model_decision.action == "ask_user"
    assert result.assistant_message == "搜索页已经打开，你想先看哪一个岗位？"
    assert len(tools.calls) == 1
    assert len(decisions.contexts) == 2
    observation = decisions.contexts[1].model_context()["tool_observations"][0]
    assert observation == {
        "tool_name": "open_job_search",
        "state": "job_search_page_ready",
        "message": "已准备打开 BOSS 搜索“AI Engineer”。请正常浏览，并只保存你感兴趣的岗位。",
        "facts": {},
        # What the model wrote, so two calls to one capability stay distinct.
        # The projected form is not shown: it carries what the handler needs,
        # including the ids the projection boundary keeps from the model.
        "arguments": {"keyword": "AI Engineer"},
    }
    serialized = str(observation)
    assert "zhipin.com" not in serialized


def test_conversation_span_is_turn_local_observation_not_recent_history(
    tmp_path,
) -> None:
    store = CareerContextStore(tmp_path / "context.sqlite3")
    manager = ContextManager(
        store,
        recent_message_limit=2,
        summary_batch_size=2,
        summary_worker=StaticSummaryWorker(),
        max_recent_context_chars=60,
        compact_occupancy_threshold=0.7,
    )
    manager.upsert_profile(CareerProfileContext(user_id="u1"))
    for index in range(3):
        context = manager.load_for_turn(
            user_id="u1",
            conversation_id="c1",
            user_message=f"private-old-user-{index}",
        )
        manager.commit_turn(
            context=context,
            task=ConversationTaskState(),
            assistant_message=f"private-old-assistant-{index}",
        )
    decisions = SequenceDecisionMaker(
        AgentDecision(
            action="tool_call",
            tool_call=ToolCall(
                name="read_conversation_span",
                arguments={"from_sequence": 1, "through_sequence": 2},
            ),
        ),
        AgentDecision(action="final", message="我已根据那段历史继续处理。"),
    )
    tools = CountingRegistry(conversation_store=store)
    agent = MainAgentRuntime(
        context_manager=manager,
        decision_maker=decisions,
        tools=tools,
    )

    agent.run_turn(
        user_id="u1", conversation_id="c1", user_message="回看缺失的开头。"
    )

    assert decisions.contexts[0].through_sequence == 4
    assert decisions.contexts[0].recent_from_sequence == 5
    observation = decisions.contexts[1].tool_observations[-1]
    assert observation.facts == {
        "from_sequence": 1,
        "through_sequence": 2,
        "returned": 2,
        "total": 2,
        "body_clipped": False,
        "content_clipped": False,
            "resource_ref_count": 0,
            "resource_ref_total": 0,
    }
    assert observation.body is not None
    assert "private-old-user-0" in observation.body
    assert "private-old-assistant-0" in observation.body
    assert tools.calls == [
        (
            "read_conversation_span",
            {
                "user_id": "u1",
                "conversation_id": "c1",
                "from_sequence": 1,
                "through_sequence": 2,
            },
        )
    ]
    recent = manager.load_for_turn(
        user_id="u1", conversation_id="c1", user_message="next"
    ).recent_messages
    assert [message.content for message in recent] == [
        "回看缺失的开头。",
        "我已根据那段历史继续处理。",
    ]
    assert "private-old-user-0" not in str(recent)


def test_conversation_span_tells_the_model_when_its_body_is_clipped(
    tmp_path,
) -> None:
    store = CareerContextStore(tmp_path / "context.sqlite3")
    manager = ContextManager(store)
    context = manager.load_for_turn(
        user_id="u1",
        conversation_id="c1",
        user_message="u" * 5000,
    )
    manager.commit_turn(
        context=context,
        task=ConversationTaskState(),
        assistant_message="a" * 5000,
    )
    tools = MainAgentToolRegistry(conversation_store=store)

    result = tools.invoke_atomic_tool(
        "read_conversation_span",
        {
            "user_id": "u1",
            "conversation_id": "c1",
            "from_sequence": 1,
            "through_sequence": 2,
        },
    )
    observation = MainAgentRuntime._tool_observation(
        "read_conversation_span", result
    )
    projected = MainAgentContext(
        conversation_id="c1",
        profile=CareerProfileContext(user_id="u1"),
        tool_observations=(observation,),
        user_message="继续",
    ).model_context()["tool_observations"][0]

    assert result.facts["returned"] == result.facts["total"] == 2
    assert result.facts["content_clipped"] is True
    assert result.facts["body_clipped"] is True
    assert all(len(item["content"]) == 4000 for item in result.payload["messages"])
    assert observation.body is not None
    assert len(observation.body) == DECISION_OBSERVATION_BODY_LIMIT
    assert observation.body.endswith("…")
    assert "不能视为完整回读" in observation.body
    assert projected["facts"]["body_clipped"] is True


def test_conversation_span_body_clipping_is_exact_at_the_observation_limit(
    tmp_path,
) -> None:
    """The clip flag is measured on the unmarked render.

    The warning sentence enters the presenter only after ``body_clipped`` is
    True. Unmarked, the observation body is that same render: at 6000 it is
    not warned and not clamped; at 6001 it is marked, then warned, then cut.
    """
    store = CareerContextStore(tmp_path / "context.sqlite3")
    manager = ContextManager(store)

    def commit_pair(conversation_id: str, user: str, assistant: str) -> None:
        context = manager.load_for_turn(
            user_id="u1",
            conversation_id=conversation_id,
            user_message=user,
        )
        manager.commit_turn(
            context=context,
            task=ConversationTaskState(),
            assistant_message=assistant,
        )

    commit_pair("probe", "u", "a")
    probe = store.read_conversation_span(
        user_id="u1",
        conversation_id="probe",
        from_sequence=1,
        through_sequence=2,
    )
    remaining = DECISION_OBSERVATION_BODY_LIMIT - len(
        render_conversation_span(probe)
    )
    user_extra = min(3999, remaining)
    assistant_extra = remaining - user_extra
    assert 0 <= assistant_extra <= 3999

    commit_pair("exact", "u" * (user_extra + 1), "a" * (assistant_extra + 1))
    registry = MainAgentToolRegistry(conversation_store=store)
    exact_span = store.read_conversation_span(
        user_id="u1",
        conversation_id="exact",
        from_sequence=1,
        through_sequence=2,
    )
    exact_predicted = render_conversation_span(exact_span)
    exact = registry.invoke_atomic_tool(
        "read_conversation_span",
        {
            "user_id": "u1",
            "conversation_id": "exact",
            "from_sequence": 1,
            "through_sequence": 2,
        },
    )
    exact_observation = MainAgentRuntime._tool_observation(
        "read_conversation_span", exact
    )

    assert len(exact_predicted) == DECISION_OBSERVATION_BODY_LIMIT
    assert exact.facts["body_clipped"] is False
    assert exact.facts["content_clipped"] is False
    assert exact_observation.body == exact_predicted
    assert "不能视为完整回读" not in exact_observation.body

    commit_pair(
        "over",
        "u" * (user_extra + 1),
        "a" * (assistant_extra + 2),
    )
    over_span = store.read_conversation_span(
        user_id="u1",
        conversation_id="over",
        from_sequence=1,
        through_sequence=2,
    )
    over_predicted = render_conversation_span(over_span)
    over_marked = render_conversation_span(
        over_span.model_copy(update={"body_clipped": True})
    )
    over = registry.invoke_atomic_tool(
        "read_conversation_span",
        {
            "user_id": "u1",
            "conversation_id": "over",
            "from_sequence": 1,
            "through_sequence": 2,
        },
    )
    over_observation = MainAgentRuntime._tool_observation(
        "read_conversation_span", over
    )

    assert len(over_predicted) == DECISION_OBSERVATION_BODY_LIMIT + 1
    assert over.facts["body_clipped"] is True
    assert over.facts["content_clipped"] is False
    assert "不能视为完整回读" not in over_predicted
    assert "不能视为完整回读" in over_marked
    assert over_observation.body == clamp(
        over_marked, limit=DECISION_OBSERVATION_BODY_LIMIT
    )
    assert over_observation.body.endswith("…")


def test_empty_conversation_span_does_not_substitute_a_nearby_message(
    tmp_path,
) -> None:
    store = CareerContextStore(tmp_path / "context.sqlite3")
    manager = ContextManager(store)
    context = manager.load_for_turn(
        user_id="u1", conversation_id="c1", user_message="nearest"
    )
    manager.commit_turn(
        context=context,
        task=ConversationTaskState(),
        assistant_message="also-nearby",
    )

    result = MainAgentToolRegistry(conversation_store=store).invoke_atomic_tool(
        "read_conversation_span",
        {
            "user_id": "u1",
            "conversation_id": "c1",
            "from_sequence": 100,
            "through_sequence": 110,
        },
    )

    assert result.state == "conversation_span_empty"
    assert result.facts["returned"] == result.facts["total"] == 0
    assert result.payload["messages"] == []
    assert "nearest" not in result.model_dump_json()


def test_repeated_tool_call_is_stopped_without_duplicate_execution(tmp_path) -> None:
    manager = ContextManager(CareerContextStore(tmp_path / "context.sqlite3"))
    manager.upsert_profile(CareerProfileContext(user_id="u1"))
    tools = CountingRegistry()
    decisions = SequenceDecisionMaker(
        AgentDecision(action="tool_call", tool_call=ToolCall(name="open_job_search", arguments={"keyword": "AI Engineer"})),
        AgentDecision(action="tool_call", tool_call=ToolCall(name="open_job_search", arguments={"keyword": "AI Engineer"})),
        AgentDecision(action="final", message="已有搜索页，不再重复打开。"),
    )
    agent = MainAgentRuntime(context_manager=manager, decision_maker=decisions, tools=tools)

    result = agent.run_turn(user_id="u1", conversation_id="c1", user_message="Find work.")

    assert len(tools.calls) == 1
    assert [item.state for item in result.tool_results] == ["job_search_page_ready"]
    assert result.context.tool_observations[-1].state == "authorization_refused"
    # F: the model, not the presenter, closes the turn it decided to end —
    # here by telling the reader why the repeat was not issued.
    assert result.assistant_message == "已有搜索页，不再重复打开。"
    assert len(decisions.contexts) == 3


def test_tool_loop_stops_at_configured_limit(tmp_path) -> None:
    class ReadRegistry(CountingRegistry):
        def __init__(self):
            super().__init__()
            self._atomic_handlers["find_saved_jobs"] = self._find

        def _find(self, arguments):
            return ToolObservation(
                tool_name="find_saved_jobs",
                state="saved_jobs_found",
                message=f"已读取 {arguments['query']}。",
                payload={"items": [], "query": arguments["query"]},
            )

    manager = ContextManager(CareerContextStore(tmp_path / "context.sqlite3"))
    manager.upsert_profile(CareerProfileContext(user_id="u1"))
    tools = ReadRegistry()
    decisions = SequenceDecisionMaker(
        AgentDecision(action="tool_call", tool_call=ToolCall(name="find_saved_jobs", arguments={"query": "Role A"})),
        AgentDecision(action="tool_call", tool_call=ToolCall(name="find_saved_jobs", arguments={"query": "Role B"})),
        AgentDecision(action="tool_call", tool_call=ToolCall(name="find_saved_jobs", arguments={"query": "Role C"})),
        AgentDecision(action="final", message="本轮读取预算已经用完。"),
    )
    agent = MainAgentRuntime(
        context_manager=manager,
        decision_maker=decisions,
        tools=tools,
        max_read_calls=2,
    )

    result = agent.run_turn(user_id="u1", conversation_id="c1", user_message="Research several roles.")

    assert [arguments["query"] for _, arguments in tools.calls] == ["Role A", "Role B"]
    # The refusal reached the model, and the model — not the harness —
    # tells the reader why the turn stopped.
    assert result.assistant_message == "本轮读取预算已经用完。"
    assert [item.state for item in result.tool_results] == [
        "saved_jobs_found",
        "saved_jobs_found",
    ]
    assert decisions.contexts[-1].tool_observations[-1].state == "authorization_refused"
    assert result.delegated_read_count == 2
    assert result.delegated_write_count == 0


def test_one_workflow_advance_is_one_main_loop_delegation(tmp_path) -> None:
    runtime, _, manager = build_runtime(
        tmp_path, AgentDecision(action="final", message="done")
    )
    context = manager.load_for_turn(
        user_id="u1", conversation_id="c1", user_message="推进 workflow"
    )
    # The workflow may have performed many private graph nodes before returning;
    # L1 receives one closed result and therefore records one delegated write.
    result = ToolObservation(
        tool_name="synthetic_workflow",
        state="workflow_completed",
        message="内部执行了多个节点后完成。",
    )
    observed = runtime._observe(
        {
            "context": context,
            "decision": AgentDecision(
                action="tool_call",
                tool_call=ToolCall(name="synthetic_workflow", arguments={}),
            ),
            "pending": {
                "name": "synthetic_workflow",
                "kind": "workflow",
                "effect": "WRITE",
                "arguments": {},
                "result": result,
            },
            "tool_results": (),
            "artifact_ids": (),
            "control": {
                "read_calls": 0,
                "write_calls": 0,
                "projection_refusals": 0,
                "authorization_refusals": 0,
            },
        }
    )

    assert observed["control"]["write_calls"] == 1
    assert observed["control"]["read_calls"] == 0


def test_projection_and_authorization_refusals_have_independent_budgets(
    tmp_path,
) -> None:
    class ReadRegistry(CountingRegistry):
        def __init__(self):
            super().__init__()
            self._atomic_handlers["find_saved_jobs"] = lambda arguments: None

    manager = ContextManager(CareerContextStore(tmp_path / "split-refusal.sqlite3"))
    manager.upsert_profile(CareerProfileContext(user_id="u1"))
    runtime = MainAgentRuntime(
        context_manager=manager,
        decision_maker=SequenceDecisionMaker(),
        tools=ReadRegistry(),
        max_read_calls=1,
    )
    context = manager.load_for_turn(
        user_id="u1", conversation_id="c1", user_message="继续"
    )
    state = {
        "context": context,
        "decision": AgentDecision(
            action="tool_call",
            tool_call=ToolCall(name="find_saved_jobs", arguments={"query": "AI"}),
        ),
        "tool_results": (),
        "artifact_ids": (),
        "control": {
            "read_calls": 1,
            "write_calls": 0,
            "projection_refusals": 2,
            "authorization_refusals": 0,
            "fingerprints": (),
            "retryable_fingerprints": (),
            "retry_counts": {},
        },
    }

    authorized = runtime._authorize(state)
    assert authorized["authorization_route"] == "observe"
    assert authorized["pending"]["synthetic_kind"] == "authorization"
    observed = runtime._observe({**state, **authorized})

    assert observed["control"]["projection_refusals"] == 2
    assert observed["control"]["authorization_refusals"] == 1
    assert observed["tool_results"] == ()
    assert observed["context"].tool_observations[-1].state == "authorization_refused"


def test_retryable_failure_allows_at_most_two_same_call_retries(tmp_path) -> None:
    class AlwaysRetryableRegistry(CountingRegistry):
        def __init__(self):
            super().__init__()
            self._atomic_handlers["find_saved_jobs"] = lambda arguments: None

        def invoke_atomic_tool(self, name, arguments):
            self.calls.append((name, dict(arguments)))
            return ToolObservation(
                tool_name=name,
                state="failed",
                message="上游暂时不可用。",
                payload={"error_code": "TEMPORARY", "retryable": True},
            )

    manager = ContextManager(CareerContextStore(tmp_path / "retry-context.sqlite3"))
    manager.upsert_profile(CareerProfileContext(user_id="u1"))
    same_call = AgentDecision(
        action="tool_call",
        tool_call=ToolCall(
            name="find_saved_jobs", arguments={"query": "AI Engineer"}
        ),
    )
    decisions = SequenceDecisionMaker(
        same_call,
        same_call,
        same_call,
        same_call,
        AgentDecision(action="final", message="连续重试仍未成功。"),
    )
    tools = AlwaysRetryableRegistry()
    runtime = MainAgentRuntime(
        context_manager=manager,
        decision_maker=decisions,
        tools=tools,
    )

    result = runtime.run_turn(
        user_id="u1", conversation_id="c1", user_message="打开岗位搜索"
    )

    assert len(tools.calls) == 3  # initial delegation + two retries
    assert all(item.state == "failed" for item in result.tool_results)
    assert result.context.tool_observations[-1].state == "authorization_refused"
    assert "重试上限" in result.context.tool_observations[-1].message
    assert result.context.tool_observations[-2].facts == {"retryable": True}


def test_non_retryable_failure_cannot_repeat_the_same_call(tmp_path) -> None:
    class NonRetryableRegistry(CountingRegistry):
        def __init__(self):
            super().__init__()
            self._atomic_handlers["find_saved_jobs"] = lambda arguments: None

        def invoke_atomic_tool(self, name, arguments):
            self.calls.append((name, dict(arguments)))
            return ToolObservation(
                tool_name=name,
                state="failed",
                message="请求不能重试。",
                payload={"error_code": "PERMANENT", "retryable": False},
            )

    manager = ContextManager(CareerContextStore(tmp_path / "no-retry-context.sqlite3"))
    manager.upsert_profile(CareerProfileContext(user_id="u1"))
    same_call = AgentDecision(
        action="tool_call",
        tool_call=ToolCall(name="find_saved_jobs", arguments={"query": "AI"}),
    )
    decisions = SequenceDecisionMaker(
        same_call,
        same_call,
        AgentDecision(action="final", message="不能自动重试。"),
    )
    tools = NonRetryableRegistry()
    runtime = MainAgentRuntime(
        context_manager=manager,
        decision_maker=decisions,
        tools=tools,
    )

    result = runtime.run_turn(
        user_id="u1", conversation_id="c1", user_message="打开岗位搜索"
    )

    assert len(tools.calls) == 1
    assert [item.state for item in result.tool_results] == ["failed"]
    assert result.context.tool_observations[-1].state == "authorization_refused"
    assert "没有声明为可重试" in result.context.tool_observations[-1].message


def test_target_and_search_overrides_do_not_mutate_profile(tmp_path) -> None:
    agent, tools, manager = build_runtime(tmp_path, AgentDecision(action="tool_call", tool_call=ToolCall(name="open_job_search", arguments={"keyword": "Backend Engineer", "city": "杭州"})))

    result = agent.run_turn(user_id="u1", conversation_id="c1", user_message="Search backend roles in Hangzhou this time.")

    action = result.tool_results[0].payload["client_action"]
    parsed = urlparse(action["url"])
    assert parse_qs(parsed.query) == {"query": ["Backend Engineer"], "city": ["101210100"]}
    profile = manager.load_for_turn(user_id="u1", conversation_id="c1", user_message="next").profile
    assert profile.default_city == "Shanghai"


def test_saved_job_exposes_only_the_bounded_presenter_body_not_internal_payload() -> None:
    sentinel = "PRIVATE-PAYLOAD-DO-NOT-PROMPT"
    result = ToolResult(
        tool_name="get_saved_job",
        state="saved_job_ready",
        message="已读取已保存岗位。",
        next_action="match_resume_to_job",
        payload={"jd_snapshot": {"content": sentinel}, "job_posting_id": "secret-id"},
    )

    observation = MainAgentRuntime._tool_observation("get_saved_job", result)

    assert observation.model_dump() == {
        "tool_name": "get_saved_job",
        "state": "saved_job_ready",
        "message": "已读取已保存岗位。",
        "body": sentinel,
        "facts": {},
        "arguments": {},
        "next_action": "match_resume_to_job",
    }
    assert sentinel in observation.model_dump_json()
    assert "secret-id" not in observation.model_dump_json()
    assert len(observation.model_dump_json()) < DECISION_OBSERVATION_BODY_LIMIT + 900
    with pytest.raises(ValidationError):
        DecisionObservation.model_validate(
            {**observation.model_dump(), "payload": {"content": sentinel}}
        )


def test_decision_observation_clamps_the_receipt_at_its_boundary() -> None:
    result = ToolResult(
        tool_name="start_mock_interview",
        state="mock_interview_answer_required",
        message="模拟面试题：" + "请说明你的设计。" * 200,
    )

    observation = MainAgentRuntime._tool_observation(
        "start_mock_interview", result
    )

    assert len(observation.message) == 600
    assert observation.message.endswith("…")
    assert DecisionObservation.model_validate(observation.model_dump()) == observation
    with pytest.raises(ValidationError):
        DecisionObservation(
            tool_name="get_saved_job",
            state="saved_job_ready",
            message="已读取完整 JD。",
            body="x" * (DECISION_OBSERVATION_BODY_LIMIT + 1),
        )


def test_condensed_result_body_is_bounded_and_matches_the_presenter() -> None:
    result = ToolResult(
        tool_name="get_daily_brief",
        state="daily_brief_ready",
        message="今日职业简报包含 1 个待办事项。",
        payload={
            "overdue": [
                {
                    "title": "跟进岗位" + "很重要" * 2500,
                    "summary": "发送跟进消息",
                    "due_at": "2026-09-02T09:00:00+08:00",
                }
            ],
            "due_today": [],
            "upcoming": [],
            "no_due_date": [],
        },
    )

    observation = MainAgentRuntime._tool_observation("get_daily_brief", result)
    rendered = MainAgentRuntime._assistant_message(result)

    assert observation.body is not None
    assert len(observation.body) == DECISION_OBSERVATION_BODY_LIMIT
    assert observation.body.endswith("…")
    assert observation.body == rendered[: DECISION_OBSERVATION_BODY_LIMIT - 1].rstrip() + "…"
    # Facts travel from the declaring capability rather than being re-derived
    # here, so a hand-built result carries whatever it declared — nothing.
    assert observation.facts == {}


def test_plain_result_does_not_carry_payload_as_body() -> None:
    observation = MainAgentRuntime._tool_observation(
        "find_saved_jobs",
        ToolResult(
            tool_name="find_saved_jobs",
            state="saved_jobs_found",
            message="找到 1 个岗位。",
            payload={"private": "NEVER-PROMPT-THIS"},
        ),
    )

    assert observation.body is None
    assert "NEVER-PROMPT-THIS" not in observation.model_dump_json()


def test_only_newest_observation_retains_body_without_losing_receipt_or_facts() -> None:
    first = DecisionObservation(
        tool_name="get_daily_brief",
        state="daily_brief_ready",
        message="今日职业简报包含 1 个待办事项。",
        body="# 今日职业简报\n\n- 跟进岗位",
        facts={"overdue": 1, "due_today": 0, "waiting": 0},
    )
    second = DecisionObservation(
        tool_name="list_action_items",
        state="action_items_found",
        message="找到 1 个行动项。",
    )

    observations = append_decision_observation((first,), second)

    visible = MainAgentContext(
        conversation_id="c1",
        profile=CareerProfileContext(user_id="u1"),
        tool_observations=(first,),
        user_message="继续。",
    ).model_context()["tool_observations"][0]
    assert visible["body"] == first.body
    assert observations[0].body is None
    assert observations[0].message == first.message
    assert observations[0].facts == first.facts
    assert observations[1] == second
    cleared = MainAgentContext(
        conversation_id="c1",
        profile=CareerProfileContext(user_id="u1"),
        tool_observations=observations,
        user_message="继续。",
    ).model_context()["tool_observations"][0]
    assert "body" not in cleared
    with pytest.raises(ValidationError, match="newest decision observation"):
        MainAgentContext(
            conversation_id="c1",
            profile=CareerProfileContext(user_id="u1"),
            tool_observations=(first, second),
            user_message="继续。",
        )


def test_failed_observation_exposes_only_explicit_retryability() -> None:
    retryable = MainAgentRuntime._tool_observation(
        "research_job",
        ToolResult(
            tool_name="research_job",
            state="job_research_failed",
            message="岗位调研暂时失败。",
            payload={"retryable": True, "error_code": "UPSTREAM_TIMEOUT"},
        ),
    )
    unknown = MainAgentRuntime._tool_observation(
        "research_job",
        ToolResult(
            tool_name="research_job",
            state="job_research_failed",
            message="岗位调研失败。",
            payload={"error_code": "UNKNOWN"},
        ),
    )

    assert retryable.facts == {"retryable": True}
    assert unknown.facts == {}
    assert "error_code" not in retryable.model_dump_json()


def test_non_streaming_interrupt_enforces_renderer_completeness(
    tmp_path, monkeypatch
) -> None:
    """CLI/run_turn cannot bypass the interaction contract checked by SSE."""

    class BrokenInteractionRegistry(CountingRegistry):
        def invoke_atomic_tool(self, name, arguments):
            self.calls.append((name, dict(arguments)))
            return ToolObservation(
                tool_name=name,
                state="calendar_approval_required",
                message="需要确认。",
                execution_outcome="committed",
            )

    manager = ContextManager(CareerContextStore(tmp_path / "context.sqlite3"))
    manager.upsert_profile(CareerProfileContext(user_id="u1"))
    runtime = MainAgentRuntime(
        context_manager=manager,
        decision_maker=DecisionMaker(
            AgentDecision(
                action="tool_call",
                tool_call=ToolCall(
                    name="open_job_search", arguments={"keyword": "AI Engineer"}
                ),
            )
        ),
        tools=BrokenInteractionRegistry(),
    )
    monkeypatch.setattr(
        MainAgentRuntime,
        "_INTERACTION_RENDERER_STATES",
        MainAgentRuntime._INTERACTION_RENDERER_STATES
        - {"calendar_approval_required"},
    )

    with pytest.raises(ValueError, match="has no interaction renderer"):
        runtime.run_turn(
            user_id="u1",
            conversation_id="c1",
            user_message="打开岗位搜索",
        )


def test_main_agent_context_keeps_the_full_observation_turn_window() -> None:
    observations = tuple(
        DecisionObservation(
            tool_name=f"read_step_{index}",
            state="read_complete",
            message=f"第 {index} 步读取完成。",
        )
        for index in range(MAX_DECISION_OBSERVATIONS)
    )

    context = MainAgentContext(
        conversation_id="c1",
        profile=CareerProfileContext(user_id="u1"),
        tool_observations=observations,
        user_message="继续处理。",
    )

    assert context.tool_observations == observations
    assert len(context.model_context()["tool_observations"]) == MAX_DECISION_OBSERVATIONS

    with pytest.raises(ValidationError):
        MainAgentContext(
            conversation_id="c1",
            profile=CareerProfileContext(user_id="u1"),
            tool_observations=(
                *observations,
                DecisionObservation(
                    tool_name=f"read_step_{MAX_DECISION_OBSERVATIONS}",
                    state="read_complete",
                    message=f"第 {MAX_DECISION_OBSERVATIONS} 步读取完成。",
                ),
            ),
            user_message="继续处理。",
        )


def test_observation_count_and_character_budgets_fit_the_declared_worst_shape() -> None:
    observations = tuple(
        DecisionObservation(
            tool_name="t" * 80,
            state="s" * 80,
            message="m" * DECISION_OBSERVATION_RECEIPT_LIMIT,
            next_action="n" * 80,
            arguments={"a": "a" * 180},
        )
        for _ in range(MAX_DECISION_OBSERVATIONS - 1)
    ) + (
        DecisionObservation(
            tool_name="t" * 80,
            state="job_research_ready",
            message="m" * DECISION_OBSERVATION_RECEIPT_LIMIT,
            body="b" * DECISION_OBSERVATION_BODY_LIMIT,
            facts={
                "cached": True,
                "finding_count": 1_000_000,
                "status": "superseded",
            },
            next_action="n" * 80,
            arguments={"a": "a" * 180},
        ),
    )

    assert (
        MAX_DECISION_OBSERVATION_BODIES * DECISION_OBSERVATION_BODY_LIMIT
        + MAX_DECISION_OBSERVATIONS * DECISION_OBSERVATION_RECEIPT_LIMIT
    ) == 12_600
    assert decision_observation_chars(observations) == 18_367
    assert decision_observation_chars(observations) <= (
        MAX_DECISION_OBSERVATION_CHARS
    )
    MainAgentContext(
        conversation_id="c1",
        profile=CareerProfileContext(user_id="u1"),
        tool_observations=observations,
        user_message="继续。",
    )


def test_blank_receipt_degrades_after_a_tool_result_instead_of_raising() -> None:
    observation = MainAgentRuntime._tool_observation(
        "write_side_effect",
        ToolResult(
            tool_name="write_side_effect",
            state="write_complete",
            message="   ",
        ),
    )

    assert observation.message == "工具已返回，但没有提供结果摘要。"


def test_decision_facts_reject_nested_values_ids_and_unbounded_shapes() -> None:
    base = {
        "tool_name": "get_daily_brief",
        "state": "daily_brief_ready",
        "message": "已读取简报。",
    }

    with pytest.raises(ValidationError):
        DecisionObservation.model_validate({**base, "facts": {"report_id": "secret"}})
    with pytest.raises(ValidationError):
        DecisionObservation.model_validate({**base, "facts": {"counts": {"due": 1}}})
    with pytest.raises(ValidationError):
        DecisionObservation.model_validate(
            {**base, "facts": {f"fact_{index}": index for index in range(9)}}
        )
    # A partial set is no longer rejected: membership moved to the capability
    # that declares it, and the contract now bounds shape rather than schema.
    assert DecisionObservation.model_validate(
        {**base, "facts": {"overdue": 1}}
    ).facts == {"overdue": 1}
    with pytest.raises(ValidationError):
        DecisionObservation.model_validate(
            {**base, "facts": {"anchor": "a" * 32}}
        )


def test_internal_tool_result_requires_a_durable_receipt() -> None:
    """Summary/message delivery must never commit an invisible empty row."""
    with pytest.raises(ValidationError):
        ToolResult(
            tool_name="get_daily_brief",
            state="daily_brief_ready",
            message="",
        )


def test_the_cards_shown_live_are_the_references_the_transcript_keeps(
    tmp_path,
) -> None:
    """Live delivery and the reloaded transcript must name the same reports.

    This is the invariant the plural ``resource_refs`` exists for, and it is not
    implied by the card test above: emitting every card while committing only
    ``tool_results[-1]`` passes that one and still shows two cards live and one
    after refresh. Pinning both sides against the *same* turn is what makes the
    regression impossible to reintroduce quietly.
    """

    class TwoReportRegistry(MainAgentToolRegistry):
        def capability_kind(self, name):
            return "atomic_tool"

        def invoke_atomic_tool(self, name, arguments):
            kinds = {
                "get_job_research": ("job_research_report", "report-1"),
                "get_resume_job_match": ("resume_job_match", "match-1"),
            }
            kind, resource_id = kinds[name]
            extra = (
                {"status_at_delivery": "current", "anchored_by_other_job": False}
                if kind == "job_research_report"
                else {}
            )
            return ToolResult(
                tool_name=name,
                state=(
                    "job_research_ready"
                    if kind == "job_research_report"
                    else "resume_job_match_ready"
                ),
                message=f"已读取 {kind}。",
                resource_ref=ConversationResourceReference(
                    kind=kind, resource_id=resource_id, **extra
                ),
            )

    class DirectRuntime(MainAgentRuntime):
        """Argument projection is not what this test is about."""

        @staticmethod
        def _project_atomic_tool_arguments(context, name, arguments):
            return dict(arguments)

    manager = ContextManager(CareerContextStore(tmp_path / "context.sqlite3"))
    manager.upsert_profile(CareerProfileContext(user_id="u1"))
    decisions = SequenceDecisionMaker(
        AgentDecision(
            action="tool_call",
            tool_call=ToolCall(name="get_job_research", arguments={}),
        ),
        AgentDecision(
            action="tool_call",
            tool_call=ToolCall(name="get_resume_job_match", arguments={}),
        ),
        AgentDecision(action="final", message="两份都读好了。"),
    )
    runtime = DirectRuntime(
        context_manager=manager,
        decision_maker=decisions,
        tools=TwoReportRegistry(),
    )

    events: list[object] = []
    runtime.run_turn(
        user_id="u1",
        conversation_id="c1",
        user_message="调研和匹配都看看",
        event_sink=events.append,
    )

    shown = [
        event.resource_id for event in events if event.type == "report_ready"
    ]
    stored = manager.load_for_turn(
        user_id="u1", conversation_id="c1", user_message="继续"
    ).recent_messages[-1]
    kept = [reference.resource_id for reference in stored.resource_refs]

    assert shown == ["report-1", "match-1"]
    assert kept == shown


def _interrupted_turn_runtime(tmp_path, *, first_tool: str):
    """A turn that succeeds at ``first_tool`` and is then killed by a hard refusal.

    The second decision names a capability projection does not know, which
    ``_reraise_security_refusal`` throws straight through the turn — the real
    path, not a synthetic error.
    """

    class Registry(MainAgentToolRegistry):
        def capability_kind(self, name):
            return "atomic_tool"

        def invoke_atomic_tool(self, name, arguments):
            return ToolResult(
                tool_name=name,
                state="application_created",
                message="已创建投递记录。",
                execution_outcome="committed",
            )

    class DirectRuntime(MainAgentRuntime):
        @staticmethod
        def _project_atomic_tool_arguments(context, name, arguments):
            if name == "get_daily_brief":
                raise ValueError("Unknown capability: get_daily_brief")
            return dict(arguments)

    manager = ContextManager(CareerContextStore(tmp_path / "context.sqlite3"))
    manager.upsert_profile(CareerProfileContext(user_id="u1"))
    runtime = DirectRuntime(
        context_manager=manager,
        decision_maker=SequenceDecisionMaker(
            AgentDecision(
                action="tool_call",
                tool_call=ToolCall(name=first_tool, arguments={}),
            ),
            AgentDecision(
                action="tool_call",
                tool_call=ToolCall(name="get_daily_brief", arguments={}),
            ),
        ),
        tools=Registry(),
        # Preparation is what makes an interrupted write reportable: the record
        # is read from the ledger, not from an in-process list that a killed
        # process would take with it.
        action_execution_store=SQLiteActionExecutionStore(
            tmp_path / "context.sqlite3"
        ),
    )
    return runtime, manager


def test_a_turn_killed_after_a_write_still_leaves_the_write_in_the_conversation(
    tmp_path,
) -> None:
    """A durable write must never be invisible at the conversation layer.

    The turn runs to completion and commits afterwards, so an exception in the
    middle used to drop the entire conversation record — the user's own message
    included — while the application it had already created stayed in its store.
    Three states disagreed: the store said it happened, the conversation had zero
    messages, and the task still said no application was active.

    The record was never lost; ``list_applications`` would still find it. What
    was missing is any reason for the model to look. ``max_write_calls = 1``
    makes a write deliberate, and this path made a deliberate write vanish.

    The boundary itself stays hard: the refusal is recorded, then re-raised.
    """
    runtime, manager = _interrupted_turn_runtime(
        tmp_path, first_tool="create_application"
    )

    with pytest.raises(ValueError, match="Unknown capability"):
        runtime.run_turn(
            user_id="u1",
            conversation_id="c1",
            user_message="帮我记一下这次投递，再看看今天的待办",
        )

    context = manager.load_for_turn(
        user_id="u1", conversation_id="c1", user_message="继续"
    )
    stored = context.recent_messages
    # The user's turn survives, and the note names what already landed.
    assert [message.role for message in stored[-2:]] == ["user", "assistant"]
    assert "create_application" in stored[-1].content
    assert "帮我记一下这次投递" in stored[-2].content


def test_the_mock_interview_graph_path_reports_its_write(tmp_path) -> None:
    """The standard act ledger survives a later workflow observation failure."""

    class Tools:
        runtime_workflow_names = ("handle_mock_interview_input",)

        def handle_mock_interview_input(self, *, user_id, session_id, message):
            return ToolObservation(
                tool_name="handle_mock_interview_input",
                state="ok",
                message="下一题。",
                execution_outcome="committed",
            )

        def invoke_runtime_workflow(self, name, arguments):
            return self.handle_mock_interview_input(**arguments)

    class ExplodingRuntime(MainAgentRuntime):
        def _update_mock_interview_task(self, context, result):
            raise RuntimeError("checkpoint store unavailable")

    manager = ContextManager(CareerContextStore(tmp_path / "context.sqlite3"))
    manager.upsert_profile(CareerProfileContext(user_id="u1"))
    seeded = manager.load_for_turn(
        user_id="u1", conversation_id="c1", user_message="开始模拟面试"
    )
    manager.commit_turn(
        context=seeded,
        task=ConversationTaskState(
            active_workflow="mock_interview", run_id="s1", phase="mock_interview_running"
        ),
        assistant_message="第一题：介绍一下你自己。",
    )

    runtime = ExplodingRuntime(
        context_manager=manager,
        decision_maker=SequenceDecisionMaker(),
        tools=Tools(),
        # The record is read from the ledger now, so a runtime without one has
        # nothing to write down. That is a true statement about the system
        # rather than a test detail: preparation is what makes an interrupted
        # write reportable at all.
        action_execution_store=SQLiteActionExecutionStore(
            tmp_path / "context.sqlite3"
        ),
    )

    with pytest.raises(RuntimeError, match="checkpoint store unavailable"):
        runtime.run_turn(
            user_id="u1",
            conversation_id="c1",
            user_message="我做过检索系统的端到端优化。",
        )

    stored = manager.load_for_turn(
        user_id="u1", conversation_id="c1", user_message="继续"
    ).recent_messages
    assert "handle_mock_interview_input" in stored[-1].content
    # The note rides on the context as loaded, so workflow-owned input stays
    # withheld: reconciling a durable write must not become a way to leak the
    # candidate's answer into the main transcript.
    assert stored[-2].role == "user"
    assert "我做过检索系统的端到端优化。" not in stored[-2].content


def test_a_write_that_failed_is_not_reported_as_written(tmp_path) -> None:
    """``WRITE`` is what the tool may do, not what this call did.

    ``execute_calendar_proposal`` is a WRITE that can return
    ``calendar_write_failed`` with nothing changed on the calendar. Recording it
    on effect alone made the note claim an effect that never happened — the
    original bug with its sign flipped. The trajectory is ordinary:
    ``_after_observe`` sends a failed result back to ``decide``, so "the write
    fails, the model tries something else, that hard-throws" is a normal turn.

    "请先核对这些记录的实际状态" would cover for it, but only by asking the user
    to look up something ``result.disposition`` already answered.
    """

    class Registry(MainAgentToolRegistry):
        def capability_kind(self, name):
            return "atomic_tool"

        def invoke_atomic_tool(self, name, arguments):
            return ToolResult(
                tool_name=name,
                state="calendar_write_failed",
                message="日历写入失败，日程未创建。",
                execution_outcome="not_committed",
            )

    class DirectRuntime(MainAgentRuntime):
        @staticmethod
        def _project_atomic_tool_arguments(context, name, arguments):
            if name == "get_daily_brief":
                raise ValueError("Unknown capability: get_daily_brief")
            return dict(arguments)

    manager = ContextManager(CareerContextStore(tmp_path / "context.sqlite3"))
    manager.upsert_profile(CareerProfileContext(user_id="u1"))
    runtime = DirectRuntime(
        context_manager=manager,
        decision_maker=SequenceDecisionMaker(
            AgentDecision(
                action="tool_call",
                tool_call=ToolCall(name="execute_calendar_proposal", arguments={}),
            ),
            AgentDecision(
                action="tool_call",
                tool_call=ToolCall(name="get_daily_brief", arguments={}),
            ),
        ),
        tools=Registry(),
    )

    with pytest.raises(ValueError, match="Unknown capability"):
        runtime.run_turn(
            user_id="u1",
            conversation_id="c1",
            user_message="把面试加进日历，再看看今天的待办",
        )

    context = manager.load_for_turn(
        user_id="u1", conversation_id="c1", user_message="继续"
    )
    # Nothing happened, so there is no disagreement to reconcile and no note.
    assert context.recent_messages == ()


def test_a_turn_killed_before_any_write_leaves_the_conversation_clean(
    tmp_path,
) -> None:
    """No write, no disagreement — and a clean retry beats a failure note.

    The row exists to reconcile a domain effect with a conversation that does not
    mention it. A read-only turn has no such effect, so writing a note would add
    a permanent apology to the recent window for something the user can simply
    send again.
    """
    runtime, manager = _interrupted_turn_runtime(
        tmp_path, first_tool="list_applications"
    )

    with pytest.raises(ValueError, match="Unknown capability"):
        runtime.run_turn(
            user_id="u1",
            conversation_id="c1",
            user_message="看看我的投递和今天的待办",
        )

    context = manager.load_for_turn(
        user_id="u1", conversation_id="c1", user_message="继续"
    )
    assert context.recent_messages == ()


def test_a_mixed_turn_streams_the_card_less_body_and_keeps_the_whole_reply(
    tmp_path,
) -> None:
    """The card ceiling is a property of the turn, and three forks must agree.

    ``_present`` learned this when the reply limit was fixed; the other two
    forks kept asking ``tool_results[-1]``. A turn that reads a JD (no card) and
    then a research report (card) ends with a card-backed last result, so:

    * the stream handed delivery to the card and never wrote the JD out — the
      body had nowhere else to go, so it was simply lost;
    * the row re-clamped an already-bounded reply to card length, storing 600
      characters of an answer the reader was shown in full.

    Both are invisible in a single-result turn, which is why every existing test
    passed. This one is mixed on purpose.
    """
    jd_body = "岗位职责\n\n" + "负责端到端的检索系统。" * 60
    reply = "先说 JD：" + "这个岗位要求的是检索与排序的工程能力。" * 40

    class MixedRegistry(MainAgentToolRegistry):
        def capability_kind(self, name):
            return "atomic_tool"

        def invoke_atomic_tool(self, name, arguments):
            if name == "get_saved_job":
                return ToolResult(
                    tool_name=name,
                    state="saved_job_ready",
                    message="已读取该岗位的 JD。",
                    payload={"jd_snapshot": {"content": jd_body}},
                )
            return ToolResult(
                tool_name=name,
                state="job_research_ready",
                message="已读取公司调研。",
                resource_ref=ConversationResourceReference(
                    kind="job_research_report",
                    resource_id="report-1",
                    status_at_delivery="current",
                    anchored_by_other_job=False,
                ),
            )

    class DirectRuntime(MainAgentRuntime):
        @staticmethod
        def _project_atomic_tool_arguments(context, name, arguments):
            return dict(arguments)

    manager = ContextManager(CareerContextStore(tmp_path / "context.sqlite3"))
    manager.upsert_profile(CareerProfileContext(user_id="u1"))
    runtime = DirectRuntime(
        context_manager=manager,
        decision_maker=SequenceDecisionMaker(
            AgentDecision(
                action="tool_call",
                tool_call=ToolCall(name="get_saved_job", arguments={}),
            ),
            AgentDecision(
                action="tool_call",
                tool_call=ToolCall(name="get_job_research", arguments={}),
            ),
            AgentDecision(action="final", message=reply),
        ),
        tools=MixedRegistry(),
    )

    events: list[object] = []
    result = runtime.run_turn(
        user_id="u1",
        conversation_id="c1",
        user_message="看下 JD 和调研",
        event_sink=events.append,
    )

    streamed = "".join(
        event.delta for event in events if event.type == "content_delta"
    )
    stored = manager.load_for_turn(
        user_id="u1", conversation_id="c1", user_message="继续"
    ).recent_messages[-1]

    # Fork one: the JD has no card behind it, so the stream is the only place it
    # can appear. A card elsewhere in the turn does not excuse dropping it.
    assert jd_body in streamed
    # Fork two: the reply was already cut at the ceiling this turn earns.
    assert len(reply) > DELIVERY_SUMMARY_LIMIT
    assert result.model_message == reply
    assert stored.content == reply


def test_every_stored_report_in_the_turn_gets_its_own_card() -> None:
    """One card per report, not one per turn.

    Four card-backed reads fit inside the read budget, so a turn can end holding
    two stored reports. Emitting only ``tool_results[-1]`` would leave a durable
    report the reader is never handed — the same last-result assumption that
    dropped composed prose, in the delivery layer.
    """

    def _card(kind: str, resource_id: str, state: str, tool: str) -> ToolResult:
        # Only job research carries delivery-time render metadata; the other
        # kinds derive current state when read.
        extra = (
            {"status_at_delivery": "current", "anchored_by_other_job": False}
            if kind == "job_research_report"
            else {}
        )
        return ToolResult(
            tool_name=tool,
            state=state,
            message=f"已读取 {kind}。",
            resource_ref=ConversationResourceReference(
                kind=kind, resource_id=resource_id, **extra
            ),
        )

    events: list[object] = []
    runtime = MainAgentRuntime.__new__(MainAgentRuntime)
    result = MainAgentTurnResult(
        origin=ModelDecision(AgentDecision(action="final", message="两份都给你了。")),
        context=MainAgentContext(
            conversation_id="c1",
            profile=CareerProfileContext(user_id="u1"),
            user_message="调研和匹配都看看",
        ),
        assistant_message="两份都给你了。",
        model_message="两份都给你了。",
        tool_results=(
            _card(
                "job_research_report",
                "report-1",
                "job_research_ready",
                "get_job_research",
            ),
            _card(
                "resume_job_match",
                "match-1",
                "resume_job_match_ready",
                "get_resume_job_match",
            ),
        ),
    )
    result.tool_result = result.tool_results[-1]

    token = _STREAM_SINK.set(events.append)
    try:
        runtime._deliver_stream_events(
            result=result, turn_id="t1", conversation_id="c1"
        )
    finally:
        _STREAM_SINK.reset(token)

    cards = [event for event in events if event.type == "report_ready"]
    assert [(item.kind, item.resource_id) for item in cards] == [
        ("job_research_report", "report-1"),
        ("resume_job_match", "match-1"),
    ]


def test_every_card_less_body_in_the_turn_is_delivered_not_just_the_last() -> None:
    """Pins ``_present``/``_undelivered_bodies`` against a last-result relapse.

    Two condensed states with no card in one turn — reading a JD, then the daily
    brief — each hold a body nothing else will ever show. Delivering only
    ``tool_results[-1]`` silently drops the first, which is precisely what the
    presenter used to do to composed turns.
    """
    jd = ToolResult(
        tool_name="get_saved_job",
        state="saved_job_ready",
        message="已读取完整 JD。",
        payload={"jd_snapshot": {"content": "JD BODY: 精通 Rust。"}},
    )
    brief = ToolResult(
        tool_name="get_daily_brief",
        state="daily_brief_ready",
        message="今日没有待办事项。",
        payload={
            "timezone": "Asia/Shanghai",
            "generated_at": "2026-09-02T00:00:00+00:00",
            "overdue": [],
            "due_today": [],
            "upcoming": [],
            "no_due_date": [],
        },
    )

    update = MainAgentRuntime._present(
        {
            "decision": AgentDecision(action="final", message="两件事都看过了。"),
            "tool_results": (jd, brief),
        }
    )

    assert update["model_message"] == "两件事都看过了。"
    assert "JD BODY: 精通 Rust。" in update["assistant_message"]
    assert "今日职业简报" in update["assistant_message"]
    assert update["assistant_message"].index("JD BODY") < update[
        "assistant_message"
    ].index("今日职业简报")


def test_a_reply_is_bounded_by_what_else_carries_the_delivery() -> None:
    """The card ceiling applies to prose about a card, not to every answer.

    The old writer only ever restated card-backed reports, so its 600-character
    limit never touched an ordinary reply. Applying it to everything would
    truncate explanations and multi-step answers the writer never saw, so the
    bound follows the whole turn: prose beside cards keeps the card ceiling,
    and an answer that *is* part of the delivery gets the message bound.
    """
    plain = MainAgentRuntime._present(
        {
            "decision": AgentDecision(action="final", message="长" * 5_000),
            "tool_results": (),
        }
    )
    assert len(plain["assistant_message"]) == 5_000

    huge = MainAgentRuntime._present(
        {
            "decision": AgentDecision(action="final", message="长" * 20_000),
            "tool_results": (),
        }
    )
    assert len(huge["assistant_message"]) == MODEL_REPLY_LIMIT
    assert huge["assistant_message"].endswith("…")

    card = ToolResult(
        tool_name="research_job",
        state="job_research_ready",
        message="已完成岗位研究。",
        resource_ref=ConversationResourceReference(
            kind="job_research_report",
            resource_id="report-1",
            status_at_delivery="current",
            anchored_by_other_job=False,
        ),
    )
    carded = MainAgentRuntime._present(
        {
            "decision": AgentDecision(action="final", message="长" * 5_000),
            "tool_results": (card,),
        }
    )
    assert len(carded["assistant_message"]) == DELIVERY_SUMMARY_LIMIT

    # A turn that mixes a card with a body nothing else delivers is no longer
    # "prose about a card": the reply has to introduce that body too, so the
    # ceiling follows the whole turn rather than its last result.
    mixed = MainAgentRuntime._present(
        {
            "decision": AgentDecision(action="final", message="长" * 5_000),
            "tool_results": (
                ToolResult(
                    tool_name="get_saved_job",
                    state="saved_job_ready",
                    message="已读取完整 JD。",
                    payload={"jd_snapshot": {"content": "JD BODY"}},
                ),
                card,
            ),
        }
    )
    assert len(mixed["model_message"]) == 5_000
    assert "JD BODY" in mixed["assistant_message"]


def test_the_model_narrates_and_the_presenter_is_the_fallback() -> None:
    """F: the answer is the model's; the presenter covers turns without one.

    Before H the model had never seen the result it was answering about, so
    ``_present`` overrode its prose unconditionally — an override that also
    dropped every result but the last, which is what made a composed turn
    deliver half an answer. The model now sees the same bounded presenter text
    the reader will, so it narrates; grounding is enforced by what reaches its
    context, not by taking away the pen.
    """
    result = ToolResult(
        tool_name="analyze_resume",
        state="resume_analysis_ready",
        message="已分析简历并生成待确认候选事实。",
        payload={"records": [{"title": "PRIVATE RESULT"}]},
    )

    answered = MainAgentRuntime._present(
        {
            "decision": AgentDecision(
                action="final",
                message="已提取 1 段候选经历，等你确认。",
            ),
            "tool_results": (result,),
        }
    )
    # A condensed state with no card has nowhere else to put its body, so the
    # reply introduces it rather than standing in for it. The row keeps only
    # the reply — that is what ``model_message`` carries separately.
    assert answered["model_message"] == "已提取 1 段候选经历，等你确认。"
    assert answered["assistant_message"].startswith("已提取 1 段候选经历，等你确认。")
    assert result.message in answered["assistant_message"]

    # No prose from the model — the authoritative presenter still delivers.
    silent = MainAgentRuntime._present(
        {
            "decision": AgentDecision(action="final", message=""),
            "tool_results": (result,),
        }
    )
    assert silent["assistant_message"] == result.message


def test_a_blank_reply_is_no_reply_at_all() -> None:
    """``model_message`` is non-empty exactly when the model wrote the answer.

    Downstream code reads that flag as the whole answer to "did the model
    narrate this turn?" — ``_durable_screen`` picks the row's text from it, and
    the trace records ``composed`` from it. A truthiness guard let a
    whitespace-only message through: the branch fired, clamp stripped it to "",
    and the flag said no while the branch said yes. The answer landed in the
    right place by accident, with a blank line in front of the body. An
    invariant the code states has to hold, or the next reader builds on it.
    """
    result = ToolResult(
        tool_name="analyze_resume",
        state="resume_analysis_ready",
        message="已分析简历并生成待确认候选事实。",
        payload={"records": []},
    )

    blank = MainAgentRuntime._present(
        {
            "decision": AgentDecision(action="final", message="   \n  "),
            "tool_results": (result,),
        }
    )

    # The presenter path leaves the flag unset rather than writing an empty one,
    # so both readings of "the model did not narrate" agree.
    assert blank.get("model_message", "") == ""
    # Same delivery as an absent message: the presenter body alone, with no
    # blank line where a stripped-away reply used to sit.
    assert blank["assistant_message"] == MainAgentRuntime._assistant_message(result)
    assert not blank["assistant_message"].startswith("\n")


def test_decision_tool_schema_recursively_removes_internal_ids() -> None:
    schema = {
        "type": "function",
        "function": {
            "name": "example",
            "parameters": {
                "type": "object",
                "properties": {
                    "selection_index": {"type": "integer"},
                    "application_id": {"type": "string"},
                    "details": {
                        "type": "object",
                        "properties": {"source_id": {"type": "string"}},
                        "required": ["source_id"],
                    },
                },
                "required": ["application_id", "selection_index"],
            },
        },
    }

    projected = MainAgentToolRegistry._decision_tool_schema(schema)
    serialized = str(projected)

    assert "application_id" not in serialized
    assert "source_id" not in serialized
    assert projected["function"]["parameters"]["required"] == [
        "selection_index"
    ]


def test_normal_answer_commits_history_without_tool(tmp_path) -> None:
    agent, tools, manager = build_runtime(tmp_path, AgentDecision(action="final", message="AI Engineers build AI products."))

    result = agent.run_turn(user_id="u1", conversation_id="c1", user_message="What is an AI Engineer?")
    loaded = manager.load_for_turn(user_id="u1", conversation_id="c1", user_message="next")

    assert result.tool_result is None
    assert tools.calls == []
    assert [message.content for message in loaded.recent_messages] == ["What is an AI Engineer?", "AI Engineers build AI products."]


def _seed_saved_job(repository: SQLiteJobPostingRepository, *, user_id: str = "u1", source_job_id: str = "saved-1") -> str:
    from datetime import datetime, timezone

    captured_at = datetime(2026, 8, 23, tzinfo=timezone.utc)
    record = repository.save_detail(
        user_id=user_id,
        run_id=f"run-{source_job_id}",
        result_ref=f"ref-{source_job_id}",
        selection_index=1,
        detail=JobDetail(
            source_name="boss",
            source_job_id=source_job_id,
            title="RAG Engineer",
            company_name="Acme",
            description="PRIVATE SAVED JD: Build production RAG systems.",
            city="Shanghai",
            captured_at=captured_at,
            provenance=Provenance(source_name="boss", source_job_id=source_job_id, captured_at=captured_at, operation="detail", adapter_version="test-v1"),
        ),
    )
    repository.save_analysis(
        user_id=user_id,
        jd_snapshot_id=record.snapshot.id,
        analyzer_version="jd-analysis-v1",
        analysis=JDAnalysisPayload(
            job_summary="构建生产级 RAG 系统。",
            responsibilities=("建设 RAG 系统",),
            required_skills=("Python",),
        ),
    )
    return record.posting.id


def test_saved_job_tools_are_registered_and_find_returns_only_summaries(tmp_path) -> None:
    manager = ContextManager(CareerContextStore(tmp_path / "context.sqlite3"))
    manager.upsert_profile(CareerProfileContext(user_id="u1"))
    repository = SQLiteJobPostingRepository(tmp_path / "jobs.sqlite3")
    job_posting_id = _seed_saved_job(repository)
    _seed_saved_job(repository, user_id="other", source_job_id="saved-other")
    decisions = SequenceDecisionMaker(
        AgentDecision(action="tool_call", tool_call=ToolCall(name="find_saved_jobs", arguments={"query": "RAG"})),
        AgentDecision(action="final", message="找到了以前看过的岗位。"),
    )
    tools = MainAgentToolRegistry(job_repository=repository)
    agent = MainAgentRuntime(context_manager=manager, decision_maker=decisions, tools=tools)

    result = agent.run_turn(user_id="u1", conversation_id="c1", user_message="找一下我以前看过的 RAG 岗位")

    assert tuple(spec["function"]["name"] for spec in tools.schemas()) == ("open_job_search", "find_saved_jobs", "get_saved_job")
    assert all("user_id" not in spec["function"]["parameters"].get("properties", {}) for spec in tools.schemas())
    observation = decisions.contexts[1].tool_observations[0]
    tool_result = result.tool_results[0]
    assert observation.tool_name == "find_saved_jobs"
    assert not hasattr(observation, "payload")
    assert tool_result.payload["items"][0]["job_posting_id"] == job_posting_id
    assert len(tool_result.payload["items"]) == 1
    assert result.context.model_context()["task"]["saved_jobs"][0]["selection_index"] == 1
    assert "PRIVATE SAVED JD" not in observation.model_dump_json()
    assert result.assistant_message == "找到了以前看过的岗位。"


def test_ask_user_after_listing_emits_structured_public_options(tmp_path) -> None:
    manager = ContextManager(CareerContextStore(tmp_path / "context.sqlite3"))
    manager.upsert_profile(CareerProfileContext(user_id="u1"))
    repository = SQLiteJobPostingRepository(tmp_path / "jobs.sqlite3")
    internal_job_id = _seed_saved_job(repository)
    decisions = SequenceDecisionMaker(
        AgentDecision(
            action="tool_call",
            tool_call=ToolCall(name="find_saved_jobs", arguments={"query": "RAG"}),
        ),
        AgentDecision(action="ask_user", message="你想打开哪一个岗位？"),
    )
    runtime = MainAgentRuntime(
        context_manager=manager,
        decision_maker=decisions,
        tools=MainAgentToolRegistry(job_repository=repository),
    )
    events = []

    runtime.run_turn(
        user_id="u1",
        conversation_id="c1",
        user_message="找一下之前的岗位",
        event_sink=events.append,
    )

    interaction = next(
        event for event in events if isinstance(event, InteractionRequiredEvent)
    )
    assert interaction.kind == "single_selection"
    assert interaction.options[0].label == "RAG Engineer｜Acme"
    assert interaction.options[0].selection_index == 1
    assert internal_job_id not in interaction.model_dump_json()
    assert events[-1].type == "turn_suspended"


def test_get_saved_job_injects_user_scope_and_returns_complete_jd(tmp_path) -> None:
    manager = ContextManager(CareerContextStore(tmp_path / "context.sqlite3"))
    manager.upsert_profile(CareerProfileContext(user_id="u1"))
    repository = SQLiteJobPostingRepository(tmp_path / "jobs.sqlite3")
    job_posting_id = _seed_saved_job(repository)
    decisions = SequenceDecisionMaker(
        AgentDecision(action="tool_call", tool_call=ToolCall(name="find_saved_jobs", arguments={"query": "RAG"})),
        AgentDecision(action="tool_call", tool_call=ToolCall(name="get_saved_job", arguments={"selection_index": 1})),
        AgentDecision(action="final", message="这是该岗位的完整 JD。"),
    )
    agent = MainAgentRuntime(
        context_manager=manager,
        decision_maker=decisions,
        tools=MainAgentToolRegistry(job_repository=repository),
    )

    result = agent.run_turn(user_id="u1", conversation_id="c1", user_message="打开这个职位")

    observation = decisions.contexts[2].tool_observations[-1]
    tool_result = result.tool_results[-1]
    assert observation.tool_name == "get_saved_job"
    assert observation.body == "PRIVATE SAVED JD: Build production RAG systems."
    assert tool_result.payload["jd_snapshot"]["content"] == "PRIVATE SAVED JD: Build production RAG systems."
    assert tool_result.payload["analysis"]["required_skills"] == ["Python"]
    # The JD reaches the model as an observation body (H). It reaches the reader
    # here and nowhere else — saved_job_ready has no card — so the model's reply
    # introduces the body rather than replacing it.
    assert result.model_message == "这是该岗位的完整 JD。"
    assert result.assistant_message == (
        "这是该岗位的完整 JD。\n\nPRIVATE SAVED JD: Build production RAG systems."
    )


@pytest.mark.parametrize("tool_name,arguments", [
    ("find_saved_jobs", {"query": "RAG", "user_id": "other"}),
    ("get_saved_job", {"job_posting_id": "job-1", "user_id": "other"}),
])
def test_saved_job_tools_reject_model_supplied_user_id(tmp_path, tool_name, arguments) -> None:
    manager = ContextManager(CareerContextStore(tmp_path / "context.sqlite3"))
    manager.upsert_profile(CareerProfileContext(user_id="u1"))
    repository = SQLiteJobPostingRepository(tmp_path / "jobs.sqlite3")
    agent = MainAgentRuntime(
        context_manager=manager,
        decision_maker=SequenceDecisionMaker(AgentDecision(action="tool_call", tool_call=ToolCall(name=tool_name, arguments=arguments))),
        tools=MainAgentToolRegistry(job_repository=repository),
    )

    with pytest.raises(ValueError, match="cannot accept internal identifier"):
        agent.run_turn(user_id="u1", conversation_id="c1", user_message="越权读取")


@pytest.mark.parametrize("forbidden", ["user_id", "conversation_id", "run_id", "result_ref", "security_id", "job_id", "jd_text"])
def test_internal_arguments_are_rejected_without_commit(tmp_path, forbidden) -> None:
    agent, _, manager = build_runtime(tmp_path, AgentDecision(action="tool_call", tool_call=ToolCall(name="open_job_search", arguments={"keyword": "AI Engineer", forbidden: "hidden"})))

    # Identifier-shaped keys are refused by the shared guard; anything else the
    # model invents is refused by the tool contract itself. Either way the turn
    # must die before it commits.
    with pytest.raises(ValueError, match="internal identifiers|Extra inputs are not permitted"):
        agent.run_turn(user_id="u1", conversation_id="c1", user_message="Do it.")

    assert manager.load_for_turn(user_id="u1", conversation_id="c1", user_message="next").recent_messages == ()


def test_mock_interview_refusal_can_reroute_before_a_run_is_entered(tmp_path) -> None:
    """A selector refusal is not a workflow entry; its question does not exist."""
    from career_agent.agent.main_agent_contracts import (
        ApplicationCandidateContextItem,
        ConversationTaskState,
        MainAgentContext,
        AgentDecision,
        CareerProfileContext,
    )
    from career_agent.agent.main_agent_tools import ToolObservation

    def state_for(
        capability,
        result_state,
        *,
        refusal_count=1,
        disposition="completed",
    ):
        return {
            "context": MainAgentContext(
                conversation_id="c1",
                profile=CareerProfileContext(user_id="u1"),
                task=ConversationTaskState(
                    application_candidates=(
                        ApplicationCandidateContextItem(
                            application_id="app-1",
                            title="算法",
                            company_name="Acme",
                            status="submitted",
                        ),
                    ),
                ),
                user_message="走起",
            ),
            "pending": {
                "name": capability,
                "synthetic_kind": (
                    "projection" if result_state == "invalid_input" else None
                ),
                "result": ToolObservation(
                    tool_name=capability,
                    state=result_state,
                    message="x",
                    disposition=disposition,
                ),
            },
            "control": {"projection_refusals": refusal_count},
        }

    # Projection failed before the graph started, so candidates can still
    # repair the selector in the same turn.
    assert (
        MainAgentRuntime._after_observe(
            state_for("start_mock_interview", "invalid_input")
        )
        == "decide"
    )
    # Once the workflow really starts, its typed interaction bypasses another
    # model call without relying on capability-name routing.
    assert (
        MainAgentRuntime._after_observe(
            state_for(
                "start_mock_interview",
                "mock_interview_answer_required",
                disposition="interaction_required",
            )
        )
        == "interrupt"
    )
    # An observed refusal always returns once; authorize prevents a second
    # refusal from being appended after the configured synthetic limit.
    assert (
        MainAgentRuntime._after_observe(
            state_for("start_mock_interview", "invalid_input", refusal_count=2)
        )
        == "decide"
    )


def test_observe_routes_by_typed_disposition_not_tool_or_state_name() -> None:
    base = {
        "pending": {"name": "either_resume_tool"},
        "control": {"projection_refusals": 0, "authorization_refusals": 0},
    }

    assert MainAgentRuntime._after_observe(
        {
            **base,
            "pending": {
                **base["pending"],
                "result": ToolObservation(
                    tool_name="get_resume_analysis",
                    state="resume_analysis_ready",
                    message="已读取分析。",
                    disposition="completed",
                ),
            },
        }
    ) == "decide"
    assert MainAgentRuntime._after_observe(
        {
            **base,
            "pending": {
                **base["pending"],
                "result": ToolObservation(
                    tool_name="analyze_resume",
                    state="resume_analysis_ready",
                    message="分析完成，等待确认。",
                    disposition="interaction_required",
                ),
            },
        }
    ) == "interrupt"


def test_unknown_capability_is_rejected_without_commit(tmp_path) -> None:
    agent, _, manager = build_runtime(tmp_path, AgentDecision(action="tool_call", tool_call=ToolCall(name="boss.detail", arguments={})))

    with pytest.raises(ValueError, match="Unknown main-agent capability"):
        agent.run_turn(user_id="u1", conversation_id="c1", user_message="Do it.")

    assert manager.load_for_turn(user_id="u1", conversation_id="c1", user_message="next").recent_messages == ()


@pytest.mark.parametrize(
    ("phase", "expected"),
    [
        ("mock_interview_answer_required", "resume"),
        # The answer for the current turn is already durable, so the next
        # message must not be consumed as a new one.
        ("failed", "retry"),
    ],
)
def test_a_failed_mock_interview_step_retries_instead_of_taking_a_new_answer(
    tmp_path, phase: str, expected: str
) -> None:
    class Tools:
        runtime_workflow_names = (
            "handle_mock_interview_input",
            "retry_mock_interview",
        )

        def __init__(self) -> None:
            self.calls: list[tuple[str, str | None]] = []

        def handle_mock_interview_input(self, *, user_id, session_id, message):
            self.calls.append(("resume", message))
            return ToolObservation(
                tool_name="handle_mock_interview_input",
                state="ok",
                message="m",
                execution_outcome="committed",
            )

        def retry_mock_interview(self, *, user_id, session_id):
            self.calls.append(("retry", None))
            return ToolObservation(
                tool_name="retry_mock_interview",
                state="ok",
                message="m",
                execution_outcome="committed",
            )

        def invoke_runtime_workflow(self, name, arguments):
            if name == "retry_mock_interview":
                return self.retry_mock_interview(**arguments)
            return self.handle_mock_interview_input(**arguments)

    agent = MainAgentRuntime(
        context_manager=ContextManager(
            CareerContextStore(tmp_path / f"context-{phase}.sqlite3")
        ),
        decision_maker=SequenceDecisionMaker(),
        tools=Tools(),
    )
    context = MainAgentContext(
        conversation_id="c1",
        profile=CareerProfileContext(user_id="u1"),
        task=ConversationTaskState(
            active_workflow="mock_interview", run_id="s1", phase=phase
        ),
        user_message="随便说点别的",
    )

    result = agent._run_owned_workflow_turn(
        context=context,
        user_message="随便说点别的",
    )

    assert [name for name, _ in agent._tools.calls] == [expected]
    assert result.delegated_write_count == 1
    if expected == "retry":
        # Recovery must not depend on what the candidate can retype.
        assert agent._tools.calls[0][1] is None
    else:
        assert agent._tools.calls[0][1] == "随便说点别的"


@pytest.mark.parametrize(
    "phase",
    [
        "mock_interview_checkpoint_missing",
        "mock_interview_graph_incompatible",
    ],
)
def test_unresumable_mock_interview_returns_control_to_main_agent(
    tmp_path, phase: str
) -> None:
    manager = ContextManager(CareerContextStore(tmp_path / "context.sqlite3"))
    seeded = manager.load_for_turn(
        user_id="u1", conversation_id="c1", user_message="seed"
    )
    manager.commit_turn(
        context=seeded,
        task=ConversationTaskState(
            active_workflow="mock_interview",
            run_id="mock-session-1",
            phase=phase,
        ),
        assistant_message="The workflow cannot resume.",
    )

    class NeverResumeTools:
        def schemas(self):
            return ()

        def handle_mock_interview_input(self, **kwargs):
            raise AssertionError("an unresumable workflow must not be resumed")

    decision_maker = SequenceDecisionMaker(
        AgentDecision(action="final", message="我来处理你的新请求。")
    )
    runtime = MainAgentRuntime(
        context_manager=manager,
        decision_maker=decision_maker,
        tools=NeverResumeTools(),
    )

    result = runtime.run_turn(
        user_id="u1",
        conversation_id="c1",
        user_message="那算了，帮我看看简历。",
    )

    assert len(decision_maker.contexts) == 1
    assert result.assistant_message == "我来处理你的新请求。"


def _reference(kind: str, resource_id: str) -> ConversationResourceReference:
    # Job research alone carries delivery-time render context; the contract
    # rejects it both missing here and present on any other kind.
    extra = (
        {"status_at_delivery": "current", "anchored_by_other_job": False}
        if kind == "job_research_report"
        else {}
    )
    return ConversationResourceReference(kind=kind, resource_id=resource_id, **extra)


def test_a_report_made_this_turn_can_be_named_before_the_turn_is_stored() -> None:
    """The number has to exist while the turn is still running.

    A reference lives on a conversation row, and that row is written when the
    turn commits. Mid-turn the report is already durable and the line that would
    name it is not, so an observation could only say "a report exists". With
    ``MAX_DECISION_OBSERVATION_BODIES = 1`` the next call clears its body, and
    the model that wanted to re-read it had nothing to select it with — it could
    only call the read tool bare and hope the active resource was still the one
    it meant. Inferring, in a loop whose whole direction has been to stop making
    the model infer.
    """
    context = MainAgentContext(
        conversation_id="c1",
        profile=CareerProfileContext(user_id="u1"),
        user_message="研究一下这两个岗位",
        recent_messages=(
            ConversationMessageContext(
                role="assistant",
                content="上次的调研。",
                created_at=datetime(2026, 9, 1, tzinfo=timezone.utc),
                resource_refs=(_reference("job_research_report", "old-1"),),
            ),
        ),
        tool_observations=(
            DecisionObservation(
                tool_name="research_job",
                state="job_research_ready",
                message="岗位研究已完成。",
                resource_ref=_reference("job_research_report", "new-1"),
            ),
        ),
    )

    projected = context.model_context()["tool_observations"][0]

    # The handle names this turn's report, not last turn's, and resolves to it.
    # The one failure mode that matters here is silent: being handed a different
    # report.
    assert projected["reference"].startswith("report_")
    assert (
        context.resolve_reference(
            reference=projected["reference"], kind="job_research_report"
        )
        == "new-1"
    )


def test_a_report_read_back_in_the_same_turn_keeps_one_number() -> None:
    """Producing and re-reading one report must not mint two handles.

    ``research_job`` then ``get_job_research`` land on the same resource. Two
    numbers for one report is the drift ``referenced_resources`` exists to
    prevent, and it is silent: every index after the duplicate shifts.
    """
    reference = _reference("job_research_report", "r-1")
    context = MainAgentContext(
        conversation_id="c1",
        profile=CareerProfileContext(user_id="u1"),
        user_message="再看一眼那份调研",
        tool_observations=(
            DecisionObservation(
                tool_name="research_job",
                state="job_research_ready",
                message="岗位研究已完成。",
                resource_ref=reference,
            ),
            DecisionObservation(
                tool_name="get_job_research",
                state="job_research_ready",
                message="已读取这份调研。",
                resource_ref=reference,
            ),
        ),
    )

    observations = context.model_context()["tool_observations"]

    handles = {line["reference"] for line in observations}
    assert len(handles) == 1
    assert len(context.referenced_resources()) == 1
    assert (
        context.resolve_reference(
            reference=handles.pop(), kind="job_research_report"
        )
        == "r-1"
    )


def test_every_number_the_model_is_shown_resolves_to_what_it_was_shown_for() -> None:
    """The projection and the resolver must never be two walks that can drift.

    They are now one: ``reference_handles`` is derived from
    ``referenced_resources``, which is what ``resolve_reference`` reads.
    This checks the property across a mixed context rather than the wiring, so
    it keeps holding if either side is rewritten.
    """
    context = MainAgentContext(
        conversation_id="c1",
        profile=CareerProfileContext(user_id="u1"),
        user_message="继续",
        archived_resources=(
            ConversationMessageContext(
                role="assistant",
                content="很久以前的调研。",
                created_at=datetime(2026, 8, 1, tzinfo=timezone.utc),
                resource_refs=(_reference("job_research_report", "arch-1"),),
            ),
        ),
        recent_messages=(
            ConversationMessageContext(
                role="assistant",
                content="上一轮的两份。",
                created_at=datetime(2026, 9, 1, tzinfo=timezone.utc),
                resource_refs=(
                    _reference("job_research_report", "recent-1"),
                    _reference("interview_preparation", "recent-2"),
                ),
            ),
        ),
        tool_observations=(
            DecisionObservation(
                tool_name="research_job",
                state="job_research_ready",
                message="岗位研究已完成。",
                resource_ref=_reference("job_research_report", "turn-1"),
            ),
        ),
    )

    projection = context.model_context()
    shown = {
        line["reference"]: line["kind"]
        for line in projection["archived_reports"]["items"]
    }
    for message in projection["recent_messages"]:
        for resource in message.get("resources", ()):
            shown[resource["reference"]] = resource["kind"]
    for line in projection["tool_observations"]:
        if "reference" in line:
            shown[line["reference"]] = "job_research_report"

    expected = ["arch-1", "recent-1", "recent-2", "turn-1"]
    assert len(shown) == len(expected)
    assert [
        context.resolve_reference(reference=handle, kind=kind)
        for handle, kind in shown.items()
    ] == expected


def test_two_calls_to_one_capability_stay_distinct_after_the_body_is_cleared(
    tmp_path,
) -> None:
    """Researching job 1 then job 2 must not project as two identical lines.

    ``MAX_DECISION_OBSERVATION_BODIES = 1`` clears the first body as soon as the
    second call lands. Without arguments, what remained was tool_name, state and
    a receipt — identical for both — so the model could not tell which
    observation was which job, or even reconstruct what it had just done.

    Anthropic's context editing keeps the ``tool_use`` block, arguments and all,
    and clears only the result. This is that record. It answers a different
    question from the resource handle: arguments say what was done, the handle
    says where the output is.
    """
    observations = ()
    for index in (1, 2):
        result = ToolObservation(
            tool_name="research_job",
            state="job_research_ready",
            message="岗位研究已完成。",
        )
        observations = append_decision_observation(
            observations,
            MainAgentRuntime._tool_observation(
                "research_job", result, {"job_selection_index": index}
            ),
        )

    projected = decision_observation_projection(observations)

    assert [line["arguments"] for line in projected] == [
        {"job_selection_index": 1},
        {"job_selection_index": 2},
    ]
    # The first body is gone, exactly as designed; the call record is not.
    assert "body" not in projected[0]


def test_an_oversized_argument_set_is_dropped_rather_than_half_recorded() -> None:
    """Arguments are model-authored, so nothing upstream bounds them.

    Ten observations carrying an unbounded dict would break the character budget
    this contract declares. Truncating instead would be worse than dropping: a
    half-recorded call reads as a call made with different arguments, and the
    whole point of the record is that the model can trust what it says.
    """
    observation = DecisionObservation(
        tool_name="research_job",
        state="job_research_ready",
        message="岗位研究已完成。",
        arguments={"note": "n" * OBSERVATION_ARGUMENTS_LIMIT},
    )

    assert observation.arguments == {}


def test_the_catalogue_and_the_window_are_numbered_by_the_same_walk() -> None:
    """Not "they agree today" — they must come from one walk.

    ``model_context`` used to number the archived catalogue with a local
    ``enumerate`` and the recent window with its own counter, while
    ``resolve_reference`` read a third construction. All three agreed, and
    only because each happened to expand messages in the same order. That is the
    state ``referenced_resources`` warns about: the day one of them changes, the
    model picks index 2 and is handed index 3's report, silently.

    This checks the property from the outside — every number the projection
    shows resolves to the resource it was shown for — so it survives a rewrite
    of either side.
    """
    context = MainAgentContext(
        conversation_id="c1",
        profile=CareerProfileContext(user_id="u1"),
        user_message="继续",
        archived_resources=(
            ConversationMessageContext(
                role="assistant",
                content="旧的两份。",
                created_at=datetime(2026, 8, 1, tzinfo=timezone.utc),
                resource_refs=(
                    _reference("job_research_report", "arch-1"),
                    _reference("interview_preparation", "arch-2"),
                ),
            ),
        ),
        recent_messages=(
            ConversationMessageContext(
                role="assistant",
                content="没有产出物的一行。",
                created_at=datetime(2026, 9, 1, tzinfo=timezone.utc),
            ),
            ConversationMessageContext(
                role="assistant",
                content="上一轮的两份。",
                created_at=datetime(2026, 9, 2, tzinfo=timezone.utc),
                resource_refs=(
                    _reference("job_research_report", "recent-1"),
                    _reference("resume_job_match", "recent-2"),
                ),
            ),
        ),
    )

    projection = context.model_context()
    shown: list[tuple[str, str]] = [
        (line["reference"], line["kind"])
        for line in projection["archived_reports"]["items"]
    ]
    for message in projection["recent_messages"]:
        for resource in message.get("resources", ()):
            shown.append((resource["reference"], resource["kind"]))

    expected = ["arch-1", "arch-2", "recent-1", "recent-2"]
    assert [
        context.resolve_reference(reference=handle, kind=kind)
        for handle, kind in shown
    ] == expected
    assert set(context.reference_handles().values()) == set(expected)


def test_an_unsettled_write_blocks_a_different_one_without_killing_the_turn(
    tmp_path,
) -> None:
    """A slot whose outcome nobody knows refuses a different write, and says so.

    Preparation records intent before the tool runs, so a process that dies
    mid-flight leaves the row PENDING: the write may have reached the outside
    and may not have. Starting a *different* write in that slot has to be
    refused — but throwing would end the turn with a crash the user cannot act
    on, and would leave the stuck row invisible.

    So it is fed back as an observation, the same treatment projection and
    authorization refusals already get: the model is told, explains, and does
    not retry. The operator finds the row through ``actions reconcile``, which
    is also why the action id stays out of the reply — it is a runtime-generated
    internal identifier.
    """
    class Registry(MainAgentToolRegistry):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def capability_kind(self, name):
            return "atomic_tool"

        def invoke_atomic_tool(self, name, arguments):  # pragma: no cover - guarded
            self.calls += 1
            raise AssertionError("the blocked write must never reach the tool")

    class Runtime(MainAgentRuntime):
        @staticmethod
        def _project_atomic_tool_arguments(context, name, arguments):
            return {"user_id": context.profile.user_id, **arguments}

    manager = ContextManager(CareerContextStore(tmp_path / "context.sqlite3"))
    manager.upsert_profile(CareerProfileContext(user_id="u1"))
    ledger = SQLiteActionExecutionStore(tmp_path / "context.sqlite3")
    # What a crash between preparation and settlement leaves behind.
    stranded, _ = ledger.prepare(
        user_id="u1",
        conversation_id="c1",
        anchor="request-1",
        request_id="request-1",
        write_slot=0,
        tool_name="create_application",
        fingerprint="f" * 64,
        policy_epoch=1,
    )
    registry = Registry()

    result = Runtime(
        context_manager=manager,
        decision_maker=SequenceDecisionMaker(
            AgentDecision(
                action="tool_call",
                tool_call=ToolCall(
                    name="create_application", arguments={"note": "内推投递"}
                ),
            ),
            AgentDecision(action="final", message="有一次未确认的操作需要先核对。"),
        ),
        tools=registry,
        action_execution_store=ledger,
    ).run_turn(
        user_id="u1",
        conversation_id="c1",
        user_message="再记一次投递",
        request_id="request-1",
    )

    assert registry.calls == 0
    assert result.tool_result.state == "action_reconciliation_required"
    # The turn completed rather than raising, and the model got to answer.
    assert result.model_decision.action == "final"
    # The stranded row is untouched: only reconciliation may settle it.
    assert [item.action_id for item in ledger.list_pending()] == [stranded.action_id]
    assert stranded.action_id not in result.assistant_message


def test_an_unsettled_write_is_only_reissued_when_something_downstream_dedupes(
    tmp_path,
) -> None:
    """All writes register intent; only declared capabilities may be reissued.

    ``create_application`` is replay-safe. ``create_interview`` still has a
    durable PENDING row but cannot be reissued without reconciliation.
    """
    class Registry(MainAgentToolRegistry):
        def __init__(self, tool: str) -> None:
            super().__init__()
            self.tool = tool
            self.calls = 0

        def capability_kind(self, name):
            return "atomic_tool"

        def invoke_atomic_tool(self, name, arguments):
            self.calls += 1
            return ToolObservation(
                tool_name=name,
                state="interview_ready" if "interview" in name else "application_ready",
                message="完成。",
                payload={"interview_round_id": "round-1", "application_id": "app-1"},
                execution_outcome="committed",
            )

    class Runtime(MainAgentRuntime):
        @staticmethod
        def _project_atomic_tool_arguments(context, name, arguments):
            return {"user_id": context.profile.user_id, **arguments}

    outcomes = {}
    for tool in ("create_application", "create_interview"):
        manager = ContextManager(CareerContextStore(tmp_path / f"{tool}.sqlite3"))
        manager.upsert_profile(CareerProfileContext(user_id="u1"))
        ledger = SQLiteActionExecutionStore(tmp_path / f"{tool}.sqlite3")
        # A crash between preparation and settlement, with the same arguments:
        # the fingerprint matches, so this is a replay rather than a conflict.
        registry = Registry(tool)

        def run():
            return Runtime(
                context_manager=manager,
                decision_maker=SequenceDecisionMaker(
                    AgentDecision(
                        action="tool_call",
                        tool_call=ToolCall(name=tool, arguments={}),
                    ),
                    AgentDecision(action="final", message="完成。"),
                ),
                tools=registry,
                action_execution_store=ledger,
            ).run_turn(
                user_id="u1",
                conversation_id="c1",
                user_message="执行",
                request_id="request-1",
            )

        run()
        with sqlite3.connect(tmp_path / f"{tool}.sqlite3") as connection:
            connection.execute(
                "UPDATE action_executions SET status='PENDING', settled_at=NULL"
            )
        before = registry.calls
        replayed = run()
        outcomes[tool] = (registry.calls - before, replayed.tool_result.state)

    assert outcomes["create_application"] == (1, "application_ready")
    assert outcomes["create_interview"] == (0, "action_reconciliation_required")


def test_an_interrupted_turn_separates_confirmed_writes_from_unconfirmed_ones(
    tmp_path,
) -> None:
    """"It was written" and "it may have been written" ask for different things.

    The in-process list this replaced could only ever report the first: it
    appended after the call returned, so a process that died *during* a call
    left it empty — the case the record exists for. The ledger records intent
    before the call, which is what makes "started, outcome unknown" expressible.

    Conflating the two either invites a duplicate (treating unknown as done) or
    hides a real effect (treating done as nothing happened).
    """
    class Registry(MainAgentToolRegistry):
        def capability_kind(self, name):
            return "atomic_tool"

        def invoke_atomic_tool(self, name, arguments):  # pragma: no cover - unused
            raise AssertionError("this turn is seeded, not executed")

    class DirectRuntime(MainAgentRuntime):
        @staticmethod
        def _project_atomic_tool_arguments(context, name, arguments):
            raise ValueError("Unknown capability: get_daily_brief")

    manager = ContextManager(CareerContextStore(tmp_path / "context.sqlite3"))
    manager.upsert_profile(CareerProfileContext(user_id="u1"))
    ledger = SQLiteActionExecutionStore(tmp_path / "context.sqlite3")
    for slot, tool in ((0, "create_application"), (1, "create_interview")):
        execution, _ = ledger.prepare(
            user_id="u1",
            conversation_id="c1",
            anchor="request-1",
            request_id="request-1",
            write_slot=slot,
            tool_name=tool,
            fingerprint="a" * 64,
            policy_epoch=1,
        )
        if tool == "create_application":
            ledger.succeed(
                action_id=execution.action_id, output={"application_id": "app-1"}
            )

    with pytest.raises(ValueError, match="Unknown capability"):
        DirectRuntime(
            context_manager=manager,
            decision_maker=SequenceDecisionMaker(
                AgentDecision(
                    action="tool_call",
                    tool_call=ToolCall(name="get_daily_brief", arguments={}),
                ),
            ),
            tools=Registry(),
            action_execution_store=ledger,
        ).run_turn(
            user_id="u1",
            conversation_id="c1",
            user_message="继续",
            request_id="request-1",
        )

    note = manager.load_for_turn(
        user_id="u1", conversation_id="c1", user_message="然后呢"
    ).recent_messages[-1].content
    assert "已经写入：create_application" in note
    assert "没有确认结果" in note and "create_interview" in note
    # The settled write is not described as uncertain, and the unsettled one is
    # not described as done.
    assert note.index("create_application") < note.index("没有确认结果")


def test_a_rejected_turn_is_recorded_without_a_turn_of_its_own(tmp_path) -> None:
    """The gate rejects before a turn exists, so the record cannot ride on one.

    Measured rather than assumed: replacing the process-local gate with a lease
    or with an optimistic version number are opposite answers, suited to
    opposite contention levels, and nothing in this deployment has ever
    established which it has.
    """
    from career_agent.storage.run_events import SQLiteTraceRecorder

    recorder = SQLiteTraceRecorder(tmp_path / "run_events.sqlite3")
    runtime = MainAgentRuntime(
        context_manager=ContextManager(CareerContextStore(tmp_path / "c.sqlite3")),
        decision_maker=SequenceDecisionMaker(),
        tools=MainAgentToolRegistry(),
        trace_recorder=recorder,
    )

    runtime.record_rejected_turn(user_id="u1", conversation_id="c1")

    with sqlite3.connect(tmp_path / "run_events.sqlite3") as connection:
        rows = connection.execute(
            "SELECT event_type, stage, outcome FROM run_events"
        ).fetchall()
    assert rows == [("turn_rejected", "gate", "interrupted")]


def test_recording_a_rejection_never_turns_a_refusal_into_a_failure(tmp_path) -> None:
    """Best effort, like every other trace write.

    The turn is already refused with a 409. A telemetry store that cannot be
    written must not escalate that into a 500 — the caller would then retry a
    request that was correctly rejected.
    """
    class BrokenRecorder:
        def record(self, *args, **kwargs):
            raise RuntimeError("trace store unavailable")

    runtime = MainAgentRuntime(
        context_manager=ContextManager(CareerContextStore(tmp_path / "c.sqlite3")),
        decision_maker=SequenceDecisionMaker(),
        tools=MainAgentToolRegistry(),
        trace_recorder=BrokenRecorder(),
    )

    runtime.record_rejected_turn(user_id="u1", conversation_id="c1")


@pytest.mark.parametrize(
    ("preference", "expected_state", "expected_calls"),
    (
        ("on_user_report", "application_ready", 1),
        ("always_ask", "capability_confirmation_required", 0),
    ),
)
def test_an_owner_rule_gates_a_capability_before_it_runs(
    tmp_path, preference, expected_state, expected_calls
) -> None:
    """``review`` stops the action before any side effect and seals it."""

    class Registry(MainAgentToolRegistry):
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

    class Runtime(MainAgentRuntime):
        @staticmethod
        def _project_atomic_tool_arguments(context, name, arguments):
            return {"user_id": context.profile.user_id, **arguments}

    store = CareerContextStore(tmp_path / "context.sqlite3")
    manager = ContextManager(store)
    manager.upsert_profile(CareerProfileContext(user_id="u1"))
    store.upsert_preferences(
        "u1", AgentPreferencesContext(application_confirmation=preference)
    )
    registry = Registry()

    result = Runtime(
        context_manager=manager,
        decision_maker=SequenceDecisionMaker(
            AgentDecision(
                action="tool_call",
                tool_call=ToolCall(name="create_application", arguments={}),
            ),
            AgentDecision(action="final", message="好的。"),
        ),
        tools=registry,
        capability_confirmation_store=SQLiteCapabilityConfirmationStore(
            tmp_path / "context.sqlite3"
        ),
    ).run_turn(
        user_id="u1", conversation_id="c1", user_message="记一下这次投递"
    )

    assert result.context.tool_observations[-1].state == expected_state
    assert registry.calls == expected_calls


def test_an_owner_rule_can_be_satisfied_across_a_restart_and_runs_once(tmp_path) -> None:
    """The whole loop: set the rule, stop the action, confirm it, run it once.

    This is the test the previous slice could not have passed. ``review`` was a
    refusal plus a request that the model ask; the user would say yes, the next
    turn would consult the same unchanged rule, and refuse again. A rule the
    owner cannot satisfy is a permanent block, not a review gate.

    Every runtime here is constructed fresh from the same files, because the
    guarantee is about durable state and not about one process's memory: the
    seal has to be answerable by a process that never saw it created.
    """

    database = tmp_path / "context.sqlite3"

    class Registry(MainAgentToolRegistry):
        calls: list[dict] = []

        def capability_kind(self, name):
            return "atomic_tool"

        def invoke_atomic_tool(self, name, arguments):
            Registry.calls.append(arguments)
            return ToolObservation(
                tool_name=name,
                state="application_ready",
                message="已创建投递记录。",
                payload={"application_id": "app-1"},
                execution_outcome="committed",
            )

    class Runtime(MainAgentRuntime):
        @staticmethod
        def _project_atomic_tool_arguments(context, name, arguments):
            return {"user_id": context.profile.user_id, **arguments}

    def runtime(*decisions):
        return Runtime(
            context_manager=ContextManager(CareerContextStore(database)),
            decision_maker=SequenceDecisionMaker(*decisions),
            tools=Registry(),
            capability_confirmation_store=SQLiteCapabilityConfirmationStore(database),
        )

    Registry.calls = []
    manager = ContextManager(CareerContextStore(database))
    manager.upsert_profile(CareerProfileContext(user_id="u1"))

    # 1. The owner sets the rule, through the surface the model cannot reach.
    manager.upsert_preferences(
        user_id="u1",
        preferences=AgentPreferencesContext(application_confirmation="always_ask"),
    )

    # 2. A fresh process proposes the action; the rule stops it before it runs.
    stopped = runtime(
        AgentDecision(
            action="tool_call",
            tool_call=ToolCall(name="create_application", arguments={"note": "n1"}),
        ),
        AgentDecision(action="final", message="好的。"),
    ).run_turn(user_id="u1", conversation_id="c1", user_message="记一下这次投递")

    assert stopped.tool_result.state == "capability_confirmation_required"
    assert Registry.calls == []
    gate = MainAgentRuntime._interaction_event(result=stopped, conversation_id="c1")
    assert gate is not None and gate.scope == "capability_confirmation"

    # 3. Another fresh process — nothing in memory — receives the owner's yes.
    #    The decision maker would raise if consulted: confirming an action the
    #    model already chose must not re-ask it.
    confirmed = Runtime(
        context_manager=ContextManager(CareerContextStore(database)),
        decision_maker=_never_called_decision_maker(),
        tools=Registry(),
        capability_confirmation_store=SQLiteCapabilityConfirmationStore(database),
    ).run_turn(
        user_id="u1",
        conversation_id="c1",
        user_message="确认",
        interaction_response=InteractionResponse(
            interaction_id=gate.interaction_id,
            scope="capability_confirmation",
            action="confirm",
        ),
    )

    assert confirmed.tool_result.state == "application_ready"
    assert confirmed.origin == InteractionReceipt(
        scope="capability_confirmation", action="confirm"
    )
    # The sealed arguments, not a fresh guess: what ran is what was approved.
    assert Registry.calls == [{"user_id": "u1", "note": "n1"}]

    # 4. The same click again, from yet another process. The seal is spent, so
    #    the action does not run a second time.
    replayed = Runtime(
        context_manager=ContextManager(CareerContextStore(database)),
        decision_maker=_never_called_decision_maker(),
        tools=Registry(),
        capability_confirmation_store=SQLiteCapabilityConfirmationStore(database),
    ).run_turn(
        user_id="u1",
        conversation_id="c1",
        user_message="确认",
        interaction_response=InteractionResponse(
            interaction_id=gate.interaction_id,
            scope="capability_confirmation",
            action="confirm",
        ),
    )

    assert replayed.tool_result.state == "capability_confirmation_expired"
    assert len(Registry.calls) == 1


def test_declining_a_sealed_action_settles_it_without_running_it(tmp_path) -> None:
    """"Cancel" has to be as durable as "confirm", or the gate leaks pending rows."""

    database = tmp_path / "context.sqlite3"
    store = SQLiteCapabilityConfirmationStore(database)
    manager = ContextManager(CareerContextStore(database))
    manager.upsert_profile(CareerProfileContext(user_id="u1"))
    manager.upsert_preferences(
        user_id="u1",
        preferences=AgentPreferencesContext(application_confirmation="always_ask"),
    )

    class Registry(MainAgentToolRegistry):
        calls: list[dict] = []

        def capability_kind(self, name):
            return "atomic_tool"

        def invoke_atomic_tool(self, name, arguments):
            Registry.calls.append(arguments)
            return ToolObservation(
                tool_name=name, state="application_ready", message="已创建。"
            )

    class Runtime(MainAgentRuntime):
        @staticmethod
        def _project_atomic_tool_arguments(context, name, arguments):
            return {"user_id": context.profile.user_id, **arguments}

    Registry.calls = []
    stopped = Runtime(
        context_manager=manager,
        decision_maker=SequenceDecisionMaker(
            AgentDecision(
                action="tool_call",
                tool_call=ToolCall(name="create_application", arguments={}),
            ),
            AgentDecision(action="final", message="好的。"),
        ),
        tools=Registry(),
        capability_confirmation_store=store,
    ).run_turn(user_id="u1", conversation_id="c1", user_message="记一下")
    gate = MainAgentRuntime._interaction_event(result=stopped, conversation_id="c1")

    declined = Runtime(
        context_manager=ContextManager(CareerContextStore(database)),
        decision_maker=_never_called_decision_maker(),
        tools=Registry(),
        capability_confirmation_store=store,
    ).run_turn(
        user_id="u1",
        conversation_id="c1",
        user_message="不要",
        interaction_response=InteractionResponse(
            interaction_id=gate.interaction_id,
            scope="capability_confirmation",
            action="cancel",
        ),
    )

    assert declined.tool_result.state == "capability_confirmation_cancelled"
    assert Registry.calls == []
    # Settled, not merely unanswered: nothing is left pending for a later click.
    assert store.pending_for_conversation(
        user_id="u1", conversation_id="c1", policy_revision=None
    ) == ()


def test_default_owner_rule_does_not_expand_the_model_context() -> None:
    context = MainAgentContext(
        conversation_id="c1",
        profile=CareerProfileContext(user_id="u1"),
        user_message="测试",
    )
    assert context.model_context()["preferences"] == {
        "boss_search": "explicit_request_only"
    }
    assert "behavior_policy" not in context.model_context()

    reviewed = context.model_copy(
        update={
            "preferences": AgentPreferencesContext(
                application_confirmation="always_ask"
            )
        }
    )
    assert reviewed.model_context()["behavior_policy"]["application_confirmation"] == "always_ask"


def test_bound_owner_confirmation_executes_once_and_uses_a_durable_action_anchor(
    tmp_path,
) -> None:
    class Registry(MainAgentToolRegistry):
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

    class Runtime(MainAgentRuntime):
        @staticmethod
        def _project_atomic_tool_arguments(context, name, arguments):
            return {"user_id": context.profile.user_id, **arguments}

    path = tmp_path / "context.sqlite3"
    context_store = CareerContextStore(path)
    manager = ContextManager(context_store)
    manager.upsert_profile(CareerProfileContext(user_id="u1"))
    manager.upsert_preferences(
        user_id="u1",
        preferences=AgentPreferencesContext(application_confirmation="always_ask"),
    )
    confirmations = SQLiteCapabilityConfirmationStore(path)
    ledger = SQLiteActionExecutionStore(path)
    registry = Registry()
    runtime = Runtime(
        context_manager=manager,
        decision_maker=DecisionMaker(
            AgentDecision(
                action="tool_call",
                tool_call=ToolCall(name="create_application", arguments={}),
            )
        ),
        tools=registry,
        capability_confirmation_store=confirmations,
        action_execution_store=ledger,
    )

    held = runtime.run_turn(
        user_id="u1", conversation_id="c1", user_message="记录投递"
    )
    event = runtime._interaction_event(result=held, conversation_id="c1")
    executed = runtime.run_turn(
        user_id="u1",
        conversation_id="c1",
        user_message="确认",
        interaction_response=InteractionResponse(
            interaction_id=event.interaction_id,
            scope="capability_confirmation",
            action="confirm",
        ),
    )
    replayed_click = runtime.run_turn(
        user_id="u1",
        conversation_id="c1",
        user_message="确认",
        interaction_response=InteractionResponse(
            interaction_id=event.interaction_id,
            scope="capability_confirmation",
            action="confirm",
        ),
    )

    assert registry.calls == 1
    assert executed.tool_result.state == "application_ready"
    assert replayed_click.tool_result.state == "capability_confirmation_expired"
    confirmation_id = held.tool_result.payload["confirmation_id"]
    assert confirmations.get(confirmation_id).status == "EXECUTED"
    actions = ledger.list_for_anchor(
        user_id="u1",
        conversation_id="c1",
        anchor=f"confirmation:{confirmation_id}",
    )
    assert len(actions) == 1 and actions[0].status == "SUCCEEDED"


def test_a_policy_change_invalidates_the_exact_action_waiting_for_approval(tmp_path):
    class Registry(MainAgentToolRegistry):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def capability_kind(self, name):
            return "atomic_tool"

        def invoke_atomic_tool(self, name, arguments):
            self.calls += 1
            return ToolObservation(tool_name=name, state="application_ready", message="done")

    class Runtime(MainAgentRuntime):
        @staticmethod
        def _project_atomic_tool_arguments(context, name, arguments):
            return {"user_id": context.profile.user_id}

    path = tmp_path / "context.sqlite3"
    store = CareerContextStore(path)
    manager = ContextManager(store)
    manager.upsert_profile(CareerProfileContext(user_id="u1"))
    initial = manager.preferences(user_id="u1")
    guarded = manager.update_owner_settings(
        user_id="u1",
        desired=initial.model_copy(
            update={
                "behavior_policy": initial.behavior_policy.model_copy(
                    update={"application_confirmation": "always_ask"}
                )
            }
        ),
        expected_revision=0,
        actor_type="cli",
        actor_id="test",
    )
    confirmations = SQLiteCapabilityConfirmationStore(path)
    registry = Registry()
    runtime = Runtime(
        context_manager=manager,
        decision_maker=DecisionMaker(
            AgentDecision(action="tool_call", tool_call=ToolCall(name="create_application", arguments={}))
        ),
        tools=registry,
        capability_confirmation_store=confirmations,
    )
    held = runtime.run_turn(user_id="u1", conversation_id="c1", user_message="记录")
    event = runtime._interaction_event(result=held, conversation_id="c1")
    manager.update_owner_settings(
        user_id="u1",
        desired=guarded.model_copy(
            update={
                "behavior_policy": guarded.behavior_policy.model_copy(
                    update={"application_confirmation": "on_user_report"}
                )
            }
        ),
        expected_revision=guarded.revision,
        actor_type="cli",
        actor_id="test",
    )

    result = runtime.run_turn(
        user_id="u1", conversation_id="c1", user_message="确认",
        interaction_response=InteractionResponse(
            interaction_id=event.interaction_id,
            scope="capability_confirmation",
            action="confirm",
        ),
    )

    assert registry.calls == 0
    assert result.tool_result.state == "capability_confirmation_expired"
    assert "行为规则" in result.assistant_message


def test_agent_can_only_propose_owner_settings_and_the_bound_confirmation_applies_them(
    tmp_path,
) -> None:
    path = tmp_path / "context.sqlite3"
    store = CareerContextStore(path)
    manager = ContextManager(store)
    manager.upsert_profile(CareerProfileContext(user_id="u1"))
    confirmations = SQLiteCapabilityConfirmationStore(path)
    runtime = MainAgentRuntime(
        context_manager=manager,
        decision_maker=SequenceDecisionMaker(
            AgentDecision(
                action="tool_call",
                tool_call=ToolCall(
                    name="update_owner_settings",
                    arguments={"application_confirmation": "always_ask"},
                ),
            )
        ),
        tools=MainAgentToolRegistry(owner_settings_store=store),
        capability_confirmation_store=confirmations,
        action_execution_store=SQLiteActionExecutionStore(path),
    )

    proposal = runtime.run_turn(
        user_id="u1",
        conversation_id="c1",
        user_message="以后记录投递前都问我",
    )
    assert proposal.tool_result.state == "capability_confirmation_required"
    assert manager.preferences(user_id="u1").application_confirmation == "on_user_report"
    event = runtime._interaction_event(result=proposal, conversation_id="c1")

    applied = runtime.run_turn(
        user_id="u1",
        conversation_id="c1",
        user_message="确认",
        interaction_response=InteractionResponse(
            interaction_id=event.interaction_id,
            scope="capability_confirmation",
            action="confirm",
        ),
    )

    settings = manager.preferences(user_id="u1")
    assert applied.tool_result.state == "owner_settings_updated"
    assert settings.application_confirmation == "always_ask"
    event_record = store.list_owner_settings_events(user_id="u1")[0]
    assert event_record.actor_type == "confirmed_agent_proposal"
    assert event_record.actor_id == proposal.tool_result.payload["confirmation_id"]


def test_settings_review_is_a_system_invariant_not_an_owner_editable_rule() -> None:
    settings = AgentPreferencesContext()

    assert settings.behavior_policy.capability_verdict("update_owner_settings") == "permit"
    assert settings.capability_verdict("update_owner_settings") == "review"


def test_external_writes_default_to_review_below_any_owner_rule() -> None:
    """An event on the user's calendar cannot be undone by a later turn.

    So the floor for an external write is the owner's button, not the model's
    reading of "yes". No owner-editable rule reaches this verdict: the
    behaviour policy alone still says permit, and the system says review.
    """

    settings = AgentPreferencesContext()

    assert settings.behavior_policy.capability_verdict("execute_calendar_proposal") == "permit"
    assert settings.capability_verdict("execute_calendar_proposal") == "review"
    # The local preview beside it is an ordinary internal write.
    assert settings.capability_verdict("prepare_interview_calendar_sync") == "permit"
    assert settings.capability_verdict("create_application") == "permit"


def test_confirm_before_is_a_canonical_set_of_declared_write_capabilities() -> None:
    from career_agent.agent.main_agent_contracts import (
        BehaviorPolicyContext,
        UpdateOwnerSettingsToolArguments,
        canonical_confirm_before,
    )

    assert canonical_confirm_before(
        ("update_application_status", "create_application", "create_application")
    ) == ("create_application", "update_application_status")
    assert canonical_confirm_before(()) == ()
    with pytest.raises(ValueError, match="search_career_history"):
        canonical_confirm_before(("search_career_history",))
    with pytest.raises(ValueError, match="unknown: drop_tables"):
        canonical_confirm_before(("drop_tables",))

    policy = BehaviorPolicyContext(
        confirm_before=["update_application_status", "create_application"]
    )
    assert policy.confirm_before == ("create_application", "update_application_status")
    assert policy.capability_verdict("create_application") == "review"
    assert policy.capability_verdict("update_interview") == "permit"
    with pytest.raises(ValidationError):
        BehaviorPolicyContext(confirm_before=["get_daily_brief"])

    # The model's proposal is validated to the same vocabulary, and an empty
    # list is a change (clear the rule) rather than "nothing to do".
    assert UpdateOwnerSettingsToolArguments(confirm_before=[]).confirm_before == ()
    with pytest.raises(ValidationError):
        UpdateOwnerSettingsToolArguments()
    with pytest.raises(ValidationError):
        UpdateOwnerSettingsToolArguments(confirm_before=["get_daily_brief"])


def test_confirm_before_gates_the_named_capability_and_is_shown_to_the_model(
    tmp_path,
) -> None:
    class Registry(MainAgentToolRegistry):
        def __init__(self) -> None:
            super().__init__()
            self.calls: list[str] = []

        def capability_kind(self, name):
            return "atomic_tool"

        def invoke_atomic_tool(self, name, arguments):
            self.calls.append(name)
            return ToolObservation(
                tool_name=name,
                state="interview_ready",
                message="已记录。",
                payload={"interview_round_id": "ir-1"},
                execution_outcome="committed",
            )

    class Runtime(MainAgentRuntime):
        @staticmethod
        def _project_atomic_tool_arguments(context, name, arguments):
            return {"user_id": context.profile.user_id, **arguments}

    path = tmp_path / "context.sqlite3"
    store = CareerContextStore(path)
    manager = ContextManager(store)
    manager.upsert_profile(CareerProfileContext(user_id="u1"))
    initial = manager.preferences(user_id="u1")
    manager.update_owner_settings(
        user_id="u1",
        desired=initial.model_copy(
            update={
                "behavior_policy": initial.behavior_policy.model_copy(
                    update={"confirm_before": ("create_interview",)}
                )
            }
        ),
        expected_revision=initial.revision,
        actor_type="cli",
        actor_id="test",
    )
    registry = Registry()
    decisions = SequenceDecisionMaker(
        AgentDecision(
            action="tool_call",
            tool_call=ToolCall(name="update_interview", arguments={}),
        ),
        AgentDecision(
            action="tool_call",
            tool_call=ToolCall(name="create_interview", arguments={}),
        ),
        AgentDecision(action="final", message="好的。"),
    )

    result = Runtime(
        context_manager=manager,
        decision_maker=decisions,
        tools=registry,
        capability_confirmation_store=SQLiteCapabilityConfirmationStore(path),
        max_read_calls=5,
        max_write_calls=2,
    ).run_turn(user_id="u1", conversation_id="c1", user_message="记一下面试")

    # The unlisted write ran; the listed one stopped at the seal.
    assert registry.calls == ["update_interview"]
    assert result.tool_result.state == "capability_confirmation_required"
    assert "你设置了此操作需要确认" in result.tool_result.message
    assert decisions.contexts[0].model_context()["behavior_policy"] == {
        "confirm_before": ["create_interview"]
    }


def test_internal_and_external_writes_draw_on_separate_budgets(tmp_path) -> None:
    """One local record plus one external booking fits in a turn.

    Under a single WRITE bucket the second call below was refused for budget
    before authorization ever looked at it; now it reaches the review seal.
    The external ceiling is its own counter, so an exhausted internal budget
    still refuses internal writes and never borrows the external slot.
    """

    class Registry(MainAgentToolRegistry):
        def __init__(self) -> None:
            super().__init__()
            self.calls: list[str] = []

        def capability_kind(self, name):
            return "atomic_tool"

        def invoke_atomic_tool(self, name, arguments):
            self.calls.append(name)
            return ToolObservation(
                tool_name=name,
                state="application_ready",
                message="已记录。",
                payload={"application_id": "app-1"},
                execution_outcome="committed",
            )

    class Runtime(MainAgentRuntime):
        @staticmethod
        def _project_atomic_tool_arguments(context, name, arguments):
            return {"user_id": context.profile.user_id, **arguments}

    path = tmp_path / "context.sqlite3"
    manager = ContextManager(CareerContextStore(path))
    manager.upsert_profile(CareerProfileContext(user_id="u1"))
    registry = Registry()

    result = Runtime(
        context_manager=manager,
        decision_maker=SequenceDecisionMaker(
            AgentDecision(
                action="tool_call",
                tool_call=ToolCall(name="create_application", arguments={}),
            ),
            AgentDecision(
                action="tool_call",
                tool_call=ToolCall(name="update_application_status", arguments={}),
            ),
            AgentDecision(
                action="tool_call",
                tool_call=ToolCall(name="execute_calendar_proposal", arguments={}),
            ),
            AgentDecision(action="final", message="请确认。"),
        ),
        tools=registry,
        capability_confirmation_store=SQLiteCapabilityConfirmationStore(path),
    ).run_turn(user_id="u1", conversation_id="c1", user_message="记录并加日历")

    states = [item.state for item in result.context.tool_observations]
    # The seal reads the proposal back to describe it to the owner; nothing
    # external ran.
    assert registry.calls == ["create_application", "get_calendar_proposal"]
    assert states == [
        "application_ready",
        "authorization_refused",
        "capability_confirmation_required",
    ]
    assert "本轮 WRITE 委派预算已经用完" in result.context.tool_observations[1].message
    assert result.delegated_write_count == 1

    runtime = Runtime(
        context_manager=manager,
        decision_maker=_never_called_decision_maker(),
        tools=registry,
    )
    control = {"read_calls": 2, "write_calls": 3, "external_write_calls": 1}
    assert runtime._budget_bucket(control, name="search_career_history", effect="READ") == (
        "READ", 2, 6
    )
    assert runtime._budget_bucket(control, name="create_application", effect="WRITE") == (
        "WRITE", 2, 1
    )
    assert runtime._budget_bucket(
        control, name="execute_calendar_proposal", effect="WRITE"
    ) == ("WRITE_EXTERNAL", 1, 1)
    with pytest.raises(ValueError, match="max_external_write_calls"):
        Runtime(
            context_manager=manager,
            decision_maker=_never_called_decision_maker(),
            tools=registry,
            max_external_write_calls=0,
        )


def test_reproposing_an_applying_confirmation_reports_in_progress_not_waiting(
    tmp_path,
) -> None:
    path = tmp_path / "context.sqlite3"
    store = CareerContextStore(path)
    manager = ContextManager(store)
    manager.upsert_profile(CareerProfileContext(user_id="u1"))
    confirmations = SQLiteCapabilityConfirmationStore(path)
    arguments = {
        "user_id": "u1",
        "expected_revision": 0,
        "application_confirmation": "always_ask",
    }
    sealed = confirmations.seal(
        user_id="u1", conversation_id="c1", capability="update_owner_settings",
        display_summary="准备更新持久设置。", arguments=arguments, policy_revision=0,
    )
    confirmations.claim(confirmation_id=sealed.confirmation_id, user_id="u1")
    runtime = MainAgentRuntime(
        context_manager=manager,
        decision_maker=SequenceDecisionMaker(
            AgentDecision(
                action="tool_call",
                tool_call=ToolCall(
                    name="update_owner_settings",
                    arguments={"application_confirmation": "always_ask"},
                ),
            ),
            AgentDecision(action="final", message="操作仍在执行。"),
        ),
        tools=MainAgentToolRegistry(owner_settings_store=store),
        capability_confirmation_store=confirmations,
    )

    result = runtime.run_turn(
        user_id="u1", conversation_id="c1", user_message="再设置一次"
    )

    assert result.tool_results[0].state == "capability_confirmation_in_progress"
    assert "正在执行" in result.tool_results[0].message
    assert "等你确认" not in result.tool_results[0].message
