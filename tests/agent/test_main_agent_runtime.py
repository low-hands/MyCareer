from __future__ import annotations

from datetime import datetime, timezone
from urllib.parse import parse_qs, urlparse

import pytest
from pydantic import ValidationError

from career_agent.agent.context_manager import ContextManager
from career_agent.agent.main_agent_contracts import AgentDecision, CareerMemoryContext, CareerMemoryRecord, CareerProfileContext, ConversationTaskState, DECISION_OBSERVATION_BODY_LIMIT, DECISION_OBSERVATION_RECEIPT_LIMIT, MAX_DECISION_OBSERVATION_BODIES, MAX_DECISION_OBSERVATION_CHARS, DecisionObservation, MainAgentContext, MAX_DECISION_OBSERVATIONS, ToolCall, ToolObservation, ToolResult, append_decision_observation, decision_observation_chars
from career_agent.agent.main_agent_runtime import MainAgentRuntime
from career_agent.agent.main_agent_tools import MainAgentToolRegistry
from career_agent.domain.job_discovery import JobDetail, Provenance
from career_agent.storage.context import CareerContextStore
from career_agent.storage.jobs import JDAnalysisPayload, SQLiteJobPostingRepository
from career_agent.harness.streaming import ClientActionEvent, InteractionRequiredEvent


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


def test_main_graph_has_distinct_delivery_and_suspension_exits(tmp_path) -> None:
    agent, _, _ = build_runtime(tmp_path, AgentDecision(action="final", message="done"))
    graph = agent._graph.get_graph()

    ends = {edge.source for edge in graph.edges if edge.target == "__end__"}
    assert ends == {"present", "interrupt"}


def test_loop_state_and_budget_window_are_structurally_bounded(tmp_path) -> None:
    from career_agent.agent.main_agent_runtime import (
        DEFAULT_MAX_AUTHORIZATION_REFUSALS,
        DEFAULT_MAX_PROJECTION_REFUSALS,
        DEFAULT_MAX_READ_CALLS,
        DEFAULT_MAX_WRITE_CALLS,
        MainAgentState,
    )

    assert len(MainAgentState.__annotations__) <= 9
    assert (
        DEFAULT_MAX_READ_CALLS
        + DEFAULT_MAX_WRITE_CALLS
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
            max_projection_refusals=2,
            max_authorization_refusals=1,
        )


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
                        confirmed_highlights=("Led an AI product",),
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

    assert result.decision.action == "ask_user"
    assert result.assistant_message == "搜索页已经打开，你想先看哪一个岗位？"
    assert len(tools.calls) == 1
    assert len(decisions.contexts) == 2
    observation = decisions.contexts[1].model_context()["tool_observations"][0]
    assert observation == {
        "tool_name": "open_job_search",
        "state": "job_search_page_ready",
        "message": "已准备打开 BOSS 搜索“AI Engineer”。请正常浏览，并只保存你感兴趣的岗位。",
        "facts": {},
        "next_action": "browse_and_save_job",
    }
    serialized = str(observation)
    assert "zhipin.com" not in serialized


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
    assert result.assistant_message.startswith("已准备打开 BOSS 搜索")
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
    assert result.assistant_message == "已读取 Role B。"
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
    assert observation.facts == {"overdue": 1, "due_today": 0, "waiting": 0}


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
        ),
    )

    assert (
        MAX_DECISION_OBSERVATION_BODIES * DECISION_OBSERVATION_BODY_LIMIT
        + MAX_DECISION_OBSERVATIONS * DECISION_OBSERVATION_RECEIPT_LIMIT
    ) == 12_000
    assert decision_observation_chars(observations) == 15_204
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


@pytest.mark.parametrize(
    ("result", "expected"),
    (
        (
            ToolResult(
                tool_name="get_daily_brief",
                state="daily_brief_ready",
                message="今日职业简报包含 13 个待办事项。",
                payload={
                    "overdue": [{}] * 6,
                    "due_today": [{}] * 4,
                    "no_due_date": [{}] * 3,
                },
            ),
            {"overdue": 6, "due_today": 4, "waiting": 3},
        ),
        (
            ToolResult(
                tool_name="analyze_resume",
                state="resume_analysis_ready",
                message="已分析简历。",
                payload={
                    "records": [{}] * 12,
                    "clarification_questions": ["请确认时间"],
                    "warnings": [],
                },
            ),
            {
                "record_count": 12,
                "clarification_count": 1,
                "has_warnings": False,
            },
        ),
        (
            ToolResult(
                tool_name="research_job",
                state="job_research_ready",
                message="已复用岗位研究。",
                payload={
                    "cached": True,
                    "status": "current",
                    "research": {"findings": [{}] * 8},
                },
            ),
            {"cached": True, "finding_count": 8, "status": "current"},
        ),
        (
            ToolResult(
                tool_name="match_resume_to_job",
                state="resume_job_match_ready",
                message="已完成逐项匹配，整体匹配度为 moderate。",
                payload={"result": {"overall_fit": "moderate"}},
            ),
            {},
        ),
    ),
)
def test_decision_facts_are_state_whitelisted(result, expected) -> None:
    observation = MainAgentRuntime._tool_observation(result.tool_name, result)

    assert observation.facts == expected
    assert all(key != "id" and not key.endswith("_id") for key in observation.facts)


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
    with pytest.raises(ValidationError):
        DecisionObservation.model_validate({**base, "facts": {"overdue": 1}})


def test_internal_tool_result_requires_a_durable_receipt() -> None:
    """Summary/message delivery must never commit an invisible empty row."""
    with pytest.raises(ValidationError):
        ToolResult(
            tool_name="get_daily_brief",
            state="daily_brief_ready",
            message="",
        )


def test_final_model_message_cannot_characterize_an_opaque_tool_result() -> None:
    result = ToolResult(
        tool_name="analyze_resume",
        state="resume_analysis_ready",
        message="已分析简历并生成待确认候选事实。",
        payload={"records": [{"title": "PRIVATE RESULT"}]},
    )

    update = MainAgentRuntime._present(
        {
            "decision": AgentDecision(
                action="final",
                message="看起来很不错，经历非常有竞争力。",
            ),
            "tool_results": (result,),
        }
    )

    assert update["assistant_message"] == result.message
    assert "很不错" not in update["assistant_message"]


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
    assert result.assistant_message == "找到 1 个已保存职位。"


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
    assert result.assistant_message == "PRIVATE SAVED JD: Build production RAG systems."


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
    phase: str, expected: str
) -> None:
    class Tools:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str | None]] = []

        def handle_mock_interview_input(self, *, user_id, session_id, message):
            self.calls.append(("resume", message))
            return ToolObservation(tool_name="start_mock_interview", state="ok", message="m")

        def retry_mock_interview(self, *, user_id, session_id):
            self.calls.append(("retry", None))
            return ToolObservation(tool_name="start_mock_interview", state="ok", message="m")

    agent = MainAgentRuntime.__new__(MainAgentRuntime)
    agent._tools = Tools()
    context = MainAgentContext(
        conversation_id="c1",
        profile=CareerProfileContext(user_id="u1"),
        task=ConversationTaskState(
            active_workflow="mock_interview", run_id="s1", phase=phase
        ),
        user_message="随便说点别的",
    )

    agent._run_active_mock_interview(
        context=context,
        user_message="随便说点别的",
    )

    assert [name for name, _ in agent._tools.calls] == [expected]
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
        def schemas(self, context=None):
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
