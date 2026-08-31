from __future__ import annotations

from datetime import datetime, timezone
from urllib.parse import parse_qs, urlparse

import pytest
from pydantic import ValidationError

from career_agent.agent.context_manager import ContextManager
from career_agent.agent.main_agent_contracts import AgentDecision, CareerMemoryContext, CareerMemoryRecord, CareerProfileContext, ConversationTaskState, DecisionObservation, MainAgentContext, ToolCall, ToolObservation, ToolResult
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


def test_main_graph_separates_atomic_tools_from_workflows(tmp_path) -> None:
    agent, _, _ = build_runtime(tmp_path, AgentDecision(action="final", message="done"))

    assert set(agent._graph.get_graph().nodes) == {
        "__start__",
        "hydrate_career_context",
        "decide",
        "invoke_atomic_tool",
        "run_workflow",
        "observe",
        "finish",
        "present",
        "__end__",
    }


def test_every_exit_that_renders_a_tool_result_is_one_node(tmp_path) -> None:
    """One presenter exit, not three.

    ``present_workflow`` and ``fallback`` were separate nodes with near-identical
    bodies, and ``finish`` calls the same presenter for a tool-backed final
    answer, so the names implied a division of labour that did not exist. Both
    conditional edges now name the same landing spot, which is the only reason
    it has to be a node at all.
    """
    agent, _, _ = build_runtime(tmp_path, AgentDecision(action="final", message="done"))
    graph = agent._graph.get_graph()

    assert "fallback" not in graph.nodes
    assert "present_workflow" not in graph.nodes
    ends = {edge.source for edge in graph.edges if edge.target == "__end__"}
    assert ends == {"finish", "present"}


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
        "next_action": "browse_and_save_job",
    }
    serialized = str(observation)
    assert "zhipin.com" not in serialized


def test_repeated_tool_call_is_stopped_without_duplicate_execution(tmp_path) -> None:
    agent, tools, _ = build_runtime(
        tmp_path,
        AgentDecision(action="tool_call", tool_call=ToolCall(name="open_job_search", arguments={"keyword": "AI Engineer"})),
    )

    result = agent.run_turn(user_id="u1", conversation_id="c1", user_message="Find work.")

    assert len(tools.calls) == 1
    assert result.assistant_message.startswith("已准备打开 BOSS 搜索“AI Engineer”")


def test_tool_loop_stops_at_configured_limit(tmp_path) -> None:
    manager = ContextManager(CareerContextStore(tmp_path / "context.sqlite3"))
    manager.upsert_profile(CareerProfileContext(user_id="u1"))
    tools = CountingRegistry()
    decisions = SequenceDecisionMaker(
        AgentDecision(action="tool_call", tool_call=ToolCall(name="open_job_search", arguments={"keyword": "Role A"})),
        AgentDecision(action="tool_call", tool_call=ToolCall(name="open_job_search", arguments={"keyword": "Role B"})),
        AgentDecision(action="tool_call", tool_call=ToolCall(name="open_job_search", arguments={"keyword": "Role C"})),
    )
    agent = MainAgentRuntime(
        context_manager=manager,
        decision_maker=decisions,
        tools=tools,
        max_tool_calls=2,
    )

    result = agent.run_turn(user_id="u1", conversation_id="c1", user_message="Research several roles.")

    assert [arguments["keyword"] for _, arguments in tools.calls] == ["Role A", "Role B"]
    assert result.assistant_message.startswith("已准备打开 BOSS 搜索“Role B”")


def test_target_and_search_overrides_do_not_mutate_profile(tmp_path) -> None:
    agent, tools, manager = build_runtime(tmp_path, AgentDecision(action="tool_call", tool_call=ToolCall(name="open_job_search", arguments={"keyword": "Backend Engineer", "city": "杭州"})))

    result = agent.run_turn(user_id="u1", conversation_id="c1", user_message="Search backend roles in Hangzhou this time.")

    action = result.tool_results[0].payload["client_action"]
    parsed = urlparse(action["url"])
    assert parse_qs(parsed.query) == {"query": ["Backend Engineer"], "city": ["101210100"]}
    profile = manager.load_for_turn(user_id="u1", conversation_id="c1", user_message="next").profile
    assert profile.default_city == "Shanghai"


def test_internal_tool_result_cannot_expand_decision_prompt() -> None:
    sentinel = "PRIVATE-PAYLOAD-DO-NOT-PROMPT"
    result = ToolResult(
        tool_name="get_saved_job",
        state="saved_job_ready",
        message=f"message:{sentinel}",
        next_action="match_resume_to_job",
        payload={"jd_snapshot": {"content": sentinel}, "job_posting_id": "secret-id"},
    )

    observation = MainAgentRuntime._tool_observation("get_saved_job", result)

    assert observation.model_dump() == {
        "tool_name": "get_saved_job",
        "state": "saved_job_ready",
        "next_action": "match_resume_to_job",
    }
    assert sentinel not in observation.model_dump_json()
    assert len(observation.model_dump_json()) < 256
    with pytest.raises(ValidationError):
        DecisionObservation.model_validate(
            {**observation.model_dump(), "payload": {"content": sentinel}}
        )


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

    update = MainAgentRuntime._finish(
        {
            "decision": AgentDecision(
                action="final",
                message="看起来很不错，经历非常有竞争力。",
            ),
            "last_tool_result": result,
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
    assert "PRIVATE SAVED JD" not in observation.model_dump_json()
    assert tool_result.payload["jd_snapshot"]["content"] == "PRIVATE SAVED JD: Build production RAG systems."
    assert tool_result.payload["analysis"]["required_skills"] == ["Python"]
    assert result.assistant_message == "已读取 RAG Engineer（Acme）的完整 JD。"


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
