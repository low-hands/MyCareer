"""An explicitly named sequence span is read by the runtime, not re-asked."""

from __future__ import annotations

import pytest

from career_agent.agent.context_manager import ContextManager
from career_agent.agent.conversation_memory_contracts import ConversationSummaryContent
from career_agent.agent.conversation_span_requests import (
    ExplicitSequenceSpan,
    explicit_sequence_span,
)
from career_agent.agent.main_agent_contracts import (
    AgentDecision,
    CareerProfileContext,
    ConversationTaskState,
    ToolCall,
)
from career_agent.agent.main_agent_runtime import MainAgentRuntime, ModelDecision
from career_agent.agent.main_agent_tools import MainAgentToolRegistry
from career_agent.storage.context import CareerContextStore


class SequenceDecisionMaker:
    def __init__(self, *decisions: AgentDecision) -> None:
        self.decisions = list(decisions)
        self.contexts = []

    def decide(self, context, tool_specs):
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
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.calls: list[tuple[str, dict]] = []

    def invoke_atomic_tool(self, name, arguments):
        self.calls.append((name, dict(arguments)))
        return super().invoke_atomic_tool(name, arguments)


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("把序号 100 到 110 的对话读回来。", ExplicitSequenceSpan(100, 110)),
        ("序号3-7的消息说了什么", ExplicitSequenceSpan(3, 7)),
        ("回看编号 12 至 15 的聊天记录", ExplicitSequenceSpan(12, 15)),
        ("第 4 到 6 条消息里我提过哪家公司？", ExplicitSequenceSpan(4, 6)),
        ("序号 9 那条对话原话是什么", ExplicitSequenceSpan(9, 9)),
        ("read back conversation sequence 2 through 5", ExplicitSequenceSpan(2, 5)),
        ("会话序号 8 说了什么", ExplicitSequenceSpan(8, 8)),
        ("第3至7句对话再读一遍", ExplicitSequenceSpan(3, 7)),
    ],
)
def test_explicit_sequence_span_reads_a_named_range(message, expected) -> None:
    assert explicit_sequence_span(message) == expected


@pytest.mark.parametrize(
    "message",
    [
        # Numbers without a sequence cue, or a sequence cue about something
        # other than the conversation, are not a history read.
        "薪资 30 到 50K 的对话怎么谈",
        "第 3 到 5 条岗位帮我对比",
        "序号 3 到 5 的岗位再看看",
        # A conversation word elsewhere does not make the number a message
        # number: these are about jobs and interview rounds.
        "对话里序号3到5的岗位再看看",
        "回看第3轮面试的聊天记录",
        "第2轮面试时我们的对话说了什么",
        "第 3 段经历在聊天里提过吗",
        # Invalid spans need the user, not a guess.
        "把序号 10 到 3 的对话读回来",
        "把序号 0 到 3 的对话读回来",
        # Two different spans: the runtime does not pick one.
        "序号 3 到 5 和序号 8 到 9 的对话都读回来",
        "我一开始指定的目标公司全名叫什么？",
    ],
)
def test_explicit_sequence_span_leaves_unclear_requests_alone(message) -> None:
    assert explicit_sequence_span(message) is None


def _compacted_runtime(tmp_path, *decisions):
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
    maker = SequenceDecisionMaker(*decisions)
    tools = CountingRegistry(conversation_store=store)
    runtime = MainAgentRuntime(
        context_manager=manager, decision_maker=maker, tools=tools
    )
    return runtime, maker, tools


def test_named_span_is_read_before_the_model_is_asked(tmp_path) -> None:
    runtime, maker, tools = _compacted_runtime(
        tmp_path,
        AgentDecision(action="final", message="序号 1 到 2 是你最早的两句。"),
    )

    result = runtime.run_turn(
        user_id="u1",
        conversation_id="c1",
        user_message="把序号 1 到 2 的对话读回来。",
    )

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
    # The model decides exactly once, with the span already observed.
    assert len(maker.contexts) == 1
    seen = maker.contexts[0]
    assert seen.through_sequence == 4
    assert seen.recent_from_sequence == 5
    observation = seen.tool_observations[-1]
    assert observation.tool_name == "read_conversation_span"
    assert observation.state == "conversation_span_found"
    assert observation.arguments == {"from_sequence": 1, "through_sequence": 2}
    assert observation.body is not None
    assert "private-old-user-0" in observation.body
    assert isinstance(result.origin, ModelDecision)
    assert result.tool_result is not None
    assert result.tool_result.state == "conversation_span_found"
    # Delivered like any model-chosen span read: reply first, span body after.
    assert result.assistant_message.startswith("序号 1 到 2 是你最早的两句。")
    assert "private-old-user-0" in result.assistant_message
    assert result.delegated_read_count == 1


def test_named_empty_span_is_reported_not_filled(tmp_path) -> None:
    runtime, maker, tools = _compacted_runtime(
        tmp_path,
        AgentDecision(action="final", message="序号 100 到 110 不存在。"),
    )

    result = runtime.run_turn(
        user_id="u1",
        conversation_id="c1",
        user_message="把序号 100 到 110 的对话读回来。",
    )

    assert [name for name, _ in tools.calls] == ["read_conversation_span"]
    assert tools.calls[0][1]["from_sequence"] == 100
    assert tools.calls[0][1]["through_sequence"] == 110
    assert len(maker.contexts) == 1
    observation = maker.contexts[0].tool_observations[-1]
    assert observation.state == "conversation_span_empty"
    assert observation.facts["returned"] == 0
    assert result.tool_result is not None
    assert result.tool_result.state == "conversation_span_empty"


def test_model_still_owns_its_own_calls_after_the_prelude(tmp_path) -> None:
    """The prelude's policy ownership must not leak into the model's turn.

    A model call after the prelude is authorised, observed, and handed back to
    the model like any other; it does not end the turn at ``present``.
    """

    runtime, maker, tools = _compacted_runtime(
        tmp_path,
        AgentDecision(
            action="tool_call",
            tool_call=ToolCall(
                name="read_conversation_span",
                arguments={"from_sequence": 3, "through_sequence": 4},
            ),
        ),
        AgentDecision(action="final", message="两段都看过了。"),
    )

    result = runtime.run_turn(
        user_id="u1",
        conversation_id="c1",
        user_message="把序号 1 到 2 的对话读回来，再看看后面。",
    )

    assert [(name, a["from_sequence"], a["through_sequence"]) for name, a in tools.calls] == [
        ("read_conversation_span", 1, 2),
        ("read_conversation_span", 3, 4),
    ]
    assert len(maker.contexts) == 2
    assert [o.arguments for o in maker.contexts[1].tool_observations[-2:]] == [
        {"from_sequence": 1, "through_sequence": 2},
        {"from_sequence": 3, "through_sequence": 4},
    ]
    assert result.assistant_message.startswith("两段都看过了。")
    assert result.delegated_read_count == 2


def test_named_span_without_compacted_history_stays_with_the_model(
    tmp_path,
) -> None:
    store = CareerContextStore(tmp_path / "context.sqlite3")
    manager = ContextManager(store)
    manager.upsert_profile(CareerProfileContext(user_id="u1"))
    maker = SequenceDecisionMaker(AgentDecision(action="final", message="好。"))
    tools = CountingRegistry(conversation_store=store)
    runtime = MainAgentRuntime(
        context_manager=manager, decision_maker=maker, tools=tools
    )

    runtime.run_turn(
        user_id="u1",
        conversation_id="c1",
        user_message="把序号 1 到 2 的对话读回来。",
    )

    assert tools.calls == []
    assert len(maker.contexts) == 1
    assert maker.contexts[0].through_sequence == 0
    assert maker.contexts[0].tool_observations == ()


def test_named_span_without_a_conversation_store_stays_with_the_model(
    tmp_path,
) -> None:
    manager = ContextManager(CareerContextStore(tmp_path / "context.sqlite3"))
    manager.upsert_profile(CareerProfileContext(user_id="u1"))
    maker = SequenceDecisionMaker(AgentDecision(action="final", message="好。"))
    tools = CountingRegistry()
    runtime = MainAgentRuntime(
        context_manager=manager, decision_maker=maker, tools=tools
    )

    runtime.run_turn(
        user_id="u1",
        conversation_id="c1",
        user_message="把序号 1 到 2 的对话读回来。",
    )

    assert tools.calls == []
    assert len(maker.contexts) == 1
    assert "read_conversation_span" not in MainAgentToolRegistry().atomic_tool_names
