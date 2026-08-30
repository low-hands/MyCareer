from __future__ import annotations

from types import SimpleNamespace

import pytest

from career_agent.agent.answer_writer import (
    AnswerCompositionRequest,
    OpenAIStreamingAnswerWriter,
)
from career_agent.agent.context_manager import ContextManager
from career_agent.agent.main_agent_contracts import (
    AgentDecision,
    CareerProfileContext,
    ToolCall,
)
from career_agent.agent.main_agent_runtime import MainAgentRuntime
from career_agent.agent.main_agent_tools import MainAgentToolRegistry
from career_agent.agent.openai_compatible_client import (
    AgentWorkerError,
    OpenAICompatibleAgentConfig,
)
from career_agent.storage.context import CareerContextStore


class Decisions:
    def __init__(self, *items: AgentDecision) -> None:
        self.items = list(items)

    def decide(self, context, tool_specs):
        return self.items.pop(0)


class RecordingWriter:
    def __init__(self, *chunks: str) -> None:
        self.chunks = chunks
        self.requests = []

    def stream(self, request):
        self.requests.append(request)
        yield from self.chunks


def _manager(tmp_path) -> ContextManager:
    manager = ContextManager(CareerContextStore(tmp_path / "context.sqlite3"))
    manager.upsert_profile(CareerProfileContext(user_id="u1"))
    return manager


def test_runtime_streams_writer_tokens_and_commits_the_same_answer(tmp_path) -> None:
    manager = _manager(tmp_path)
    writer = RecordingWriter("这是", "真正的", "流式回答。")
    runtime = MainAgentRuntime(
        context_manager=manager,
        decision_maker=Decisions(
            AgentDecision(action="final", message="结构化阶段形成的展示草稿。")
        ),
        tools=MainAgentToolRegistry(),
        answer_writer=writer,
    )
    events = []

    result = runtime.run_turn(
        user_id="u1",
        conversation_id="c1",
        user_message="请用自然语言解释",
        event_sink=events.append,
    )

    assert result.assistant_message == "这是真正的流式回答。"
    assert result.content_streamed is True
    assert [
        event.delta for event in events if event.type == "content_delta"
    ] == ["这是", "真正的", "流式回答。"]
    assert writer.requests[0].grounded_draft == "结构化阶段形成的展示草稿。"
    history = manager.load_for_turn(
        user_id="u1", conversation_id="c1", user_message="next"
    ).recent_messages
    assert [message.content for message in history][-2:] == [
        "请用自然语言解释",
        "这是真正的流式回答。",
    ]


def test_interaction_bypasses_writer(tmp_path) -> None:
    writer = RecordingWriter("不应生成")
    runtime = MainAgentRuntime(
        context_manager=_manager(tmp_path),
        decision_maker=Decisions(
            AgentDecision(
                action="tool_call",
                tool_call=ToolCall(name="open_job_search", arguments={"keyword": "AI Engineer"}),
            ),
            AgentDecision(action="ask_user", message="请选择岗位。"),
        ),
        tools=MainAgentToolRegistry(),
        answer_writer=writer,
    )
    events = []

    runtime.run_turn(
        user_id="u1",
        conversation_id="c1",
        user_message="帮我找岗位",
        event_sink=events.append,
    )

    assert writer.requests == []
    assert events[-1].type == "turn_suspended"


def test_writer_failure_before_first_token_falls_back_to_grounded_draft(
    tmp_path,
) -> None:
    class FailingWriter:
        def stream(self, request):
            raise AgentWorkerError(
                "ANSWER_WRITER_TRANSPORT_ERROR",
                "failed",
                retryable=True,
            )
            yield "unreachable"

    runtime = MainAgentRuntime(
        context_manager=_manager(tmp_path),
        decision_maker=Decisions(
            AgentDecision(action="final", message="可靠的确定性回答。")
        ),
        tools=MainAgentToolRegistry(),
        answer_writer=FailingWriter(),
    )
    events = []

    result = runtime.run_turn(
        user_id="u1",
        conversation_id="c1",
        user_message="回答我",
        event_sink=events.append,
    )

    assert result.assistant_message == "可靠的确定性回答。"
    assert "".join(
        event.delta for event in events if event.type == "content_delta"
    ) == "可靠的确定性回答。"
    assert events[-1].type == "turn_completed"


def test_writer_failure_after_partial_stream_does_not_commit_or_append_fallback(
    tmp_path,
) -> None:
    class PartialWriter:
        def stream(self, request):
            yield "已经展示的部分"
            raise AgentWorkerError(
                "ANSWER_WRITER_TRANSPORT_ERROR",
                "failed",
                retryable=True,
            )

    manager = _manager(tmp_path)
    runtime = MainAgentRuntime(
        context_manager=manager,
        decision_maker=Decisions(
            AgentDecision(action="final", message="不应追加的回退回答。")
        ),
        tools=MainAgentToolRegistry(),
        answer_writer=PartialWriter(),
    )
    events = []

    with pytest.raises(AgentWorkerError):
        runtime.run_turn(
            user_id="u1",
            conversation_id="c1",
            user_message="回答我",
            event_sink=events.append,
        )

    assert [
        event.delta for event in events if event.type == "content_delta"
    ] == ["已经展示的部分"]
    assert events[-1].type == "turn_failed"
    loaded = manager.load_for_turn(
        user_id="u1", conversation_id="c1", user_message="next"
    )
    assert loaded.recent_messages == ()


def test_openai_writer_requests_provider_stream_and_yields_deltas() -> None:
    class Completions:
        def __init__(self) -> None:
            self.kwargs = None

        def create(self, **kwargs):
            self.kwargs = kwargs
            return iter(
                (
                    SimpleNamespace(
                        choices=(
                            SimpleNamespace(
                                delta=SimpleNamespace(content="第一段")
                            ),
                        )
                    ),
                    SimpleNamespace(
                        choices=(
                            SimpleNamespace(delta=SimpleNamespace(content="第二段")),
                        )
                    ),
                )
            )

    completions = Completions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    writer = OpenAIStreamingAnswerWriter(
        OpenAICompatibleAgentConfig(
            endpoint="https://example.com/v1/chat/completions",
            api_key="test",
            model="test-model",
        ),
        client=client,
    )
    request = AnswerCompositionRequest(
        response_type="general",
        user_request="解释一下",
        grounded_draft="只使用这段可靠内容。",
    )

    assert tuple(writer.stream(request)) == ("第一段", "第二段")
    assert completions.kwargs["stream"] is True
    assert "只使用这段可靠内容。" in completions.kwargs["messages"][1]["content"]
