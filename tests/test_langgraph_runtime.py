import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from career_agent.contracts import AgentRequest
from career_agent.runtime.langgraph_runtime import LangGraphRuntime


@pytest.mark.asyncio
async def test_runtime_returns_framework_neutral_result() -> None:
    model = GenericFakeChatModel(messages=iter([AIMessage("岗位匹配分析完成")]))
    runtime = LangGraphRuntime(model=model)

    result = await runtime.run(AgentRequest(message="分析这个岗位", thread_id="thread-1"))

    assert result.thread_id == "thread-1"
    assert result.output == "岗位匹配分析完成"
    assert result.metadata["runtime"] == "langgraph"
    assert result.trace_id


@pytest.mark.asyncio
async def test_runtime_reuses_checkpointed_thread_without_storing_system_messages() -> None:
    model = GenericFakeChatModel(messages=iter([AIMessage("第一轮"), AIMessage("第二轮")]))
    runtime = LangGraphRuntime(model=model)

    await runtime.run(AgentRequest(message="第一问", thread_id="same-thread"))
    result = await runtime.run(AgentRequest(message="继续追问", thread_id="same-thread"))

    assert result.output == "第二轮"
    assert result.model_messages == 4
