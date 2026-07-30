from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from pydantic import Field

from career_agent.contracts import AgentRequest, ToolPermission
from career_agent.jobs.grounding_guard import JobGroundingGuard
from career_agent.runtime.agent_loop_runtime import AgentLoopRuntime
from career_agent.tools import ToolRegistry


class ScriptedToolModel(BaseChatModel):
    """A deterministic model that still exercises bind_tools and the real loop."""

    responses: list[AIMessage]
    response_index: int = 0
    bound_tool_names: list[str] = Field(default_factory=list)
    seen_messages: list[list[AnyMessage]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted-tool-model"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ScriptedToolModel":
        self.bound_tool_names = [item.name for item in tools]
        return self

    def _generate(
        self,
        messages: list[AnyMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.seen_messages.append(list(messages))
        response = self.responses[self.response_index]
        self.response_index += 1
        return ChatResult(generations=[ChatGeneration(message=response)])


@tool
def get_resume(candidate_id: str) -> dict[str, Any]:
    """Load a parsed resume by candidate ID."""

    return {"candidate_id": candidate_id, "skills": ["Python", "FastAPI"]}


@pytest.mark.asyncio
async def test_agent_loop_observes_tool_result_before_answering() -> None:
    model = ScriptedToolModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_resume",
                        "args": {"candidate_id": "candidate-1"},
                        "id": "call-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="候选人具备 Python 和 FastAPI 技能。"),
        ]
    )
    registry = ToolRegistry()
    registry.register(get_resume, permission=ToolPermission.READ)
    runtime = AgentLoopRuntime(model=model, tools=registry)

    result = await runtime.run(AgentRequest(message="分析 candidate-1 的技术背景", thread_id="tool-thread"))

    assert model.bound_tool_names == ["get_resume"]
    assert result.output == "候选人具备 Python 和 FastAPI 技能。"
    assert result.model_messages == 2
    assert result.tool_calls == 1
    assert result.metadata == {"runtime": "agent_loop", "steps": 2}
    observation = model.seen_messages[1][-1]
    assert isinstance(observation, ToolMessage)
    assert '"skills": ["Python", "FastAPI"]' in str(observation.content)


@pytest.mark.asyncio
async def test_disallowed_tool_is_not_exposed_to_model() -> None:
    model = ScriptedToolModel(responses=[AIMessage(content="请先确认是否允许投递。")])
    registry = ToolRegistry()
    registry.register(get_resume, permission=ToolPermission.EXTERNAL)
    runtime = AgentLoopRuntime(model=model, tools=registry)

    result = await runtime.run(AgentRequest(message="读取简历"))

    assert model.bound_tool_names == []
    assert result.tool_calls == 0


@tool("career_search_jobs")
def search_jobs(query: str) -> str:
    """Search grounded job postings."""

    return '{"status":"no_results","message":"No grounded jobs found."}'


@pytest.mark.asyncio
async def test_response_guard_retries_ungrounded_answer_then_accepts_tool_backed_answer() -> None:
    model = ScriptedToolModel(
        responses=[
            AIMessage(content="我推荐三家公司的 Agent 岗位。"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "career_search_jobs",
                        "args": {"query": "Agent"},
                        "id": "search-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="没有检索到符合条件的真实岗位。"),
        ]
    )
    registry = ToolRegistry()
    registry.register(search_jobs, permission=ToolPermission.READ)
    runtime = AgentLoopRuntime(
        model=model,
        tools=registry,
        response_guard=JobGroundingGuard(),
    )

    result = await runtime.run(
        AgentRequest(
            message="帮我搜索 Agent 开发岗位",
            thread_id="grounded-search",
        )
    )

    assert result.output == "没有检索到符合条件的真实岗位。"
    assert result.model_messages == 3
    assert result.tool_calls == 1
    assert result.metadata["guard_rejections"] == 1


@pytest.mark.asyncio
async def test_response_guard_returns_safe_output_after_retry_budget_is_exhausted() -> None:
    model = ScriptedToolModel(
        responses=[
            AIMessage(content="这是我记忆中的岗位。"),
            AIMessage(content="我还是直接给出一个岗位。"),
        ]
    )
    registry = ToolRegistry()
    registry.register(search_jobs, permission=ToolPermission.READ)
    runtime = AgentLoopRuntime(
        model=model,
        tools=registry,
        response_guard=JobGroundingGuard(),
        max_guard_retries=1,
    )

    result = await runtime.run(
        AgentRequest(
            message="帮我搜索北京的 Agent 岗位",
            thread_id="blocked-search",
        )
    )

    assert "无法在没有岗位工具证据的情况下回答" in result.output
    assert result.tool_calls == 0
    assert result.metadata["guard_blocked"] is True
