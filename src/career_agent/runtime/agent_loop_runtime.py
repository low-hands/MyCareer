"""Framework-independent function-calling Agent Loop."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from career_agent.context import BasicContextManager, ContextManager
from career_agent.contracts import AgentRequest, AgentResult
from career_agent.state import ConversationStore, InMemoryConversationStore
from career_agent.tools import ToolRegistry
from career_agent.tracing import InMemoryTraceSink, RunTrace, TraceSink


class AgentLoopLimitError(RuntimeError):
    """Raised when the model does not reach a terminal answer within its budget."""


class AgentLoopRuntime:
    """Owns the observe-decide-act loop without delegating it to LangGraph.

    One iteration is:

    1. send the goal, history, context, and tool schemas to the model;
    2. receive either a final answer or function calls;
    3. validate and execute requested tools;
    4. append observations and let the model decide again.
    """

    def __init__(
        self,
        *,
        model: BaseChatModel,
        tools: ToolRegistry | None = None,
        context_manager: ContextManager | None = None,
        state_store: ConversationStore | None = None,
        trace_sink: TraceSink | None = None,
        max_steps: int = 12,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        self._model = model
        self._tools = tools or ToolRegistry()
        self._context_manager = context_manager or BasicContextManager()
        self._state_store = state_store or InMemoryConversationStore()
        self._trace_sink = trace_sink or InMemoryTraceSink()
        self._max_steps = max_steps
        self._thread_locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    async def run(self, request: AgentRequest) -> AgentResult:
        async with self._thread_locks[request.thread_id]:
            return await self._run_locked(request)

    async def _run_locked(self, request: AgentRequest) -> AgentResult:
        trace = self._trace_sink.start(request.thread_id)
        history = await self._state_store.load(request.thread_id)
        history.append(HumanMessage(request.message))
        tool_call_count = 0
        model_call_count = 0

        try:
            context = await self._context_manager.prepare(request)
            allowed_tools = self._tools.allowed(request.allowed_permissions)
            model = self._model.bind_tools(allowed_tools) if allowed_tools else self._model
            self._trace_sink.event(
                trace,
                "loop.started",
                max_steps=self._max_steps,
                allowed_tools=[tool.name for tool in allowed_tools],
            )

            for step in range(1, self._max_steps + 1):
                self._trace_sink.event(trace, "model.requested", step=step)
                response = await model.ainvoke([SystemMessage(context.system_prompt), *history])
                if not isinstance(response, AIMessage):
                    raise TypeError(f"Chat model returned {type(response).__name__}, expected AIMessage")

                model_call_count += 1
                history.append(response)
                self._trace_sink.event(
                    trace,
                    "model.responded",
                    step=step,
                    requested_tools=[tool_call["name"] for tool_call in response.tool_calls],
                )

                if not response.tool_calls:
                    await self._state_store.save(request.thread_id, history)
                    self._trace_sink.event(
                        trace,
                        "loop.completed",
                        steps=step,
                        tool_calls=tool_call_count,
                    )
                    self._trace_sink.finish(trace)
                    return AgentResult(
                        thread_id=request.thread_id,
                        trace_id=trace.trace_id,
                        output=self._message_text(response),
                        model_messages=model_call_count,
                        tool_calls=tool_call_count,
                        metadata={"runtime": "agent_loop", "steps": step},
                    )

                for tool_call in response.tool_calls:
                    observation = await self._execute_tool_call(
                        tool_call=tool_call,
                        request=request,
                        trace=trace,
                        step=step,
                    )
                    history.append(observation)
                    tool_call_count += 1

                await self._state_store.save(request.thread_id, history)

            raise AgentLoopLimitError(f"Agent exceeded its {self._max_steps}-step execution budget")
        except Exception as exc:
            await self._state_store.save(request.thread_id, history)
            self._trace_sink.event(trace, "loop.failed", error_type=type(exc).__name__)
            self._trace_sink.finish(trace, exc)
            raise

    async def _execute_tool_call(
        self,
        *,
        tool_call: dict[str, Any],
        request: AgentRequest,
        trace: RunTrace,
        step: int,
    ) -> ToolMessage:
        name = tool_call["name"]
        arguments = tool_call.get("args") or {}
        call_id = tool_call.get("id") or f"{name}-{step}"
        self._trace_sink.event(
            trace,
            "tool.started",
            step=step,
            tool=name,
            call_id=call_id,
        )
        result = await self._tools.execute(
            name=name,
            arguments=arguments,
            permissions=request.allowed_permissions,
        )
        self._trace_sink.event(
            trace,
            "tool.completed",
            step=step,
            tool=name,
            call_id=call_id,
        )
        return ToolMessage(content=result, tool_call_id=call_id, name=name)

    @staticmethod
    def _message_text(message: AIMessage) -> str:
        if isinstance(message.content, str):
            return message.content
        return "\n".join(
            str(block["text"]) for block in message.content if isinstance(block, dict) and block.get("type") == "text"
        )
