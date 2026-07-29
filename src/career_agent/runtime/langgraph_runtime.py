from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.prebuilt import ToolNode

from career_agent.context import BasicContextManager, ContextManager
from career_agent.contracts import AgentRequest, AgentResult
from career_agent.tools import ToolRegistry
from career_agent.tracing import InMemoryTraceSink, TraceSink


class _GraphState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


class LangGraphRuntime:
    """A small ReAct-style loop behind the framework-neutral runtime boundary."""

    def __init__(
        self,
        *,
        model: BaseChatModel,
        tools: ToolRegistry | None = None,
        context_manager: ContextManager | None = None,
        trace_sink: TraceSink | None = None,
        max_steps: int = 20,
    ) -> None:
        if max_steps < 2:
            raise ValueError("max_steps must be at least 2")
        self._model = model
        self._tools = tools or ToolRegistry()
        self._context_manager = context_manager or BasicContextManager()
        self._trace_sink = trace_sink or InMemoryTraceSink()
        self._max_steps = max_steps
        self._checkpointer = InMemorySaver()

    def _build_graph(self, allowed_tools: list[Any], system_text: str):
        bound_model = self._model.bind_tools(allowed_tools) if allowed_tools else self._model

        async def call_model(state: _GraphState) -> dict[str, list[AIMessage]]:
            # System context is injected for every model call but is not stored in
            # thread history. This prevents duplicate/mid-conversation system
            # messages when a checkpointed thread receives another user turn.
            response = await bound_model.ainvoke([SystemMessage(system_text), *state["messages"]])
            return {"messages": [response]}

        def route(state: _GraphState) -> str:
            last_message = state["messages"][-1]
            if isinstance(last_message, AIMessage) and last_message.tool_calls:
                return "tools"
            return END

        builder = StateGraph(_GraphState)
        builder.add_node("agent", call_model)
        builder.add_edge(START, "agent")

        if allowed_tools:
            builder.add_node("tools", ToolNode(allowed_tools))
            builder.add_conditional_edges("agent", route, {"tools": "tools", END: END})
            builder.add_edge("tools", "agent")
        else:
            builder.add_edge("agent", END)

        return builder.compile(checkpointer=self._checkpointer)

    async def run(self, request: AgentRequest) -> AgentResult:
        trace = self._trace_sink.start(request.thread_id)
        try:
            context = await self._context_manager.prepare(request)
            allowed_tools = self._tools.allowed(request.allowed_permissions)
            self._trace_sink.event(
                trace,
                "run.started",
                model=self._model.__class__.__name__,
                allowed_tools=[tool.name for tool in allowed_tools],
            )

            graph = self._build_graph(allowed_tools, context.system_prompt)
            result = await graph.ainvoke(
                {"messages": [HumanMessage(request.message)]},
                config={
                    "configurable": {"thread_id": request.thread_id},
                    "recursion_limit": self._max_steps,
                },
            )
            messages = result["messages"]
            tool_calls = sum(len(message.tool_calls) for message in messages if isinstance(message, AIMessage))
            output = self._last_text(messages)
            self._trace_sink.event(
                trace,
                "run.completed",
                model_messages=len(messages),
                tool_calls=tool_calls,
            )
            self._trace_sink.finish(trace)
            return AgentResult(
                thread_id=request.thread_id,
                trace_id=trace.trace_id,
                output=output,
                model_messages=len(messages),
                tool_calls=tool_calls,
                metadata={"runtime": "langgraph"},
            )
        except Exception as exc:
            self._trace_sink.event(trace, "run.failed", error_type=type(exc).__name__)
            self._trace_sink.finish(trace, exc)
            raise

    @staticmethod
    def _last_text(messages: list[AnyMessage]) -> str:
        for message in reversed(messages):
            if not isinstance(message, AIMessage):
                continue
            if isinstance(message.content, str):
                return message.content
            text_parts = [
                str(block["text"])
                for block in message.content
                if isinstance(block, dict) and block.get("type") == "text"
            ]
            return "\n".join(text_parts)
        return ""
