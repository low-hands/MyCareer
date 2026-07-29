"""MCP client adapter built on top of LangChain's official MCP adapters.

We deliberately do not re-implement the MCP protocol, JSON Schema
conversion, or tool error handling here — the official
``langchain-mcp-adapters`` package handles all of that and is what the
wider LangGraph ecosystem (DeerFlow, Open Deep Research, etc.) has
converged on. This module is a thin project-specific facade that:

* exposes one configuration type (:data:`MCPServerConfig`) so callers do
  not import the official package directly;
* provides a :func:`load_tools` helper that fits the async style used by
  the rest of this project;
* reserves a hook point for future concerns like DeerFlow-style session
  pooling or trace events without leaking the upstream API.

The official package already supports stdio / SSE / streamable HTTP
transports, dynamic tool discovery, JSON-Schema-to-Pydantic conversion,
and graceful tool-error handling (errors become ``ToolMessage`` so the
agent can self-correct instead of the run crashing). We inherit all of
that for free.
"""

from __future__ import annotations

from typing import TypedDict

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient


class MCPServerConfig(TypedDict, total=False):
    """One MCP server's connection parameters.

    Matches the dict shape accepted by :class:`MultiServerMCPClient`:

    * stdio transport: ``{"command": "uv", "args": ["run", "boss-mcp"], "transport": "stdio"}``
    * HTTP / streamable HTTP: ``{"url": "http://.../mcp", "transport": "http"}``
    * SSE: ``{"url": "http://.../sse", "transport": "sse"}``

    ``transport`` defaults to ``"stdio"`` when ``command`` is present and
    to ``"http"`` when ``url`` is present; the upstream package fills in
    the default.
    """

    command: str
    args: list[str]
    url: str
    transport: str
    env: dict[str, str]
    headers: dict[str, str]


async def load_tools(client: MultiServerMCPClient) -> list[BaseTool]:
    """Discover every tool on every connected server.

    The returned :class:`BaseTool` instances can be registered into
    :class:`career_agent.tools.ToolRegistry` like any other LangChain
    tool. Tool errors are returned to the model as ``ToolMessage``
    rather than raised — matching the policy used by ``ToolRegistry``
    itself.
    """
    return list(await client.get_tools())


__all__ = ["MCPServerConfig", "MultiServerMCPClient", "load_tools"]
