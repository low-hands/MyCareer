"""Local CLI entry point for smoke-testing the agent runtime.

This module only wires dependencies together — it owns no agent logic.
The ``--no-mcp`` flag lets the CLI still boot when ``boss-agent-cli`` is
unavailable (useful for contract tests and offline demos); without it,
the CLI launches the ``boss-mcp`` MCP server as a subprocess and
registers every tool it advertises into the project's
:class:`ToolRegistry` with ``READ`` permission.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from dotenv import load_dotenv

from career_agent.contracts import AgentRequest, ToolPermission
from career_agent.mcp_client import MultiServerMCPClient, load_tools
from career_agent.models import model_from_env
from career_agent.runtime.agent_loop_runtime import AgentLoopRuntime
from career_agent.tools import ToolRegistry
from career_agent.tracing import InMemoryTraceSink, TraceSink

# MCP servers sometimes use non-standard JSON Schema types.
# OpenAI API rejects "int" (must be "integer"), "float" (must be "number"), etc.
_TYPE_FIXES = {"int": "integer", "float": "number", "bool": "boolean"}


def _fix_schema_dict(schema: dict) -> bool:
    """Recursively fix type values in a JSON Schema dict. Returns True if changed."""
    changed = False
    for key, value in list(schema.items()):
        if key == "type" and isinstance(value, str) and value in _TYPE_FIXES:
            schema[key] = _TYPE_FIXES[value]
            changed = True
        elif isinstance(value, dict):
            changed |= _fix_schema_dict(value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    changed |= _fix_schema_dict(item)
    return changed


def _sanitize_tools(tools: list) -> None:
    """Fix non-standard schema types in all tools for OpenAI API compat."""
    patched: list[str] = []
    for tool in tools:
        # boss-agent-cli tools have args_schema as a raw dict, not Pydantic model
        schema = getattr(tool, "args_schema", None)
        if isinstance(schema, dict) and _fix_schema_dict(schema):
            patched.append(tool.name)
    if patched:
        print(f"  [schema-fix] patched types in: {', '.join(patched)}")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Career Agent CLI")
    parser.add_argument(
        "--no-mcp",
        action="store_true",
        help="Skip launching the boss-agent-cli MCP server (bare chat mode).",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Print tool call trace after each agent response.",
    )
    return parser.parse_args(argv)


async def _chat(no_mcp: bool, trace: bool) -> None:
    model = model_from_env("orchestrator")
    registry = ToolRegistry()

    if no_mcp:
        print("Career Agent CLI (no MCP). Type 'exit' to quit.")
        await _run_loop(model=model, registry=registry, trace_sink=InMemoryTraceSink(), show_trace=trace)
        return

    try:
        # NOTE: langchain-mcp-adapters 0.1.0+ no longer uses async with.
        # Each tool call creates its own session on the fly, so there is
        # no long-lived subprocess to manage here.
        client = MultiServerMCPClient(
            {
                "boss": {
                    "command": "uv",
                    "args": ["run", "boss-mcp"],
                    "transport": "stdio",
                },
            }
        )
        tools = await load_tools(client)
        _sanitize_tools(tools)
        for tool in tools:
            registry.register(tool, permission=ToolPermission.READ)
        print(
            f"Career Agent CLI. boss-agent online with {len(tools)} tool(s). "
            f"Type 'exit' to quit."
        )
        sink = InMemoryTraceSink()
        await _run_loop(model=model, registry=registry, trace_sink=sink, show_trace=trace)
    except Exception as exc:
        print(f"error: failed to start boss-mcp ({type(exc).__name__}: {exc})", file=sys.stderr)
        print("Hint: install with `uv add 'boss-agent-cli[mcp]'`", file=sys.stderr)
        raise SystemExit(1) from exc


async def _run_loop(
    *, model, registry: ToolRegistry,
    trace_sink: TraceSink | None = None,
    show_trace: bool = False,
) -> None:
    runtime = AgentLoopRuntime(
        model=model, tools=registry,
        trace_sink=trace_sink or InMemoryTraceSink(),
    )
    thread_id = "local-cli"
    while True:
        try:
            message = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not message:
            continue
        if message.casefold() in {"exit", "quit"}:
            break
        try:
            result = await runtime.run(
                AgentRequest(
                    message=message,
                    thread_id=thread_id,
                    allowed_permissions={ToolPermission.READ},
                )
            )
        except Exception as exc:
            print(f"agent error: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        print(f"agent> {result.output}")
        if show_trace and trace_sink:
            run_trace = trace_sink.traces.get(result.trace_id)
            if run_trace:
                tool_events = [
                    e for e in run_trace.events
                    if e.name in ("tool.started", "tool.completed")
                ]
                if tool_events:
                    steps = []
                    for e in tool_events:
                        if e.name == "tool.started":
                            steps.append(f"  → {e.attributes.get('tool', '?')} (step {e.attributes.get('step', '?')})")
                    print(f"  [trace] {' | '.join(steps)}")


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    load_dotenv(override=True)
    asyncio.run(_chat(no_mcp=args.no_mcp, trace=args.trace))


if __name__ == "__main__":
    main()
