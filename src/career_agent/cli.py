"""Local CLI entry point for smoke-testing the agent runtime.

This module only wires dependencies together — it owns no agent logic.
The ``--no-mcp`` flag lets the CLI still boot when ``boss-agent-cli`` is
unavailable (useful for contract tests and offline demos). Without it,
the CLI uses selected read-only Boss MCP tools behind the project's
request-scoped Career tools; raw provider tools are never model-visible.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import shlex
import sys
from pathlib import Path

from dotenv import load_dotenv

from career_agent.contracts import AgentRequest, ToolPermission
from career_agent.jobs.agent_tools import JobDiscoveryToolContext
from career_agent.jobs.discovery_service import JobDiscoveryService
from career_agent.jobs.grounding_guard import JobGroundingGuard
from career_agent.jobs.wiring import (
    build_job_discovery_registry,
    build_job_discovery_service,
)
from career_agent.mcp_client import MultiServerMCPClient, load_tools
from career_agent.models import model_from_env
from career_agent.resumes.repository import DuplicateResumeError, ResumeNotFoundError
from career_agent.resumes.service import ResumePoolService
from career_agent.resumes.sqlite_repo import SQLiteResumeVersionRepository
from career_agent.runtime.agent_loop_runtime import AgentLoopRuntime
from career_agent.state import InMemoryConversationStore
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
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a resume file (PDF or text) to load at startup.",
    )
    parser.add_argument(
        "--resume-role",
        type=str,
        default="未分类",
        help="Role pool for --resume (default: 未分类).",
    )
    parser.add_argument(
        "--tenant-id",
        type=str,
        default=os.getenv("CAREER_AGENT_TENANT_ID", "local"),
        help="Tenant scope for local data (default: CAREER_AGENT_TENANT_ID or local).",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=os.getenv("CAREER_AGENT_DATA_DIR", "~/.career-agent"),
        help="Persistent data directory (default: CAREER_AGENT_DATA_DIR or ~/.career-agent).",
    )
    return parser.parse_args(argv)


async def _chat(
    no_mcp: bool,
    trace: bool,
    resume_path: str | None = None,
    resume_role: str = "未分类",
    tenant_id: str = "local",
    data_dir: str = "~/.career-agent",
) -> None:
    model = model_from_env("orchestrator")
    resume_model = model_from_env("resume")
    resolved_data_dir = Path(data_dir).expanduser().resolve()
    database_path = resolved_data_dir / "career-agent.db"
    resume_service = ResumePoolService(
        repository=SQLiteResumeVersionRepository(database_path),
        parser_model=resume_model,
        storage_dir=resolved_data_dir / "resumes",
    )

    # Load resume if provided
    if resume_path:
        try:
            print(f"Loading resume from {resume_path}...")
            created = await resume_service.add_version(
                tenant_id=tenant_id,
                role_type=resume_role,
                file_path=resume_path,
            )
            resume_service.use_version(tenant_id, created.version_id)
            print(
                f"  [resume] [{created.version_id[:8]}] {created.display_name} | "
                f"{created.parsed_data.name or 'Unknown'}"
            )
        except DuplicateResumeError as exc:
            print(f"  [resume] duplicate: {exc}", file=sys.stderr)
        except Exception as exc:
            print(f"  [resume] warning: failed to parse ({type(exc).__name__}: {exc})", file=sys.stderr)

    if no_mcp:
        print("Career Agent CLI (no MCP). Type 'exit' to quit.")
        await _run_loop(
            model=model,
            resume_model=resume_model,
            job_discovery_service=None,
            resume_service=resume_service,
            tenant_id=tenant_id,
            trace_sink=InMemoryTraceSink(),
            show_trace=trace,
        )
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
        job_discovery_service = build_job_discovery_service(
            mcp_tools=tools,
            database_path=database_path,
        )
        print("Career Agent CLI. Boss MCP online behind 2 Career tools. Type 'exit' to quit.")
        sink = InMemoryTraceSink()
        await _run_loop(
            model=model,
            resume_model=resume_model,
            job_discovery_service=job_discovery_service,
            resume_service=resume_service,
            tenant_id=tenant_id,
            trace_sink=sink,
            show_trace=trace,
        )
    except Exception as exc:
        print(f"error: failed to start boss-mcp ({type(exc).__name__}: {exc})", file=sys.stderr)
        print("Hint: install with `uv add 'boss-agent-cli[mcp]'`", file=sys.stderr)
        raise SystemExit(1) from exc


async def _run_loop(
    *,
    model,
    resume_model,
    job_discovery_service: JobDiscoveryService | None,
    resume_service: ResumePoolService,
    tenant_id: str,
    trace_sink: TraceSink | None = None,
    show_trace: bool = False,
) -> None:
    sink = trace_sink or InMemoryTraceSink()
    conversation_store = InMemoryConversationStore()
    response_guard = JobGroundingGuard() if job_discovery_service is not None else None
    thread_id = f"local-cli:{tenant_id}"
    active_version = resume_service.get_active_version(tenant_id)
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

        # Handle /resume commands
        if message.lower().startswith("/resume"):
            try:
                parts = shlex.split(message)
            except ValueError as exc:
                print(f"  [resume] invalid command: {exc}", file=sys.stderr)
                continue
            cmd = parts[1].casefold() if len(parts) > 1 else "help"

            if cmd == "list":
                role_type = parts[2] if len(parts) > 2 else None
                versions = resume_service.list_versions(tenant_id, role_type)
                _print_resume_versions(versions, active_version)
                continue

            if cmd == "use":
                selector = parts[2] if len(parts) > 2 else ""
                try:
                    active_version = resume_service.use_version(tenant_id, selector)
                    print(f"  [resume] using [{active_version.version_id[:8]}] {active_version.display_name}")
                except ResumeNotFoundError as exc:
                    print(f"  [resume] {exc}", file=sys.stderr)
                continue

            if cmd == "show":
                selector = parts[2] if len(parts) > 2 else ""
                try:
                    selected = resume_service.resolve_version(tenant_id, selector)
                    print(
                        f"  [{selected.version_id[:8]}] {selected.display_name}\n"
                        f"  source: {selected.original_filename}\n"
                        f"{selected.parsed_data.model_dump_json(indent=2)}"
                    )
                except ResumeNotFoundError as exc:
                    print(f"  [resume] {exc}", file=sys.stderr)
                continue

            if cmd == "add":
                if len(parts) < 4:
                    _print_resume_help()
                    continue
                role_type = parts[2]
                file_path = parts[3]
                note = parts[4] if len(parts) > 4 else ""
                try:
                    print(f"Loading resume from {file_path}...")
                    created = await resume_service.add_version(
                        tenant_id=tenant_id,
                        role_type=role_type,
                        file_path=file_path,
                        note=note,
                    )
                    active_version = resume_service.use_version(
                        tenant_id,
                        created.version_id,
                    )
                    print(
                        f"  [resume] added [{created.version_id[:8]}] "
                        f"{created.display_name} | "
                        f"{created.parsed_data.name or 'Unknown'}"
                    )
                except DuplicateResumeError as exc:
                    print(f"  [resume] duplicate: {exc}", file=sys.stderr)
                except Exception as exc:
                    print(
                        f"  [resume] error: {type(exc).__name__}: {exc}",
                        file=sys.stderr,
                    )
                continue

            _print_resume_help()
            continue

        # Build facts from resume for prompt injection
        facts = active_version.parsed_data.to_facts() if active_version else []
        request_metadata: dict[str, object] = {"facts": facts}
        if active_version:
            request_metadata.update(
                {
                    "resume_version_id": active_version.version_id,
                    "resume_role_type": active_version.role_type,
                }
            )
        registry = (
            build_job_discovery_registry(
                service=job_discovery_service,
                context=JobDiscoveryToolContext(
                    tenant_id=tenant_id,
                    thread_id=thread_id,
                    user_query=message,
                ),
            )
            if job_discovery_service is not None
            else ToolRegistry()
        )
        runtime = AgentLoopRuntime(
            model=model,
            tools=registry,
            state_store=conversation_store,
            trace_sink=sink,
            response_guard=response_guard,
        )
        try:
            result = await runtime.run(
                AgentRequest(
                    message=message,
                    thread_id=thread_id,
                    allowed_permissions={ToolPermission.READ},
                    metadata=request_metadata,
                )
            )
        except Exception as exc:
            print(f"agent error: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        print(f"agent> {result.output}")
        if show_trace and trace_sink:
            run_trace = trace_sink.traces.get(result.trace_id)
            if run_trace:
                tool_events = [e for e in run_trace.events if e.name in ("tool.started", "tool.completed")]
                if tool_events:
                    steps = []
                    for e in tool_events:
                        if e.name == "tool.started":
                            steps.append(f"  → {e.attributes.get('tool', '?')} (step {e.attributes.get('step', '?')})")
                    print(f"  [trace] {' | '.join(steps)}")


def _print_resume_versions(versions, active_version) -> None:
    if not versions:
        print("  [resume] no resume versions found.")
        return

    current_role: str | None = None
    for version in versions:
        if version.role_type != current_role:
            current_role = version.role_type
            print(f"  {current_role}")
        active = " *" if active_version and version.version_id == active_version.version_id else ""
        note = f" · {version.note}" if version.note else ""
        print(f"    [{version.version_id[:8]}]{active} v{version.version_number}{note} · {version.original_filename}")


def _print_resume_help() -> None:
    print(
        "Usage:\n"
        '  /resume add "<role>" "<path>" ["note"]\n'
        '  /resume list ["role"]\n'
        "  /resume show <version-id>\n"
        "  /resume use <version-id>"
    )


def main(argv: list[str] | None = None) -> None:
    load_dotenv(override=True)
    args = _parse_args(argv)
    asyncio.run(
        _chat(
            no_mcp=args.no_mcp,
            trace=args.trace,
            resume_path=args.resume,
            resume_role=args.resume_role,
            tenant_id=args.tenant_id,
            data_dir=args.data_dir,
        )
    )


if __name__ == "__main__":
    main()
