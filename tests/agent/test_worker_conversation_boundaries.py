"""Workers return results; only the Main Agent writes the conversation and stream.

A worker (a workflow or a specialist model provider) hands back a
``ToolResult``-shaped value. Persisting messages, compacting context and
pushing events to the web UI belong to the Main Agent runtime, so no worker
module may import the modules that can do those things.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


SOURCE_ROOT = Path(__file__).parents[2] / "src"
PACKAGE_ROOT = SOURCE_ROOT / "career_agent"

# Specialist providers that run one bounded model task for a capability.
# ``conversation_summary`` reads conversation content by design, but only as
# arguments from ``ContextManager``, which persists the summary it returns; it
# is held to the same rule as every other worker.
WORKER_PROVIDERS = frozenset(
    {
        "agent_worker",
        "conversation_summary",
        "email_tracking",
        "interview_preparation",
        "job_analysis",
        "resume_job_match",
        "resume_transcription",
        "structured_response_retry",
        "structured_responses",
    }
)
# The Main Agent's own model adapter and the plumbing every provider shares.
NON_WORKER_PROVIDERS = frozenset(
    {
        "__init__",
        "interaction_output",
        "main_agent",
        "main_agent_vendors",
        "openai_client",
        "tiktoken_assets",
        "token_budget",
    }
)

# A module listed here is forbidden together with all of its submodules.
FORBIDDEN_MODULES = {
    # Conversation state: messages, summaries, sessions, turn context.
    "career_agent.agent.context.manager": "conversation context manager",
    "career_agent.agent.context.session_manager": "conversation sessions",
    "career_agent.agent.context.turn_builder": "Main Agent turn context",
    "career_agent.storage.context": "conversation message store",
    "career_agent.storage.turn_receipts": "stored turn events",
    "career_agent.storage.checkpoints": "Main Agent graph checkpoints",
    # The Main Agent runtime: turn lifecycle, STREAM_SINK, RuntimeObservability.
    "career_agent.agent.runtime": "Main Agent runtime",
    "career_agent.harness.agent_loop": "Main Agent loop",
    "career_agent.harness.turn_router": "Main Agent loop",
    "career_agent.harness.confirmation_coordinator": "Main Agent loop",
    # What the user sees and how it reaches them.
    "career_agent.agent.presentation": "delivery layer",
    "career_agent.harness.streaming": "UI stream events",
    "career_agent.api": "HTTP and live turn streams",
}
# Deliberately allowed: ``harness.capability_steps`` (the stream-agnostic step
# observer the runtime installs around a tool call), ``harness.observability``
# (telemetry traces, never shown to the user) and ``agent.context.deployment``
# (context-window configuration).


def _module_name(path: Path) -> str:
    parts = list(path.relative_to(SOURCE_ROOT).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _imported_modules(tree: ast.AST, *, module: str, is_package: bool) -> set[str]:
    package = module if is_package else module.rpartition(".")[0]
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                anchor = package.split(".")[: len(package.split(".")) - node.level + 1]
                base = ".".join([*anchor, base] if base else anchor)
            imported.add(base)
            # ``from career_agent.agent import runtime`` imports a module too.
            imported.update(f"{base}.{alias.name}" for alias in node.names)
    return imported


def _forbidden_imports(
    source: str, *, module: str, is_package: bool = False
) -> list[str]:
    tree = ast.parse(source)
    return sorted(
        f"{imported} ({reason})"
        for imported in _imported_modules(tree, module=module, is_package=is_package)
        for forbidden, reason in FORBIDDEN_MODULES.items()
        if imported == forbidden or imported.startswith(forbidden + ".")
    )


def _worker_paths() -> list[Path]:
    workflows = sorted((PACKAGE_ROOT / "agent" / "workflows").rglob("*.py"))
    providers = [
        PACKAGE_ROOT / "agent" / "providers" / f"{name}.py"
        for name in sorted(WORKER_PROVIDERS)
    ]
    return workflows + providers


def test_every_provider_is_classified_as_worker_or_not() -> None:
    providers = {
        path.stem for path in (PACKAGE_ROOT / "agent" / "providers").glob("*.py")
    }

    assert WORKER_PROVIDERS.isdisjoint(NON_WORKER_PROVIDERS)
    assert providers == WORKER_PROVIDERS | NON_WORKER_PROVIDERS


def test_forbidden_modules_exist() -> None:
    known = {_module_name(path) for path in PACKAGE_ROOT.rglob("*.py")}

    assert [module for module in FORBIDDEN_MODULES if module not in known] == []


def test_workers_do_not_import_conversation_or_stream_owners() -> None:
    violations: dict[str, list[str]] = {}

    for path in _worker_paths():
        found = _forbidden_imports(
            path.read_text(encoding="utf-8"),
            module=_module_name(path),
            is_package=path.name == "__init__.py",
        )
        if found:
            violations[path.relative_to(PACKAGE_ROOT).as_posix()] = found

    assert violations == {}


WORKER = "career_agent.agent.workflows.job_research.worker"


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            "from career_agent.agent.context.manager import ContextManager",
            "career_agent.agent.context.manager",
        ),
        (
            "from career_agent.agent.runtime.turn_coordinator import STREAM_SINK",
            "career_agent.agent.runtime.turn_coordinator",
        ),
        (
            "from career_agent.agent.runtime.observability import RuntimeObservability",
            "career_agent.agent.runtime.observability",
        ),
        (
            "import career_agent.harness.streaming as streaming",
            "career_agent.harness.streaming",
        ),
        (
            "from career_agent.storage import context",
            "career_agent.storage.context",
        ),
        (
            "from career_agent.agent import runtime",
            "career_agent.agent.runtime",
        ),
        (
            "from ...presentation.stream_adapter import StreamAdapter",
            "career_agent.agent.presentation.stream_adapter",
        ),
        (
            "from typing import TYPE_CHECKING\n"
            "if TYPE_CHECKING:\n"
            "    from career_agent.api.live_turns import LiveTurn\n",
            "career_agent.api.live_turns",
        ),
        (
            "def run():\n"
            "    from career_agent.storage.turn_receipts import SQLiteTurnReceiptStore\n",
            "career_agent.storage.turn_receipts",
        ),
    ],
)
def test_boundary_check_catches_violations(source: str, expected: str) -> None:
    found = _forbidden_imports(source, module=WORKER)

    assert any(entry.startswith(f"{expected} (") for entry in found), found


@pytest.mark.parametrize(
    "source",
    [
        "from career_agent.harness.capability_steps import notify_capability_step",
        "from career_agent.harness.observability import traced_model_call",
        "from career_agent.agent.context.deployment import DEFAULT_SUMMARY_BATCH_SIZE",
        "from career_agent.agent.contracts.observations import ToolResult",
        "from career_agent.storage.api_keys import ApiKeyStore",
        "from .contracts import JobResearchRequest",
    ],
)
def test_boundary_check_allows_result_side_imports(source: str) -> None:
    assert _forbidden_imports(source, module=WORKER) == []
