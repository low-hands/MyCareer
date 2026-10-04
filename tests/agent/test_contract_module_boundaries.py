"""Architecture checks for the decomposed Main Agent contracts."""

from __future__ import annotations

import ast
from pathlib import Path


COMPATIBILITY_MODULE = "career_agent.agent.contracts.main_agent"


def test_production_code_does_not_import_the_compatibility_facade() -> None:
    source_root = Path(__file__).parents[2] / "src" / "career_agent"
    violations: list[str] = []

    for path in source_root.rglob("*.py"):
        if path.name == "main_agent.py" and path.parent.name == "contracts":
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        if any(
            isinstance(node, ast.ImportFrom)
            and node.module == COMPATIBILITY_MODULE
            for node in ast.walk(tree)
        ):
            violations.append(str(path.relative_to(source_root)))

    assert violations == []


def test_compatibility_facade_still_reexports_public_contracts() -> None:
    from career_agent.agent.contracts.main_agent import (
        AgentDecision,
        ConversationTaskState,
        MainAgentContext,
        ToolObservation,
    )

    assert AgentDecision.__module__.endswith(".decisions")
    assert ConversationTaskState.__module__.endswith(".task_state")
    assert MainAgentContext.__module__.endswith(".context")
    assert ToolObservation.__module__.endswith(".observations")
