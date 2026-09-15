"""Production startup settles the receipts a dead process left ``RUNNING``.

``tests/storage/test_turn_receipts.py`` proves ``fail_orphaned_running`` does
the right thing when called; ``tests/agent/test_main_agent_runtime.py`` proves
what a retry then does per ledger state. Neither proves production *calls* it.
Constructing the real runtime needs the whole dependency graph, so this pins
the wiring structurally, the way the trace recorder guard does:

    ``build_main_agent_runtime`` calls ``fail_orphaned_running`` on the
    ``SQLiteTurnReceiptStore`` it hands to ``MainAgentRuntime``, and both
    callers of the builder take the workspace lock before calling it.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

from career_agent import cli as cli_module


def _builder() -> ast.FunctionDef:
    tree = ast.parse(Path(inspect.getsourcefile(cli_module)).read_text())
    builder = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "build_main_agent_runtime"
    )
    return builder


def _method_calls(node: ast.AST, method: str) -> list[ast.Call]:
    return [
        call
        for call in ast.walk(node)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == method
    ]


def test_the_production_builder_recovers_orphaned_receipts_on_the_store_it_wires() -> None:
    builder = _builder()

    recoveries = _method_calls(builder, "fail_orphaned_running")
    assert len(recoveries) == 1, "startup must settle orphaned RUNNING receipts exactly once"
    recovered_on = recoveries[0].func.value
    assert isinstance(recovered_on, ast.Name)

    runtime_call = next(
        call
        for call in ast.walk(builder)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "MainAgentRuntime"
    )
    wired = next(
        keyword.value
        for keyword in runtime_call.keywords
        if keyword.arg == "turn_receipt_store"
    )
    assert isinstance(wired, ast.Name) and wired.id == recovered_on.id, (
        "the store that is recovered must be the store the runtime uses"
    )

    assigned = next(
        node
        for node in ast.walk(builder)
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == recovered_on.id for t in node.targets)
    )
    assert isinstance(assigned.value, ast.Call)
    assert isinstance(assigned.value.func, ast.Name)
    assert assigned.value.func.id == "SQLiteTurnReceiptStore"
