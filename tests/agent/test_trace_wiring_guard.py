"""A structural guard: production really sends a trace recorder into the runtime.

``test_trace_recording`` proves the runtime *can* record once handed a recorder.
It does not prove production *does* hand one — that test injects its own. That
gap is the exact death the component already died once: it was fully implemented
and unit-tested while nothing in ``src/`` imported it, and no test noticed.

This test scans the wiring module rather than constructing the whole runtime
(which needs a live dependency graph). The invariant it pins is structural:

    a ``SQLiteTraceRecorder`` is built and passed to ``MainAgentRuntime``
    as ``trace_recorder=`` in the production wiring path.

Removing that wiring must fail here, not silently a year later.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

from career_agent.agent.main_agent_runtime import MainAgentRuntime
from career_agent import cli as cli_module


def _wiring_source() -> str:
    # Located from the module object so moving or renaming the file fails loudly
    # instead of silently scanning nothing.
    path = Path(inspect.getsourcefile(cli_module))
    return path.read_text()


def test_production_wiring_builds_and_passes_a_sqlite_trace_recorder() -> None:
    tree = ast.parse(_wiring_source())
    builder = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "build_main_agent_runtime"
        ),
        None,
    )
    assert builder is not None, "production Main Agent builder is missing"

    runtime_calls = [
        node
        for node in ast.walk(builder)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "MainAgentRuntime"
    ]
    assert len(runtime_calls) == 1, (
        "production wiring must have one unambiguous MainAgentRuntime construction"
    )
    trace_keyword = next(
        (
            keyword.value
            for keyword in runtime_calls[0].keywords
            if keyword.arg == "trace_recorder"
        ),
        None,
    )
    assert isinstance(trace_keyword, ast.Call), (
        "production MainAgentRuntime no longer receives a concrete trace recorder"
    )
    assert isinstance(trace_keyword.func, ast.Name)
    assert trace_keyword.func.id == "SQLiteTraceRecorder", (
        "production MainAgentRuntime must receive SQLiteTraceRecorder, not a Noop"
    )


def _runtime_module_source() -> str:
    path = Path(inspect.getsourcefile(MainAgentRuntime))
    return path.read_text()


def test_the_runtime_accepts_a_trace_recorder_parameter() -> None:
    # The wiring passes a keyword the runtime must actually accept, or the two
    # sides drift and production still silently records nothing.
    params = inspect.signature(MainAgentRuntime.__init__).parameters
    assert "trace_recorder" in params
