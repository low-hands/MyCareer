from __future__ import annotations

import ast
import inspect
import re
from pathlib import Path

from career_agent.agent.main_agent_tools import MainAgentToolRegistry
from career_agent.agent.tool_effects import TOOL_EFFECTS, effect_for


_TOOLS_SOURCE = Path(inspect.getsourcefile(MainAgentToolRegistry))


def _registered_handler_names() -> set[str]:
    """Read every registry assignment, including optional service branches."""
    tree = ast.parse(_TOOLS_SOURCE.read_text())
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if not (
                    isinstance(key, ast.Constant)
                    and isinstance(key.value, str)
                    and isinstance(value, (ast.Attribute, ast.IfExp))
                ):
                    continue
                if isinstance(value, ast.Attribute) and value.attr.startswith("_"):
                    names.add(key.value)
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Attribute):
            for target in node.targets:
                if (
                    isinstance(target, ast.Subscript)
                    and isinstance(target.slice, ast.Constant)
                    and isinstance(target.slice.value, str)
                    and node.value.attr.startswith("_")
                ):
                    names.add(target.slice.value)
    return names


def test_effect_registry_and_handler_registry_are_closed_both_directions() -> None:
    handlers = _registered_handler_names()
    declared = set(TOOL_EFFECTS)

    assert handlers - declared == set()
    assert declared - handlers == set()
    assert all(effect_for(name) in {"READ", "WRITE"} for name in handlers)


_MUTATING_CALL = re.compile(
    r"\.(?:"
    r"analyze_version|cancel|complete_action|confirm_analysis|create|create_application|"
    r"create_draft|dismiss_action|execute_proposal|finalize_draft|match|prepare_export|"
    r"prepare_interview|prepare_interview_sync|record|reject_analysis|research|resolve|"
    r"retry|revise_draft|review_draft|snooze_action|start|sync|update|upsert"
    r")[a-z_]*\("
)


def test_handlers_that_call_write_shaped_methods_are_a_subset_of_write_tools() -> None:
    """Direct write-shaped calls may not remain classified READ.

    This is deliberately a local-handler superset scan, not a call-graph or
    effect-system proof. Transitive writes remain explicit review obligations:
    ``get_daily_brief`` is the explicit exception: ``daily_brief -> refresh``
    materializes a derived action index one service layer below the handler,
    which this scan cannot discover. The exact registry test prevents omissions;
    this scan catches the common direct-call mistake without pretending to
    establish purity or transitive effects.
    """
    detected: set[str] = set()
    for name in _registered_handler_names():
        handler = getattr(MainAgentToolRegistry, f"_{name}")
        if _MUTATING_CALL.search(inspect.getsource(handler)):
            detected.add(name)

    assert detected <= {
        name for name, effect in TOOL_EFFECTS.items() if effect == "WRITE"
    }


def test_known_derived_and_external_effect_decisions_remain_explicit() -> None:
    # Refresh writes derived candidates, but does not commit user intent and
    # must not consume the one write needed to complete an item from the brief.
    assert effect_for("get_daily_brief") == "READ"
    # Opening an external browser page is still a client-side effect.
    assert effect_for("open_job_search") == "WRITE"


def test_read_and_write_sets_are_disjoint_and_nonempty() -> None:
    reads = {name for name, effect in TOOL_EFFECTS.items() if effect == "READ"}
    writes = {name for name, effect in TOOL_EFFECTS.items() if effect == "WRITE"}
    assert reads
    assert writes
    assert reads & writes == set()
