from __future__ import annotations

import ast
import inspect
import re
from pathlib import Path

import pytest

from career_agent.agent.main_agent_contracts import ToolObservation
from career_agent.agent.main_agent_tools import MainAgentToolRegistry
from career_agent.agent.tool_effects import (
    TOOL_EFFECTS,
    declared_write_capabilities,
    effect_for,
    is_external_write,
)


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
    assert all(effect_for(name) in {"READ", "WRITE", "CONTROL"} for name in handlers)


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


def test_a_new_write_handler_cannot_omit_the_execution_outcome_axis() -> None:
    registry = MainAgentToolRegistry()
    registry._atomic_handlers["create_application"] = lambda arguments: ToolObservation(
        tool_name="create_application",
        state="application_ready",
        message="遗漏了执行结果轴。",
    )

    with pytest.raises(ValueError, match="returned without execution_outcome"):
        registry.invoke_atomic_tool("create_application", {})


def test_every_write_handler_result_constructor_declares_execution_outcome() -> None:
    """Keep branch-level coverage from regressing to a capability-name count.

    The registry boundary above is the runtime backstop. This source check makes
    an omitted branch fail before that branch needs a perfectly shaped fixture
    to execute it; merely adding one declaration somewhere in the handler is
    deliberately insufficient.
    """

    tree = ast.parse(_TOOLS_SOURCE.read_text())
    registry = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "MainAgentToolRegistry"
    )
    handlers = {
        node.name.removeprefix("_"): node
        for node in registry.body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("_")
    }
    shared_result_producers = {
        "resolve_action_item",
        "drive_mock_interview",
        "mock_interview_observation",
        "job_research_failure",
        "tailoring_observation",
    }
    missing: list[tuple[str, int]] = []
    producers = {
        name: handlers[name]
        for name, effect in TOOL_EFFECTS.items()
        if effect == "WRITE"
    } | {name: handlers[name] for name in shared_result_producers}
    for name, producer in producers.items():
        for call in ast.walk(producer):
            if not (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Name)
                and call.func.id in {"ToolObservation", "ToolResult"}
            ):
                continue
            if not any(
                keyword.arg == "execution_outcome" for keyword in call.keywords
            ):
                missing.append((name, call.lineno))

    # This helper is deliberately shared with a READ, so it cannot declare a
    # constant outcome itself. Every WRITE caller must pass the axis through.
    for name, effect in TOOL_EFFECTS.items():
        if effect != "WRITE":
            continue
        for call in ast.walk(handlers[name]):
            if not (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr == "_tailoring_observation"
            ):
                continue
            if not any(
                keyword.arg == "execution_outcome" for keyword in call.keywords
            ):
                missing.append((name, call.lineno))

    assert missing == []


def test_external_writes_are_a_declared_subset_of_writes() -> None:
    """Where an effect lands is a registry decision, never a guess from a name.

    Only writes can be external, and only ``execute_calendar_proposal`` leaves
    the deployment today: the preview beside it writes a local proposal, and
    the calendar/email sync tools pull into local stores. Widening this set is
    a review decision, because every member defaults to the owner's button.
    """

    writes = {name for name, effect in TOOL_EFFECTS.items() if effect == "WRITE"}
    external = {name for name in TOOL_EFFECTS if is_external_write(name)}

    assert external == {"execute_calendar_proposal"}
    assert external <= writes
    assert not is_external_write("prepare_interview_calendar_sync")
    assert not is_external_write("search_career_history")
    assert declared_write_capabilities() == frozenset(writes)


def _runtime_workflow_handler_names() -> set[str]:
    """Keys assigned into ``_runtime_workflow_handlers`` in the registry source."""
    tree = ast.parse(_TOOLS_SOURCE.read_text())
    names: set[str] = set()
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "update"
            and isinstance(node.func.value, ast.Attribute)
            and node.func.value.attr == "_runtime_workflow_handlers"
        ):
            continue
        for argument in node.args:
            if isinstance(argument, ast.Dict):
                names.update(
                    key.value
                    for key in argument.keys
                    if isinstance(key, ast.Constant) and isinstance(key.value, str)
                )
    return names


def test_owner_rules_cannot_name_a_write_the_runtime_invokes_on_its_own() -> None:
    """``confirm_before`` may only name writes ``_authorize`` actually judges.

    Runtime-owned workflow entries are permitted without consulting owner rules
    (``verdict = "permit" if runtime_owned ...``), so accepting them in a rule
    would tell the owner they are protected when nothing will ever stop the
    call. The declared set must mirror the registry, both directions.
    """
    from career_agent.agent.tool_effects import is_runtime_owned, owner_rule_capabilities

    runtime_owned = {name for name in TOOL_EFFECTS if is_runtime_owned(name)}

    assert runtime_owned == _runtime_workflow_handler_names()
    assert runtime_owned == {"handle_mock_interview_input", "retry_mock_interview"}
    assert owner_rule_capabilities() == declared_write_capabilities() - runtime_owned
    assert "execute_calendar_proposal" in owner_rule_capabilities()
    assert "create_interview" in owner_rule_capabilities()
