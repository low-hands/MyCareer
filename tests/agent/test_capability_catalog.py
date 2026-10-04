from __future__ import annotations

from career_agent.agent.capabilities.catalog import CAPABILITIES, capability
from career_agent.agent.capabilities.registry import MainAgentToolRegistry
from career_agent.agent.capabilities.effects import (
    TOOL_EFFECTS,
    approval_policy,
    effect_for,
    is_external_write,
    is_runtime_owned,
    replay_safe,
)
from career_agent.agent.capabilities.profiles import ROUTABLE_TOOLS, profile_tools
from career_agent.agent.capabilities.reachability import (
    PRECONDITIONS,
    REQUIREMENTS,
    STATE_GATED_TOOLS,
)


def test_compatibility_views_are_derived_from_the_catalog() -> None:
    assert set(TOOL_EFFECTS) == set(CAPABILITIES)
    assert ROUTABLE_TOOLS == frozenset(
        name for name, descriptor in CAPABILITIES.items()
        if descriptor.model_callable
    )
    assert PRECONDITIONS == {
        name: descriptor.precondition
        for name, descriptor in CAPABILITIES.items()
        if descriptor.precondition is not None
    }
    assert REQUIREMENTS == {
        name: descriptor.requirement
        for name, descriptor in CAPABILITIES.items()
        if descriptor.requirement is not None
    }
    assert STATE_GATED_TOOLS == frozenset(
        name for name, descriptor in CAPABILITIES.items()
        if descriptor.schema_gated
    )


def test_effect_and_safety_queries_read_the_same_descriptor() -> None:
    for name, descriptor in CAPABILITIES.items():
        assert effect_for(name) == descriptor.effect
        assert is_external_write(name) is descriptor.external_write
        assert replay_safe(name) is descriptor.replay_safe
        assert approval_policy(name) == descriptor.approval_policy
        assert is_runtime_owned(name) is descriptor.runtime_owned


def test_core_membership_expands_to_every_profile() -> None:
    for name, descriptor in CAPABILITIES.items():
        if "core" in descriptor.profiles:
            assert all(name in profile_tools(profile) for profile in (
                "core", "job", "resume", "application", "interview", "memory"
            ))


def test_registry_refuses_a_handler_bound_under_the_wrong_execution_kind() -> None:
    registry = MainAgentToolRegistry()
    registry._workflow_handlers["route_to_capability"] = registry._route_to_capability

    try:
        registry._validate_catalog_bindings()
    except RuntimeError as error:
        assert "registered more than once" in str(error)
    else:
        raise AssertionError("duplicate cross-kind binding was accepted")


def test_descriptor_owns_the_handler_name() -> None:
    assert capability("execute_calendar_proposal").handler_name == (
        "_execute_calendar_proposal"
    )
    assert all(
        hasattr(MainAgentToolRegistry, descriptor.handler_name)
        for descriptor in CAPABILITIES.values()
    )


def test_every_model_capability_owns_its_complete_tool_schema() -> None:
    for descriptor in CAPABILITIES.values():
        if not descriptor.model_callable:
            assert descriptor.description is None
            assert descriptor.arguments_model is None
            continue
        schema = descriptor.tool_schema()
        assert schema["function"]["name"] == descriptor.name
        assert schema["function"]["description"] == descriptor.description
        assert schema["function"]["parameters"]["type"] == "object"


def test_every_capability_declares_an_enforced_output_envelope() -> None:
    for descriptor in CAPABILITIES.values():
        schema = descriptor.output_schema()
        assert descriptor.output_model == "ToolObservation"
        assert {"tool_name", "state", "message"} <= set(schema["properties"])


def test_approval_and_replay_policies_are_explicit() -> None:
    assert capability("get_saved_job").approval_policy == "never"
    assert capability("create_application").approval_policy == "owner_rule"
    assert capability("update_owner_settings").approval_policy == "always"
    assert capability("confirm_memory_tombstone").approval_policy == "always"
    assert capability("execute_calendar_proposal").approval_policy == "always"

    assert capability("get_saved_job").replay_policy == "not_applicable"
    assert capability("update_application_status").replay_policy == "never"
    assert capability("create_application").replay_policy == "idempotent"
    assert capability("get_saved_job").recovery_policy == "not_applicable"
    assert capability("create_application").recovery_policy == "retry"
    assert capability("update_application_status").recovery_policy == "reconcile"
    assert capability("execute_calendar_proposal").recovery_policy == "reconcile"


def test_output_contract_rejects_a_result_for_another_capability() -> None:
    from career_agent.agent.contracts.observations import ToolObservation

    registry = MainAgentToolRegistry()
    result = ToolObservation(
        tool_name="get_saved_job",
        state="completed",
        message="ok",
    )
    registry._atomic_handlers["route_to_capability"] = lambda arguments: result
    try:
        registry.invoke_atomic_tool("route_to_capability", {})
    except ValueError as error:
        assert "returned result for" in str(error)
    else:
        raise AssertionError("cross-capability result was accepted")
