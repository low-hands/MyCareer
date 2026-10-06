from __future__ import annotations

from collections import Counter
from dataclasses import replace
import hashlib
import json

import pytest

import career_agent.agent.capabilities.catalog as catalog_module
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
from career_agent.agent.contracts.tools.job import FindSavedJobsToolArguments
from career_agent.agent.capabilities.reachability import (
    PRECONDITIONS,
    REQUIREMENTS,
    STATE_GATED_TOOLS,
)
from career_agent.evaluation.main_agent_scenarios import SCENARIOS
from career_agent.evaluation.trajectory import (
    prompt_fingerprint,
    trajectory_prompt_fingerprint,
    trajectory_tool_specs,
)


def test_discovery_metadata_covers_the_model_catalogue() -> None:
    model_tools = [descriptor for descriptor in CAPABILITIES.values() if descriptor.model_callable]
    assert len(model_tools) == 72
    assert len({descriptor.name for descriptor in model_tools}) == 72
    assert max(Counter(descriptor.namespace for descriptor in model_tools).values()) <= 10
    aliases = [alias for descriptor in model_tools for alias in descriptor.aliases_zh]
    assert len(aliases) == len(set(aliases))
    for descriptor in model_tools:
        assert descriptor.namespace
        assert descriptor.summary
        assert 2 <= len(descriptor.aliases_zh) <= 5
        assert all(
            successor != descriptor.name and CAPABILITIES[successor].model_callable
            for successor in descriptor.successors
        )
    assert all(
        not (descriptor.namespace or descriptor.summary or descriptor.aliases_zh or descriptor.successors)
        for descriptor in CAPABILITIES.values()
        if not descriptor.model_callable
    )
    assert "research_job" not in capability("analyze_job").successors
    assert all(
        5 <= len(descriptor.example_queries) <= 10
        for descriptor in model_tools
        if descriptor.name not in {"search_capabilities", "route_to_capability"}
    )
    assert not capability("search_capabilities").example_queries
    assert not capability("route_to_capability").example_queries


def test_a_read_never_suggests_an_ungated_write() -> None:
    for descriptor in CAPABILITIES.values():
        if descriptor.effect == "WRITE":
            continue
        for successor in descriptor.successors:
            target = CAPABILITIES[successor]
            assert target.effect != "WRITE" or target.schema_gated, (
                f"{descriptor.name} -> {successor}"
            )
    # Reading a job or an interview is not a request to act on it.
    assert "create_application" not in capability("get_saved_job").successors
    assert "analyze_job" not in capability("get_saved_job").successors
    assert "prepare_interview_calendar_sync" not in capability("get_interview").successors
    # A flow that is already under way keeps its next step.
    assert "match_resume_to_job" in capability("analyze_job").successors
    assert "confirm_memory_amendment" in capability("propose_memory_amendment").successors


@pytest.mark.parametrize(
    ("changes", "error"),
    [
        ({"namespace": ""}, "needs a namespace"),
        ({"summary": ""}, "needs a summary"),
        ({"aliases_zh": ("加载技能",)}, "needs 2-5 Chinese aliases"),
        ({"aliases_zh": ("一", "二", "三", "四", "五", "六")}, "needs 2-5 Chinese aliases"),
        ({"aliases_zh": ("列出简历", "另一种加载")}, "belongs to both"),
        ({"namespace": "memory.proposals"}, "exceeds ten tools"),
        ({"successors": ("load_skill",)}, "invalid capability successor"),
        ({"successors": ("missing_tool",)}, "invalid capability successor"),
        ({"successors": ("handle_mock_interview_input",)}, "invalid capability successor"),
        ({"successors": ("create_application",)}, "a read cannot suggest an ungated write"),
        ({"profiles": frozenset()}, "needs a profile"),
        ({"legacy_profile_exposed": False}, "hidden legacy capability cannot have a profile"),
        ({"example_queries": ()}, "needs 5-10 example queries"),
        ({"example_queries": ("帮我找一下",) * 11}, "needs 5-10 example queries"),
        ({"example_queries": ("帮我",) * 5}, "must have 4-40 characters"),
        ({"example_queries": ("加载技能",) * 5}, "duplicates name or alias"),
    ],
)
def test_catalog_rejects_invalid_discovery_metadata(monkeypatch, changes, error) -> None:
    descriptors = tuple(
        replace(descriptor, **changes) if descriptor.name == "load_skill" else descriptor
        for descriptor in CAPABILITIES.values()
    )
    monkeypatch.setattr(catalog_module, "_descriptors", lambda: iter(descriptors))
    with pytest.raises(RuntimeError, match=error):
        catalog_module._build_catalog()


def test_excluded_and_runtime_capabilities_reject_examples(monkeypatch) -> None:
    for name in ("search_capabilities", "route_to_capability", "handle_mock_interview_input"):
        descriptors = tuple(
            replace(descriptor, example_queries=("给我看看具体内容",) * 5)
            if descriptor.name == name else descriptor
            for descriptor in CAPABILITIES.values()
        )
        monkeypatch.setattr(catalog_module, "_descriptors", lambda: iter(descriptors))
        with pytest.raises(RuntimeError, match="example queries|discovery metadata"):
            catalog_module._build_catalog()


def test_discovery_metadata_does_not_change_model_schemas_or_prompt_fingerprint() -> None:
    registered = trajectory_tool_specs()
    assert registered[-1]["function"]["name"] == "search_capabilities"
    encoded_registered = json.dumps(
        registered, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    assert hashlib.sha256(encoded_registered).hexdigest() == (
        "47d2012a1dc66fb7dd6355cd3cd0c55186ebc0e2506ba0be7a832a976ca62a78"
    )
    schemas = registered[:-1]
    encoded = json.dumps(
        schemas, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    assert hashlib.sha256(encoded).hexdigest() == (
        "f801c8fd86e99a8704f1d5f1189099598f37134703dd285078bec64b013ffdef"
    )
    assert prompt_fingerprint(schemas) == (
        "077b9db3d56cdd8b7f07e8b364b875e5a7db6dbcbc3f448b5500d4f1edf2fe40"
    )
    assert trajectory_prompt_fingerprint(SCENARIOS[0], schemas) == (
        "a27d3ecd951f05a387a030a017419c82f8581efdf9aa5b08a4cf9287561dfb38"
    )


def test_compatibility_views_are_derived_from_the_catalog() -> None:
    assert set(TOOL_EFFECTS) == set(CAPABILITIES)
    assert ROUTABLE_TOOLS == frozenset(
        name for name, descriptor in CAPABILITIES.items()
        if descriptor.model_callable and descriptor.legacy_profile_exposed
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


def test_saved_job_search_can_list_recent_jobs_without_a_filter() -> None:
    assert FindSavedJobsToolArguments.model_validate({}).query == ""
    schema = capability("find_saved_jobs").tool_schema()["function"]["parameters"]
    assert "query" not in schema.get("required", ())


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
