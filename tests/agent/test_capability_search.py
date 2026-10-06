from __future__ import annotations

import pytest

from career_agent.agent.capabilities.catalog import CAPABILITIES, TOOL_PROFILE_NAMES
from career_agent.agent.capabilities.registry import MainAgentToolRegistry
from career_agent.agent.capabilities.search import (
    MAX_SEMANTIC_CANDIDATES,
    SemanticCapabilityIndex,
    search_catalog,
    searchable_capabilities,
)
from career_agent.agent.execution.capability_executor import CapabilityExecutor
from career_agent.agent.contracts.context import MainAgentContext
from career_agent.agent.contracts.profile import CareerProfileContext
from career_agent.agent.contracts.task_state import ConversationTaskState
from career_agent.agent.contracts.tools.core_memory import SearchCapabilitiesToolArguments
from career_agent.agent.middleware.argument_projection import project_atomic_arguments
from career_agent.agent.runtime.decision_engine import DecisionEngine
from career_agent.agent.runtime.reducers import reduce_task_state


@pytest.mark.parametrize("arguments", [
    {}, {"query": "x", "names": ["list_resumes"]}, {"query": " "},
    {"query": "x" * 201}, {"names": []},
    {"names": ["list_resumes"] * 11}, {"query": "简历", "limit": 0},
    {"query": "简历", "limit": 11},
])
def test_search_arguments_reject_invalid_modes_and_bounds(arguments) -> None:
    with pytest.raises(ValueError):
        SearchCapabilitiesToolArguments.model_validate(arguments)


def test_exact_names_namespaces_and_exclusions() -> None:
    assert search_catalog(query="match_resume_to_job")[0] == "match_resume_to_job"
    assert search_catalog(query="resume.match", limit=2) == (
        "match_resume_to_job", "get_resume_job_match",
    )
    assert search_catalog(names=("resume.match",), limit=2) == (
        "match_resume_to_job", "get_resume_job_match",
    )
    assert search_catalog(names=("list_resumes",)) == ("list_resumes",)
    expanded = search_catalog(names=("memory.proposals",), limit=1)
    assert len(expanded) == 10
    assert "confirm_memory_tombstone" in expanded
    assert "propose_career_fact" in expanded
    with pytest.raises(ValueError, match="unknown capability"):
        search_catalog(names=("route_to_capability",))
    with pytest.raises(ValueError, match="unknown capability"):
        search_catalog(names=("not_a_capability",))
    searchable = {item.name for item in searchable_capabilities()}
    assert "search_capabilities" not in searchable
    assert "route_to_capability" not in searchable
    assert "handle_mock_interview_input" not in searchable


def test_chinese_reordering_aliases_and_tie_order_are_stable() -> None:
    assert search_catalog(query="天气怎么样") == ()
    assert search_catalog(query="的") == ()
    assert "match_resume_to_job" in search_catalog(query="岗位跟简历匹配一下")
    assert "match_resume_to_job" in search_catalog(query="简历和岗位匹配一下")
    for descriptor in searchable_capabilities():
        for alias in descriptor.aliases_zh:
            assert descriptor.name in search_catalog(query=alias, limit=5), (
                descriptor.name, alias,
            )
    assert search_catalog(query="岗位", limit=10) == search_catalog(query="岗位", limit=10)


def test_registry_result_reducer_and_legacy_state_roundtrip() -> None:
    registry = MainAgentToolRegistry()
    context = MainAgentContext(
        conversation_id="search-test", profile=CareerProfileContext(user_id="u"),
        user_message="找简历匹配工具", task=ConversationTaskState(),
    )
    projected = project_atomic_arguments(
        context, "search_capabilities", {"names": ["resume.match"], "limit": 2},
        source_turn_id=None,
    )
    result = registry.invoke_atomic_tool("search_capabilities", projected)
    assert result.state == "capabilities_found"
    assert result.payload["loaded"] == ["match_resume_to_job", "get_resume_job_match"]
    assert result.payload["items"][0]["reachable"] is False
    assert result.payload["items"][0]["requirement"]
    expanded = registry.invoke_atomic_tool("search_capabilities", {
        "current_task": {}, "names": ["memory.proposals"], "limit": 1,
    })
    assert len(expanded.payload["items"]) == 10
    assert len(expanded.payload["loaded"]) == 10
    task = reduce_task_state(context.task, result)
    assert task.loaded_capabilities == (
        "match_resume_to_job", "get_resume_job_match",
    )
    restored = ConversationTaskState.model_validate(task.model_dump(mode="json"))
    assert restored.loaded_capabilities == task.loaded_capabilities
    old = ConversationTaskState.model_validate({"tool_profile": "core"})
    assert old.loaded_capabilities == ()
    stale = ConversationTaskState.model_validate({
        "loaded_capabilities": ["made_up", "get_resume_job_match", "get_resume_job_match"],
    })
    assert stale.loaded_capabilities == ("get_resume_job_match",)
    loaded_context = context.model_copy(update={"task": task})
    assert "loaded_capabilities" not in loaded_context.model_context().get("task", {})
    with pytest.raises(ValueError, match="unknown capability"):
        registry.invoke_atomic_tool("search_capabilities", {
            "current_task": {}, "names": ["made_up"],
        })


def test_search_is_registered_but_absent_from_every_legacy_decision_offer() -> None:
    assert CAPABILITIES["search_capabilities"].legacy_profile_exposed is False
    registry = MainAgentToolRegistry()
    assert registry.schemas()[-1]["function"]["name"] == "search_capabilities"
    engine = DecisionEngine(
        emit=lambda _: None, decision_heartbeat=lambda _: None,
        record_trace_event=lambda *args, **kwargs: None,
        project_atomic_tool_arguments=lambda *args: {},
        project_workflow_arguments=lambda *args: {},
        context_manager=object(), decision_maker_provider=lambda: object(),
        tools=registry, career_memory_enabled=False,
    )
    for profile in TOOL_PROFILE_NAMES:
        names = {schema["function"]["name"] for schema in engine.tool_schemas(
            profile, ConversationTaskState(tool_profile=profile)
        )}
        assert "search_capabilities" not in names


def test_semantic_catalog_vectors_are_cached_by_catalogue_hash() -> None:
    class Client:
        model_id = "test"

        def __init__(self) -> None:
            self.calls: list[tuple[str, ...]] = []

        def embed(self, texts):
            self.calls.append(tuple(texts))
            return [(1.0, 0.0) for _ in texts]

    client = Client()
    index = SemanticCapabilityIndex(client)
    assert index.scores("简历") == {}  # request never warms 69 catalogue vectors
    assert client.calls == []
    index.warm()
    assert len(index.scores("简历")) == MAX_SEMANTIC_CANDIDATES
    assert len(index.scores("岗位")) == MAX_SEMANTIC_CANDIDATES
    assert len(client.calls) == 3  # catalogue once, two query vectors


def test_small_positive_semantic_scores_cannot_fill_zero_lexical_results() -> None:
    scores = {item.name: 0.01 for item in searchable_capabilities()}
    assert search_catalog(query="天气怎么样", semantic_scores=scores) == ()


def test_semantic_query_uses_injected_client_and_falls_back_to_lexical() -> None:
    class WarmClient:
        model_id = "warm-client"

        def __init__(self) -> None:
            self.calls: list[tuple[str, ...]] = []

        def embed(self, texts):
            self.calls.append(tuple(texts))
            return [(1.0, 0.0) for _ in texts]

    class FailingQueryClient:
        model_id = "query-client"

        def embed(self, texts):
            raise TimeoutError("query embedding timed out")

    warm_client = WarmClient()
    index = SemanticCapabilityIndex(warm_client, FailingQueryClient())
    index.warm()
    registry = MainAgentToolRegistry()
    registry.configure_capability_search(index)
    result = registry.invoke_atomic_tool("search_capabilities", {
        "current_task": {}, "query": "简历和岗位匹配一下",
    })
    assert "match_resume_to_job" in result.payload["loaded"]
    assert len(warm_client.calls) == 1  # no catalogue or query embedding on request


def test_empty_search_records_a_trace_event() -> None:
    events = []
    executor = CapabilityExecutor(
        tools=MainAgentToolRegistry(), action_execution_store=None,
        action_policy_epoch=1, emit_capability_started=lambda _: None,
        emit_capability_completed=lambda *_: None,
        run_capability=lambda pending, run: run(),
        record_trace_event=lambda *args, **kwargs: events.append((args, kwargs)),
    )
    update = executor.act({"pending": {
        "name": "search_capabilities", "kind": "atomic_tool", "effect": "CONTROL",
        "arguments": {"current_task": {}, "query": "zzzxxyyunknownword"},
    }})
    assert update["pending"]["result"].state == "no_capabilities_found"
    assert events == [(('capability_search_empty', 'act'), {
        'outcome': 'succeeded', 'details': {'tool_name': 'search_capabilities'},
    })]
