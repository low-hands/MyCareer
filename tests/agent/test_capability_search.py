from __future__ import annotations

from collections import Counter
from dataclasses import replace

import pytest

from career_agent.agent.capabilities.catalog import CAPABILITIES
from career_agent.agent.capabilities.registry import MainAgentToolRegistry
from career_agent.agent.capabilities.search import (
    MAX_SEMANTIC_CANDIDATES,
    SemanticCapabilityIndex,
    _common_example_terms,
    search_catalog,
    searchable_capabilities,
)
from career_agent.agent.execution.capability_executor import CapabilityExecutor
from career_agent.agent.contracts.context import MainAgentContext
from career_agent.agent.contracts.profile import CareerProfileContext
from career_agent.agent.contracts.task_state import ConversationTaskState
from career_agent.agent.contracts.tools.core_memory import SearchCapabilitiesToolArguments
from career_agent.agent.middleware.argument_projection import project_atomic_arguments
from career_agent.agent.runtime.reducers import reduce_task_state
from career_agent.evaluation.tool_selection_scenarios import SELECTION_DEV
from career_agent.evaluation.tool_selection_scenarios import SELECTION_HOLDOUT
from career_agent.evaluation.independent_tool_selection_holdout import (
    SELECTION_INDEPENDENT_HOLDOUT,
)


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
    assert search_catalog(names=("create_application",)) == ("create_application",)
    assert search_catalog(names=("confirm_memory_tombstone",)) == (
        "confirm_memory_tombstone",
    )
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
    assert search_catalog(query="zzzxxyyunknownword") == ()
    assert "match_resume_to_job" in search_catalog(query="岗位跟简历匹配一下")
    assert "match_resume_to_job" in search_catalog(query="简历和岗位匹配一下")
    assert sum(len(item.aliases_zh) for item in searchable_capabilities()) == 141
    for descriptor in searchable_capabilities():
        for alias in descriptor.aliases_zh:
            assert descriptor.name in search_catalog(query=alias, limit=5), (
                descriptor.name, alias,
            )
    assert search_catalog(query="岗位", limit=10) == search_catalog(query="岗位", limit=10)


@pytest.mark.parametrize("query", [
    "天气怎么样", "的", "这个怎么弄", "帮我看一下", "zzzxxyyunknownword",
])
def test_unrelated_queries_return_no_capabilities(query: str) -> None:
    assert search_catalog(query=query) == ()


@pytest.mark.parametrize("query", [
    "帮我看看我的经历", "看看我的面试安排", "我的投递记录有哪些", "这个岗位不错",
])
def test_read_or_observational_queries_report_action_exposure(
    query: str, record_property,
) -> None:
    offered = search_catalog(query=query, semantic_scores={
        "propose_memory_tombstone": 1.0,
        "confirm_career_fact": 1.0,
        "create_application": 1.0,
    })
    record_property("query", query)
    record_property("offered", offered)
    record_property("action_tools", tuple(
        name for name in offered
        if CAPABILITIES[name].effect == "WRITE" or name.startswith("propose_")
    ))
    assert all(name in CAPABILITIES for name in offered)


def test_explicit_application_creation_still_retrieves_create_tool() -> None:
    assert "create_application" in search_catalog(query="我想把这个岗位加入投递")


def test_common_example_terms_derive_from_tool_and_namespace_frequency() -> None:
    entries = searchable_capabilities()
    one_per_namespace = {}
    for item in entries:
        one_per_namespace.setdefault(item.namespace, item.name)
    selected = tuple(one_per_namespace.values())
    assert len(selected) >= 11

    def with_token(count: int):
        names = set(selected[:count])
        return tuple(
            replace(item, example_queries=(*item.example_queries, "这是闲词测试"))
            if item.name in names else item
            for item in entries
        )

    assert "闲词" not in _common_example_terms(with_token(10))
    assert "闲词" in _common_example_terms(with_token(11))


def test_search_runs_on_both_frozen_holdouts_without_locking_scores() -> None:
    assert (len(SELECTION_HOLDOUT), len(SELECTION_INDEPENDENT_HOLDOUT)) == (20, 12)
    for case in (*SELECTION_HOLDOUT, *SELECTION_INDEPENDENT_HOLDOUT):
        offered = search_catalog(query=case.scenario.context.user_message)
        assert isinstance(offered, tuple)
        assert all(name in CAPABILITIES for name in offered)


def test_development_query_recall_and_control_write_exposure() -> None:
    bare = tuple(replace(item, example_queries=()) for item in searchable_capabilities())

    def counts(descriptors=None):
        hits: Counter[int] = Counter()
        writes = 0
        demand_count = 0
        for case in SELECTION_DEV:
            offered = search_catalog(
                query=case.scenario.context.user_message, descriptors=descriptors,
            )
            first = case.scenario.steps[0]
            expected = set(first.expect_tools or ({first.expect_tool} if first.expect_tool else set()))
            if expected:
                demand_count += 1
                for rank in (1, 3, 5):
                    hits[rank] += bool(expected.intersection(offered[:rank]))
            if case.kind == "control":
                writes += sum(CAPABILITIES[name].effect == "WRITE" for name in offered)
        return demand_count, dict(hits), writes

    assert counts(bare) == (43, {1: 11, 3: 17, 5: 20}, 1)
    assert counts() == (43, {1: 17, 3: 28, 5: 34}, 8)


def test_registry_result_reducer_and_old_state_roundtrip() -> None:
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
    write = registry.invoke_atomic_tool("search_capabilities", {
        "current_task": {}, "names": ["create_application"],
    })
    assert write.payload["loaded"] == ["create_application"]
    assert "已加载" in write.message
    if write.payload["items"][0]["reachable"]:
        assert "请重新发起调用" in write.message
    else:
        assert "先满足前置条件" in write.message
    discovered_write = registry.invoke_atomic_tool("search_capabilities", {
        "current_task": {}, "query": "我想把这个岗位加入投递",
    })
    assert "create_application" in discovered_write.payload["loaded"]
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
    executed, *rest = events
    assert executed[0] == ('capability_executed', 'act')
    assert executed[1]['duration_ms'] >= 0
    assert {k: v for k, v in executed[1].items() if k != 'duration_ms'} == {
        'outcome': 'succeeded', 'error_code': None,
        'details': {
            'tool_name': 'search_capabilities', 'effect': 'CONTROL',
            'result_state': 'no_capabilities_found', 'disposition': 'completed',
        },
    }
    assert rest == [(('capability_search_empty', 'act'), {
        'outcome': 'succeeded', 'details': {'tool_name': 'search_capabilities'},
    })]
