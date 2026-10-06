"""Search-mode selection and one-snapshot model/execution boundary."""

from __future__ import annotations

from threading import Event
from pathlib import Path

from career_agent.agent.capabilities.catalog import CAPABILITIES
from career_agent.agent.capabilities.selection import ALWAYS_OFFERED_TOOLS
from career_agent.agent.capabilities.selection_strategy import SearchStrategy, strategy_for_mode
from career_agent.agent.capabilities.selection_strategy import capability_directory
from career_agent.agent.capabilities.search import searchable_capabilities
from career_agent.agent.capabilities.waiting import (
    SELECTION_ONLY_WAITING_STATES, WAITING_FOR_USER_STATES,
)
from career_agent.agent.presentation.delivery_policy import DELIVERY_POLICIES, is_waiting
import pytest
from career_agent.agent.contracts.context import MainAgentContext
from career_agent.agent.contracts.decisions import AgentDecision, ToolCall
from career_agent.agent.contracts.observations import DecisionObservation
from career_agent.agent.contracts.profile import CareerProfileContext
from career_agent.agent.contracts.task_state import ConversationTaskState
from career_agent.agent.middleware.budget import BudgetMiddleware
from career_agent.agent.runtime.decision_engine import DecisionEngine
from career_agent.agent.runtime.decision_messages import project_decision_messages
from career_agent.harness.turn_router import TurnRouter
from career_agent.agent.context.manager import ContextManager
from career_agent.agent.capabilities.registry import MainAgentToolRegistry
from career_agent.agent.runtime.main_agent_runtime import MainAgentRuntime
from career_agent.storage.context import CareerContextStore
from career_agent.agent.providers.main_agent import OpenAICompatibleMainAgentDecisionMaker
from career_agent.evaluation.trajectory import prompt_fingerprint, trajectory_prompt_fingerprint
from career_agent.evaluation.main_agent_scenarios import SCENARIOS


SCHEMAS = tuple(
    descriptor.tool_schema() for descriptor in CAPABILITIES.values()
    if descriptor.model_callable
)


def _context(task=None, observations=()):
    return MainAgentContext(
        conversation_id="c1", profile=CareerProfileContext(user_id="u1"),
        user_message="看看这个岗位和简历是否匹配",
        task=task or ConversationTaskState(), tool_observations=observations,
    )


def test_search_offer_starts_small_and_retains_loaded_tools() -> None:
    strategy = SearchStrategy()
    cold = strategy.select(_context(), SCHEMAS)
    assert set(cold.offered_names) == set(ALWAYS_OFFERED_TOOLS)
    assert "route_to_capability" not in cold.selected_names
    assert "create_application" not in cold.offered_names
    task = ConversationTaskState(loaded_capabilities=(
        "list_resumes", "create_application", "match_resume_to_job",
    ), active_job_posting_id="job-1", active_jd_snapshot_id="jd-1")
    loaded = strategy.select(_context(task), SCHEMAS)
    assert {"list_resumes", "create_application"} <= set(loaded.offered_names)
    assert ("match_resume_to_job", CAPABILITIES["match_resume_to_job"].requirement) in loaded.blocked_requirements
    assert "match_resume_to_job" not in loaded.offered_names
    assert "route_to_capability" not in loaded.offered_names
    assert strategy.select(_context(task), SCHEMAS).schemas is loaded.schemas


def test_newly_satisfied_state_gate_is_offered_without_search() -> None:
    selection = SearchStrategy().select(
        _context(ConversationTaskState(active_job_research_run_id="research-1")),
        SCHEMAS,
    )
    assert "retry_job_research" in selection.offered_names
    assert dict(selection.sources)["retry_job_research"] == "state"


def test_w_offers_bound_reads_and_reviewed_successor_transiently() -> None:
    strategy = SearchStrategy()
    task = ConversationTaskState(active_application_id="app-1")
    initial = strategy.select(_context(task), SCHEMAS)
    assert "get_application" in initial.offered_names
    assert dict(initial.sources)["get_application"] == "proactive"
    assert "list_email_events" not in initial.offered_names
    assert task.loaded_capabilities == ()

    result = DecisionObservation(
        tool_name="get_application", state="application_ready", message="已读取。",
    )
    following = strategy.select(_context(task, (result,)), SCHEMAS)
    assert "list_email_events" in following.offered_names
    assert dict(following.sources)["list_email_events"] == "proactive"
    assert task.loaded_capabilities == ()
    assert "list_email_events" not in strategy.select(_context(task), SCHEMAS).offered_names


def test_w_does_not_follow_a_failed_or_waiting_observation() -> None:
    strategy = SearchStrategy()
    task = ConversationTaskState(active_application_id="app-1")
    for state in ("failed", "capability_confirmation_required"):
        observation = DecisionObservation(
            tool_name="get_application", state=state, message="未完成。",
        )
        offer = strategy.select(_context(task, (observation,)), SCHEMAS)
        assert "list_email_events" not in offer.offered_names


def test_w_keeps_successors_after_intervening_search_within_turn() -> None:
    strategy = SearchStrategy()
    task = ConversationTaskState(
        active_job_posting_id="job-1", active_jd_snapshot_id="jd-1",
        active_job_analysis_id="analysis-1",
        active_job_analysis_jd_snapshot_id="jd-1",
        job_analysis_status="ready", active_resume_version_id="resume-1",
    )
    analysis = DecisionObservation(
        tool_name="analyze_job", state="job_analysis_ready", message="岗位已分析。",
    )
    searched = DecisionObservation(
        tool_name="search_capabilities", state="capabilities_found",
        message="已搜索。",
    )
    before_search = strategy.select(_context(task, (analysis,)), SCHEMAS)
    after_search = strategy.select(_context(task, (analysis, searched)), SCHEMAS)
    assert "match_resume_to_job" in before_search.offered_names
    assert "match_resume_to_job" in after_search.offered_names
    assert "match_resume_to_job" not in strategy.select(_context(task), SCHEMAS).offered_names
    assert task.loaded_capabilities == ()

    stale = task.model_copy(update={
        "domain_context": task.domain_context.model_copy(update={
            "job": task.domain_context.job.model_copy(update={"analysis_status": None})
        })
    })
    assert "match_resume_to_job" not in strategy.select(
        _context(stale, (analysis, searched)), SCHEMAS,
    ).offered_names


def test_w_unions_successors_from_two_business_results() -> None:
    task = ConversationTaskState(
        active_application_id="app-1", active_job_posting_id="job-1",
        active_jd_snapshot_id="jd-1", active_job_analysis_id="analysis-1",
        active_job_analysis_jd_snapshot_id="jd-1",
        job_analysis_status="ready", active_resume_version_id="resume-1",
    )
    observations = (
        DecisionObservation(
            tool_name="get_application", state="application_ready", message="已读取投递。",
        ),
        DecisionObservation(
            tool_name="analyze_job", state="job_analysis_ready", message="已分析岗位。",
        ),
    )
    offered = SearchStrategy().select(_context(task, observations), SCHEMAS).offered_names
    assert {"list_email_events", "match_resume_to_job"} <= set(offered)


def test_w_skips_failed_and_waiting_observations_among_multiple_results() -> None:
    task = ConversationTaskState(
        active_application_id="app-1", active_job_posting_id="job-1",
        active_jd_snapshot_id="jd-1", active_job_analysis_id="analysis-1",
        active_job_analysis_jd_snapshot_id="jd-1",
        job_analysis_status="ready", active_resume_version_id="resume-1",
    )
    observations = (
        DecisionObservation(tool_name="get_application", state="failed", message="失败。"),
        DecisionObservation(
            tool_name="analyze_job", state="capability_confirmation_required",
            message="等待确认。",
        ),
        DecisionObservation(
            tool_name="search_capabilities", state="capabilities_found", message="已搜索。",
        ),
    )
    offered = SearchStrategy().select(_context(task, observations), SCHEMAS).offered_names
    assert "list_email_events" not in offered
    assert "match_resume_to_job" not in offered


def test_waiting_tool_is_suppressed_until_a_new_user_turn() -> None:
    strategy = SearchStrategy()
    task = ConversationTaskState(loaded_capabilities=("update_working_notes",))
    observation = DecisionObservation(
        tool_name="update_working_notes", state="working_notes_derived_argument",
        message="需要用户确认。",
    )
    waiting = strategy.select(_context(task, (observation,)), SCHEMAS)
    assert "update_working_notes" in waiting.waiting_suppressed
    assert "update_working_notes" not in waiting.offered_names
    resumed = strategy.select(_context(task), SCHEMAS)
    assert "update_working_notes" in resumed.offered_names


def test_waiting_selection_tracks_delivery_policy() -> None:
    assert WAITING_FOR_USER_STATES == frozenset(
        state for state in DELIVERY_POLICIES if is_waiting(state)
    ) | SELECTION_ONLY_WAITING_STATES
    assert SELECTION_ONLY_WAITING_STATES == frozenset({
        "working_notes_derived_argument", "job_intent_proposed",
    })


def test_new_user_turn_clears_waiting_observation_but_keeps_loaded_tool(tmp_path) -> None:
    store = CareerContextStore(tmp_path / "context.sqlite3")
    manager = ContextManager(store)
    manager.upsert_profile(CareerProfileContext(user_id="u1"))
    task = ConversationTaskState(loaded_capabilities=("update_working_notes",))
    store.upsert_task(user_id="u1", conversation_id="c1", task=task)
    prior = manager.load_for_turn(user_id="u1", conversation_id="c1", user_message="记下我的偏好")
    waiting = prior.model_copy(update={"tool_observations": (DecisionObservation(
        tool_name="update_working_notes", state="working_notes_derived_argument",
        message="需要确认。",
    ),)})
    assert "update_working_notes" not in SearchStrategy().select(waiting, SCHEMAS).offered_names
    next_turn = manager.load_for_turn(user_id="u1", conversation_id="c1", user_message="我确认这个偏好")
    assert next_turn.tool_observations == ()
    assert next_turn.turn_proactive_capabilities == ()
    assert "update_working_notes" in SearchStrategy().select(next_turn, SCHEMAS).offered_names


def test_search_projection_and_prompt_exclude_profile_routing() -> None:
    strategy = SearchStrategy()
    context = _context(ConversationTaskState(loaded_capabilities=("list_resumes",)))
    selection = strategy.select(context, SCHEMAS)
    projected = context.model_copy(update={"capability_selection": selection})
    task = projected.model_context()["task"]
    assert task["available_now"] == list(selection.offered_names)
    assert task["loaded_capabilities"] == ["list_resumes"]
    assert "tool_profile" not in task
    control = project_decision_messages(projected).control["task"]
    assert control["available_now"] == task["available_now"]
    assert "tool_profile" not in control
    assert "route_to_capability" not in strategy.tool_policy()


def test_directory_is_stable_complete_and_metadata_only() -> None:
    directory = capability_directory()
    for item in searchable_capabilities():
        assert f"- {item.name}: {item.summary}" in directory
        assert item.example_queries[0] not in directory
    assert "route_to_capability" not in directory
    assert "handle_mock_interview_input" not in directory
    assert directory == capability_directory()
    prompt = OpenAICompatibleMainAgentDecisionMaker._system_prompt(SearchStrategy().tool_policy())
    assert "route_to_capability" not in prompt
    assert "tool_profile" not in prompt


def test_search_mode_fingerprint_has_its_own_namespace() -> None:
    selection = SearchStrategy().select(_context(), SCHEMAS)
    assert prompt_fingerprint(selection.schemas, mode="search") != prompt_fingerprint(selection.schemas)
    assert trajectory_prompt_fingerprint(SCENARIOS[0], SCHEMAS, mode="search") != trajectory_prompt_fingerprint(SCENARIOS[0], SCHEMAS)


def test_search_budget_is_independent_of_legacy_control_budget() -> None:
    budget = BudgetMiddleware(
        max_read_calls=6, max_write_calls=1, max_external_write_calls=1,
        search_mode=True,
    )
    assert budget.check({"search_calls": 4}, name="search_capabilities", effect="CONTROL") is None
    assert budget.check({"search_calls": 5}, name="search_capabilities", effect="CONTROL") is not None


def test_unknown_mode_fails_during_strategy_construction() -> None:
    with pytest.raises(ValueError, match="MAIN_AGENT_TOOL_SELECTION"):
        strategy_for_mode("typo")


def test_decide_uses_one_selection_for_model_and_execution() -> None:
    class Registry:
        def schemas(self):
            return SCHEMAS

    class ContextManager:
        def mark_episodes_projected(self, **kwargs):
            pass

    class Maker:
        def decide(self, context, schemas):
            self.context = context
            self.schemas = schemas
            return AgentDecision(action="final", message="完成。")

    maker = Maker()
    engine = DecisionEngine(
        emit=lambda event: None, decision_heartbeat=lambda sink: Event(),
        record_trace_event=lambda *args, **kwargs: None,
        project_atomic_tool_arguments=lambda *args: {},
        project_workflow_arguments=lambda *args: {},
        context_manager=ContextManager(), decision_maker_provider=lambda: maker,
        tools=Registry(), career_memory_enabled=False,
        selection_strategy=SearchStrategy(),
    )
    result = engine.decide({
        "context": _context(ConversationTaskState(active_application_id="app-1")),
        "control": {},
    })
    names = tuple(schema["function"]["name"] for schema in maker.schemas)
    assert "get_application" in names
    assert result["control"]["offered_tool_names"] == names
    assert maker.context.model_context()["task"]["available_now"] == list(names)


def test_explicit_span_prelude_uses_strategy_offer() -> None:
    router = TurnRouter(
        context_manager=None, confirmation_store=None, agent_loop=None,
        selection_strategy=SearchStrategy(),
    )
    context = _context().model_copy(update={
        "user_message": "回看第 2 到 3 条消息",
        "through_sequence": 3, "recent_from_sequence": 4,
    })
    assert router.explicit_span_prelude(context)["decision"].tool_call.name == "read_conversation_span"


def test_search_mode_is_wired_at_runtime_startup(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MAIN_AGENT_TOOL_SELECTION", "search")
    store = CareerContextStore(tmp_path / "context.sqlite3")
    manager = ContextManager(store)
    manager.upsert_profile(CareerProfileContext(user_id="u1"))

    class Maker:
        def decide(self, context, schemas):
            self.context = context
            self.schemas = schemas
            return AgentDecision(action="final", message="完成。")

    maker = Maker()
    tools = MainAgentToolRegistry(
        conversation_store=store, career_history_store=object(),
        skills_root=Path("skills"), resume_store=object(),
    )
    runtime = MainAgentRuntime(context_manager=manager, decision_maker=maker, tools=tools)
    result = runtime.run_turn(
        user_id="u1", conversation_id="c1", user_message="列出我的简历",
    )
    assert set(schema["function"]["name"] for schema in maker.schemas) == set(ALWAYS_OFFERED_TOOLS)
    assert maker.context.model_context()["task"]["available_now"] == [
        schema["function"]["name"] for schema in maker.schemas
    ]
    assert result.context.task.tool_profile == "core"


def test_search_result_loads_schema_on_the_next_decision(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MAIN_AGENT_TOOL_SELECTION", "search")
    store = CareerContextStore(tmp_path / "context.sqlite3")
    manager = ContextManager(store)
    manager.upsert_profile(CareerProfileContext(user_id="u1"))

    class Maker:
        def __init__(self):
            self.offers = []

        def decide(self, context, schemas):
            names = tuple(schema["function"]["name"] for schema in schemas)
            self.offers.append(names)
            if len(self.offers) == 1:
                return AgentDecision(
                    action="tool_call", tool_call=ToolCall(
                        name="search_capabilities", arguments={"names": ["list_resumes"]},
                    ),
                )
            assert "list_resumes" in context.task.loaded_capabilities
            return AgentDecision(action="final", message="已找到简历列表能力。")

    maker = Maker()
    tools = MainAgentToolRegistry(
        conversation_store=store, career_history_store=object(),
        skills_root=Path("skills"), resume_store=object(),
    )
    runtime = MainAgentRuntime(context_manager=manager, decision_maker=maker, tools=tools)
    result = runtime.run_turn(
        user_id="u1", conversation_id="c1", user_message="列出我的简历",
    )
    assert len(maker.offers) == 2
    assert "list_resumes" not in maker.offers[0]
    assert "list_resumes" in maker.offers[1]
    assert result.context.task.loaded_capabilities == ("list_resumes",)
