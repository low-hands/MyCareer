"""Tool profiles and the route that switches them.

``tool_profile`` is its own axis of task state: which fixed tool group the next
decision sees. It must not lean on ``active_workflow``, which answers whether a
suspended run needs resuming. These tests hold that separation, the route's
CONTROL classification, and the control-slot disclosure derived from the same
tables the runtime enforces.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from career_agent.agent.decision_messages import project_decision_messages
from career_agent.agent.main_agent_contracts import (
    DOMAIN_TOOL_PROFILES,
    TOOL_PROFILE_NAMES,
    AgentDecision,
    ConversationTaskState,
    RouteToCapabilityToolArguments,
    ToolCall,
    ToolResult,
)
from career_agent.agent.main_agent_reducers import reduce_task_state
from career_agent.agent.main_agent_runtime import MainAgentRuntime
from career_agent.agent.tool_effects import effect_for
from career_agent.agent.tool_profiles import (
    CORE_TOOLS,
    MAX_NEXT_REQUIREMENTS,
    ROUTE_TOOL,
    TOOL_PROFILES,
    project_tool_availability,
    profile_tools,
)
from career_agent.agent.context_manager import ContextManager
from career_agent.agent.main_agent_contracts import CareerProfileContext
from career_agent.agent.main_agent_tools import MainAgentToolRegistry
from career_agent.storage.context import CareerContextStore


class SequenceDecisionMaker:
    def __init__(self, *decisions: AgentDecision) -> None:
        self.decisions = list(decisions)
        self.contexts = []

    def decide(self, context, tool_names):
        self.contexts.append(context)
        if not self.decisions:
            raise AssertionError("Main Agent requested more decisions than expected")
        return self.decisions.pop(0)


class CountingRegistry(MainAgentToolRegistry):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.calls: list[tuple[str, dict]] = []

    def invoke_atomic_tool(self, name, arguments):
        self.calls.append((name, dict(arguments)))
        return super().invoke_atomic_tool(name, arguments)


def _registry() -> MainAgentToolRegistry:
    return MainAgentToolRegistry(
        job_repository=object(),
        career_profile_store=object(),
        resume_store=object(),
        resume_analysis_service=object(),
        resume_job_match_service=object(),
        resume_tailoring_service=object(),
        resume_export_service=object(),
        application_service=object(),
        email_tracking_service=object(),
        interview_service=object(),
        interview_preparation_service=object(),
        action_center_service=object(),
        calendar_service=object(),
        mock_interview_graph=object(),
        mock_interview_store=object(),
        job_research_service=object(),
        job_comparison_service=object(),
        conversation_store=object(),
        career_history_store=object(),
        episode_store=object(),
        working_notes_store=object(),
        owner_settings_store=object(),
    )


def test_profile_defaults_to_core_and_rejects_unknown_values() -> None:
    assert ConversationTaskState().tool_profile == "core"
    assert set(TOOL_PROFILE_NAMES) == {
        "core", "job", "resume", "application", "interview", "memory",
    }
    with pytest.raises(ValidationError):
        ConversationTaskState.model_validate({"tool_profile": "workflow"})
    with pytest.raises(ValidationError):
        RouteToCapabilityToolArguments.model_validate({"domain": "jobs"})


def test_profile_round_trips_and_is_independent_of_active_workflow() -> None:
    task = ConversationTaskState(tool_profile="resume")
    restored = ConversationTaskState.model_validate_json(task.model_dump_json())
    assert restored.tool_profile == "resume"
    assert restored.active_workflow == "none"
    # The workflow slot's own invariant is untouched by the profile axis.
    with pytest.raises(ValidationError):
        ConversationTaskState(active_workflow="job_discovery", tool_profile="job")
    suspended = ConversationTaskState(
        active_workflow="mock_interview", run_id="run-1", tool_profile="interview"
    )
    assert suspended.tool_profile == "interview"


def test_every_profile_contains_core_and_every_tool_has_a_home() -> None:
    universe = set(_registry().names)
    for profile in TOOL_PROFILE_NAMES:
        assert CORE_TOOLS <= profile_tools(profile)
        assert profile_tools(profile) <= universe
    assert ROUTE_TOOL in CORE_TOOLS
    for domain in DOMAIN_TOOL_PROFILES:
        assert TOOL_PROFILES[domain] > CORE_TOOLS


def test_route_is_a_control_capability() -> None:
    assert effect_for(ROUTE_TOOL) == "CONTROL"


def test_route_reducer_switches_profile_only_on_success() -> None:
    task = ConversationTaskState()
    switched = reduce_task_state(
        task,
        ToolResult(
            tool_name=ROUTE_TOOL,
            state="tool_profile_switched",
            message="ok",
            payload={"tool_profile": "resume"},
        ),
    )
    assert switched.tool_profile == "resume"
    unchanged = reduce_task_state(
        switched,
        ToolResult(
            tool_name=ROUTE_TOOL,
            state="tool_profile_unchanged",
            message="ok",
            payload={"tool_profile": "job"},
        ),
    )
    assert unchanged.tool_profile == "resume"
    bogus = reduce_task_state(
        switched,
        ToolResult(
            tool_name=ROUTE_TOOL,
            state="tool_profile_switched",
            message="ok",
            payload={"tool_profile": "nope"},
        ),
    )
    assert bogus.tool_profile == "resume"


def test_availability_is_derived_from_profile_and_preconditions() -> None:
    cold = project_tool_availability(ConversationTaskState())
    assert cold["tool_profile"] == "core"
    assert "find_saved_jobs" in cold["available_now"]
    assert ROUTE_TOOL in cold["available_now"]
    assert "analyze_resume" not in cold["available_now"]
    # Core discloses only its own gaps; the resume chain's are not mentioned.
    assert len(cold["next_requirements"]) <= MAX_NEXT_REQUIREMENTS
    assert not any("match_resume_to_job" in line for line in cold["next_requirements"])

    resume = project_tool_availability(ConversationTaskState(tool_profile="resume"))
    assert "list_resumes" in resume["available_now"]
    assert "analyze_resume" not in resume["available_now"]
    assert 0 < len(resume["next_requirements"]) <= MAX_NEXT_REQUIREMENTS
    assert all(isinstance(line, str) for line in resume["next_requirements"])

    ready = project_tool_availability(
        ConversationTaskState(
            tool_profile="resume",
            active_resume_version_id="rv-1",
            active_job_posting_id="job-1",
        )
    )
    assert {"analyze_resume", "match_resume_to_job"} <= set(ready["available_now"])
    assert "draft_resume_tailoring" not in ready["available_now"]


def _manager(tmp_path):
    manager = ContextManager(CareerContextStore(tmp_path / "context.sqlite3"))
    manager.upsert_profile(CareerProfileContext(user_id="u1"))
    return manager


def test_control_slot_carries_profile_and_availability_not_blocked_map(tmp_path) -> None:
    manager = _manager(tmp_path)
    context = manager.load_for_turn(
        user_id="u1", conversation_id="c1", user_message="帮我改简历"
    )
    control = project_decision_messages(context).control["task"]
    assert control["tool_profile"] == "core"
    assert set(control["available_now"]) <= CORE_TOOLS
    assert isinstance(control["next_requirements"], list)
    assert "blocked" not in control
    data = project_decision_messages(context).data["task"]
    assert "available_now" not in data


def test_route_persists_profile_re_enters_decide_and_spends_no_budget(tmp_path) -> None:
    manager = _manager(tmp_path)
    tools = CountingRegistry()
    decisions = SequenceDecisionMaker(
        AgentDecision(action="tool_call", tool_call=ToolCall(name=ROUTE_TOOL, arguments={"domain": "resume"})),
        AgentDecision(action="tool_call", tool_call=ToolCall(name=ROUTE_TOOL, arguments={"domain": "interview"})),
        AgentDecision(action="final", message="已进入面试工具档。"),
    )
    runtime = MainAgentRuntime(
        context_manager=manager,
        decision_maker=decisions,
        tools=tools,
        max_read_calls=1,
        max_write_calls=1,
    )

    result = runtime.run_turn(user_id="u1", conversation_id="c1", user_message="改简历并准备面试")

    assert [name for name, _ in tools.calls] == [ROUTE_TOOL, ROUTE_TOOL]
    assert [item.state for item in result.tool_results] == [
        "tool_profile_switched",
        "tool_profile_switched",
    ]
    assert result.context.task.tool_profile == "interview"
    assert result.delegated_read_count == 0
    assert result.delegated_write_count == 0
    # The second decision already saw the first switch.
    assert decisions.contexts[1].task.tool_profile == "resume"
    assert decisions.contexts[2].task.tool_profile == "interview"
    reloaded = manager.load_for_turn(
        user_id="u1", conversation_id="c1", user_message="继续"
    )
    assert reloaded.task.tool_profile == "interview"


def test_routing_to_the_current_profile_does_not_switch_and_is_deduplicated(tmp_path) -> None:
    manager = _manager(tmp_path)
    tools = CountingRegistry()
    decisions = SequenceDecisionMaker(
        AgentDecision(action="tool_call", tool_call=ToolCall(name=ROUTE_TOOL, arguments={"domain": "core"})),
        AgentDecision(action="tool_call", tool_call=ToolCall(name=ROUTE_TOOL, arguments={"domain": "core"})),
        AgentDecision(action="final", message="仍在核心档。"),
    )
    runtime = MainAgentRuntime(context_manager=manager, decision_maker=decisions, tools=tools)

    result = runtime.run_turn(user_id="u1", conversation_id="c1", user_message="你好")

    assert len(tools.calls) == 1
    assert [item.state for item in result.tool_results] == ["tool_profile_unchanged"]
    assert result.context.tool_observations[-1].state == "authorization_refused"
    assert result.context.task.tool_profile == "core"
