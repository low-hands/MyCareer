"""Tool profiles and the route that switches them.

``tool_profile`` is its own axis of task state: which fixed tool group the next
decision sees. It must not lean on ``active_workflow``, which answers whether a
suspended run needs resuming. These tests hold that separation, the route's
CONTROL classification, and the control-slot disclosure derived from the same
tables the runtime enforces.
"""

from __future__ import annotations

from pathlib import Path

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
from career_agent.agent.tool_reachability import REQUIREMENTS, reachable
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
from career_agent.storage.resumes import ResumeStore


class SequenceDecisionMaker:
    def __init__(self, *decisions: AgentDecision) -> None:
        self.decisions = list(decisions)
        self.contexts = []
        self.schemas = []

    def decide(self, context, tool_names):
        self.contexts.append(context)
        self.schemas.append(tool_names)
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
        skills_root=Path("skills"),
        resume_job_match_service=object(),
        job_analysis_service=object(),
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


def test_the_resume_profile_offers_the_step_that_unlocks_matching() -> None:
    """Matching needs a current JD analysis, so its profile must offer one.

    Without analyze_job the resume profile offered neither the match tool nor
    the step that unlocks it, and the model wandered through job lookups.
    """
    unanalyzed = project_tool_availability(
        ConversationTaskState(
            tool_profile="resume",
            active_resume_version_id="rv-1",
            active_job_posting_id="job-1",
        )
    )

    assert "analyze_job" in unanalyzed["available_now"]
    assert "match_resume_to_job" not in unanalyzed["available_now"]
    assert "analyze_job" in REQUIREMENTS["match_resume_to_job"]
    # The cap keeps the step that is actionable now. Ordering by name used to
    # cut it behind tailoring steps that wait on matching itself.
    assert any(
        line.startswith("match_resume_to_job:")
        for line in unanalyzed["next_requirements"]
    )
    assert not any(
        line.startswith("draft_resume_tailoring")
        for line in unanalyzed["next_requirements"]
    )


def test_availability_is_derived_from_profile_and_preconditions() -> None:
    cold = project_tool_availability(ConversationTaskState())
    assert cold["tool_profile"] == "core"
    assert "find_saved_jobs" in cold["available_now"]
    assert ROUTE_TOOL in cold["available_now"]
    assert "export_resume_artifact" not in cold["available_now"]
    # Core discloses only its own gaps; the resume chain's are not mentioned.
    assert len(cold["next_requirements"]) <= MAX_NEXT_REQUIREMENTS
    assert not any("match_resume_to_job" in line for line in cold["next_requirements"])

    resume = project_tool_availability(ConversationTaskState(tool_profile="resume"))
    assert "list_resumes" in resume["available_now"]
    assert "export_resume_artifact" not in resume["available_now"]
    assert 0 < len(resume["next_requirements"]) <= MAX_NEXT_REQUIREMENTS
    assert all(isinstance(line, str) for line in resume["next_requirements"])

    ready = project_tool_availability(
        ConversationTaskState(
            tool_profile="resume",
            active_resume_version_id="rv-1",
            active_job_posting_id="job-1",
            active_jd_snapshot_id="jd-1",
            active_job_analysis_id="analysis-1",
            active_job_analysis_jd_snapshot_id="jd-1",
            job_analysis_status="ready",
        )
    )
    assert {"export_resume_artifact", "match_resume_to_job"} <= set(ready["available_now"])
    assert "draft_resume_tailoring" not in ready["available_now"]


@pytest.mark.parametrize("profile", TOOL_PROFILE_NAMES)
@pytest.mark.parametrize("inputs_present", [False, True])
def test_unmet_requirements_name_only_the_tools_they_block(
    profile, inputs_present
) -> None:
    task = ConversationTaskState(
        tool_profile=profile,
        active_resume_version_id="rv-1" if inputs_present else None,
        active_job_posting_id="job-1" if inputs_present else None,
    )
    availability = project_tool_availability(task)
    requirements = availability["next_requirements"]
    assert 0 < len(requirements) <= MAX_NEXT_REQUIREMENTS
    for entry in requirements:
        names, separator, requirement = entry.partition(": ")
        assert separator
        for name in names.split(", "):
            assert name in profile_tools(profile)
            assert name not in availability["available_now"]
            assert not reachable(name, task)
            assert REQUIREMENTS[name] == requirement


def _manager(tmp_path):
    manager = ContextManager(CareerContextStore(tmp_path / "context.sqlite3"))
    manager.upsert_profile(CareerProfileContext(user_id="u1"))
    return manager


@pytest.mark.parametrize("profile", TOOL_PROFILE_NAMES)
def test_shared_read_runs_in_every_profile_without_routing(tmp_path, profile) -> None:
    store = CareerContextStore(tmp_path / "context.sqlite3")
    store.upsert_task(
        user_id="u1",
        conversation_id="c1",
        task=ConversationTaskState(tool_profile=profile),
    )
    manager = ContextManager(store)
    manager.upsert_profile(CareerProfileContext(user_id="u1"))
    tools = CountingRegistry(resume_store=ResumeStore(tmp_path / "resumes.sqlite3"))
    decisions = SequenceDecisionMaker(
        AgentDecision(
            action="tool_call",
            tool_call=ToolCall(name="list_resumes", arguments={}),
        ),
        AgentDecision(action="final", message="已读取简历列表。"),
    )
    runtime = MainAgentRuntime(
        context_manager=manager, decision_maker=decisions, tools=tools
    )

    result = runtime.run_turn(
        user_id="u1", conversation_id="c1", user_message="列出我的简历"
    )

    assert [name for name, _ in tools.calls] == ["list_resumes"]
    assert result.context.task.tool_profile == (
        "resume" if profile == "core" else profile
    )
    assert result.delegated_read_count == 1
    assert result.delegated_write_count == 0
    assert all("list_resumes" in _offered(schemas) for schemas in decisions.schemas)


@pytest.mark.parametrize("has_resume", [False, True])
@pytest.mark.parametrize("has_job", [False, True])
@pytest.mark.parametrize("has_current_analysis", [False, True])
@pytest.mark.parametrize("has_match", [False, True])
def test_input_and_artifact_preconditions_remain_independent(
    has_resume, has_job, has_current_analysis, has_match
) -> None:
    task = ConversationTaskState(
        tool_profile="resume",
        active_resume_version_id="rv-1" if has_resume else None,
        active_job_posting_id="job-1" if has_job else None,
        active_jd_snapshot_id="jd-1" if has_job else None,
        active_job_analysis_id=(
            "analysis-1" if has_job and has_current_analysis else None
        ),
        active_job_analysis_jd_snapshot_id=(
            "jd-1" if has_job and has_current_analysis else None
        ),
        job_analysis_status=(
            "ready" if has_job and has_current_analysis else None
        ),
        active_resume_job_match_id="match-1" if has_match else None,
    )
    available = project_tool_availability(task)["available_now"]

    assert ("export_resume_artifact" in available) is has_resume
    assert ("match_resume_to_job" in available) is (
        has_resume and has_job and has_current_analysis
    )
    assert ("draft_resume_tailoring" in available) is has_match


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


def _offered(schemas) -> set[str]:
    return {schema["function"]["name"] for schema in schemas}


def test_model_receives_exactly_the_profile_schemas_and_the_same_tuple_within_a_profile(
    tmp_path,
) -> None:
    manager = _manager(tmp_path)
    tools = _registry()
    registered = _offered(tools.schemas())
    assert len(registered) > len(profile_tools("resume"))
    decisions = SequenceDecisionMaker(
        AgentDecision(action="tool_call", tool_call=ToolCall(name=ROUTE_TOOL, arguments={"domain": "resume"})),
        # Routing to the current profile re-enters decide without switching.
        AgentDecision(action="tool_call", tool_call=ToolCall(name=ROUTE_TOOL, arguments={"domain": "resume"})),
        AgentDecision(action="tool_call", tool_call=ToolCall(name=ROUTE_TOOL, arguments={"domain": "core"})),
        AgentDecision(action="final", message="完成。"),
    )
    runtime = MainAgentRuntime(context_manager=manager, decision_maker=decisions, tools=tools)

    runtime.run_turn(user_id="u1", conversation_id="c1", user_message="继续")

    core, resume, resume_again, core_again = decisions.schemas
    assert _offered(core) == profile_tools("core") & registered
    assert _offered(resume) == profile_tools("resume") & registered
    assert "export_resume_artifact" in _offered(resume)
    assert "export_resume_artifact" not in _offered(core)
    assert "match_resume_to_job" not in _offered(core)
    assert not (_offered(resume) - profile_tools("resume"))
    # The prefix cache depends on the same object being reused within a
    # profile and swapped exactly at the switch.
    assert resume is resume_again
    assert core is core_again
    assert core is not resume


def test_schema_filtering_only_offers_tools_the_registry_installs(tmp_path) -> None:
    manager = _manager(tmp_path)
    tools = MainAgentToolRegistry(job_repository=object(), resume_store=object())
    registered = _offered(tools.schemas())
    assert "match_resume_to_job" not in registered
    assert "list_resumes" in registered
    decisions = SequenceDecisionMaker(
        AgentDecision(action="tool_call", tool_call=ToolCall(name=ROUTE_TOOL, arguments={"domain": "resume"})),
        AgentDecision(action="final", message="完成。"),
    )
    runtime = MainAgentRuntime(context_manager=manager, decision_maker=decisions, tools=tools)

    runtime.run_turn(user_id="u1", conversation_id="c1", user_message="看看我的简历")

    offered = _offered(decisions.schemas[1])
    assert offered == profile_tools("resume") & registered
    assert offered <= registered


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


class _EmailService:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def sync(self, **kwargs):
        self.calls.append(kwargs)
        raise AssertionError("an out-of-profile tool must never reach its service")

    def list_events(self, **kwargs):
        return ()


def test_a_tool_outside_the_current_profile_is_refused_before_it_runs(tmp_path) -> None:
    """Filtering the schemas hides a tool; authorization has to refuse it too.

    A model can name a tool from memory that its current profile never offered.
    The refusal is a soft observation so the model can route and retry, and the
    same name runs once the profile is right.
    """

    manager = _manager(tmp_path)
    service = _EmailService()
    tools = MainAgentToolRegistry(email_tracking_service=service)
    decisions = SequenceDecisionMaker(
        AgentDecision(action="tool_call", tool_call=ToolCall(name="sync_application_emails", arguments={})),
        AgentDecision(action="final", message="先不查邮箱。"),
    )
    runtime = MainAgentRuntime(context_manager=manager, decision_maker=decisions, tools=tools)

    result = runtime.run_turn(user_id="u1", conversation_id="c1", user_message="查邮箱")

    assert service.calls == []
    assert result.context.task.email_sync_phase is None
    assert result.context.task.tool_profile == "core"
    refusal = result.context.tool_observations[-1]
    assert refusal.tool_name == "sync_application_emails"
    assert refusal.state == "authorization_refused"
    assert "core" in refusal.message
    assert ROUTE_TOOL in (refusal.next_action or "")
    assert "sync_application_emails" not in _offered(decisions.schemas[0])
