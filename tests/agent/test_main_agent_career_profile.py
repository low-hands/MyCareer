from __future__ import annotations

import pytest

from career_agent.agent.context_manager import ContextManager
from career_agent.agent.main_agent_contracts import (
    AgentDecision,
    CareerProfileContext,
    CareerProfileUpdate,
    ConversationTaskState,
    ToolCall,
)
from career_agent.agent.main_agent_runtime import MainAgentRuntime
from career_agent.agent.main_agent_tools import MainAgentToolRegistry
from career_agent.storage.context import CareerContextStore


class SequenceDecisionMaker:
    def __init__(self, *decisions: AgentDecision) -> None:
        self.decisions = list(decisions)
        self.contexts = []

    def decide(self, context, tool_specs):
        self.contexts.append(context)
        return self.decisions.pop(0)


def build(tmp_path, *decisions: AgentDecision, profile: CareerProfileContext | None = None):
    store = CareerContextStore(tmp_path / "context.sqlite3")
    manager = ContextManager(store)
    if profile is not None:
        manager.upsert_profile(profile)
    decision_maker = SequenceDecisionMaker(*decisions)
    runtime = MainAgentRuntime(
        context_manager=manager,
        decision_maker=decision_maker,
        tools=MainAgentToolRegistry(career_profile_store=store),
    )
    return runtime, store, decision_maker


def propose(**fields) -> AgentDecision:
    return AgentDecision(
        action="tool_call",
        tool_call=ToolCall(name="propose_career_profile_update", arguments=fields),
    )


def confirm() -> AgentDecision:
    return AgentDecision(
        action="tool_call",
        tool_call=ToolCall(name="confirm_career_profile_update", arguments={}),
    )


def test_the_profile_had_no_writer_before_this_tool(tmp_path) -> None:
    """The regression this closes: the schema existed with nothing to fill it."""
    registry = MainAgentToolRegistry()

    assert "propose_career_profile_update" not in registry.atomic_tool_names
    assert "confirm_career_profile_update" not in registry.atomic_tool_names

    store = CareerContextStore(tmp_path / "context.sqlite3")
    wired = MainAgentToolRegistry(career_profile_store=store)

    assert "propose_career_profile_update" in wired.atomic_tool_names
    assert "confirm_career_profile_update" in wired.atomic_tool_names


def test_proposing_reads_back_without_saving(tmp_path) -> None:
    runtime, store, _ = build(
        tmp_path,
        propose(default_city="上海", target_roles=["AI Agent 工程师"]),
        AgentDecision(action="final", message=""),
    )

    result = runtime.run_turn(
        user_id="u1", conversation_id="c1", user_message="我想找上海的 AI Agent 岗位"
    )

    assert "默认城市：上海" in result.assistant_message
    assert "确认后才会保存" in result.assistant_message
    assert store.get_profile("u1") is None
    assert result.context.task.pending_career_profile_update == CareerProfileUpdate(
        default_city="上海", target_roles=("AI Agent 工程师",)
    )


def test_confirming_writes_the_profile(tmp_path) -> None:
    runtime, store, _ = build(
        tmp_path,
        propose(default_city="上海"),
        AgentDecision(action="final", message=""),
    )
    runtime.run_turn(user_id="u1", conversation_id="c1", user_message="找上海的岗位")

    runtime, store2, _ = build(tmp_path, confirm(), AgentDecision(action="final", message=""))
    result = runtime.run_turn(user_id="u1", conversation_id="c1", user_message="对")

    stored = store2.get_profile("u1")
    assert stored is not None
    assert stored.default_city == "上海"
    assert "已记录" in result.assistant_message
    # The confirmation is spent, so a later turn cannot silently reuse it.
    assert result.context.task.pending_career_profile_update is None


def test_confirming_without_a_readback_is_refused(tmp_path) -> None:
    """Confirmation must point at something the user was actually shown."""
    runtime, store, _ = build(tmp_path, confirm())

    with pytest.raises(ValueError, match="proposed update the user has seen"):
        runtime.run_turn(user_id="u1", conversation_id="c1", user_message="好的")

    assert store.get_profile("u1") is None


def test_an_omitted_field_is_left_alone_rather_than_cleared(tmp_path) -> None:
    """Mentioning a city must not erase a salary stated three turns ago."""
    existing = CareerProfileContext(
        user_id="u1", default_city="北京", salary_preference="25K以上"
    )
    runtime, store, _ = build(
        tmp_path,
        propose(default_city="上海"),
        AgentDecision(action="final", message=""),
        profile=existing,
    )
    runtime.run_turn(user_id="u1", conversation_id="c1", user_message="改成上海")

    runtime, store2, _ = build(
        tmp_path, confirm(), AgentDecision(action="final", message="")
    )
    result = runtime.run_turn(user_id="u1", conversation_id="c1", user_message="确认")

    stored = store2.get_profile("u1")
    assert stored.default_city == "上海"
    assert stored.salary_preference == "25K以上"
    assert "北京 → 上海" in result.assistant_message


def test_the_readback_shows_the_previous_value_when_it_changes(tmp_path) -> None:
    runtime, _, _ = build(
        tmp_path,
        propose(default_city="上海"),
        AgentDecision(action="final", message=""),
        profile=CareerProfileContext(user_id="u1", default_city="北京"),
    )

    result = runtime.run_turn(
        user_id="u1", conversation_id="c1", user_message="改成上海"
    )

    assert "默认城市：北京 → 上海" in result.assistant_message


def test_the_tool_cannot_be_used_to_record_skills(tmp_path) -> None:
    """Capability claims stay on the resume path, which demands source quotes."""
    store = CareerContextStore(tmp_path / "context.sqlite3")
    registry = MainAgentToolRegistry(career_profile_store=store)
    schema = next(
        spec
        for spec in registry.schemas()
        if spec["function"]["name"] == "propose_career_profile_update"
    )

    assert set(schema["function"]["parameters"]["properties"]) == {
        "target_roles",
        "default_city",
        "salary_preference",
        "experience",
        "education",
    }


def test_an_update_that_changes_nothing_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one field"):
        CareerProfileUpdate()
    with pytest.raises(ValueError, match="cannot be set to an empty list"):
        CareerProfileUpdate(target_roles=())
