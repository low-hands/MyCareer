from __future__ import annotations

import pytest

from career_agent.agent.context_manager import ContextManager
from career_agent.agent.main_agent_contracts import (
    AgentDecision,
    CareerProfileContext,
    JobIntentUpdate,
    ToolCall,
)
from career_agent.agent.main_agent_runtime import MainAgentRuntime
from career_agent.agent.main_agent_tools import MainAgentToolRegistry
from career_agent.storage.context import CareerContextStore
from career_agent.storage.resumes import ResumeStore


class SequenceDecisionMaker:
    def __init__(self, *decisions: AgentDecision) -> None:
        self.decisions = list(decisions)
        self.contexts = []

    def decide(self, context, tool_specs):
        self.contexts.append(context)
        return self.decisions.pop(0)


def build(tmp_path, *decisions, profile=None):
    store = CareerContextStore(tmp_path / "context.sqlite3")
    resumes = ResumeStore(tmp_path / "resumes.sqlite3")
    manager = ContextManager(store)
    if profile is not None:
        manager.upsert_profile(profile)
    runtime = MainAgentRuntime(
        context_manager=manager,
        decision_maker=SequenceDecisionMaker(*decisions),
        tools=MainAgentToolRegistry(
            career_profile_store=store, resume_store=resumes
        ),
    )
    return runtime, store, resumes


def propose(**fields) -> AgentDecision:
    return AgentDecision(
        action="tool_call",
        tool_call=ToolCall(name="propose_job_intent", arguments=fields),
    )


def confirm() -> AgentDecision:
    return AgentDecision(
        action="tool_call", tool_call=ToolCall(name="confirm_job_intent", arguments={})
    )


def listed_roles() -> AgentDecision:
    return AgentDecision(
        action="tool_call",
        tool_call=ToolCall(name="list_target_roles", arguments={}),
    )


def final() -> AgentDecision:
    return AgentDecision(action="final", message="")


def test_intent_had_no_writer_before_these_tools(tmp_path) -> None:
    """The regression this closes: the schema existed with nothing to fill it."""
    assert "propose_job_intent" not in MainAgentToolRegistry().atomic_tool_names

    store = CareerContextStore(tmp_path / "context.sqlite3")
    wired = MainAgentToolRegistry(career_profile_store=store)

    assert "propose_job_intent" in wired.atomic_tool_names
    assert "confirm_job_intent" in wired.atomic_tool_names


def test_a_salary_cannot_be_recorded_against_the_person(tmp_path) -> None:
    """The scoping is a type rule, not a prompt instruction.

    A salary stored on the person cannot be un-mixed later: the two roles it was
    meant to distinguish have already collapsed into one number.
    """
    with pytest.raises(ValueError, match="belong to a target role"):
        JobIntentUpdate(salary_expectation="35K以上")
    with pytest.raises(ValueError, match="belong to a target role"):
        JobIntentUpdate(experience="3-5年")

    assert JobIntentUpdate(
        target_role_id="role-1", salary_expectation="35K以上"
    ).is_role_scoped


def test_a_city_without_a_role_is_the_persons_default(tmp_path) -> None:
    runtime, store, _ = build(tmp_path, propose(city="上海"), final())
    runtime.run_turn(user_id="u1", conversation_id="c1", user_message="我在上海找工作")

    runtime, store2, _ = build(tmp_path, confirm(), final())
    result = runtime.run_turn(user_id="u1", conversation_id="c1", user_message="对")

    assert store2.get_profile("u1").default_city == "上海"
    assert "整体求职意向" in result.assistant_message


def test_role_scoped_intent_lands_on_that_role_alone(tmp_path) -> None:
    """Two tracks, two salary bands: the whole reason this is not one profile."""
    resumes = ResumeStore(tmp_path / "resumes.sqlite3")
    agent_role = resumes.create_target_role(
        user_id="u1", title="AI Agent 工程师", priority=0
    )
    resumes.create_target_role(user_id="u1", title="大模型应用工程师", priority=1)

    runtime, _, _ = build(
        tmp_path,
        listed_roles(),
        propose(target_role_selection_index=1, salary_expectation="40K以上"),
        final(),
    )
    runtime.run_turn(user_id="u1", conversation_id="c1", user_message="这个方向我要 40K")

    runtime, _, resumes2 = build(tmp_path, confirm(), final())
    result = runtime.run_turn(user_id="u1", conversation_id="c1", user_message="确认")

    roles = {role.title: role for role in resumes2.list_target_roles(user_id="u1")}
    assert roles["AI Agent 工程师"].salary_expectation == "40K以上"
    assert roles["大模型应用工程师"].salary_expectation is None
    assert "AI Agent 工程师" in result.assistant_message
    assert roles["AI Agent 工程师"].id == agent_role.id


def test_a_role_city_overrides_without_touching_the_person_default(tmp_path) -> None:
    """"本地找产品岗、异地愿意去大厂" has to be expressible."""
    resumes = ResumeStore(tmp_path / "resumes.sqlite3")
    resumes.create_target_role(user_id="u1", title="AI Agent 工程师", priority=0)

    runtime, _, _ = build(
        tmp_path,
        listed_roles(),
        propose(target_role_selection_index=1, city="北京"),
        final(),
        profile=CareerProfileContext(user_id="u1", default_city="上海"),
    )
    runtime.run_turn(user_id="u1", conversation_id="c1", user_message="这个方向我愿意去北京")

    runtime, store2, resumes2 = build(tmp_path, confirm(), final())
    runtime.run_turn(user_id="u1", conversation_id="c1", user_message="确认")

    assert store2.get_profile("u1").default_city == "上海"
    assert resumes2.list_target_roles(user_id="u1")[0].city == "北京"


def test_proposing_saves_nothing(tmp_path) -> None:
    runtime, store, _ = build(tmp_path, propose(city="上海"), final())

    result = runtime.run_turn(
        user_id="u1", conversation_id="c1", user_message="我在上海找工作"
    )

    assert "确认后才会保存" in result.assistant_message
    assert store.get_profile("u1") is None
    assert result.context.task.pending_job_intent_update == JobIntentUpdate(city="上海")


def test_confirming_without_a_readback_is_refused(tmp_path) -> None:
    runtime, store, _ = build(tmp_path, confirm(), final())

    result = runtime.run_turn(user_id="u1", conversation_id="c1", user_message="好的")

    # A consumed-or-missing confirmation is now a grounded soft result, not a
    # turn-killing exception; nothing was written either way.
    assert result.tool_result is not None
    assert result.tool_result.state == "invalid_input"
    assert "proposed update the user has seen" in result.assistant_message
    assert store.get_profile("u1") is None


def test_a_spent_confirmation_cannot_be_reused(tmp_path) -> None:
    runtime, _, _ = build(tmp_path, propose(city="上海"), final())
    runtime.run_turn(user_id="u1", conversation_id="c1", user_message="上海")

    runtime, _, _ = build(tmp_path, confirm(), final())
    result = runtime.run_turn(user_id="u1", conversation_id="c1", user_message="确认")

    assert result.context.task.pending_job_intent_update is None


def test_an_omitted_field_is_left_alone_rather_than_cleared(tmp_path) -> None:
    """Naming a salary must not withdraw a city named last week."""
    resumes = ResumeStore(tmp_path / "resumes.sqlite3")
    role = resumes.create_target_role(user_id="u1", title="AI Agent 工程师", priority=0)
    resumes.update_target_role_intent(
        user_id="u1", target_role_id=role.id, city="北京", experience="3-5年"
    )

    updated = resumes.update_target_role_intent(
        user_id="u1", target_role_id=role.id, salary_expectation="40K以上"
    )

    assert updated.city == "北京"
    assert updated.experience == "3-5年"
    assert updated.salary_expectation == "40K以上"


def test_an_out_of_range_role_index_is_rejected(tmp_path) -> None:
    runtime, _, _ = build(
        tmp_path,
        propose(target_role_selection_index=3, salary_expectation="40K"),
        final(),
    )

    # Same soft-refusal contract as confirm-without-readback: the model sees a
    # grounded observation before the presenter delivers it, and nothing is
    # recorded.
    result = runtime.run_turn(
        user_id="u1", conversation_id="c1", user_message="记一下"
    )
    assert result.tool_result is not None
    assert result.tool_result.state == "invalid_input"
    assert "target-role selection index is out of range" in result.assistant_message


def test_the_tool_cannot_be_used_to_record_skills(tmp_path) -> None:
    """Capability claims stay on the resume path, which demands source quotes."""
    store = CareerContextStore(tmp_path / "context.sqlite3")
    schema = next(
        spec
        for spec in MainAgentToolRegistry(career_profile_store=store).schemas()
        if spec["function"]["name"] == "propose_job_intent"
    )

    assert set(schema["function"]["parameters"]["properties"]) == {
        "target_role_selection_index",
        "city",
        "salary_expectation",
        "experience",
        "education",
    }


def test_an_update_that_changes_nothing_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one field"):
        JobIntentUpdate()
