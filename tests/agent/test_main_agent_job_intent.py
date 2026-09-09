from __future__ import annotations

import pytest

from career_agent.harness.memory_telemetry import content_digest
from career_agent.agent.context_manager import ContextManager
from career_agent.agent.main_agent_contracts import (
    AgentDecision,
    CareerProfileContext,
    HardConstraintContext,
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


def build_with(tmp_path, decision_maker, profile=None):
    """``build`` with a caller-owned decision maker, so its contexts stay visible."""
    store = CareerContextStore(tmp_path / "context.sqlite3")
    resumes = ResumeStore(tmp_path / "resumes.sqlite3")
    manager = ContextManager(store)
    if profile is not None:
        manager.upsert_profile(profile)
    runtime = MainAgentRuntime(
        context_manager=manager,
        decision_maker=decision_maker,
        tools=MainAgentToolRegistry(career_profile_store=store, resume_store=resumes),
    )
    return runtime, store, resumes


def build(tmp_path, *decisions, profile=None):
    return build_with(tmp_path, SequenceDecisionMaker(*decisions), profile=profile)


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


def test_named_situational_intent_is_scoped_without_overwriting_global(
    tmp_path,
) -> None:
    runtime, _, _ = build(
        tmp_path,
        propose(
            city="北京",
            pref_scope="startup_interview",
            timescale="situational",
        ),
        final(),
        profile=CareerProfileContext(user_id="u1", default_city="上海"),
    )
    runtime.run_turn(
        user_id="u1",
        conversation_id="c1",
        user_message="创业公司面试时我可以考虑北京",
    )

    runtime, store, _ = build(tmp_path, confirm(), final())
    result = runtime.run_turn(
        user_id="u1",
        conversation_id="c1",
        user_message="确认",
    )

    assert store.get_profile("u1").default_city == "上海"
    scoped = store.list_profile_intent_versions(
        user_id="u1",
        scope_key="person_intent/self/default_city",
        pref_scope="startup_interview",
        active_only=True,
    )
    assert len(scoped) == 1
    assert scoped[0].value == "北京"
    assert scoped[0].layer == "transient"
    assert scoped[0].capture_action == "narrow-to-scope"
    assert "startup_interview" in result.assistant_message


def test_situational_intent_requires_a_named_scope() -> None:
    with pytest.raises(ValueError, match="named situation"):
        JobIntentUpdate(city="北京", timescale="situational")


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
    decisions = SequenceDecisionMaker(
        confirm(),
        AgentDecision(action="final", message="我还没给你看过要记的内容，先说一下？"),
    )
    runtime, store, _ = build_with(tmp_path, decisions)

    result = runtime.run_turn(user_id="u1", conversation_id="c1", user_message="好的")

    # A consumed-or-missing confirmation is a grounded soft result, not a
    # turn-killing exception; nothing was written either way.
    assert result.tool_result is None
    assert result.context.tool_observations[-1].state == "invalid_input"
    # The reason goes to the model, not to the user: it is the model that turns
    # "no proposal the user has seen" into a sentence worth reading.
    assert (
        "proposed update the user has seen"
        in decisions.contexts[-1].tool_observations[-1].message
    )
    assert result.assistant_message == "我还没给你看过要记的内容，先说一下？"
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
    decisions = SequenceDecisionMaker(
        propose(target_role_selection_index=3, salary_expectation="40K"),
        AgentDecision(action="final", message="你只有两个目标岗位，第 3 个不存在。"),
    )
    runtime, _, _ = build_with(tmp_path, decisions)

    # Same soft-refusal contract as confirm-without-readback: the model sees a
    # grounded observation and answers from it, and nothing is recorded.
    result = runtime.run_turn(
        user_id="u1", conversation_id="c1", user_message="记一下"
    )
    assert result.tool_result is None
    assert result.context.tool_observations[-1].state == "invalid_input"
    assert (
        "target-role selection index is out of range"
        in decisions.contexts[-1].tool_observations[-1].message
    )
    assert result.assistant_message == "你只有两个目标岗位，第 3 个不存在。"


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
        "pref_scope",
        "timescale",
        "city",
        "salary_expectation",
        "experience",
        "education",
        "hard_constraints",
    }


def test_an_update_that_changes_nothing_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one field"):
        JobIntentUpdate()


def test_default_city_updates_append_and_supersede_versions(tmp_path) -> None:
    store = CareerContextStore(tmp_path / "context.sqlite3")
    store.upsert_profile(CareerProfileContext(user_id="u1", default_city="上海"))
    store.upsert_profile(CareerProfileContext(user_id="u1", default_city="北京"))
    store.upsert_profile(CareerProfileContext(user_id="u1", default_city="北京"))

    versions = store.list_profile_intent_versions(
        user_id="u1",
        scope_key="person_intent/self/default_city",
    )

    assert [item.value for item in versions] == ["上海", "北京"]
    assert [item.revision for item in versions] == [1, 2]
    assert versions[0].superseded_by == versions[1].update_id
    assert versions[0].superseded_at == versions[1].valid_from
    assert versions[1].superseded_at is None
    assert versions[1].content_digest == content_digest("北京")


def test_role_intent_fields_have_independent_version_histories(tmp_path) -> None:
    resumes = ResumeStore(tmp_path / "resumes.sqlite3")
    role = resumes.create_target_role(
        user_id="u1", title="AI Agent 工程师", priority=0
    )
    resumes.update_target_role_intent(
        user_id="u1",
        target_role_id=role.id,
        city="北京",
        salary_expectation="35K",
    )
    resumes.update_target_role_intent(
        user_id="u1",
        target_role_id=role.id,
        salary_expectation="40K",
    )

    salary_scope = f"target_role_intent/{role.id}/salary_expectation"
    salary_versions = resumes.list_target_role_intent_versions(
        user_id="u1",
        scope_key=salary_scope,
    )
    city_versions = resumes.list_target_role_intent_versions(
        user_id="u1",
        scope_key=f"target_role_intent/{role.id}/city",
    )

    assert [item.value for item in salary_versions] == ["35K", "40K"]
    assert [item.revision for item in salary_versions] == [1, 2]
    assert salary_versions[0].superseded_by == salary_versions[1].update_id
    assert [item.value for item in city_versions] == ["北京"]
    assert city_versions[0].superseded_at is None


def test_normalization_equivalent_intent_does_not_create_a_revision(
    tmp_path,
) -> None:
    resumes = ResumeStore(tmp_path / "resumes.sqlite3")
    role = resumes.create_target_role(user_id="u1", title="Agent", priority=0)
    resumes.update_target_role_intent(
        user_id="u1",
        target_role_id=role.id,
        salary_expectation="40k 60k",
    )
    resumes.update_target_role_intent(
        user_id="u1",
        target_role_id=role.id,
        salary_expectation="40k\u300060k",
    )

    versions = resumes.list_target_role_intent_versions(
        user_id="u1",
        scope_key=f"target_role_intent/{role.id}/salary_expectation",
    )

    assert [(item.revision, item.value) for item in versions] == [
        (1, "40k 60k")
    ]


def test_confirmed_hard_constraint_is_versioned_and_projected(tmp_path) -> None:
    constraint = HardConstraintContext(
        relation="work_schedule",
        value="不接受996",
    )
    runtime, proposal_store, _ = build(
        tmp_path,
        propose(hard_constraints=[constraint.model_dump()]),
        final(),
    )
    proposed = runtime.run_turn(
        user_id="u1",
        conversation_id="c1",
        user_message="不接受996",
    )
    assert "确认后才会保存" in proposed.assistant_message
    assert proposal_store.get_profile("u1") is None

    runtime, store, _ = build(tmp_path, confirm(), final())
    confirmed = runtime.run_turn(
        user_id="u1",
        conversation_id="c1",
        user_message="确认",
    )

    assert "已记录" in confirmed.assistant_message
    runtime, _, _ = build(tmp_path, final())
    projected = runtime.run_turn(
        user_id="u1",
        conversation_id="c1",
        user_message="继续",
    )
    profile_file = projected.context.model_context()["career_profile"][
        "memory/profile.md"
    ]
    assert '- Work schedule: "不接受996"' in profile_file
    assert "- Last confirmed: 今天确认" in profile_file
    versions = store.list_profile_intent_versions(
        user_id="u1",
        scope_key="person_intent/self/work_schedule",
    )
    assert [(item.revision, item.value) for item in versions] == [(1, "不接受996")]
    assert versions[0].source == "confirmed_job_intent"


def test_hard_constraints_cannot_be_role_scoped_or_silently_removed(
    tmp_path,
) -> None:
    constraint = HardConstraintContext(
        relation="work_arrangement",
        value="必须远程",
    )
    with pytest.raises(ValueError, match="belong to the person"):
        JobIntentUpdate(
            target_role_id="role-1",
            hard_constraints=(constraint,),
        )

    store = CareerContextStore(tmp_path / "context.sqlite3")
    store.upsert_profile(
        CareerProfileContext(
            user_id="u1",
            hard_constraints=(constraint,),
        )
    )
    with pytest.raises(ValueError, match="M3 forget primitive"):
        store.upsert_profile(CareerProfileContext(user_id="u1"))
