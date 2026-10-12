from __future__ import annotations

import pytest

from career_agent.harness.memory_telemetry import content_digest
from career_agent.agent.context.manager import ContextManager
from career_agent.agent.contracts.decisions import (
    AgentDecision,
    ToolCall,
)
from career_agent.agent.contracts.profile import (
    CareerProfileContext,
    HardConstraintContext,
    JobIntentUpdate,
)
from career_agent.agent.contracts.projections.memory import project_job_intent_arguments
from career_agent.agent.contracts.questionnaire import QuestionAnswer, UserQuestion
from career_agent.agent.runtime.main_agent_runtime import MainAgentRuntime
from career_agent.agent.capabilities.registry import MainAgentToolRegistry
from career_agent.harness.streaming import InteractionRequiredEvent, InteractionResponse
from career_agent.storage.context import CareerContextStore
from career_agent.storage.checkpoints import SQLiteCheckpointOwner
from career_agent.storage.resumes import ResumeStore
from agent_test_support import load_capability_family


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
    load_capability_family(store, "job")
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


def test_one_confirmation_saves_new_role_city_and_company(tmp_path) -> None:
    runtime, store, resumes = build(
        tmp_path,
        propose(new_target_role_title="产品经理", city="上海",
                add_target_companies=["ByteDance"]),
        final(),
    )
    proposed = runtime.run_turn(
        user_id="u1", conversation_id="c1", user_message="记一下上海、产品经理、ByteDance"
    )
    assert "新建目标岗位：产品经理" in proposed.assistant_message
    assert "意向城市：上海" in proposed.assistant_message
    assert "新增目标公司：ByteDance" in proposed.assistant_message
    assert resumes.list_target_roles(user_id="u1") == ()
    assert store.get_profile("u1") is None

    runtime, store, resumes = build(tmp_path, confirm(), final())
    confirmed = runtime.run_turn(
        user_id="u1", conversation_id="c1", user_message="确认"
    )
    roles = resumes.list_target_roles(user_id="u1")
    assert len(roles) == 1
    assert (roles[0].title, roles[0].city) == ("产品经理", "上海")
    assert store.get_profile("u1").target_companies == ("字节跳动",)
    assert confirmed.tool_result.state == "job_intent_recorded"


def test_existing_role_is_reused_on_new_title_proposal(tmp_path) -> None:
    resumes = ResumeStore(tmp_path / "resumes.sqlite3")
    existing = resumes.create_target_role(user_id="u1", title="产品经理", priority=0)
    runtime, _, _ = build(
        tmp_path, propose(new_target_role_title=" 产品经理 ", city="上海"), final()
    )
    proposed = runtime.run_turn(
        user_id="u1", conversation_id="c1", user_message="我找产品经理岗位，上海"
    )
    assert "使用已有岗位：产品经理" in proposed.assistant_message
    runtime, _, resumes = build(tmp_path, confirm(), final())
    runtime.run_turn(user_id="u1", conversation_id="c1", user_message="确认")
    roles = resumes.list_target_roles(user_id="u1")
    assert len(roles) == 1
    assert (roles[0].id, roles[0].city) == (existing.id, "上海")


def test_new_role_follows_existing_priority_and_scoped_intent_is_atomic(tmp_path) -> None:
    resumes = ResumeStore(tmp_path / "resumes.sqlite3")
    resumes.create_target_role(user_id="u1", title="后端工程师", priority=4)
    runtime, _, _ = build(
        tmp_path,
        propose(new_target_role_title="产品经理", salary_expectation="30K",
                pref_scope="startup", timescale="situational"),
        final(),
    )
    runtime.run_turn(user_id="u1", conversation_id="c1", user_message="创业公司产品经理岗位期望30K")
    runtime, _, resumes = build(tmp_path, confirm(), final())
    runtime.run_turn(user_id="u1", conversation_id="c1", user_message="确认")
    role = next(role for role in resumes.list_target_roles(user_id="u1") if role.title == "产品经理")
    assert role.priority == 5
    assert role.salary_expectation is None
    versions = resumes.list_target_role_intent_versions(
        user_id="u1", scope_key=f"target_role_intent/{role.id}/salary_expectation",
        pref_scope="startup", active_only=True,
    )
    assert len(versions) == 1 and versions[0].value == "30K"


def test_target_companies_are_incremental_and_canonical(tmp_path) -> None:
    for message, fields in (
        ("记下字节跳动", {"add_target_companies": ["字节跳动"]}),
        ("再加上拼多多", {"add_target_companies": ["拼多多"]}),
        ("再加 ByteDance", {"add_target_companies": ["ByteDance"]}),
        ("不考虑字节了", {"remove_target_companies": ["ByteDance"]}),
    ):
        runtime, _, _ = build(tmp_path, propose(**fields), final())
        runtime.run_turn(user_id="u1", conversation_id="c1", user_message=message)
        runtime, store, _ = build(tmp_path, confirm(), final())
        runtime.run_turn(user_id="u1", conversation_id="c1", user_message="确认")
    assert store.get_profile("u1").target_companies == ("拼多多",)
    runtime, _, _ = build(tmp_path, final())
    projected = runtime.run_turn(
        user_id="u1", conversation_id="c1", user_message="我的目标公司是哪家"
    )
    assert '"拼多多"' in projected.context.model_context()["career_profile"]["memory/profile.md"]


def test_target_company_versions_close_on_removal_and_project_confirmation_time(tmp_path) -> None:
    store = CareerContextStore(tmp_path / "context.sqlite3")
    store.upsert_profile(CareerProfileContext(user_id="u1", target_companies=("字节跳动",)))
    scope = "person_intent/self/target_company/字节跳动"
    version = store.list_profile_intent_versions(user_id="u1", scope_key=scope)
    assert len(version) == 1 and version[0].source == "career_profile_upsert"

    runtime, _, _ = build(tmp_path, final())
    projected = runtime.run_turn(user_id="u1", conversation_id="c1", user_message="继续")
    profile_file = projected.context.model_context()["career_profile"]["memory/profile.md"]
    assert '- Target company: "字节跳动"' in profile_file
    assert "- Last confirmed: 今天确认" in profile_file
    assert any(binding.entry_id == scope for binding in projected.context.profile.telemetry_bindings)

    store.upsert_profile(CareerProfileContext(user_id="u1", target_companies=()))
    closed = store.list_profile_intent_versions(user_id="u1", scope_key=scope)
    assert len(closed) == 1 and closed[0].superseded_at is not None
    assert store.list_profile_intent_versions(user_id="u1", scope_key=scope, active_only=True) == ()

    store.upsert_profile(CareerProfileContext(user_id="u1", target_companies=("ByteDance",)))
    reopened = store.list_profile_intent_versions(user_id="u1", scope_key=scope, active_only=True)
    assert len(reopened) == 1 and reopened[0].revision == 2


@pytest.mark.parametrize(
    ("user_message", "proposed_name", "canonical"),
    [
        ("目标公司加上阿里", "阿里巴巴", "阿里巴巴"),
        ("还有 PDD", "拼多多", "拼多多"),
    ],
)
def test_target_company_short_name_counts_as_user_evidence(
    tmp_path, user_message, proposed_name, canonical,
) -> None:
    runtime, _, _ = build(
        tmp_path, propose(add_target_companies=[proposed_name]), final()
    )
    proposed = runtime.run_turn(
        user_id="u1", conversation_id="c1", user_message=user_message,
    )
    assert proposed.tool_result.state == "job_intent_proposed"
    runtime, store, _ = build(tmp_path, confirm(), final())
    runtime.run_turn(user_id="u1", conversation_id="c1", user_message="确认")
    assert store.get_profile("u1").target_companies == (canonical,)


def test_recent_user_company_names_survive_a_clarification_turn(tmp_path) -> None:
    runtime, _, _ = build(
        tmp_path, AgentDecision(action="final", message="两家公司都作为目标公司吗？")
    )
    runtime.run_turn(
        user_id="u1", conversation_id="c1",
        user_message="字节跳动和拼多多",
    )
    runtime, _, _ = build(
        tmp_path,
        propose(add_target_companies=["字节跳动", "拼多多"]), final(),
    )
    proposed = runtime.run_turn(
        user_id="u1", conversation_id="c1", user_message="对，都记上",
    )
    assert proposed.tool_result.state == "job_intent_proposed"
    runtime, store, _ = build(tmp_path, confirm(), final())
    runtime.run_turn(user_id="u1", conversation_id="c1", user_message="确认")
    assert store.get_profile("u1").target_companies == ("字节跳动", "拼多多")


def test_company_evidence_survives_a_short_prompt_window(tmp_path) -> None:
    store = CareerContextStore(tmp_path / "context.sqlite3")
    writer = ContextManager(store)
    for message in ("目标公司是字节跳动", "能再确认一下吗"):
        context = writer.load_for_turn(user_id="u1", conversation_id="c1", user_message=message)
        writer.commit_turn(context=context, task=context.task, assistant_message="请确认。")
    narrow = ContextManager(store, recent_message_limit=2)
    context = narrow.load_for_turn(user_id="u1", conversation_id="c1", user_message="对，都记上")
    assert all("字节跳动" not in message.content for message in context.recent_messages)
    projected = project_job_intent_arguments(
        context, "propose_job_intent", {"add_target_companies": ["字节跳动"]},
    )
    assert projected["update"].add_target_companies == ("字节跳动",)


def test_company_proposal_after_persistent_checkpoint_resume(tmp_path) -> None:
    context_path = tmp_path / "context.sqlite3"
    checkpoint_path = tmp_path / "checkpoints.sqlite3"
    store = CareerContextStore(context_path)
    load_capability_family(store, "job")
    questions = tuple(
        UserQuestion(question_id=f"q{index}", prompt=f"问题 {index}", kind="free_text")
        for index in range(1, 6)
    )
    first_owner = SQLiteCheckpointOwner(checkpoint_path)
    first_runtime = MainAgentRuntime(
        context_manager=ContextManager(store),
        decision_maker=SequenceDecisionMaker(
            AgentDecision(action="questionnaire", message="请确认。", questions=questions)
        ),
        tools=MainAgentToolRegistry(career_profile_store=store),
        checkpointer=first_owner.saver,
    )
    events = []
    first_runtime.run_turn(
        user_id="u1", conversation_id="c1",
        user_message="字节跳动和拼多多都想记为目标公司",
        event_sink=events.append,
    )
    interaction = next(event for event in events if isinstance(event, InteractionRequiredEvent))
    first_owner.close()

    second_owner = SQLiteCheckpointOwner(checkpoint_path)
    try:
        resumed_decisions = SequenceDecisionMaker(
            propose(add_target_companies=["字节跳动", "拼多多"]), final(),
        )
        resumed_runtime = MainAgentRuntime(
            context_manager=ContextManager(CareerContextStore(context_path)),
            decision_maker=resumed_decisions,
            tools=MainAgentToolRegistry(career_profile_store=CareerContextStore(context_path)),
            checkpointer=second_owner.saver,
        )
        result = resumed_runtime.run_turn(
            user_id="u1", conversation_id="c1", user_message="对，都记上",
            interaction_response=InteractionResponse(
                interaction_id=interaction.interaction_id,
                scope="questionnaire", action="submit",
                answers=tuple(
                    QuestionAnswer(question_id=f"q{index}", free_text="确认")
                    for index in range(1, 6)
                ),
            ),
        )
        assert result.tool_result.state == "job_intent_proposed"
        assert "字节跳动" in resumed_decisions.contexts[0].recent_user_messages_for_intent[-1]
    finally:
        second_owner.close()


def test_target_company_must_be_in_recent_user_message(tmp_path) -> None:
    runtime, _, _ = build(
        tmp_path, AgentDecision(action="final", message="JD 和调研结果里提到了字节跳动。")
    )
    runtime.run_turn(
        user_id="u1", conversation_id="c1", user_message="帮我看一下这份 JD",
    )
    runtime, store, _ = build(
        tmp_path, propose(add_target_companies=["字节跳动"]), final()
    )
    result = runtime.run_turn(
        user_id="u1", conversation_id="c1",
        user_message="根据刚看的 JD 帮我记一下目标公司",
    )
    with pytest.raises(ValueError, match="recent user message"):
        project_job_intent_arguments(
            result.context, "propose_job_intent",
            {"add_target_companies": ["字节跳动"]},
        )
    assert result.tool_result is None
    assert store.get_profile("u1") is None


def test_short_name_and_full_name_share_one_target_company(tmp_path) -> None:
    for message, name in (
        ("目标公司加上阿里", "阿里"),
        ("阿里巴巴也记上", "阿里巴巴"),
    ):
        runtime, _, _ = build(tmp_path, propose(add_target_companies=[name]), final())
        runtime.run_turn(user_id="u1", conversation_id="c1", user_message=message)
        runtime, store, _ = build(tmp_path, confirm(), final())
        runtime.run_turn(user_id="u1", conversation_id="c1", user_message="确认")
    assert store.get_profile("u1").target_companies == ("阿里巴巴",)


def test_company_scope_key_is_case_insensitive_across_removal(tmp_path) -> None:
    store = CareerContextStore(tmp_path / "context.sqlite3")
    scope = "person_intent/self/target_company/apple"
    store.upsert_profile(CareerProfileContext(user_id="u1", target_companies=("Apple",)))
    store.upsert_profile(CareerProfileContext(user_id="u1", target_companies=()))
    store.upsert_profile(CareerProfileContext(user_id="u1", target_companies=("apple",)))
    versions = store.list_profile_intent_versions(user_id="u1", scope_key=scope)
    assert [(item.revision, item.value) for item in versions] == [(1, "Apple"), (2, "apple")]
    assert versions[0].superseded_by == versions[0].update_id
    assert versions[1].superseded_at is None


def test_same_title_created_during_confirmation_is_reused(tmp_path, monkeypatch) -> None:
    store = CareerContextStore(tmp_path / "context.sqlite3")
    resumes = ResumeStore(tmp_path / "resumes.sqlite3")
    registry = MainAgentToolRegistry(career_profile_store=store, resume_store=resumes)
    def concurrent_create(**kwargs):
        resumes.create_target_role(user_id=kwargs["user_id"], title=kwargs["title"], priority=0)
        raise ValueError("target role already exists")

    monkeypatch.setattr(resumes, "create_target_role_with_intent", concurrent_create)
    result = registry._confirm_job_intent({
        "user_id": "u1", "update": JobIntentUpdate(new_target_role_title="产品经理", city="上海"),
    })
    assert result.state == "job_intent_recorded"
    roles = resumes.list_target_roles(user_id="u1")
    assert len(roles) == 1 and roles[0].city == "上海"


def test_failed_compensation_preserves_original_write_error(tmp_path, monkeypatch) -> None:
    store = CareerContextStore(tmp_path / "context.sqlite3")
    resumes = ResumeStore(tmp_path / "resumes.sqlite3")
    registry = MainAgentToolRegistry(career_profile_store=store, resume_store=resumes)

    def fail_write(*args, **kwargs):
        raise RuntimeError("original profile failure")

    def fail_compensation(*args, **kwargs):
        raise RuntimeError("compensation failure")

    monkeypatch.setattr(store, "upsert_profile", fail_write)
    monkeypatch.setattr(resumes, "delete_unreferenced_target_role", fail_compensation)
    with pytest.raises(RuntimeError, match="original profile failure"):
        registry._confirm_job_intent({
            "user_id": "u1",
            "update": JobIntentUpdate(new_target_role_title="产品经理", add_target_companies=("字节跳动",)),
        })


def test_failed_profile_write_removes_new_role_and_its_intent(tmp_path, monkeypatch) -> None:
    store = CareerContextStore(tmp_path / "context.sqlite3")
    resumes = ResumeStore(tmp_path / "resumes.sqlite3")
    registry = MainAgentToolRegistry(career_profile_store=store, resume_store=resumes)

    def fail_profile_write(*args, **kwargs):
        raise RuntimeError("profile write failed")

    monkeypatch.setattr(store, "upsert_profile", fail_profile_write)
    with pytest.raises(RuntimeError, match="profile write failed"):
        registry._confirm_job_intent({
            "user_id": "u1",
            "update": JobIntentUpdate(
                new_target_role_title="产品经理", city="上海",
                add_target_companies=("字节跳动",),
            ),
        })
    assert resumes.list_target_roles(user_id="u1") == ()
    assert resumes.list_target_role_intent_versions(user_id="u1") == ()


def test_job_intent_cannot_select_and_create_role_at_once() -> None:
    with pytest.raises(ValueError, match="select an existing role or name a new role"):
        JobIntentUpdate(target_role_id="role-1", new_target_role_title="产品经理")


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

    # A consumed-or-missing confirmation is rejected before execution; nothing
    # was written either way.
    assert result.tool_result is None
    assert result.context.tool_observations[-1].state == "authorization_refused"
    assert "propose_job_intent" in decisions.contexts[-1].tool_observations[-1].message
    assert result.assistant_message == "我还没给你看过要记的内容，先说一下？"
    assert store.get_profile("u1") is None


def test_a_spent_confirmation_cannot_be_reused(tmp_path) -> None:
    runtime, _, _ = build(tmp_path, propose(city="上海"), final())
    runtime.run_turn(user_id="u1", conversation_id="c1", user_message="上海")

    runtime, _, _ = build(tmp_path, confirm(), final())
    result = runtime.run_turn(user_id="u1", conversation_id="c1", user_message="确认")

    assert result.context.task.pending_job_intent_update is None


def test_stale_bare_confirmation_cannot_confirm_older_job_intent(
    tmp_path,
) -> None:
    runtime, _, _ = build(tmp_path, propose(city="上海"), final())
    runtime.run_turn(
        user_id="u1",
        conversation_id="c1",
        user_message="我在上海找工作",
    )

    unrelated_maker = SequenceDecisionMaker(final())
    unrelated, _, _ = build_with(tmp_path, unrelated_maker)
    unrelated.run_turn(
        user_id="u1",
        conversation_id="c1",
        user_message="先帮我看看新岗位",
    )
    assert len(unrelated_maker.contexts) == 1

    confirmation_maker = SequenceDecisionMaker(final())
    confirmation, _, _ = build_with(tmp_path, confirmation_maker)
    result = confirmation.run_turn(
        user_id="u1",
        conversation_id="c1",
        user_message="可以",
    )

    assert len(confirmation_maker.contexts) == 1
    assert result.origin.label == "model:final"
    assert result.tool_result is None
    assert result.context.task.pending_job_intent_update == JobIntentUpdate(
        city="上海"
    )
    assert result.context.task.bare_confirmation_target is None
    assert CareerContextStore(
        tmp_path / "context.sqlite3"
    ).get_profile("u1") is None


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
        "new_target_role_title",
        "pref_scope",
        "timescale",
        "city",
        "salary_expectation",
        "experience",
        "education",
        "add_target_companies",
        "remove_target_companies",
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
