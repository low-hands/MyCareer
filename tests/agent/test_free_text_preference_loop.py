import sqlite3
from datetime import datetime, timezone

from career_agent.agent.context_manager import ContextManager
from career_agent.agent.decision_messages import project_decision_messages
from career_agent.agent.main_agent_contracts import (
    AgentDecision,
    ToolCall,
)
from career_agent.agent.main_agent_runtime import (
    MainAgentRuntime,
    RuntimePolicyAction,
)
from career_agent.agent.main_agent_tools import MainAgentToolRegistry
from career_agent.storage.context import CareerContextStore


class SequenceDecisionMaker:
    def __init__(self, *decisions: AgentDecision) -> None:
        self.decisions = list(decisions)
        self.contexts = []

    def decide(self, context, tool_specs):
        self.contexts.append(context)
        return self.decisions.pop(0)


def test_free_text_preference_cannot_be_confirmed_before_readback(tmp_path) -> None:
    store = CareerContextStore(tmp_path / "context.sqlite3")
    manager = ContextManager(store)
    manager.load_for_turn(
        user_id="u1",
        conversation_id="c1",
        user_message="我想清楚了，不去大厂。",
    )
    decisions = SequenceDecisionMaker(
        AgentDecision(
            action="tool_call",
            tool_call=ToolCall(name="confirm_free_text_preference", arguments={}),
        ),
        AgentDecision(action="final", message="需要先向你展示待确认内容。"),
    )
    runtime = MainAgentRuntime(
        context_manager=manager,
        decision_maker=decisions,
        tools=MainAgentToolRegistry(career_profile_store=store),
    )

    result = runtime.run_turn(
        user_id="u1",
        conversation_id="c1",
        user_message="继续",
    )

    assert result.tool_result is None
    assert result.context.tool_observations[-1].state == "invalid_input"
    assert store.list_free_text_preferences(
        user_id="u1", statuses=("active",)
    ) == ()


def test_fixed_free_text_preference_acceptance_loop(tmp_path) -> None:
    store = CareerContextStore(tmp_path / "context.sqlite3")
    manager = ContextManager(store)

    first = manager.load_for_turn(
        user_id="u1",
        conversation_id="c1",
        user_message="我想清楚了，不去大厂。",
    )
    first_projection = first.model_context()["free_text_preferences"]
    assert "已确认的自由文本偏好（可用于推荐）\n- 无" in first_projection
    assert "1. 我想清楚了，不去大厂。" in first_projection
    stored_track = store.list_free_text_preferences(user_id="u1")
    assert len(stored_track) == 1
    assert stored_track[0].pref_scope == "freeform.person_default"
    with sqlite3.connect(store.path) as connection:
        assert connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' "
            "AND name='free_text_preference_versions'"
        ).fetchone() is None

    # A new conversation remembers the candidate, but it remains quarantine
    # and therefore has no recommendation authority.
    second = manager.load_for_turn(
        user_id="u1",
        conversation_id="c2",
        user_message="帮我推荐几个合适的岗位",
    )
    second_projection = second.model_context()["free_text_preferences"]
    assert "不得用于筛选、排序或推荐" in second_projection
    assert "1. 我想清楚了，不去大厂。" in second_projection
    assert "1. 我想清楚了，不去大厂。" in (
        project_decision_messages(second).volatile_data["free_text_preferences"]
    )

    unrelated_decisions = SequenceDecisionMaker(
        AgentDecision(action="final", message="我们先准备面试。")
    )
    tools = MainAgentToolRegistry(career_profile_store=store)
    unrelated_runtime = MainAgentRuntime(
        context_manager=manager,
        decision_maker=unrelated_decisions,
        tools=tools,
    )
    unrelated = unrelated_runtime.run_turn(
        user_id="u1",
        conversation_id="c2",
        user_message="帮我准备 Python 面试",
    )
    assert unrelated.tool_result is None
    assert unrelated.assistant_message == "我们先准备面试。"

    proposal_decisions = SequenceDecisionMaker()
    runtime = MainAgentRuntime(
        context_manager=manager,
        decision_maker=proposal_decisions,
        tools=tools,
    )
    proposed = runtime.run_turn(
        user_id="u1",
        conversation_id="c2",
        user_message="帮我推荐几个合适的岗位",
    )
    assert proposed.tool_result is not None
    assert proposed.tool_result.state == "free_text_preference_confirmation_proposed"
    assert "不会影响岗位推荐" in proposed.assistant_message
    assert proposed.origin == RuntimePolicyAction(
        policy="free_text_preference_confirmation"
    )
    assert proposed.model_decision is None
    assert proposal_decisions.contexts == []

    decisions = SequenceDecisionMaker(
        AgentDecision(
            action="tool_call",
            tool_call=ToolCall(
                name="confirm_free_text_preference",
                arguments={},
            ),
        ),
        AgentDecision(action="final", message="已记住。"),
    )
    runtime = MainAgentRuntime(
        context_manager=manager,
        decision_maker=decisions,
        tools=tools,
    )
    confirmed = runtime.run_turn(
        user_id="u1",
        conversation_id="c2",
        user_message="确认",
    )
    assert (
        confirmed.tool_results[0].state
        == "free_text_preference_confirmed_structured_proposed"
    )
    assert "结构化版本" in confirmed.assistant_message
    assert confirmed.context.task.pending_job_intent_update is not None

    structured_runtime = MainAgentRuntime(
        context_manager=manager,
        decision_maker=SequenceDecisionMaker(),
        tools=tools,
    )
    structured = structured_runtime.run_turn(
        user_id="u1",
        conversation_id="c2",
        user_message="确认",
    )
    assert structured.origin == RuntimePolicyAction(
        policy="job_intent_confirmation"
    )
    assert structured.tool_result is not None
    assert structured.tool_result.state == "job_intent_recorded"
    structured_rows = [
        item
        for item in store.list_profile_intent_versions(user_id="u1")
        if item.scope_key == "person_intent/self/company_scale"
        and not item.pref_scope.startswith("freeform")
        and item.superseded_at is None
    ]
    assert [item.value for item in structured_rows] == [
        "exclude_large_companies"
    ]
    assert structured_rows[0].pref_scope == "person_default"

    third = manager.load_for_turn(
        user_id="u1",
        conversation_id="c3",
        user_message="继续推荐岗位",
    ).model_context()["free_text_preferences"]
    assert "- 我想清楚了，不去大厂。（确认于" in third
    assert "待确认偏好（隔离态，不得用于筛选、排序或推荐）\n- 无" in third

    active_before_reaffirmation = store.list_free_text_preferences(
        user_id="u1", statuses=("active",)
    )[0]
    reaffirmed = manager.load_for_turn(
        user_id="u1",
        conversation_id="c3",
        user_message="我还是不考虑大厂。",
    ).model_context()["free_text_preferences"]
    current = store.list_free_text_preferences(user_id="u1")
    assert "- 我想清楚了，不去大厂。（确认于" in reaffirmed
    assert "待确认偏好（隔离态，不得用于筛选、排序或推荐）\n- 无" in reaffirmed
    assert len(current) == 1
    assert current[0].admission_status == "active"
    assert current[0].update_id == active_before_reaffirmation.update_id
    assert (
        current[0].last_corroborated_at
        > active_before_reaffirmation.last_corroborated_at
    )

    # A clear reversal deactivates the old preference immediately; the new
    # statement waits for its own confirmation.
    reversed_context = manager.load_for_turn(
        user_id="u1",
        conversation_id="c4",
        user_message="我改主意了，想去大厂。",
    ).model_context()["free_text_preferences"]
    assert "已确认的自由文本偏好（可用于推荐）\n- 无" in reversed_context
    assert "1. 我改主意了，想去大厂。" in reversed_context

    manager.load_for_turn(
        user_id="u1",
        conversation_id="c4",
        user_message="删除这条大厂偏好",
    )
    deleted = manager.load_for_turn(
        user_id="u1",
        conversation_id="c5",
        user_message="推荐岗位",
    ).model_context()["free_text_preferences"]
    assert "已确认的自由文本偏好（可用于推荐）\n- 无" in deleted
    assert "待确认偏好（隔离态，不得用于筛选、排序或推荐）\n- 无" in deleted
    with sqlite3.connect(store.path) as connection:
        tombstone = connection.execute(
            "SELECT scope_key FROM memory_deleted_scopes WHERE user_id = 'u1'"
        ).fetchone()
        suppressed_message_count = connection.execute(
            "SELECT COUNT(*) FROM memory_deletion_message_suppressions "
            "WHERE user_id = 'u1' AND scope_key = ?",
            ("person_intent/self/company_scale",),
        ).fetchone()[0]
    assert tombstone == ("person_intent/self/company_scale",)
    assert suppressed_message_count > 0
    assert [
        item
        for item in store.list_profile_intent_versions(user_id="u1")
        if item.scope_key == "person_intent/self/company_scale"
    ] == []


def test_ambiguous_role_scope_is_clarified_before_activation(tmp_path) -> None:
    store = CareerContextStore(tmp_path / "context.sqlite3")
    manager = ContextManager(store)
    manager.load_for_turn(
        user_id="u1",
        conversation_id="c1",
        user_message="这类岗位不考虑大厂。",
    )
    tools = MainAgentToolRegistry(career_profile_store=store)
    proposed = MainAgentRuntime(
        context_manager=manager,
        decision_maker=SequenceDecisionMaker(),
        tools=tools,
    ).run_turn(
        user_id="u1",
        conversation_id="c1",
        user_message="帮我看看合适的岗位",
    )

    assert "只针对这类岗位，还是以后默认都这样" in (
        proposed.assistant_message
    )
    proposal = proposed.context.task.pending_free_text_preference
    assert proposal is not None
    assert proposal.needs_scope_clarification

    confirmed = MainAgentRuntime(
        context_manager=manager,
        decision_maker=SequenceDecisionMaker(
            AgentDecision(
                action="tool_call",
                tool_call=ToolCall(
                    name="confirm_free_text_preference",
                    arguments={
                        "scope_choice": "role",
                        "scope_domain": "ai_engineering",
                    },
                ),
            )
        ),
        tools=tools,
    ).run_turn(
        user_id="u1",
        conversation_id="c1",
        user_message="只针对 AI 工程岗位。",
    )

    assert confirmed.tool_result is not None
    assert (
        confirmed.tool_result.state
        == "free_text_preference_confirmed_structured_proposed"
    )
    active = store.list_free_text_preferences(
        user_id="u1", statuses=("active",)
    )
    assert len(active) == 1
    assert active[0].pref_scope == "freeform.role.ai_engineering"
    assert store.list_free_text_preferences(
        user_id="u1", statuses=("quarantined",)
    ) == ()


def test_reopening_deleted_scope_does_not_resurrect_old_tracks(
    tmp_path,
) -> None:
    store = CareerContextStore(tmp_path / "context.sqlite3")
    old = store.capture_free_text_preference_from_message(
        user_id="u1",
        conversation_id="c1",
        message="我不考虑大厂。",
    )
    assert old is not None
    store.capture_free_text_preference_from_message(
        user_id="u1",
        conversation_id="c1",
        message="删除这条大厂偏好",
    )
    assert store.list_free_text_preferences(
        user_id="u1", statuses=("quarantined",)
    ) == ()

    replacement = store.capture_free_text_preference_from_message(
        user_id="u1",
        conversation_id="c1",
        message="这家例外，我愿意去大厂。",
    )
    assert replacement is not None
    active = store.confirm_free_text_preference(
        user_id="u1",
        update_id=replacement.update_id,
    )

    assert active is not None
    assert store.list_free_text_preferences(
        user_id="u1", statuses=("quarantined",)
    ) == ()
    with sqlite3.connect(store.path) as connection:
        old_row = connection.execute(
            """
            SELECT superseded_by
            FROM career_intent_versions
            WHERE update_id = ?
            """,
            (old.update_id,),
        ).fetchone()
    assert old_row == (active.update_id,)


def test_person_level_situational_preference_projects_without_job_scope(
    tmp_path,
) -> None:
    store = CareerContextStore(tmp_path / "context.sqlite3")
    pending = store.capture_free_text_preference_from_message(
        user_id="u1",
        conversation_id="c1",
        message="我年底前不考虑大厂。",
    )
    assert pending is not None
    assert pending.pref_scope == "freeform.person_situational"
    assert pending.timescale == "situational"
    assert pending.valid_until == datetime(
        2027, 1, 1, tzinfo=timezone.utc
    )
    assert store.confirm_free_text_preference(
        user_id="u1", update_id=pending.update_id
    ) is not None

    projected = ContextManager(store).load_for_turn(
        user_id="u1",
        conversation_id="c2",
        user_message="推荐岗位",
    ).model_context()["free_text_preferences"]

    assert "我年底前不考虑大厂。" in projected
    assert "待确认偏好（隔离态，不得用于筛选、排序或推荐）\n- 无" in (
        projected
    )
