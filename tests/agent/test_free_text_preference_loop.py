import sqlite3

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
    assert stored_track[0].pref_scope == "freeform"
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
    assert confirmed.tool_results[0].state == "free_text_preference_confirmed"

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
            ("person_intent/self/employer_scale_preference",),
        ).fetchone()[0]
    assert tombstone == ("person_intent/self/employer_scale_preference",)
    assert suppressed_message_count > 0
