from datetime import datetime, timezone

from career_agent.agent.context_manager import ContextManager
from career_agent.agent.main_agent_contracts import AgentDecision, CareerProfileContext, ToolCall
from career_agent.agent.main_agent_runtime import MainAgentRuntime
from career_agent.agent.main_agent_tools import MainAgentToolRegistry
from career_agent.domain.action_center import ActionItem, DailyBrief
from career_agent.storage.context import CareerContextStore


class ActionCenter:
    def __init__(self):
        now = datetime(2026, 8, 26, tzinfo=timezone.utc)
        self.item = ActionItem(
            id="action-1", user_id="u1",
            stable_key="application_follow_up:application-1:cycle-1",
            action_type="application_follow_up", source_type="application",
            source_id="application-1", application_id="application-1",
            title="跟进投递：Example", summary="岗位已有 7 天没有记录新进展。",
            due_at=now, status="open", created_at=now, updated_at=now,
        )
        self.complete_calls = []

    def daily_brief(self, **kwargs):
        return DailyBrief(
            user_id=kwargs["user_id"], timezone=kwargs["timezone_name"],
            generated_at=self.item.updated_at, due_today=(self.item,),
        )

    def complete_action(self, **kwargs):
        self.complete_calls.append(kwargs)
        return self.item.model_copy(
            update={"status": "completed", "resolved_at": self.item.updated_at}
        )


class Decisions:
    def __init__(self):
        self.values = [
            AgentDecision(action="tool_call", tool_call=ToolCall(name="get_daily_brief", arguments={})),
            AgentDecision(
                action="tool_call",
                tool_call=ToolCall(name="complete_action_item", arguments={"selection_index": 1}),
            ),
            AgentDecision(action="final", message="已记录完成。"),
        ]

    def decide(self, context, tool_specs):
        names = {spec["function"]["name"] for spec in tool_specs}
        assert {
            "get_daily_brief",
            "list_action_items",
        }.issubset(names)
        return self.values.pop(0)


def test_daily_brief_selection_projects_internal_action_id(tmp_path) -> None:
    manager = ContextManager(CareerContextStore(tmp_path / "context.sqlite3"))
    manager.upsert_profile(CareerProfileContext(user_id="u1"))
    action_center = ActionCenter()
    tools = MainAgentToolRegistry(action_center_service=action_center)
    runtime = MainAgentRuntime(
        context_manager=manager, decision_maker=Decisions(), tools=tools,
    )

    result = runtime.run_turn(
        user_id="u1", conversation_id="c1", user_message="看看今天的待办，第一个完成了"
    )

    assert tools.capability_kind("get_daily_brief") == "atomic_tool"
    assert action_center.complete_calls == [
        {"user_id": "u1", "action_item_id": "action-1"}
    ]
    assert result.context.task.active_action_item_id == "action-1"
    assert result.context.task.action_candidates == ()
