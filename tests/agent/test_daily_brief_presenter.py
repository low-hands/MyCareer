from career_agent.agent.main_agent_contracts import ToolObservation
from career_agent.agent.main_agent_runtime import MainAgentRuntime


def test_daily_brief_delivers_the_full_ephemeral_body_but_keeps_a_summary_row() -> None:
    observation = ToolObservation(
        tool_name="get_daily_brief",
        state="daily_brief_ready",
        message="今日职业简报包含 1 个待办事项。",
        payload={
            "timezone": "Asia/Shanghai",
            "generated_at": "2026-08-31T09:00:00+08:00",
            "overdue": [],
            "due_today": [
                {
                    "title": "跟进投递",
                    "summary": "确认招聘方是否收到材料",
                    "due_at": "2026-08-31T18:00:00+08:00",
                }
            ],
            "upcoming": [],
            "no_due_date": [],
        },
    )

    screen = MainAgentRuntime._assistant_message(observation)

    assert screen.startswith("# 今日职业简报")
    assert "跟进投递" in screen
    assert "确认招聘方是否收到材料" in screen
    assert observation.resource_ref is None
    assert MainAgentRuntime._conversation_content(
        observation, screen=screen, composed=False
    ) == observation.message
