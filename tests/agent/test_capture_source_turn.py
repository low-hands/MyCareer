from pathlib import Path

from career_agent.agent.context_manager import ContextManager
from career_agent.agent.main_agent_contracts import AgentDecision, ToolCall
from career_agent.agent.main_agent_runtime import MainAgentRuntime
from career_agent.agent.main_agent_tools import MainAgentToolRegistry
from career_agent.harness.streaming import PublicStreamEvent
from career_agent.storage.context import CareerContextStore
from career_agent.storage.job_captures import SQLiteJobCaptureStore


class SearchDecisions:
    def __init__(self) -> None:
        self.decisions = iter((
            AgentDecision(action="tool_call", tool_call=ToolCall(
                name="open_job_search", arguments={"keyword": "AI"},
            )),
            AgentDecision(action="final", message="已打开搜索页。"),
        ))

    def decide(self, context, tool_specs) -> AgentDecision:
        return next(self.decisions)


def test_capture_intent_records_runtime_source_turn_without_url_context(tmp_path: Path):
    captures = SQLiteJobCaptureStore(tmp_path / "jobs.sqlite3")
    runtime = MainAgentRuntime(
        context_manager=ContextManager(CareerContextStore(tmp_path / "context.sqlite3")),
        decision_maker=SearchDecisions(),
        tools=MainAgentToolRegistry(job_capture_store=captures),
    )
    events: list[PublicStreamEvent] = []
    runtime.run_turn(
        user_id="u1", conversation_id="original", user_message="找 AI 岗位",
        event_sink=events.append,
    )
    started = next(event for event in events if event.type == "turn_started")
    action = next(event for event in events if event.type == "client_action")
    assert action.capture_intent_id is not None
    intent = captures.get_intent(user_id="u1", intent_id=action.capture_intent_id)
    assert intent is not None and intent.source_turn_id == started.turn_id
    assert intent.conversation_id == "original"
    assert started.turn_id not in action.url
    assert intent.id not in action.url
