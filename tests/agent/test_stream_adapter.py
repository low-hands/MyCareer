from career_agent.agent.main_agent_contracts import AgentDecision, ToolObservation
from career_agent.agent.presentation.stream_adapter import StreamAdapter
from career_agent.agent.turn_models import MainAgentTurnResult, ModelDecision
from career_agent.harness.streaming import InteractionRequiredEvent


class Host:
    def __init__(self, interaction=None, *, card_backed=False) -> None:
        self.interaction = interaction
        self.card_backed = card_backed

    def event(self, *, result, conversation_id):
        return self.interaction

    def turn_resource_refs(self, results):
        return ()

    def conversation_content(self, result, *, screen, composed):
        assert screen == "durable screen"
        assert composed is True
        return "durable content"

    def durable_screen(self, result):
        return "durable screen"

    def turn_is_card_backed(self, results):
        return self.card_backed


def _turn(*, results=(), message="reply") -> MainAgentTurnResult:
    return MainAgentTurnResult(
        origin=ModelDecision(AgentDecision(action="final", message=message)),
        context=None,
        assistant_message=message,
        tool_result=results[-1] if results else None,
        tool_results=results,
        model_message=message,
    )


def test_stream_adapter_emits_client_action_before_turn_completion() -> None:
    events = []
    result = ToolObservation(
        tool_name="open_job_search",
        state="search_opened",
        message="已打开。",
        payload={
            "client_action": {
                "type": "open_url",
                "url": "https://example.com/jobs",
                "label": "打开岗位页",
            }
        },
    )
    host = Host()
    adapter = StreamAdapter(
        interaction_renderer=host,  # type: ignore[arg-type]
        presenter=host,  # type: ignore[arg-type]
        emit=events.append,
    )

    adapter.deliver_events(
        result=_turn(results=(result,)),
        turn_id="turn-1",
        conversation_id="conversation-1",
    )

    assert [event.type for event in events] == ["client_action", "turn_completed"]
    assert events[0].url == "https://example.com/jobs"


def test_stream_adapter_suspends_without_emitting_completion() -> None:
    events = []
    interaction = InteractionRequiredEvent(
        interaction_id="interaction_0123456789abcdefabcd",
        kind="free_text",
        prompt="请补充信息。",
        allow_free_text=True,
    )
    host = Host(interaction)
    adapter = StreamAdapter(
        interaction_renderer=host,  # type: ignore[arg-type]
        presenter=host,  # type: ignore[arg-type]
        emit=events.append,
    )

    adapter.deliver_events(
        result=_turn(),
        turn_id="turn-1",
        conversation_id="conversation-1",
    )

    assert [event.type for event in events] == [
        "interaction_required",
        "turn_suspended",
    ]
    assert events[-1].interaction_id == interaction.interaction_id


def test_stream_adapter_uses_durable_card_content_for_synthetic_reply() -> None:
    events = []
    result = ToolObservation(
        tool_name="get_job_research",
        state="job_research_ready",
        message="已读取。",
    )
    host = Host(card_backed=True)
    adapter = StreamAdapter(
        interaction_renderer=host,  # type: ignore[arg-type]
        presenter=host,  # type: ignore[arg-type]
        emit=events.append,
    )

    adapter.deliver_reply(
        result=_turn(results=(result,)),
        conversation_id="conversation-1",
    )

    assert events[0].type == "progress"
    assert "".join(event.delta for event in events[1:]) == "durable content"
