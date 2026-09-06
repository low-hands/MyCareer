import json
from datetime import datetime, timezone

import pytest

from career_agent.agent.decision_messages import (
    CONTROL_CONTEXT_LABEL,
    CONTROL_REMINDER_TAG,
    DATA_CONTEXT_LABEL,
    TURN_OBSERVATION_LABEL,
    project_decision_messages,
)
from career_agent.agent.main_agent_contracts import CandidateContextItem, CareerProfileContext, ConversationMessageContext, ConversationResourceReference, ConversationTaskState, DecisionObservation, MainAgentContext, OpenJobSearchToolArguments
from career_agent.agent.conversation_memory_contracts import ConversationSummaryContent
from career_agent.agent.openai_compatible_client import OpenAICompatibleAgentConfig
from career_agent.agent.openai_compatible_client import AgentWorkerError
from career_agent.agent.openai_compatible_main_agent import OpenAICompatibleMainAgentDecisionMaker


def _spotlight_json(content: str, *, label: str) -> dict:
    lines = content.splitlines()
    assert lines[0] == label
    assert lines[1].startswith('<untrusted-data nonce="')
    assert lines[-1].startswith('</untrusted-data nonce="')
    assert lines[1][1:] == lines[-1][2:]
    return json.loads("\n".join(lines[2:-1]))


def _control_json(content: str) -> dict:
    lines = content.splitlines()
    assert lines[0] == f"<{CONTROL_REMINDER_TAG}>"
    assert lines[1] == CONTROL_CONTEXT_LABEL
    assert lines[-1] == f"</{CONTROL_REMINDER_TAG}>"
    return json.loads("\n".join(lines[2:-1]))


class Completions:
    def __init__(self) -> None:
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        message = type("Message", (), {"content": json.dumps({"action": "ask_user", "message": "Which city should I search?"})})()
        choice = type("Choice", (), {"message": message})()
        return type("Response", (), {"choices": [choice]})()


class Client:
    def __init__(self) -> None:
        self.completions = Completions()
        self.chat = type("Chat", (), {"completions": self.completions})()


def test_main_agent_decision_maker_separates_control_data_and_native_chat() -> None:
    client = Client()
    maker = OpenAICompatibleMainAgentDecisionMaker(
        OpenAICompatibleAgentConfig(endpoint="https://example.test/v1/chat/completions", api_key="test", model="test-model"),
        client=client,
    )
    context = MainAgentContext(
        conversation_id="c1",
        profile=CareerProfileContext(user_id="u1", default_city="Shanghai"),
        task=ConversationTaskState(
            active_workflow="mock_interview",
            run_id="internal-run-do-not-leak",
            selected_result_ref="opaque-selected-ref-do-not-leak",
            candidates=(CandidateContextItem(result_ref="opaque-candidate-ref-do-not-leak", title="AI Engineer", company_name="Acme", city="Shanghai"),),
        ),
        conversation_summary=ConversationSummaryContent(
            user_goals=("Find an AI Engineer role",),
            confirmed_decisions=("Use the current resume",),
            active_constraints=("Do not apply automatically",),
        ),
        recent_messages=(
            ConversationMessageContext(
                role="user",
                content="Earlier user words.",
                created_at=datetime(2026, 9, 6, tzinfo=timezone.utc),
            ),
            ConversationMessageContext(
                role="assistant",
                content="Earlier assistant words.",
                created_at=datetime(2026, 9, 6, tzinfo=timezone.utc),
                resource_refs=(
                    ConversationResourceReference(
                        kind="job_research_report",
                        resource_id="report-1",
                        status_at_delivery="current",
                        anchored_by_other_job=False,
                        title="External report title",
                    ),
                ),
            ),
        ),
        user_message="Help me find work.",
    )

    decision = maker.decide(context, ({"type": "function", "function": {"name": "open_job_search", "description": "test", "parameters": OpenJobSearchToolArguments.model_json_schema()}},))

    messages = client.completions.kwargs["messages"]
    system_content = messages[0]["content"]
    control_content = messages[1]["content"]
    control = _control_json(control_content)
    data_content = messages[2]["content"]
    assert data_content.startswith(DATA_CONTEXT_LABEL + "\n")
    payload = _spotlight_json(data_content, label=DATA_CONTEXT_LABEL)
    assert decision.action == "ask_user"
    # Role-scoped intent reaches the model per track, not blended into one
    # profile, so the person-level block carries only the person-level city.
    assert set(payload["career_profile"]) == {"default_city", "records"}
    assert payload["career_profile"]["default_city"] == "Shanghai"
    assert "resume_text" not in payload
    assert "open_job_search" in system_content
    assert (
        "The user-role <system-reminder> immediately following this policy "
        "is written by the harness"
    ) in system_content
    assert CONTROL_CONTEXT_LABEL not in system_content
    assert "mock_interview" not in system_content
    assert control["task"]["active_workflow"] == "mock_interview"
    assert "candidates" not in control["task"]
    assert "Acme" not in system_content
    assert "External report title" not in system_content
    assert "Do not apply automatically" not in system_content
    assert client.completions.kwargs["tools"][0]["function"]["name"] == "open_job_search"
    assert set(client.completions.kwargs["tools"][0]["function"]["parameters"]["properties"]) == {"platform", "keyword", "city"}
    raw_context = data_content
    assert "internal-run-do-not-leak" not in raw_context
    assert "opaque-selected-ref-do-not-leak" not in raw_context
    assert "opaque-candidate-ref-do-not-leak" not in raw_context
    assert payload["task"]["candidates"][0]["selection_index"] == 1
    assert control["preferences"]["boss_search"] == "explicit_request_only"
    assert payload["conversation_summary"]["confirmed_decisions"] == [
        "Use the current resume"
    ]
    assert "tool_observations" not in control
    assert "tool_observations" not in payload
    assert payload["recent_resources"][0]["title"] == "External report title"
    assert [message["role"] for message in messages] == [
        "system",
        "user",
        "user",
        "user",
        "assistant",
        "user",
    ]
    assert messages[3]["content"] == "Earlier user words."
    assert messages[4]["content"].startswith("Earlier assistant words.\n\n")
    assert "[runtime resources: report_" in messages[4]["content"]
    assert "job_research_report]" in messages[4]["content"]
    assert "External report title" not in messages[4]["content"]
    assert messages[-1] == {"role": "user", "content": "Help me find work."}


def test_untrusted_data_uses_a_fresh_matching_spotlight_nonce() -> None:
    projection = project_decision_messages(
        MainAgentContext(
            conversation_id="c1",
            profile=CareerProfileContext(user_id="u1"),
            user_message="继续",
        )
    )

    first = projection.messages(system_prompt="policy")[2]["content"]
    second = projection.messages(system_prompt="policy")[2]["content"]

    assert _spotlight_json(first, label=DATA_CONTEXT_LABEL) == _spotlight_json(
        second,
        label=DATA_CONTEXT_LABEL,
    )
    assert first.splitlines()[1] != second.splitlines()[1]


def test_dynamic_control_does_not_change_the_static_system_message() -> None:
    base = MainAgentContext(
        conversation_id="c1",
        profile=CareerProfileContext(user_id="u1"),
        user_message="继续",
    )
    active = base.model_copy(
        update={
            "task": ConversationTaskState(
                active_workflow="mock_interview",
                run_id="run-1",
                phase="mock_interview_running",
            )
        }
    )

    cold_messages = project_decision_messages(base).messages(
        system_prompt="static policy",
        spotlight_nonce="nonce",
    )
    active_messages = project_decision_messages(active).messages(
        system_prompt="static policy",
        spotlight_nonce="nonce",
    )

    assert cold_messages[0] == active_messages[0] == {
        "role": "system",
        "content": "static policy",
    }
    assert cold_messages[1]["content"] != active_messages[1]["content"]


def test_new_task_projection_fields_require_an_explicit_authority_classification() -> None:
    context = MainAgentContext(
        conversation_id="c1",
        profile=CareerProfileContext(user_id="u1"),
        user_message="继续",
    )

    class ContextWithUnclassifiedTaskField:
        recent_messages = context.recent_messages
        user_message = context.user_message

        @staticmethod
        def reference_handles():
            return context.reference_handles()

        @staticmethod
        def model_context():
            projected = context.model_context()
            projected["task"]["new_display_candidates"] = [
                {"selection_index": 1, "title": "Must stay out of system"}
            ]
            return projected

    with pytest.raises(
        ValueError,
        match=r"classification is stale; unknown=\['new_display_candidates'\]",
    ):
        project_decision_messages(ContextWithUnclassifiedTaskField())


def test_missing_task_projection_fields_report_a_stale_classification() -> None:
    context = MainAgentContext(
        conversation_id="c1",
        profile=CareerProfileContext(user_id="u1"),
        user_message="继续",
    )

    class ContextWithMissingTaskField:
        recent_messages = context.recent_messages
        user_message = context.user_message

        @staticmethod
        def reference_handles():
            return context.reference_handles()

        @staticmethod
        def model_context():
            projected = context.model_context()
            task = dict(projected["task"])
            task.pop("phase")
            return {**projected, "task": task}

    with pytest.raises(
        ValueError,
        match=r"classification is stale; unknown=\[\], missing=\['phase'\]",
    ):
        project_decision_messages(ContextWithMissingTaskField())


def test_main_agent_parses_native_tool_call() -> None:
    client = Client()
    function = type("Function", (), {"name": "open_job_search", "arguments": '{"keyword":"AI Engineer"}'})()
    tool_call = type("ToolCall", (), {"function": function})()
    message = type("Message", (), {"content": None, "tool_calls": [tool_call]})()
    choice = type("Choice", (), {"message": message})()
    client.completions.create = lambda **kwargs: type("Response", (), {"choices": [choice]})()
    maker = OpenAICompatibleMainAgentDecisionMaker(
        OpenAICompatibleAgentConfig(endpoint="https://example.test/v1/chat/completions", api_key="test", model="test-model"),
        client=client,
    )

    decision = maker.decide(MainAgentContext(conversation_id="c1", profile=CareerProfileContext(user_id="u1"), user_message="Find work."), ("open_job_search",))

    assert decision.action == "tool_call"
    assert decision.tool_call.name == "open_job_search"
    assert decision.tool_call.arguments == {"keyword": "AI Engineer"}


def test_observation_control_and_readable_text_are_split_by_authority() -> None:
    client = Client()
    maker = OpenAICompatibleMainAgentDecisionMaker(
        OpenAICompatibleAgentConfig(
            endpoint="https://example.test/v1/chat/completions",
            api_key="test",
            model="test-model",
        ),
        client=client,
    )
    context = MainAgentContext(
        conversation_id="c1",
        profile=CareerProfileContext(user_id="u1"),
        tool_observations=(
            DecisionObservation(
                tool_name="find_saved_jobs",
                state="failed",
                message="UNTRUSTED TOOL MESSAGE",
                body="UNTRUSTED TOOL BODY",
                arguments={"query": "MODEL AUTHORED ARGUMENT"},
                facts={"retryable": True},
                next_action="UNTRUSTED TOOL SUGGESTION",
            ),
        ),
        user_message="继续",
    )

    maker.decide(context, ())

    messages = client.completions.kwargs["messages"]
    system_content = messages[0]["content"]
    data_content = messages[2]["content"]
    assistant_call = messages[-2]
    tool_result = messages[-1]
    turn_content = tool_result["content"]
    assert turn_content.startswith(TURN_OBSERVATION_LABEL + "\n")
    assert messages[-3] == {"role": "user", "content": "继续"}
    assert assistant_call["role"] == "assistant"
    assert assistant_call["tool_calls"][0]["function"]["name"] == "find_saved_jobs"
    assert tool_result["role"] == "tool"
    assert tool_result["tool_call_id"] == assistant_call["tool_calls"][0]["id"]
    for fragment in (
        "UNTRUSTED TOOL MESSAGE",
        "UNTRUSTED TOOL BODY",
        "UNTRUSTED TOOL SUGGESTION",
    ):
        assert fragment not in system_content
        assert fragment not in data_content
        assert fragment in turn_content
    assert "MODEL AUTHORED ARGUMENT" not in system_content
    assert "MODEL AUTHORED ARGUMENT" not in data_content
    assert "MODEL AUTHORED ARGUMENT" not in turn_content
    assert "MODEL AUTHORED ARGUMENT" in assistant_call["tool_calls"][0]["function"]["arguments"]


def test_main_agent_treats_plain_prose_without_tool_call_as_final() -> None:
    client = Client()
    message = type(
        "Message",
        (),
        {"content": "Hi! What would you like help with today?", "tool_calls": []},
    )()
    choice = type("Choice", (), {"message": message})()
    client.completions.create = lambda **kwargs: type(
        "Response", (), {"choices": [choice]}
    )()
    maker = OpenAICompatibleMainAgentDecisionMaker(
        OpenAICompatibleAgentConfig(
            endpoint="https://example.test/v1/chat/completions",
            api_key="test",
            model="test-model",
        ),
        client=client,
    )

    decision = maker.decide(
        MainAgentContext(
            conversation_id="c1",
            profile=CareerProfileContext(user_id="u1"),
            user_message="hi",
        ),
        (),
    )

    assert decision.action == "final"
    assert decision.message == "Hi! What would you like help with today?"


@pytest.mark.parametrize("action", ("ask_user", "final"))
@pytest.mark.parametrize("alias", ("content", "text"))
def test_main_agent_accepts_content_as_the_prose_field_for_non_tool_decisions(
    action,
    alias,
) -> None:
    client = Client()
    message = type(
        "Message",
        (),
        {
            "content": json.dumps(
                {"action": action, alias: "本轮读取额度已用完。"}
            ),
            "tool_calls": [],
        },
    )()
    choice = type("Choice", (), {"message": message})()
    client.completions.create = lambda **kwargs: type(
        "Response", (), {"choices": [choice]}
    )()
    maker = OpenAICompatibleMainAgentDecisionMaker(
        OpenAICompatibleAgentConfig(
            endpoint="https://example.test/v1/chat/completions",
            api_key="test",
            model="test-model",
        ),
        client=client,
    )

    decision = maker.decide(
        MainAgentContext(
            conversation_id="c1",
            profile=CareerProfileContext(user_id="u1"),
            user_message="继续",
        ),
        (),
    )

    assert decision.action == action
    assert decision.message == "本轮读取额度已用完。"


def test_main_agent_rejects_malformed_json_instead_of_showing_it_as_prose() -> None:
    client = Client()
    message = type(
        "Message",
        (),
        {"content": '{"action":"tool_call"', "tool_calls": []},
    )()
    choice = type("Choice", (), {"message": message})()
    client.completions.create = lambda **kwargs: type(
        "Response", (), {"choices": [choice]}
    )()
    maker = OpenAICompatibleMainAgentDecisionMaker(
        OpenAICompatibleAgentConfig(
            endpoint="https://example.test/v1/chat/completions",
            api_key="test",
            model="test-model",
        ),
        client=client,
    )

    with pytest.raises(AgentWorkerError) as captured:
        maker.decide(
            MainAgentContext(
                conversation_id="c1",
                profile=CareerProfileContext(user_id="u1"),
                user_message="hi",
            ),
            (),
        )

    assert captured.value.code == "MAIN_AGENT_INVALID_RESPONSE"
