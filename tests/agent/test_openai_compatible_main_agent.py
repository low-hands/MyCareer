import json

from career_agent.agent.main_agent_contracts import CandidateContextItem, CareerProfileContext, ConversationTaskState, JobDiscoveryToolArguments, MainAgentContext
from career_agent.agent.openai_compatible_client import OpenAICompatibleAgentConfig
from career_agent.agent.openai_compatible_main_agent import OpenAICompatibleMainAgentDecisionMaker


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


def test_main_agent_decision_maker_receives_only_structured_context() -> None:
    client = Client()
    maker = OpenAICompatibleMainAgentDecisionMaker(
        OpenAICompatibleAgentConfig(endpoint="https://example.test/v1/chat/completions", api_key="test", model="test-model"),
        client=client,
    )
    context = MainAgentContext(
        conversation_id="c1",
        profile=CareerProfileContext(user_id="u1", target_roles=("AI Engineer",), default_city="Shanghai"),
        task=ConversationTaskState(
            active_workflow="job_discovery",
            run_id="internal-run-do-not-leak",
            selected_result_ref="opaque-selected-ref-do-not-leak",
            candidates=(CandidateContextItem(result_ref="opaque-candidate-ref-do-not-leak", title="AI Engineer", company_name="Acme", city="Shanghai"),),
        ),
        user_message="Help me find work.",
    )

    decision = maker.decide(context, ({"type": "function", "function": {"name": "job_discovery", "description": "test", "parameters": JobDiscoveryToolArguments.model_json_schema()}},))

    payload = json.loads(client.completions.kwargs["messages"][1]["content"])
    assert decision.action == "ask_user"
    assert payload["career_profile"]["target_roles"] == ["AI Engineer"]
    assert "resume_text" not in payload
    assert "job_discovery" in client.completions.kwargs["messages"][0]["content"]
    assert "job_discovery.research" not in client.completions.kwargs["messages"][0]["content"]
    assert client.completions.kwargs["tools"][0]["function"]["name"] == "job_discovery"
    assert set(client.completions.kwargs["tools"][0]["function"]["parameters"]["properties"]) == {"target_role", "city", "salary", "experience", "education", "selection_index", "selection_indices", "jd_selection_index"}
    raw_context = client.completions.kwargs["messages"][1]["content"]
    assert "internal-run-do-not-leak" not in raw_context
    assert "opaque-selected-ref-do-not-leak" not in raw_context
    assert "opaque-candidate-ref-do-not-leak" not in raw_context
    assert json.loads(raw_context)["task"]["candidates"][0]["selection_index"] == 1


def test_main_agent_parses_native_tool_call() -> None:
    client = Client()
    function = type("Function", (), {"name": "job_discovery", "arguments": '{"target_role":"AI Engineer"}'})()
    tool_call = type("ToolCall", (), {"function": function})()
    message = type("Message", (), {"content": None, "tool_calls": [tool_call]})()
    choice = type("Choice", (), {"message": message})()
    client.completions.create = lambda **kwargs: type("Response", (), {"choices": [choice]})()
    maker = OpenAICompatibleMainAgentDecisionMaker(
        OpenAICompatibleAgentConfig(endpoint="https://example.test/v1/chat/completions", api_key="test", model="test-model"),
        client=client,
    )

    decision = maker.decide(MainAgentContext(conversation_id="c1", profile=CareerProfileContext(user_id="u1"), user_message="Find work."), ("job_discovery",))

    assert decision.action == "tool_call"
    assert decision.tool_call.name == "job_discovery"
    assert decision.tool_call.arguments == {"target_role": "AI Engineer"}
