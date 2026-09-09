from __future__ import annotations

import json

import pytest

from career_agent.agent.conversation_memory_contracts import (
    HARNESS_SUMMARY_COUNTER_FIELDS,
    ConversationSummaryContent,
    SummaryMessage,
)
from career_agent.agent.openai_compatible_client import (
    AgentWorkerError,
    OpenAICompatibleAgentConfig,
)
from career_agent.agent.openai_conversation_summary_worker import (
    OpenAIConversationSummaryWorker,
)


class Completions:
    def __init__(self, content: str) -> None:
        self.content = content
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        message = type("Message", (), {"content": self.content})()
        choice = type("Choice", (), {"message": message})()
        return type("Response", (), {"choices": [choice]})()


class Client:
    def __init__(self, content: str) -> None:
        self.completions = Completions(content)
        self.chat = type("Chat", (), {"completions": self.completions})()


def config() -> OpenAICompatibleAgentConfig:
    return OpenAICompatibleAgentConfig(
        endpoint="https://example.test/v1/chat/completions",
        api_key="secret",
        model="summary-model",
    )


def test_summary_worker_merges_structured_previous_and_messages() -> None:
    client = Client(
        json.dumps(
            {
                "user_goals": ["Track applications"],
                "confirmed_decisions": ["Use SQLite"],
                "unresolved_questions": ["When to send a follow-up?"],
                "active_constraints": ["Do not send email without approval"],
            }
        )
    )
    worker = OpenAIConversationSummaryWorker(config(), client=client)
    previous = ConversationSummaryContent(
        user_goals=("Build a career agent",),
        confirmed_decisions=("Use SQLite",),
        omitted_active_constraint_count=2,
    )

    result = worker.summarize(
        previous=previous,
        messages=(
            SummaryMessage(sequence=9, role="user", content="Track my applications."),
            SummaryMessage(sequence=10, role="assistant", content="I can do that."),
        ),
    )

    assert result.user_goals == ("Track applications",)
    request = json.loads(client.completions.kwargs["messages"][1]["content"])
    assert request["previous_summary"]["confirmed_decisions"] == ["Use SQLite"]
    assert "omitted_active_constraint_count" not in request["previous_summary"]
    assert HARNESS_SUMMARY_COUNTER_FIELDS.isdisjoint(request["previous_summary"])
    assert request["new_messages"][0]["sequence"] == 9
    system = client.completions.kwargs["messages"][0]["content"]
    assert "not confirmed long-term user memory" in system
    assert "resume text" in system
    assert "tools" not in client.completions.kwargs


def test_summary_worker_rejects_unstructured_response() -> None:
    worker = OpenAIConversationSummaryWorker(config(), client=Client("not-json"))

    with pytest.raises(AgentWorkerError) as error:
        worker.summarize(
            previous=None,
            messages=(SummaryMessage(sequence=1, role="user", content="Hello"),),
        )

    assert error.value.code == "CONVERSATION_SUMMARY_INVALID_RESPONSE"
