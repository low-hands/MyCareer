from __future__ import annotations

import json

import pytest

from career_agent.agent.contracts.memory import (
    HARNESS_SUMMARY_COUNTER_FIELDS,
    ConversationSummaryContent,
    SummaryMessage,
)
from career_agent.agent.providers.openai_client import (
    AgentWorkerError,
    OpenAICompatibleAgentConfig,
)
from career_agent.agent.providers.conversation_summary import (
    OpenAIConversationSummaryWorker,
)


class Completions:
    def __init__(self, content: str) -> None:
        self.content = content
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        message = type("Message", (), {"content": self.content})()
        choice = type("Choice", (), {"message": message, "finish_reason": "stop"})()
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
                "new_facts": [{"fact": "Track my applications", "source_sequence": 9}],
                "superseded_fact_ids": [],
                "user_goals": ["Track applications"],
                "confirmed_decisions": ["Use SQLite"],
                "unresolved_questions": ["When to send a follow-up?"],
                "active_constraints": ["Do not send email without approval"],
                "long_term_memory_candidates": [
                    {
                        "topic_key": "application_tracking",
                        "statement": "希望持续跟踪求职申请",
                        "stance": "prefer",
                        "source_sequence": 9,
                        "source_quote": "Track my applications.",
                        "confidence": 0.9,
                    }
                ],
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
    assert result.new_facts[0].source_sequence == 9
    assert result.long_term_memory_candidates[0].topic_key == "application_tracking"
    request = json.loads(client.completions.kwargs["messages"][1]["content"])
    assert request["previous_summary"]["confirmed_decisions"] == ["Use SQLite"]
    assert "omitted_active_constraint_count" not in request["previous_summary"]
    assert HARNESS_SUMMARY_COUNTER_FIELDS.isdisjoint(request["previous_summary"])
    assert request["new_messages"][0]["sequence"] == 9
    system = client.completions.kwargs["messages"][0]["content"]
    assert "not confirmed long-term user memory" in system
    assert "exact verbatim source_quote" in system
    assert "previous summary" in system
    assert "resume text" in system
    assert "tools" not in client.completions.kwargs
    assert client.completions.kwargs["response_format"]["type"] == "json_schema"
    assert client.completions.kwargs["response_format"]["json_schema"]["strict"] is True


def test_key_fact_requires_an_original_message_sequence() -> None:
    with pytest.raises(ValueError):
        ConversationSummaryContent.model_validate({
            "key_facts": [{"fact": "目标公司是 Pinnacle Robotics"}],
        })


def test_summary_worker_rejects_unstructured_response() -> None:
    worker = OpenAIConversationSummaryWorker(config(), client=Client("not-json"))

    with pytest.raises(AgentWorkerError) as error:
        worker.summarize(
            previous=None,
            messages=(SummaryMessage(sequence=1, role="user", content="Hello"),),
        )

    assert error.value.code == "CONVERSATION_SUMMARY_INVALID_RESPONSE"


def test_summary_worker_drops_bad_optional_candidate_without_losing_summary() -> None:
    client = Client(
        json.dumps(
            {
                "new_facts": [],
                "superseded_fact_ids": [],
                "user_goals": ["Find a suitable role"],
                "confirmed_decisions": ["Search this week"],
                "unresolved_questions": [],
                "active_constraints": ["Keep the search private"],
                "long_term_memory_candidates": [
                    {
                        "topic_key": "team_culture",
                        "statement": "偏好开放协作的团队",
                        "stance": "prefer_team_culture",
                        "source_sequence": 1,
                        "source_quote": "我偏好开放协作的团队",
                        "confidence": 0.9,
                    },
                    {
                        "topic_key": "company_stage",
                        "statement": "对公司阶段心情复杂",
                        "stance": "mixed_feelings",
                        "source_sequence": 1,
                        "source_quote": "我对公司阶段心情复杂",
                        "confidence": 0.6,
                    },
                ],
            },
            ensure_ascii=False,
        )
    )
    result = OpenAIConversationSummaryWorker(config(), client=client).summarize(
        previous=None,
        messages=(
            SummaryMessage(
                sequence=1,
                role="user",
                content="我偏好开放协作的团队，我对公司阶段心情复杂",
            ),
        ),
    )

    assert result.user_goals == ("Find a suitable role",)
    assert result.confirmed_decisions == ("Search this week",)
    assert result.active_constraints == ("Keep the search private",)
    assert len(result.long_term_memory_candidates) == 1
    assert result.long_term_memory_candidates[0].stance == "positive"


def test_summary_worker_client_does_not_retry(monkeypatch) -> None:
    captured = {}

    class RecordingOpenAI:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(
        "career_agent.agent.providers.conversation_summary.OpenAI",
        RecordingOpenAI,
    )

    OpenAIConversationSummaryWorker(config())

    # A failed summary is attempted again by the next load; client retries
    # would only lengthen how long that turn waits on an outage.
    assert captured["max_retries"] == 0


def test_summary_fact_budget_drops_one_fact_and_preserves_the_others() -> None:
    from career_agent.harness.observability import ACTIVE_TRACE_CONTEXT, InMemoryTraceRecorder
    decoded = {name: [] for name in (
        "new_facts", "superseded_fact_ids", "user_goals", "confirmed_decisions",
        "unresolved_questions", "active_constraints", "long_term_memory_candidates",
    )}
    decoded["new_facts"] = [{"fact": "x" * 194 + "薪资 300k", "source_sequence": 1}, {"fact": "Keep this", "source_sequence": 2}]
    client = Client(json.dumps(decoded))
    worker = OpenAIConversationSummaryWorker(config(), client=client, disable_thinking=True)
    recorder = InMemoryTraceRecorder()
    token = ACTIVE_TRACE_CONTEXT.set((recorder, "summary"))
    try:
        result = worker.summarize(previous=None, messages=(SummaryMessage(sequence=1, role="user", content="hello"), SummaryMessage(sequence=2, role="assistant", content="Keep this")))
    finally:
        ACTIVE_TRACE_CONTEXT.reset(token)
    assert len(result.new_facts) == 1
    assert result.new_facts[0].fact == "Keep this"
    assert result.new_facts[0].source_sequence == 2
    events = [event for event in recorder.snapshot("summary").events if event.event_type == "summary_facts_dropped"]
    assert len(events) == 1
    assert events[0].details == {"overlong_fact_count": 1}
    assert "300k" not in events[0].model_dump_json()
    assert client.completions.kwargs["extra_body"] == {"enable_thinking": False}
    schema = client.completions.kwargs["response_format"]["json_schema"]["schema"]
    assert schema["$defs"]["SummaryKeyFact"]["properties"]["fact"]["maxLength"] == 200
    system = client.completions.kwargs["messages"][0]["content"]
    assert "200 characters" in system
    assert "Aim for at most" not in system
    assert "1200 tokens" in system


def test_fact_dropping_does_not_salvage_missing_provenance() -> None:
    decoded = {name: [] for name in (
        "new_facts", "superseded_fact_ids", "user_goals", "confirmed_decisions",
        "unresolved_questions", "active_constraints", "long_term_memory_candidates",
    )}
    decoded["new_facts"] = [{"fact": "x" * 201}]
    worker = OpenAIConversationSummaryWorker(config(), client=Client(json.dumps(decoded)))
    with pytest.raises(AgentWorkerError) as error:
        worker.summarize(previous=None, messages=(SummaryMessage(sequence=1, role="user", content="hello"),))
    assert error.value.code == "CONVERSATION_SUMMARY_INVALID_RESPONSE"


def test_discard_counter_is_not_recorded_when_the_whole_delta_is_invalid() -> None:
    from career_agent.harness.observability import ACTIVE_TRACE_CONTEXT, InMemoryTraceRecorder
    decoded = {name: [] for name in (
        "new_facts", "superseded_fact_ids", "user_goals", "confirmed_decisions",
        "unresolved_questions", "active_constraints", "long_term_memory_candidates",
    )}
    decoded["new_facts"] = [{"fact": "x" * 201, "source_sequence": 1}]
    decoded["user_goals"] = ["x" * 501]
    recorder = InMemoryTraceRecorder()
    token = ACTIVE_TRACE_CONTEXT.set((recorder, "summary"))
    try:
        with pytest.raises(AgentWorkerError):
            OpenAIConversationSummaryWorker(config(), client=Client(json.dumps(decoded))).summarize(
                previous=None, messages=(SummaryMessage(sequence=1, role="user", content="hello"),))
    finally:
        ACTIVE_TRACE_CONTEXT.reset(token)
    assert not any(event.event_type == "summary_facts_dropped" for event in recorder.snapshot("summary").events)
