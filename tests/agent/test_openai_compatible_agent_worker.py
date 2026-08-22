import json

import pytest

from career_agent.agent.openai_compatible_agent_worker import OpenAICompatibleAgentWorker
from career_agent.agent.job_discovery_contracts import JDAnalysis, QueryProposal
from career_agent.agent.openai_compatible_client import AgentWorkerError, OpenAICompatibleAgentConfig


class FakeCompletions:
    def __init__(self): self.kwargs = None
    def create(self, **kwargs):
        self.kwargs = kwargs
        message = type("Message", (), {"content": json.dumps({"query": "LLM Engineer", "rationale": "resume context"})})()
        choice = type("Choice", (), {"message": message})()
        return type("Response", (), {"choices": [choice]})()


class FakeClient:
    def __init__(self):
        self.completions = FakeCompletions()
        self.chat = type("Chat", (), {"completions": self.completions})()


def test_agent_worker_parses_stage_output_with_openai_sdk_shape():
    client = FakeClient()
    worker = OpenAICompatibleAgentWorker(OpenAICompatibleAgentConfig(endpoint="https://example.test/v1/chat/completions", api_key="secret", model="openai-test"), client=client)

    result = worker.decide(stage="search_strategy", input={"target": "AI Engineer"}, output_type=QueryProposal)

    assert result.query == "LLM Engineer"
    assert client.completions.kwargs["model"] == "openai-test"
    assert "search_strategy" in client.completions.kwargs["messages"][0]["content"]


class InvalidCompletions:
    def create(self, **kwargs):
        message = type("Message", (), {"content": json.dumps({"query": "LLM Engineer"})})()
        choice = type("Choice", (), {"message": message})()
        return type("Response", (), {"choices": [choice]})()


class InvalidClient:
    def __init__(self):
        self.completions = InvalidCompletions()
        self.chat = type("Chat", (), {"completions": self.completions})()


def test_jd_analysis_prompt_is_job_only_and_grounded():
    worker = OpenAICompatibleAgentWorker(OpenAICompatibleAgentConfig(endpoint="https://example.test/v1/chat/completions", api_key="secret", model="openai-test"), client=FakeClient())

    prompt = worker._system_prompt("jd_analysis")

    assert "only the position" in prompt
    assert "only input.jd_text as evidence" in prompt
    assert "Do not evaluate any candidate" in prompt
    assert "Do not infer common industry skills" in prompt


def test_agent_worker_reports_schema_validation_details():
    worker = OpenAICompatibleAgentWorker(
        OpenAICompatibleAgentConfig(endpoint="https://example.test/v1/chat/completions", api_key="secret", model="openai-test"),
        client=InvalidClient(),
    )

    with pytest.raises(AgentWorkerError) as error:
        worker.decide(stage="search_strategy", input={"target": "AI Engineer"}, output_type=QueryProposal)

    assert error.value.code == "AGENT_WORKER_INVALID_RESPONSE"
    assert "rationale" in error.value.detail
    assert "missing" in error.value.detail
