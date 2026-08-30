from pathlib import Path

from career_agent.agent.deepagent_job_research_worker import (
    DeepAgentJobResearchWorker,
)
from career_agent.agent.job_research_contracts import JobResearchWorkerRequest
from career_agent.agent.openai_compatible_client import OpenAICompatibleAgentConfig
from career_agent.domain.job_research import JobResearchScope


DRAFT = {
    "summary": "Enterprise retrieval context.",
    "sources": [
        {
            "source_key": "S1",
            "url": "https://example.com/product",
            "title": "Product",
            "publisher": "Example Corp",
            "published_at": None,
            "relevant_excerpt": "Enterprise retrieval product.",
        }
    ],
    "findings": [
        {
            "topic": "Product",
            "statement": "The product serves enterprise retrieval.",
            "evidence_type": "fact",
            "source_keys": ["S1"],
            "confidence": "high",
        }
    ],
    "open_questions": [],
    "limitations": [],
}


class Checkpointer:
    def __init__(self) -> None:
        self.deleted: list[str] = []

    def delete_thread(self, thread_id: str) -> None:
        self.deleted.append(thread_id)


class Agent:
    def __init__(self) -> None:
        self.calls = []

    def invoke(self, payload, *, config):
        self.calls.append((payload, config))
        return {"structured_response": DRAFT}


def _config() -> OpenAICompatibleAgentConfig:
    return OpenAICompatibleAgentConfig(
        endpoint="https://api.openai.com/v1/chat/completions",
        api_key="secret",
        model="gpt-5.4",
    )


def _request() -> JobResearchWorkerRequest:
    return JobResearchWorkerRequest(
        company_name="Example Corp",
        role_title="RAG Engineer",
        jd_text="Build reliable retrieval systems.",
        scope=JobResearchScope(
            focus="business context",
            user_provided_context="The interviewer mentioned a knowledge product.",
            max_sources=5,
        ),
    )


def test_worker_uses_run_as_thread_and_resumes_without_readding_input() -> None:
    agent = Agent()
    checkpointer = Checkpointer()
    worker = DeepAgentJobResearchWorker(
        _config(),
        skills_root=Path("skills"),
        checkpointer=checkpointer,
        agent=agent,
    )

    result = worker.research(run_id="run-1", request=_request())
    worker.research(run_id="run-1", request=_request(), resume=True)
    worker.forget("run-1")

    assert result.sources[0].source_key == "S1"
    assert agent.calls[0][1] == {"configurable": {"thread_id": "run-1"}}
    assert "<job_description>" in agent.calls[0][0]["messages"][0]["content"]
    assert agent.calls[1][0] is None
    assert checkpointer.deleted == ["run-1"]


def test_worker_builds_read_only_deepagent_with_web_search_and_checkpoint() -> None:
    captured = {}
    fake_agent = Agent()

    def factory(**kwargs):
        captured.update(kwargs)
        return fake_agent

    checkpointer = Checkpointer()
    DeepAgentJobResearchWorker(
        _config(),
        skills_root=Path("skills"),
        checkpointer=checkpointer,
        agent_factory=factory,
    )

    assert captured["tools"] == [{"type": "web_search"}]
    assert captured["checkpointer"] is checkpointer
    assert captured["subagents"] == []
    assert captured["response_format"].__name__ == "JobResearchDraft"
    assert "does not prove a role belongs" in captured["system_prompt"]
    assert "infer private team projects" in captured["system_prompt"]


def test_worker_request_does_not_treat_generic_jd_as_business_evidence() -> None:
    text = DeepAgentJobResearchWorker._request_text(_request())

    assert "JD only for defensible search anchors" in text
    assert "do not invent one" in text
    assert "unverified, user-reported search lead" in text
    assert "The interviewer mentioned a knowledge product." in text
