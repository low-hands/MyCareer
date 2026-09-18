from __future__ import annotations

import json

import pytest

from career_agent.agent.openai_compatible_client import (
    AgentWorkerError,
    OpenAICompatibleAgentConfig,
)
from career_agent.agent.openai_resume_analysis_worker import (
    OpenAIResumeAnalysisWorker,
)
from career_agent.storage.resumes import StoredResumeDocument


class FakeResponses:
    def __init__(self, output: dict[str, object] | str) -> None:
        self.kwargs: dict[str, object] | None = None
        self.output = output

    def create(self, **kwargs: object) -> object:
        self.kwargs = kwargs
        output_text = (
            self.output
            if isinstance(self.output, str)
            else json.dumps(self.output, ensure_ascii=False)
        )
        return type("Response", (), {"output_text": output_text})()


class FakeClient:
    def __init__(self, output: dict[str, object] | str) -> None:
        self.responses = FakeResponses(output)


def _config() -> OpenAICompatibleAgentConfig:
    return OpenAICompatibleAgentConfig(
        endpoint="https://example.test/v1/chat/completions",
        api_key="secret",
        model="multimodal-model",
    )


def _worker(*, client: FakeClient | None = None) -> OpenAIResumeAnalysisWorker:
    return OpenAIResumeAnalysisWorker(
        _config(),
        client=client or FakeClient(_valid_output()),
    )


def _valid_output() -> dict[str, object]:
    return {
        "records": [
            {
                "record_type": "work",
                "organization": "示例公司",
                "title": "产品经理",
                "start_year": 2022,
                "start_month": 3,
                "end_year": None,
                "end_month": None,
                "is_current": True,
                "source_locator": "工作经历，第 1 项",
                "source_quote": "示例公司 产品经理 2022.03-至今",
                "evidence": [
                    {
                        "claim": "负责知识库产品规划",
                        "source_locator": "工作经历，第 1 条",
                        "source_quote": "负责知识库产品规划",
                    }
                ],
            }
        ],
        "clarification_questions": [],
        "warnings": [],
    }


def test_worker_rejects_non_utf8_text_without_calling_model() -> None:
    client = FakeClient(_valid_output())
    worker = _worker(client=client)

    with pytest.raises(AgentWorkerError) as error:
        worker.analyze(
            StoredResumeDocument(
                resume_version_id="resume_version_1",
                document_format="markdown",
                raw_bytes=b"\xff\xfe",
            )
        )

    assert error.value.code == "RESUME_ANALYSIS_INVALID_TEXT_ENCODING"
    assert client.responses.kwargs is None


@pytest.mark.parametrize("document_format", ["text", "markdown", "pdf"])
def test_worker_rejects_empty_document_without_calling_model(
    document_format: str,
) -> None:
    client = FakeClient(_valid_output())
    worker = _worker(client=client)

    with pytest.raises(AgentWorkerError) as error:
        worker.analyze(
            StoredResumeDocument(
                resume_version_id="resume_version_1",
                document_format=document_format,
                raw_bytes=b"" if document_format == "pdf" else b"  \n",
            )
        )

    assert error.value.code == "RESUME_ANALYSIS_EMPTY_DOCUMENT"
    assert client.responses.kwargs is None


def test_worker_uses_dedicated_environment_prefix() -> None:
    client = FakeClient(_valid_output())

    worker = OpenAIResumeAnalysisWorker.from_env(
        environ={
            "RESUME_ANALYSIS_AGENT_BASE_URL": "https://example.test/v1",
            "RESUME_ANALYSIS_AGENT_API_KEY": "secret",
            "RESUME_ANALYSIS_AGENT_MODEL": "multimodal-model",
        },
        client=client,
    )

    assert worker._config.model == "multimodal-model"
