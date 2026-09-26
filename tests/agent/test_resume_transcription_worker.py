"""The model copies a resume PDF's text out; nothing else is accepted as a copy."""

from __future__ import annotations

import json

import pytest

from career_agent.agent.openai_compatible_client import (
    AgentWorkerError,
    OpenAICompatibleAgentConfig,
)
from career_agent.agent.openai_resume_transcription_worker import (
    OpenAIResumeTranscriptionWorker,
)


class _Client:
    def __init__(self, output: dict) -> None:
        self.output_text = json.dumps(output, ensure_ascii=False)
        self.calls: list[dict] = []
        self.responses = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return type("Response", (), {"output_text": self.output_text})()


def _worker(output: dict) -> tuple[OpenAIResumeTranscriptionWorker, _Client]:
    client = _Client(output)
    config = OpenAICompatibleAgentConfig(
        endpoint="https://example.invalid/v1", api_key="k", model="m", timeout_seconds=30
    )
    return OpenAIResumeTranscriptionWorker(config, client=client), client


def test_the_pdf_goes_as_a_file_and_pages_come_back_in_order() -> None:
    worker, client = _worker(
        {"pages": [
            {"page": 2, "lines": ["项目经历", "  DeepSearch 多跳搜索  "]},
            {"page": 1, "lines": ["郭睿", "", "英属哥伦比亚大学"]},
        ]}
    )

    pages = worker.transcribe(pdf=b"%PDF-1.7 synthetic", page_count=2)

    assert pages == ("郭睿\n英属哥伦比亚大学", "项目经历\nDeepSearch 多跳搜索")
    (call,) = client.calls
    file_part = call["input"][0]["content"][0]
    assert file_part["type"] == "input_file"
    assert file_part["file_data"].startswith("data:application/pdf;base64,")
    assert "Do not summarize" in call["instructions"]


@pytest.mark.parametrize(
    ("pages", "page_count", "code"),
    [
        ([{"page": 1, "lines": ["a"]}], 2, "RESUME_TRANSCRIPTION_PAGE_MISMATCH"),
        ([{"page": 1, "lines": ["a"]}, {"page": 1, "lines": ["b"]}], None, "RESUME_TRANSCRIPTION_PAGE_MISMATCH"),
        ([{"page": 1, "lines": ["", "  "]}], 1, "RESUME_TRANSCRIPTION_EMPTY"),
    ],
)
def test_a_copy_that_misses_or_repeats_a_page_or_is_empty_is_refused(pages, page_count, code) -> None:
    worker, _ = _worker({"pages": pages})

    with pytest.raises(AgentWorkerError) as caught:
        worker.transcribe(pdf=b"%PDF-1.7 synthetic", page_count=page_count)
    assert caught.value.code == code
