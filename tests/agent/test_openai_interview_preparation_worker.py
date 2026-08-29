import base64
import json
from datetime import datetime, timezone

from career_agent.agent.interview_preparation_contracts import (
    InterviewLogisticsContext,
    InterviewPreparationContext,
    PriorInterviewRetroContext,
)
from career_agent.agent.openai_compatible_client import OpenAICompatibleAgentConfig
from career_agent.agent.openai_interview_preparation_worker import OpenAIInterviewPreparationWorker
from career_agent.storage.resumes import StoredResumeDocument


RESULT = {
    "summary": "Prepare RAG evaluation evidence.",
    "focus_areas": [], "evidence_stories": [], "likely_questions": [],
    "gaps": [], "questions_to_ask": [], "checklist": [], "limitations": [],
}


class Responses:
    def __init__(self):
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return type("Response", (), {"output_text": json.dumps(RESULT)})()


class Client:
    def __init__(self):
        self.responses = Responses()


def worker(client):
    return OpenAIInterviewPreparationWorker(
        OpenAICompatibleAgentConfig(
            endpoint="https://example.test/v1/chat/completions",
            api_key="secret", model="multimodal-model",
        ),
        client=client,
    )


def context():
    return InterviewPreparationContext(
        company_name="Example Corp",
        role_title="RAG Engineer",
        jd_text="Build reliable RAG systems.",
        logistics=InterviewLogisticsContext(
            scheduled_start=datetime(2026, 8, 28, tzinfo=timezone.utc),
            timezone="Asia/Shanghai", interview_format="video",
        ),
        prior_retros=(PriorInterviewRetroContext(
            sequence_number=1,
            summary="系统设计回答不够具体。",
            difficulties=("故障恢复",),
            next_focus=("量化恢复目标",),
            self_assessment="mixed",
        ),),
    )


def test_text_preparation_marks_resume_jd_and_interview_as_untrusted_data() -> None:
    client = Client()
    worker(client).prepare(
        document=StoredResumeDocument(
            resume_version_id="v1", document_format="markdown",
            raw_bytes=b"Built a RAG evaluation suite.",
        ),
        context=context(),
    )

    text = client.responses.kwargs["input"][0]["content"][0]["text"]
    assert "<resume_document>" in text
    assert "<job_description>" in text
    assert "<interview_logistics>" in text
    assert "<prior_real_interview_retros>" in text
    assert "量化恢复目标" in text
    assert "never invent achievements" in client.responses.kwargs["instructions"]


def test_pdf_preparation_sends_original_file_without_main_agent_extraction() -> None:
    raw_pdf = b"%PDF-1.7\x00\xffbinary"
    client = Client()
    worker(client).prepare(
        document=StoredResumeDocument(
            resume_version_id="pdf-v1", document_format="pdf", raw_bytes=raw_pdf,
        ),
        context=context(),
    )

    content = client.responses.kwargs["input"][0]["content"]
    assert content[0]["type"] == "input_file"
    assert content[0]["file_data"] == (
        "data:application/pdf;base64," + base64.b64encode(raw_pdf).decode("ascii")
    )
    assert "Build reliable RAG systems" in content[1]["text"]
