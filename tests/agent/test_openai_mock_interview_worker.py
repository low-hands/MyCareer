from __future__ import annotations

import base64
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from career_agent.agent.mock_interview_skill_loader import MockInterviewSkillLoader
from career_agent.agent.openai_compatible_client import (
    AgentWorkerError,
    OpenAICompatibleAgentConfig,
)
from career_agent.agent.openai_mock_interview_worker import OpenAIMockInterviewWorker
from career_agent.domain.mock_interviews import (
    MockInterviewAnswerEvaluation,
    MockInterviewPlan,
    MockInterviewPlanItem,
    MockInterviewScoreDimension,
    MockInterviewSession,
    MockInterviewTurn,
)
from career_agent.storage.resumes import StoredResumeDocument


NOW = datetime(2026, 8, 27, tzinfo=timezone.utc)


class FakeResponses:
    def __init__(self, outputs: list[dict[str, object] | str]) -> None:
        self.outputs = list(outputs)
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        output = self.outputs.pop(0)
        output_text = output if isinstance(output, str) else json.dumps(output)
        return type("Response", (), {"output_text": output_text})()


class FakeClient:
    def __init__(self, *outputs: dict[str, object] | str) -> None:
        self.responses = FakeResponses(list(outputs))


def _session(*, max_primary_questions: int = 2) -> MockInterviewSession:
    return MockInterviewSession(
        id="session-secret",
        user_id="user-secret",
        application_id="application-secret",
        job_posting_id="job-secret",
        jd_snapshot_id="snapshot-secret",
        resume_version_id="resume-v1",
        interview_type="mixed",
        max_primary_questions=max_primary_questions,
        max_follow_ups_per_question=1,
        created_at=NOW,
        updated_at=NOW,
    )


def _items() -> tuple[MockInterviewPlanItem, ...]:
    return (
        MockInterviewPlanItem(
            sequence_number=1,
            question_type="project_deep_dive",
            difficulty="intermediate",
            focus="Personal ownership",
            rationale="Resume mentions retrieval work.",
            resume_locators=("experience.1",),
            resume_quotes=("Built retrieval systems",),
        ),
        MockInterviewPlanItem(
            sequence_number=2,
            question_type="system_design",
            difficulty="advanced",
            focus="Failure recovery",
            rationale="JD asks for reliable systems.",
            jd_quotes=("Build reliable systems",),
        ),
    )


def _plan() -> MockInterviewPlan:
    return MockInterviewPlan(
        session_id="session-secret",
        summary="Test ownership, then system design.",
        items=_items(),
        created_at=NOW,
    )


def _document(
    *, document_format: str = "markdown", raw_bytes: bytes = b"Built retrieval systems."
) -> StoredResumeDocument:
    return StoredResumeDocument(
        resume_version_id="resume-v1",
        document_format=document_format,
        raw_bytes=raw_bytes,
    )


def _worker(client: FakeClient) -> OpenAIMockInterviewWorker:
    return OpenAIMockInterviewWorker(
        OpenAICompatibleAgentConfig(
            endpoint="https://example.test/v1/chat/completions",
            api_key="secret",
            model="multimodal-model",
        ),
        skill_loader=MockInterviewSkillLoader(Path("skills")),
        client=client,
    )


def _plan_output(*, item_count: int = 2) -> dict[str, object]:
    return {
        "summary": "Test ownership and system design.",
        "items": [item.model_dump(mode="json") for item in _items()[:item_count]],
        "limitations": [],
    }


def _evaluation_output() -> dict[str, object]:
    return {
        "rating": "adequate",
        "summary": "Relevant but needs more detail.",
        "dimensions": [
            {
                "dimension": "specificity",
                "score": 3,
                "feedback": "Name the evaluation method.",
            }
        ],
        "strengths": ["Answered the question"],
        "improvements": ["Add concrete evidence"],
        "unsupported_claims": [],
        "next_action": "next_question",
        "next_action_reason": "Enough evidence to proceed.",
        "follow_up_question": None,
    }


def _answered_turn() -> MockInterviewTurn:
    return MockInterviewTurn(
        id="turn-secret",
        session_id="session-secret",
        sequence_number=1,
        plan_item_number=1,
        turn_type="primary",
        question_type="project_deep_dive",
        question="What did you personally own?",
        answer="I designed the offline retrieval evaluation.",
        status="answered",
        asked_at=NOW,
        answered_at=NOW,
    )


def _evaluated_turn() -> MockInterviewTurn:
    evaluation = MockInterviewAnswerEvaluation.model_validate(_evaluation_output())
    return _answered_turn().model_copy(
        update={"status": "evaluated", "evaluation": evaluation, "evaluated_at": NOW}
    )


def test_plan_loads_mixed_skill_and_sends_exact_text_sources() -> None:
    client = FakeClient(_plan_output())
    result = _worker(client).plan(
        session=_session(),
        document=_document(),
        jd_text="Build reliable systems.",
    )

    assert len(result.items) == 2
    call = client.responses.calls[0]
    assert call["model"] == "multimodal-model"
    assert call["text"]["format"]["name"] == "mock_interview_plan_result"  # type: ignore[index]
    instructions = call["instructions"]
    assert "# Loaded reference: technical" in instructions
    assert "# Loaded reference: behavioral" in instructions
    assert "# Loaded reference: hr" in instructions
    content = call["input"][0]["content"][0]["text"]  # type: ignore[index]
    assert "<resume_document>" in content
    assert "Build reliable systems" in content
    assert "session-secret" not in content
    assert "user-secret" not in content


def test_ask_receives_only_the_current_plan_item_not_future_questions() -> None:
    client = FakeClient({"question": "What did you personally own?"})
    result = _worker(client).ask(
        session=_session(),
        plan=_plan(),
        plan_item=_items()[0],
        document=_document(),
        jd_text="Build reliable systems.",
    )

    assert result.question == "What did you personally own?"
    call = client.responses.calls[0]
    content = call["input"][0]["content"][0]["text"]  # type: ignore[index]
    assert "Personal ownership" in content
    assert "Failure recovery" not in content
    assert "# Loaded reference: technical" in call["instructions"]
    assert "# Loaded reference: behavioral" in call["instructions"]
    assert "# Loaded reference: hr" not in call["instructions"]


def test_evaluate_sends_pdf_as_input_file_and_persisted_answer_as_data() -> None:
    raw_pdf = b"%PDF-1.7\x00\xffbinary"
    client = FakeClient(_evaluation_output())
    result = _worker(client).evaluate(
        session=_session(),
        plan_item=_items()[0],
        turn=_answered_turn(),
        document=_document(document_format="pdf", raw_bytes=raw_pdf),
        jd_text="Build reliable systems.",
    )

    assert result.next_action == "next_question"
    content = client.responses.calls[0]["input"][0]["content"]  # type: ignore[index]
    assert content[0]["type"] == "input_file"
    assert content[0]["file_data"] == (
        "data:application/pdf;base64," + base64.b64encode(raw_pdf).decode("ascii")
    )
    assert "offline retrieval evaluation" in content[1]["text"]
    assert "turn-secret" not in content[1]["text"]


def test_evaluate_requires_a_persisted_answer_before_model_call() -> None:
    client = FakeClient(_evaluation_output())
    awaiting = _answered_turn().model_copy(
        update={"answer": None, "answered_at": None, "status": "awaiting_answer"}
    )

    with pytest.raises(AgentWorkerError) as error:
        _worker(client).evaluate(
            session=_session(),
            plan_item=_items()[0],
            turn=awaiting,
            document=_document(),
            jd_text="Build reliable systems.",
        )

    assert error.value.code == "MOCK_INTERVIEW_ANSWER_MISSING"
    assert client.responses.calls == []


def test_plan_rejects_more_items_than_the_session_limit() -> None:
    client = FakeClient(_plan_output(item_count=2))

    with pytest.raises(AgentWorkerError, match="invalid structured output") as error:
        _worker(client).plan(
            session=_session(max_primary_questions=1),
            document=_document(),
            jd_text="Build reliable systems.",
        )

    assert error.value.code == "MOCK_INTERVIEW_INVALID_RESPONSE"
    assert error.value.detail == "plan exceeds max_primary_questions"


def test_report_requires_exactly_one_result_per_answered_primary_item() -> None:
    invalid_report = {
        "summary": "Practice completed.",
        "question_results": [
            {
                "plan_item_number": 2,
                "question": "A question never answered",
                "final_rating": "adequate",
                "summary": "No grounded result.",
                "follow_up_count": 0,
            }
        ],
        "strengths": [],
        "development_areas": [],
        "practice_actions": [],
        "limitations": [],
    }
    client = FakeClient(invalid_report)

    with pytest.raises(AgentWorkerError) as error:
        _worker(client).report(
            session=_session(),
            plan=_plan(),
            turns=(_evaluated_turn(),),
            completion_reason="plan_completed",
            document=_document(),
            jd_text="Build reliable systems.",
        )

    assert error.value.code == "MOCK_INTERVIEW_INVALID_RESPONSE"
    assert "answered primary" in (error.value.detail or "")


def test_report_preserves_persisted_question_and_follow_up_count() -> None:
    report = {
        "summary": "Practice completed.",
        "question_results": [
            {
                "plan_item_number": 1,
                "question": "A rewritten question",
                "final_rating": "adequate",
                "summary": "Relevant but needs more detail.",
                "follow_up_count": 0,
            }
        ],
        "strengths": [],
        "development_areas": [],
        "practice_actions": [],
        "limitations": [],
    }
    client = FakeClient(report)

    with pytest.raises(AgentWorkerError) as error:
        _worker(client).report(
            session=_session(),
            plan=_plan(),
            turns=(_evaluated_turn(),),
            completion_reason="plan_completed",
            document=_document(),
            jd_text="Build reliable systems.",
        )

    assert "persisted turns" in (error.value.detail or "")


def test_worker_uses_dedicated_environment_prefix() -> None:
    client = FakeClient(_plan_output())
    worker = OpenAIMockInterviewWorker.from_env(
        skills_root=Path("skills"),
        environ={
            "MOCK_INTERVIEW_AGENT_BASE_URL": "https://example.test/v1",
            "MOCK_INTERVIEW_AGENT_API_KEY": "secret",
            "MOCK_INTERVIEW_AGENT_MODEL": "mock-model",
        },
        client=client,
    )

    assert worker._config.model == "mock-model"
