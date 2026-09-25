from __future__ import annotations

import base64
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from career_agent.agent.interview_preparation_contracts import (
    InterviewPreparationContext,
    PriorInterviewRetroContext,
)
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
            question="What did you personally own?",
            resume_locators=("experience.1",),
            resume_quotes=("Built retrieval systems",),
        ),
        MockInterviewPlanItem(
            sequence_number=2,
            question_type="system_design",
            difficulty="advanced",
            focus="Failure recovery",
            rationale="JD asks for reliable systems.",
            question="How would the service recover from a failed dependency?",
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


def _context() -> InterviewPreparationContext:
    return InterviewPreparationContext(
        company_name="Example Corp",
        role_title="Platform Engineer",
        jd_text="Build reliable systems.",
        prior_retros=(PriorInterviewRetroContext(
            sequence_number=1,
            summary="Recovery discussion lacked concrete targets.",
            difficulties=("failure recovery",),
            next_focus=("recovery objectives",),
            self_assessment="mixed",
        ),),
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


def test_resume_without_jd_carries_explicit_missing_jd_marker() -> None:
    content = OpenAIMockInterviewWorker._document_content(
        document=_document(), jd_text="", payload={"operation": "ask"}
    )
    assert "No job description supplied; do not invent one." in content[-1]["text"]


def test_practice_without_a_resume_still_carries_the_chosen_jd() -> None:
    content = OpenAIMockInterviewWorker._document_content(
        document=None, jd_text="负责大模型产品的增长。", payload={"operation": "plan"}
    )
    text = content[-1]["text"]
    assert "No resume supplied" in text
    assert "负责大模型产品的增长。" in text
    assert "No job description supplied" not in text


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
        "key_facts": ["team of 5"],
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
    evaluation = MockInterviewAnswerEvaluation(
        **_evaluation_output(),
        next_action="finish",
        next_action_reason="evaluated after the interview ended",
    )
    return _answered_turn().model_copy(
        update={"status": "evaluated", "evaluation": evaluation, "evaluated_at": NOW}
    )


def _follow_up_turn(*, answer: str = "Recall rose 12% over two weeks.") -> MockInterviewTurn:
    return MockInterviewTurn(
        id="turn-follow-up",
        session_id="session-secret",
        sequence_number=2,
        plan_item_number=1,
        turn_type="follow_up",
        parent_turn_id="turn-secret",
        question_type="project_deep_dive",
        question="What changed in the metric?",
        answer=answer,
        status="answered",
        asked_at=NOW,
        answered_at=NOW,
    )


def test_plan_loads_mixed_skill_and_sends_exact_text_sources() -> None:
    client = FakeClient(_plan_output())
    result = _worker(client).plan(
        session=_session(),
        document=_document(),
        context=_context(),
    )

    assert len(result.items) == 2
    call = client.responses.calls[0]
    assert call["model"] == "multimodal-model"
    assert call["text"]["format"]["name"] == "mock_interview_plan_result"  # type: ignore[index]
    instructions = call["instructions"]
    assert "# Loaded reference: technical" in instructions
    assert "# Loaded reference: behavioral" in instructions
    assert "# Loaded reference: hr" in instructions
    assert "# Loaded reference: company" in instructions
    content = call["input"][0]["content"][0]["text"]  # type: ignore[index]
    assert "<resume_document>" in content
    assert "Build reliable systems" in content
    assert '"target_company": "Example Corp"' in content
    assert '"target_role": "Platform Engineer"' in content
    assert '"prior_real_interview_retros"' in content
    assert "recovery objectives" in content
    assert "session-secret" not in content
    assert "user-secret" not in content


@pytest.mark.parametrize(
    ("model_action", "message"),
    [
        ("cancel", "不想练了，结束面试"),
        ("answer", "项目最后结束于灰度上线，我负责回滚指标。"),
    ],
)
def test_route_input_uses_only_the_current_question_and_local_message(
    model_action: str, message: str
) -> None:
    client = FakeClient({"action": model_action})

    result = _worker(client).route_input(
        session=_session(),
        turn=_answered_turn().model_copy(
            update={"answer": None, "answered_at": None, "status": "awaiting_answer"}
        ),
        user_message=message,
    )

    assert result.action == model_action
    call = client.responses.calls[0]
    assert call["text"]["format"]["name"] == "mock_interview_input_route_result"  # type: ignore[index]
    assert "ambiguous input" in call["instructions"]
    assert "inside a substantive answer" in call["instructions"]
    content = call["input"][0]["content"][0]["text"]  # type: ignore[index]
    assert "What did you personally own?" in content
    assert message in content
    assert "Build reliable systems" not in content
    assert "Built retrieval systems" not in content
    assert "session-secret" not in content


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
    assert "# Loaded reference: company" in call["instructions"]
    assert "# Loaded reference: hr" not in call["instructions"]


def test_evaluate_scores_one_question_chain_with_the_pdf_as_input_file() -> None:
    raw_pdf = b"%PDF-1.7\x00\xffbinary"
    client = FakeClient(_evaluation_output())
    result = _worker(client).evaluate(
        session=_session(),
        plan_item=_items()[0],
        turns=(_answered_turn(), _follow_up_turn()),
        document=_document(document_format="pdf", raw_bytes=raw_pdf),
        jd_text="Build reliable systems.",
    )

    # The workflow already routed; the stored evaluation only records the score.
    assert result.next_action == "finish"
    assert result.key_facts == ("team of 5",)
    call = client.responses.calls[0]
    assert call["text"]["format"]["name"] == "mock_interview_evaluate_result"  # type: ignore[index]
    content = call["input"][0]["content"]  # type: ignore[index]
    assert content[0]["type"] == "input_file"
    assert content[0]["file_data"] == (
        "data:application/pdf;base64," + base64.b64encode(raw_pdf).decode("ascii")
    )
    assert "offline retrieval evaluation" in content[1]["text"]
    assert "Recall rose 12%" in content[1]["text"]
    assert "turn-secret" not in content[1]["text"]


def test_evaluate_requires_an_answered_primary_question_before_model_call() -> None:
    client = FakeClient(_evaluation_output())
    awaiting = _answered_turn().model_copy(
        update={"answer": None, "answered_at": None, "status": "awaiting_answer"}
    )

    with pytest.raises(AgentWorkerError) as error:
        _worker(client).evaluate(
            session=_session(),
            plan_item=_items()[0],
            turns=(awaiting,),
            document=_document(),
            jd_text="Build reliable systems.",
        )

    assert error.value.code == "MOCK_INTERVIEW_ANSWER_MISSING"
    assert client.responses.calls == []


def test_a_follow_up_decision_sees_only_the_current_question_chain() -> None:
    client = FakeClient({"next_action": "follow_up", "follow_up_question": "Who else worked on it?"})
    decision = _worker(client).decide_follow_up(
        session=_session(),
        plan_item=_items()[0],
        turns=(_answered_turn(),),
        follow_ups_remaining=1,
        document=_document(),
        jd_text="Build reliable systems.",
    )

    assert decision.follow_up_question == "Who else worked on it?"
    call = client.responses.calls[0]
    assert call["text"]["format"]["name"] == "mock_interview_follow_up_result"  # type: ignore[index]
    assert call["max_output_tokens"] == 1024
    content = call["input"][0]["content"][0]["text"]  # type: ignore[index]
    assert "offline retrieval evaluation" in content
    assert '"follow_ups_remaining": 1' in content
    assert "Failure recovery" not in content


def test_a_long_answer_is_cut_to_the_budget_and_marked() -> None:
    client = FakeClient(_evaluation_output())
    long_answer = "细节" * 5000
    _worker(client).evaluate(
        session=_session(),
        plan_item=_items()[0],
        turns=(_answered_turn(), _follow_up_turn(answer=long_answer)),
        document=_document(),
        jd_text="Build reliable systems.",
    )

    content = client.responses.calls[0]["input"][0]["content"][0]["text"]  # type: ignore[index]
    assert long_answer not in content
    assert "answer truncated for length: 10000 characters in full" in content


def test_plan_rejects_more_items_than_the_session_limit() -> None:
    client = FakeClient(_plan_output(item_count=2))

    with pytest.raises(AgentWorkerError, match="invalid structured output") as error:
        _worker(client).plan(
                session=_session(max_primary_questions=1),
                document=_document(),
                context=_context(),
        )

    assert error.value.code == "MOCK_INTERVIEW_INVALID_RESPONSE"
    assert error.value.detail == "plan exceeds max_primary_questions"


def _synthesis_output(**overrides: object) -> dict[str, object]:
    return {
        "summary": "Practice completed.",
        "strengths": ["Clear ownership"],
        "development_areas": ["Quantify impact"],
        "practice_actions": [],
        "limitations": [],
        "consistency_issues": [],
        **overrides,
    }


def test_report_results_come_from_the_stored_questions_not_the_model() -> None:
    client = FakeClient(_synthesis_output())
    follow_up = _follow_up_turn()

    draft = _worker(client).report(
        session=_session(),
        plan=_plan(),
        turns=(_evaluated_turn(), follow_up),
        completion_reason="plan_completed",
        document=_document(),
        jd_text="Build reliable systems.",
    )

    [result] = draft.question_results
    assert result.question == "What did you personally own?"
    assert result.final_rating == "adequate"
    assert result.follow_up_count == 1
    content = client.responses.calls[0]["input"][0]["content"][0]["text"]  # type: ignore[index]
    # Evaluations and key facts, not the transcripts they were scored from.
    assert '"key_facts": ["team of 5"]' in content
    assert "offline retrieval evaluation" not in content
    assert "Recall rose 12%" not in content


def test_contradictions_across_questions_lead_the_development_areas() -> None:
    client = FakeClient(_synthesis_output(consistency_issues=["第1题说团队5人，第4题说3人"]))

    draft = _worker(client).report(
        session=_session(),
        plan=_plan(),
        turns=(_evaluated_turn(),),
        completion_reason="plan_completed",
        document=_document(),
        jd_text="Build reliable systems.",
    )

    assert draft.development_areas == (
        "前后说法不一致：第1题说团队5人，第4题说3人",
        "Quantify impact",
    )


def test_a_report_needs_at_least_one_scored_question() -> None:
    client = FakeClient(_synthesis_output())

    with pytest.raises(AgentWorkerError) as error:
        _worker(client).report(
            session=_session(),
            plan=_plan(),
            turns=(_answered_turn(),),
            completion_reason="plan_completed",
            document=_document(),
            jd_text="Build reliable systems.",
        )

    assert error.value.code == "MOCK_INTERVIEW_REPORT_EMPTY"
    assert client.responses.calls == []


def test_a_plan_without_written_questions_is_rejected() -> None:
    output = _plan_output()
    for item in output["items"]:  # type: ignore[union-attr]
        item["question"] = None
    client = FakeClient(output)

    with pytest.raises(AgentWorkerError) as error:
        _worker(client).plan(session=_session(), document=_document(), context=_context())

    assert error.value.detail == "every plan item must carry its primary question"


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


def test_a_plan_that_over_quotes_keeps_three_paired_quotes() -> None:
    """Seen live: five resume quotes per item, or one unpaired locator, failed the plan."""
    output = _plan_output()
    items = output["items"]  # type: ignore[index]
    items[0]["resume_locators"] = [f"experience.{n}" for n in range(5)]
    items[0]["resume_quotes"] = [f"quote {n}" for n in range(5)]
    items[0]["jd_quotes"] = ["a", "b", "c", "d"]
    items[1]["resume_locators"] = ["experience.1", "experience.2"]
    items[1]["resume_quotes"] = ["only one"]
    client = FakeClient(output)

    plan = _worker(client).plan(session=_session(), document=_document(), context=_context())

    assert plan.items[0].resume_quotes == ("quote 0", "quote 1", "quote 2")
    assert plan.items[0].resume_locators == ("experience.0", "experience.1", "experience.2")
    assert plan.items[0].jd_quotes == ("a", "b", "c")
    assert (plan.items[1].resume_locators, plan.items[1].resume_quotes) == (
        ("experience.1",), ("only one",),
    )


def test_an_evaluation_that_overflows_its_lists_is_trimmed_not_rejected() -> None:
    """Seen live: both end-of-interview scores failed on list overflow."""
    output = _evaluation_output()
    output["key_facts"] = [f"fact {n}" for n in range(12)]
    output["strengths"] = [f"strength {n}" for n in range(11)]
    output["dimensions"] = output["dimensions"] * 2  # type: ignore[operator]
    client = FakeClient(output)

    result = _worker(client).evaluate(
        session=_session(),
        plan_item=_items()[0],
        turns=(_answered_turn(),),
        document=_document(),
        jd_text="Build reliable systems.",
    )

    assert len(result.key_facts) == 8
    assert len(result.strengths) == 8
    assert [item.dimension for item in result.dimensions] == ["specificity"]


@pytest.mark.parametrize(
    ("company", "model_says", "expected", "sent"),
    [
        # A listed alias is decided in code, whatever the model returns.
        ("字节", None, "ByteDance and related businesses", "ByteDance and related businesses"),
        ("字节", "Tencent and related businesses", "ByteDance and related businesses",
         "ByteDance and related businesses"),
        # An unlisted name is the model's call, kept only if it names a real profile.
        ("北京字节跳动科技有限公司", "ByteDance and related businesses",
         "ByteDance and related businesses", None),
        ("某创业公司", "A made-up profile", None, None),
        ("", "ByteDance and related businesses", None, None),
    ],
)
def test_the_style_profile_is_matched_in_code_and_only_otherwise_by_the_model(
    company, model_says, expected, sent
) -> None:
    client = FakeClient({**_plan_output(), "company_style_profile": model_says})
    result = _worker(client).plan(
        session=_session(),
        document=_document(),
        context=_context().model_copy(update={"company_name": company}),
    )

    assert result.company_style_profile == expected
    content = client.responses.calls[0]["input"][0]["content"][0]["text"]  # type: ignore[index]
    assert f'"company_style_profile": {json.dumps(sent, ensure_ascii=False)}' in content
