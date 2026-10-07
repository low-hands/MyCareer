from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from career_agent.agent.contracts.questionnaire import (
    PendingQuestionnaire, QuestionAnswer, QuestionOption, UserQuestion,
)
from career_agent.agent.contracts.decisions import AgentDecision
from career_agent.agent.providers.interaction_output import interaction_schemas, parse_interaction


def _pending() -> PendingQuestionnaire:
    now = datetime.now(timezone.utc)
    return PendingQuestionnaire(
        interaction_id="interaction_1234567890abcdef1234",
        prompt="请回答两题",
        questions=(
            UserQuestion(question_id="q1", prompt="技能", kind="multiple", options=(
                QuestionOption(value="python", label="Python"),
                QuestionOption(value="none", label="没有", meaning="none"),
                QuestionOption(value="other", label="其他", meaning="other"),
            )),
            UserQuestion(question_id="q2", prompt="经历", kind="free_text"),
        ),
        created_at=now, expires_at=now + timedelta(days=7), active_workflow="none",
    )


def test_answers_bind_every_question_once_and_keep_none_distinct() -> None:
    pending = _pending()
    pending.validate_answers((
        QuestionAnswer(question_id="q1", selected_values=("none",)),
        QuestionAnswer(question_id="q2", skipped=True),
    ))
    with pytest.raises(ValueError, match="answers must match"):
        pending.validate_answers((
            QuestionAnswer(question_id="q1", selected_values=("python",)),
            QuestionAnswer(question_id="q1", selected_values=("python",)),
        ))
    with pytest.raises(ValueError, match="unknown option"):
        pending.validate_answers((
            QuestionAnswer(question_id="q1", selected_values=("invented",)),
            QuestionAnswer(question_id="q2", skipped=True),
        ))
    with pytest.raises(ValueError, match="none is exclusive"):
        pending.validate_answers((
            QuestionAnswer(question_id="q1", selected_values=("none", "python")),
            QuestionAnswer(question_id="q2", skipped=True),
        ))
    with pytest.raises(ValueError, match="other requires text"):
        pending.validate_answers((
            QuestionAnswer(question_id="q1", selected_values=("other",)),
            QuestionAnswer(question_id="q2", skipped=True),
        ))


def test_answer_text_and_question_count_are_bounded() -> None:
    with pytest.raises(ValidationError):
        QuestionAnswer(question_id="q1", free_text="x" * 1001)
    with pytest.raises(ValidationError):
        UserQuestion(question_id="q9", prompt="超界", kind="free_text")


def test_questionnaire_continuation_survives_storage_roundtrip() -> None:
    pending = _pending().model_copy(update={
        "continuation_capability": "draft_resume_tailoring",
        "resume_job_match_id": "match-1",
    })
    restored = PendingQuestionnaire.model_validate_json(pending.model_dump_json())
    assert restored.continuation_capability == "draft_resume_tailoring"
    assert restored.resume_job_match_id == "match-1"
    assert PendingQuestionnaire.model_validate(
        _pending().model_dump(exclude={"continuation_capability", "resume_job_match_id"})
    ).continuation_capability is None
    with pytest.raises(ValidationError, match="callable business capability"):
        PendingQuestionnaire.model_validate({
            **_pending().model_dump(), "continuation_capability": "route_to_capability",
        })


def test_search_questionnaire_schema_requires_explicit_continuation_binding() -> None:
    legacy = next(item["function"]["parameters"] for item in interaction_schemas()
                  if item["function"]["name"] == "questionnaire")
    search = next(item["function"]["parameters"] for item in interaction_schemas(continuation=True)
                  if item["function"]["name"] == "questionnaire")
    assert "continuation_capability" not in legacy["properties"]
    assert "continuation_capability" in search["required"]
    assert search["properties"]["continuation_capability"]["type"] == ["string", "null"]
    typed_search = next(item["function"]["parameters"] for item in interaction_schemas(continuation=True)
                        if item["function"]["name"] == "respond_to_user")
    assert "continuation_capability" in typed_search["properties"]
    assert "continuation_capability" not in typed_search["required"]


def test_questionnaire_output_binds_only_a_known_business_capability() -> None:
    questions = [
        {"question_id": "q1", "prompt": "一", "kind": "free_text"},
        {"question_id": "q2", "prompt": "二", "kind": "free_text"},
    ]
    decision = parse_interaction("questionnaire", {
        "message": "请补充", "questions": questions,
        "continuation_capability": "draft_resume_tailoring",
    })
    assert decision.continuation_capability == "draft_resume_tailoring"
    with pytest.raises(ValidationError, match="callable business capability"):
        parse_interaction("questionnaire", {
            "message": "请补充", "questions": questions,
            "continuation_capability": "route_to_capability",
        })
    with pytest.raises(ValidationError, match="requires questionnaire"):
        AgentDecision(action="final", message="完成", continuation_capability="list_resumes")


def test_typed_interaction_discards_null_continuation_outside_questionnaire() -> None:
    assert parse_interaction("respond_to_user", {
        "requires_user_input": False, "message": "好的",
        "continuation_capability": None,
    }).action == "final"
    assert parse_interaction("respond_to_user", {
        "requires_user_input": True, "message": "城市？",
        "continuation_capability": None,
    }).action == "ask_user"
    with pytest.raises(ValueError, match="requires questionnaire questions"):
        parse_interaction("respond_to_user", {
            "requires_user_input": True, "message": "城市？",
            "continuation_capability": "create_application",
        })
