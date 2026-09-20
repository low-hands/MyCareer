from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from career_agent.agent.questionnaire_contracts import (
    PendingQuestionnaire, QuestionAnswer, QuestionOption, UserQuestion,
)


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
