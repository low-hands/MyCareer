from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from career_agent.domain.mock_interviews import (
    MockInterviewAnswerEvaluation,
    MockInterviewPlan,
    MockInterviewPlanItem,
    MockInterviewQuestionResult,
    MockInterviewReport,
    MockInterviewScoreDimension,
    MockInterviewSession,
    MockInterviewTurn,
)


NOW = datetime(2026, 8, 26, 8, 0, tzinfo=timezone.utc)


def evaluation(*, next_action="next_question", follow_up_question=None):
    return MockInterviewAnswerEvaluation(
        rating="adequate",
        summary="The answer is relevant but needs a clearer trade-off.",
        dimensions=(
            MockInterviewScoreDimension(
                dimension="reasoning",
                score=3,
                feedback="One design choice was not justified.",
            ),
            MockInterviewScoreDimension(
                dimension="communication",
                score=4,
                feedback="The explanation was easy to follow.",
            ),
        ),
        strengths=("Clear decomposition",),
        improvements=("Explain the storage trade-off",),
        next_action=next_action,
        next_action_reason="One focused clarification is sufficient.",
        follow_up_question=follow_up_question,
    )


def test_session_binds_immutable_application_inputs_and_lifecycle() -> None:
    session = MockInterviewSession(
        id="mock-1",
        user_id="u1",
        application_id="app-1",
        interview_round_id="round-1",
        job_posting_id="job-1",
        jd_snapshot_id="jd-snapshot-1",
        resume_version_id="resume-v1",
        interview_type="mixed",
        status="active",
        current_turn_id="turn-1",
        created_at=NOW,
        started_at=NOW,
        updated_at=NOW,
    )

    assert session.application_id == "app-1"
    assert session.jd_snapshot_id == "jd-snapshot-1"
    assert session.resume_version_id == "resume-v1"

    with pytest.raises(ValidationError, match="paused sessions require paused_at"):
        MockInterviewSession.model_validate(
            {
                **session.model_dump(),
                "status": "paused",
                "current_turn_id": None,
            }
        )


def test_plan_requires_ordered_items_and_aligned_resume_evidence() -> None:
    item = MockInterviewPlanItem(
        sequence_number=1,
        question_type="project_deep_dive",
        difficulty="intermediate",
        focus="Explain the candidate's RAG design trade-offs.",
        rationale="The JD asks for production retrieval systems.",
        jd_quotes=("Build production RAG systems",),
        resume_locators=("Experience, bullet 1",),
        resume_quotes=("Built RAG systems",),
    )
    plan = MockInterviewPlan(
        session_id="mock-1",
        summary="Cover grounded project depth before broader design questions.",
        items=(item,),
        created_at=NOW,
    )

    assert plan.items[0].resume_quotes == ("Built RAG systems",)

    with pytest.raises(ValidationError, match="contiguous"):
        MockInterviewPlan(
            session_id="mock-1",
            summary="Invalid order.",
            items=(item.model_copy(update={"sequence_number": 2}),),
            created_at=NOW,
        )

    with pytest.raises(ValidationError, match="equal length"):
        MockInterviewPlanItem(
            **{
                **item.model_dump(),
                "resume_quotes": (),
            }
        )


def test_turn_separates_awaiting_answer_from_evaluated_data() -> None:
    awaiting = MockInterviewTurn(
        id="turn-1",
        session_id="mock-1",
        sequence_number=1,
        plan_item_number=1,
        turn_type="primary",
        question_type="system_design",
        question="How would you design the retrieval evaluation pipeline?",
        asked_at=NOW,
    )
    assert awaiting.answer is None

    completed = MockInterviewTurn(
        **{
            **awaiting.model_dump(),
            "status": "evaluated",
            "answer": "I would start with an offline labelled set.",
            "evaluation": evaluation().model_dump(),
            "answered_at": NOW + timedelta(minutes=2),
            "evaluated_at": NOW + timedelta(minutes=2, seconds=1),
        }
    )
    assert completed.evaluation.rating == "adequate"

    with pytest.raises(ValidationError, match="parent_turn_id"):
        MockInterviewTurn(
            **{
                **awaiting.model_dump(),
                "id": "turn-2",
                "sequence_number": 2,
                "turn_type": "follow_up",
            }
        )


def test_evaluation_requires_one_explicit_follow_up_question() -> None:
    follow_up = evaluation(
        next_action="follow_up",
        follow_up_question="What failure mode would you test first?",
    )
    assert follow_up.follow_up_question is not None

    with pytest.raises(ValidationError, match="requires a follow_up_question"):
        evaluation(next_action="follow_up")
    with pytest.raises(ValidationError, match="must be unique"):
        MockInterviewAnswerEvaluation(
            **{
                **evaluation().model_dump(),
                "dimensions": (
                    evaluation().dimensions[0].model_dump(),
                    evaluation().dimensions[0].model_dump(),
                ),
            }
        )


def test_report_is_practice_feedback_not_an_employer_decision() -> None:
    result = MockInterviewQuestionResult(
        plan_item_number=1,
        question="Explain one design trade-off.",
        final_rating="adequate",
        summary="The reasoning was relevant but incomplete.",
        follow_up_count=1,
    )
    report = MockInterviewReport(
        id="report-1",
        session_id="mock-1",
        completion_reason="user_ended",
        summary="Practice ended after one evaluated question.",
        question_results=(result,),
        strengths=("Clear communication",),
        development_areas=("Trade-off depth",),
        practice_actions=("Practise stating assumptions before choosing a design",),
        limitations=("This practice report does not predict an employer decision.",),
        created_at=NOW,
    )

    assert report.completion_reason == "user_ended"

    with pytest.raises(ValidationError, match="must be unique"):
        MockInterviewReport(
            **{
                **report.model_dump(),
                "question_results": (result.model_dump(), result.model_dump()),
            }
        )
