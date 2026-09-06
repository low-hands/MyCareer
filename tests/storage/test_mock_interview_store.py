from datetime import datetime, timedelta, timezone

import pytest

from career_agent.domain.mock_interviews import (
    MockInterviewAnswerEvaluation,
    MockInterviewPlan,
    MockInterviewPlanItem,
    MockInterviewQuestionResult,
    MockInterviewReport,
    MockInterviewScoreDimension,
    MockInterviewSession,
)
from career_agent.storage.mock_interviews import SQLiteMockInterviewStore


NOW = datetime(2026, 8, 26, 8, 0, tzinfo=timezone.utc)


def build_store(tmp_path) -> SQLiteMockInterviewStore:
    return SQLiteMockInterviewStore(tmp_path / "mock_interviews.sqlite3")


def create_session(store, *, user_id="u1", application_id="app-1") -> MockInterviewSession:
    return store.create_session(
        user_id=user_id,
        application_id=application_id,
        job_posting_id="job-1",
        jd_snapshot_id="jd-1",
        resume_version_id="resume-v1",
        interview_type="mixed",
        max_primary_questions=3,
        max_follow_ups_per_question=1,
    )


def plan_for(session, *, items=2) -> MockInterviewPlan:
    return MockInterviewPlan(
        session_id=session.id,
        summary="Cover grounded project depth then broader design reasoning.",
        items=tuple(
            MockInterviewPlanItem(
                sequence_number=index,
                question_type="project_deep_dive",
                difficulty="intermediate",
                focus=f"Probe evidence area {index}.",
                rationale="The JD requires production retrieval work.",
            )
            for index in range(1, items + 1)
        ),
        created_at=NOW,
    )


def evaluation(*, next_action="next_question", follow_up_question=None):
    return MockInterviewAnswerEvaluation(
        rating="adequate",
        summary="Relevant but the trade-off is unstated.",
        dimensions=(
            MockInterviewScoreDimension(
                dimension="reasoning", score=3, feedback="A design choice was unjustified."
            ),
        ),
        next_action=next_action,
        next_action_reason="One clarification is sufficient.",
        follow_up_question=follow_up_question,
    )


def started(store, session):
    store.save_plan(session=session, plan=plan_for(session))
    return store.start(session=session)


def test_create_rejects_a_second_unfinished_session_per_user(tmp_path) -> None:
    store = build_store(tmp_path)
    session = create_session(store)

    with pytest.raises(ValueError, match="unfinished mock interview"):
        create_session(store, application_id="app-2")

    other_user = create_session(store, user_id="u2")
    assert other_user.user_id == "u2"
    assert store.get_session(user_id="u2", session_id=session.id) is None


def test_start_requires_a_plan_and_survives_reload(tmp_path) -> None:
    store = build_store(tmp_path)
    session = create_session(store)

    with pytest.raises(ValueError, match="requires a plan"):
        store.start(session=session)

    plan = store.save_plan(session=session, plan=plan_for(session))
    active = store.start(session=session)

    reloaded = SQLiteMockInterviewStore(store.path)
    assert reloaded.get_session(user_id="u1", session_id=session.id) == active
    assert reloaded.get_plan(user_id="u1", session_id=session.id) == plan
    assert reloaded.find_resumable(user_id="u1").id == session.id


def test_only_one_turn_awaits_an_answer_at_a_time(tmp_path) -> None:
    store = build_store(tmp_path)
    active = started(store, create_session(store))

    active, turn = store.ask(
        session=active,
        plan_item_number=1,
        question_type="project_deep_dive",
        question="Walk me through one retrieval design decision.",
        asked_at=NOW,
    )
    assert active.current_turn_id == turn.id
    assert active.current_plan_item == 1

    with pytest.raises(ValueError, match="still in progress"):
        store.ask(
            session=active,
            plan_item_number=2,
            question_type="system_design",
            question="How would you evaluate it?",
        )
    with pytest.raises(ValueError, match="in progress"):
        store.pause(session=active)

    active, answered = store.record_answer(
        session=active,
        turn=turn,
        answer="I started from an offline labelled set.",
        answered_at=NOW + timedelta(minutes=2),
    )
    assert active.current_turn_id == turn.id
    assert answered.status == "answered"

    active, evaluated = store.record_evaluation(
        session=active,
        turn=answered,
        evaluation=evaluation(),
        evaluated_at=NOW + timedelta(minutes=2, seconds=5),
    )
    assert active.current_turn_id is None
    assert store.get_turn(user_id="u1", turn_id=turn.id) == evaluated
    assert evaluated.evaluation.rating == "adequate"


def test_answer_and_evaluation_writes_are_idempotent_and_survive_reload(
    tmp_path,
) -> None:
    store = build_store(tmp_path)
    active = started(store, create_session(store))
    active_before_answer, turn = store.ask(
        session=active,
        plan_item_number=1,
        question_type="project_deep_dive",
        question="What did you personally own?",
        asked_at=NOW,
    )

    active, answered = store.record_answer(
        session=active_before_answer,
        turn=turn,
        answer="  I owned retrieval evaluation.  ",
        answered_at=NOW + timedelta(minutes=1),
    )
    duplicate_session, duplicate_answer = store.record_answer(
        session=active_before_answer,
        turn=turn,
        answer="I owned retrieval evaluation.",
        answered_at=NOW + timedelta(minutes=2),
    )
    assert duplicate_session == active
    assert duplicate_answer == answered
    with pytest.raises(ValueError, match="different answer"):
        store.record_answer(
            session=active_before_answer,
            turn=turn,
            answer="A conflicting retry.",
        )

    reloaded = SQLiteMockInterviewStore(store.path)
    restored_session = reloaded.get_session(user_id="u1", session_id=active.id)
    restored_turn = reloaded.get_turn(user_id="u1", turn_id=turn.id)
    assert restored_session.current_turn_id == turn.id
    assert restored_turn.status == "answered"

    evaluated_session, evaluated_turn = reloaded.record_evaluation(
        session=restored_session,
        turn=restored_turn,
        evaluation=evaluation(),
        evaluated_at=NOW + timedelta(minutes=3),
    )
    duplicate_session, duplicate_turn = reloaded.record_evaluation(
        session=restored_session,
        turn=restored_turn,
        evaluation=evaluation(next_action="finish"),
        evaluated_at=NOW + timedelta(minutes=4),
    )
    assert duplicate_session == evaluated_session
    assert duplicate_turn == evaluated_turn
    assert duplicate_turn.evaluation.next_action == "next_question"


def test_follow_ups_are_bounded_and_primary_questions_advance_the_plan(tmp_path) -> None:
    store = build_store(tmp_path)
    active = started(store, create_session(store))
    active, primary = store.ask(
        session=active,
        plan_item_number=1,
        question_type="project_deep_dive",
        question="Describe the retrieval pipeline you built.",
        asked_at=NOW,
    )
    active, primary = store.record_answer(
        session=active,
        turn=primary,
        answer="It used hybrid search.",
        answered_at=NOW + timedelta(minutes=1),
    )
    active, _ = store.record_evaluation(
        session=active,
        turn=primary,
        evaluation=evaluation(
            next_action="follow_up", follow_up_question="Which failure mode did you test?"
        ),
    )
    active, follow_up = store.ask(
        session=active,
        plan_item_number=1,
        question_type="project_deep_dive",
        question="Which failure mode did you test?",
        turn_type="follow_up",
        parent_turn_id=primary.id,
        asked_at=NOW + timedelta(minutes=2),
    )
    active, follow_up = store.record_answer(
        session=active,
        turn=follow_up,
        answer="Recall drift on long queries.",
        answered_at=NOW + timedelta(minutes=3),
    )
    active, _ = store.record_evaluation(
        session=active,
        turn=follow_up,
        evaluation=evaluation(),
    )

    assert active.current_plan_item == 1
    with pytest.raises(ValueError, match="follow-up limit reached"):
        store.ask(
            session=active,
            plan_item_number=1,
            question_type="project_deep_dive",
            question="And how did you fix it?",
            turn_type="follow_up",
            parent_turn_id=primary.id,
        )
    with pytest.raises(ValueError, match="primary turn"):
        store.ask(
            session=active,
            plan_item_number=1,
            question_type="project_deep_dive",
            question="Do not create an unbounded follow-up chain.",
            turn_type="follow_up",
            parent_turn_id=follow_up.id,
        )
    with pytest.raises(ValueError, match="must advance the plan item"):
        store.ask(
            session=active,
            plan_item_number=1,
            question_type="project_deep_dive",
            question="Repeat the same plan item.",
        )
    with pytest.raises(ValueError, match="outside the saved plan"):
        store.ask(
            session=active,
            plan_item_number=3,
            question_type="system_design",
            question="Beyond the saved plan.",
        )
    with pytest.raises(ValueError, match="must match the saved plan item"):
        store.ask(
            session=active,
            plan_item_number=2,
            question_type="system_design",
            question="Wrong type for the saved plan item.",
        )

    turns = store.list_turns(user_id="u1", session_id=active.id)
    assert [turn.turn_type for turn in turns] == ["primary", "follow_up"]


def test_pause_and_resume_keep_one_active_session_per_user(tmp_path) -> None:
    store = build_store(tmp_path)
    active = started(store, create_session(store))
    paused = store.pause(session=active)
    assert paused.status == "paused" and paused.paused_at is not None

    resumed = store.resume(session=paused)
    assert resumed.status == "active" and resumed.paused_at is None

    with pytest.raises(RuntimeError, match="changed concurrently"):
        store.pause(session=active)


def test_report_must_be_grounded_in_evaluated_answers(tmp_path) -> None:
    store = build_store(tmp_path)
    active = started(store, create_session(store))
    active, turn = store.ask(
        session=active,
        plan_item_number=1,
        question_type="project_deep_dive",
        question="Explain one design trade-off.",
        asked_at=NOW,
    )
    active, turn = store.record_answer(
        session=active,
        turn=turn,
        answer="I chose latency over recall.",
        answered_at=NOW + timedelta(minutes=1),
    )
    active, _ = store.record_evaluation(
        session=active,
        turn=turn,
        evaluation=evaluation(next_action="finish"),
    )

    def report(plan_item_number: int) -> MockInterviewReport:
        return MockInterviewReport(
            id=f"mock_interview_report_{plan_item_number}",
            session_id=active.id,
            completion_reason="user_ended",
            summary="Practice ended after one evaluated question.",
            question_results=(
                MockInterviewQuestionResult(
                    plan_item_number=plan_item_number,
                    question="Explain one design trade-off.",
                    final_rating="adequate",
                    summary="Relevant but incomplete reasoning.",
                    follow_up_count=0,
                ),
            ),
            limitations=("This practice report does not predict an employer decision.",),
            created_at=NOW + timedelta(minutes=2),
        )

    with pytest.raises(ValueError, match="must cover exactly"):
        store.complete(session=active, report=report(2))

    completed, stored = store.complete(session=active, report=report(1))
    assert completed.status == "completed" and completed.completed_at is not None
    assert store.get_report(user_id="u1", session_id=active.id) == stored
    assert store.list_reports(
        user_id="u1",
        session_ids=("missing", active.id, active.id),
    ) == (stored,)
    assert store.list_reports(
        user_id="u2",
        session_ids=(active.id,),
    ) == ()
    assert store.find_resumable(user_id="u1") is None

    fresh = create_session(store)
    assert fresh.status == "created"
