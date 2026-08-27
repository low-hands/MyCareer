from pathlib import Path

import pytest

from career_agent.agent.mock_interview_contracts import (
    MockInterviewGraphResult,
    MockInterviewPlanDraft,
    MockInterviewQuestionDraft,
    MockInterviewReportDraft,
    MockInterviewStartRequest,
)
from career_agent.agent.mock_interview_graph import (
    MockInterviewGraph,
    MockInterviewSources,
)
from career_agent.domain.mock_interviews import (
    MockInterviewAnswerEvaluation,
    MockInterviewPlanItem,
    MockInterviewQuestionResult,
    MockInterviewScoreDimension,
)
from career_agent.storage.mock_interviews import SQLiteMockInterviewStore
from career_agent.storage.resumes import StoredResumeDocument


class Sources:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def load(self, *, session):
        self.calls.append(session.id)
        return MockInterviewSources(
            document=StoredResumeDocument(
                resume_version_id=session.resume_version_id,
                document_format="markdown",
                raw_bytes=b"Built retrieval systems.",
            ),
            jd_text="Design reliable retrieval and explain technical trade-offs.",
        )


def _evaluation(*, next_action, follow_up_question=None):
    return MockInterviewAnswerEvaluation(
        rating="adequate",
        summary="The answer is relevant but could be more specific.",
        dimensions=(
            MockInterviewScoreDimension(
                dimension="reasoning",
                score=3,
                feedback="Explain one more trade-off.",
            ),
        ),
        next_action=next_action,
        next_action_reason="Follow the bounded interview plan.",
        follow_up_question=follow_up_question,
    )


class Worker:
    def __init__(self) -> None:
        self.ask_calls = 0
        self.evaluate_calls = 0
        self.report_calls = 0

    def plan(self, **kwargs):
        return MockInterviewPlanDraft(
            summary="Test project depth then system reasoning.",
            items=(
                MockInterviewPlanItem(
                    sequence_number=1,
                    question_type="project_deep_dive",
                    difficulty="intermediate",
                    focus="Candidate ownership",
                    rationale="The resume mentions retrieval systems.",
                    resume_locators=("experience.1",),
                    resume_quotes=("Built retrieval systems",),
                ),
                MockInterviewPlanItem(
                    sequence_number=2,
                    question_type="system_design",
                    difficulty="advanced",
                    focus="Reliability trade-offs",
                    rationale="The JD requires reliable retrieval.",
                    jd_quotes=("Design reliable retrieval",),
                ),
            ),
        )

    def ask(self, *, plan_item, **kwargs):
        self.ask_calls += 1
        return MockInterviewQuestionDraft(
            question=(
                "What did you personally own in the retrieval system?"
                if plan_item.sequence_number == 1
                else "How would you design retrieval failure recovery?"
            )
        )

    def evaluate(self, *, turn, **kwargs):
        self.evaluate_calls += 1
        assert turn.status == "answered"
        if turn.turn_type == "primary" and turn.plan_item_number == 1:
            return _evaluation(
                next_action="follow_up",
                follow_up_question="How did you validate that decision?",
            )
        if turn.turn_type == "follow_up":
            return _evaluation(next_action="next_question")
        return _evaluation(next_action="finish")

    def report(self, *, turns, **kwargs):
        self.report_calls += 1
        primary = tuple(
            turn for turn in turns if turn.turn_type == "primary" and turn.status == "evaluated"
        )
        return MockInterviewReportDraft(
            summary="The candidate completed the bounded practice plan.",
            question_results=tuple(
                MockInterviewQuestionResult(
                    plan_item_number=turn.plan_item_number,
                    question=turn.question,
                    final_rating=turn.evaluation.rating,
                    summary=turn.evaluation.summary,
                    follow_up_count=sum(
                        item.parent_turn_id == turn.id for item in turns
                    ),
                )
                for turn in primary
            ),
            strengths=("Relevant reasoning",),
            development_areas=("More concrete validation",),
            practice_actions=("State assumptions before trade-offs",),
        )


class FlakyEvaluationWorker(Worker):
    def __init__(self) -> None:
        super().__init__()
        self.failed_once = False

    def evaluate(self, **kwargs):
        if not self.failed_once:
            self.failed_once = True
            raise RuntimeError("temporary evaluation failure")
        return super().evaluate(**kwargs)


def _request(*, max_follow_ups_per_question=1):
    return MockInterviewStartRequest(
        user_id="u1",
        application_id="app-1",
        job_posting_id="job-1",
        jd_snapshot_id="jd-1",
        resume_version_id="resume-v1",
        interview_type="mixed",
        max_primary_questions=2,
        max_follow_ups_per_question=max_follow_ups_per_question,
    )


def _graph(tmp_path: Path, *, max_follow_ups=1):
    store = SQLiteMockInterviewStore(tmp_path / "mock.sqlite3")
    worker = Worker()
    sources = Sources()
    graph = MockInterviewGraph(store=store, worker=worker, sources=sources)
    return graph, store, worker, sources


def test_graph_runs_primary_follow_up_next_question_and_report(tmp_path: Path) -> None:
    graph, store, worker, sources = _graph(tmp_path)

    first = graph.start(_request())
    assert first.state == "awaiting_answer"
    assert first.question == "What did you personally own in the retrieval system?"

    follow_up = graph.resume(
        user_id="u1",
        session_id=first.session_id,
        answer="I owned the offline evaluation design.",
    )
    assert follow_up.state == "awaiting_answer"
    assert follow_up.question == "How did you validate that decision?"
    assert follow_up.evaluation.next_action == "follow_up"

    second = graph.resume(
        user_id="u1",
        session_id=first.session_id,
        answer="I compared recall and latency on a labelled set.",
    )
    assert second.state == "awaiting_answer"
    assert second.question == "How would you design retrieval failure recovery?"

    completed = graph.resume(
        user_id="u1",
        session_id=first.session_id,
        answer="I would isolate failures and degrade to lexical retrieval.",
    )
    assert completed.state == "completed"
    assert completed.report is not None
    assert len(completed.report.question_results) == 2
    assert worker.ask_calls == 2
    assert worker.evaluate_calls == 3
    assert worker.report_calls == 1
    assert len(sources.calls) >= 4

    session = store.get_session(user_id="u1", session_id=first.session_id)
    turns = store.list_turns(user_id="u1", session_id=first.session_id)
    assert session.status == "completed"
    assert [turn.turn_type for turn in turns] == ["primary", "follow_up", "primary"]
    assert all(turn.status == "evaluated" for turn in turns)


def test_graph_overrides_a_follow_up_when_the_session_limit_is_zero(
    tmp_path: Path,
) -> None:
    graph, store, worker, _ = _graph(tmp_path)
    first = graph.start(_request(max_follow_ups_per_question=0))

    next_question = graph.resume(
        user_id="u1",
        session_id=first.session_id,
        answer="I owned evaluation.",
    )

    assert next_question.state == "awaiting_answer"
    assert next_question.question == "How would you design retrieval failure recovery?"
    turns = store.list_turns(user_id="u1", session_id=first.session_id)
    assert [turn.turn_type for turn in turns] == ["primary", "primary"]
    assert worker.ask_calls == 2


def test_graph_state_keeps_large_sources_out_of_the_checkpoint(tmp_path: Path) -> None:
    graph, _, _, _ = _graph(tmp_path)
    started = graph.start(_request())

    snapshot = graph._graph.get_state(
        {"configurable": {"thread_id": started.session_id}}
    ).values
    serialized = repr(snapshot)
    assert "Design reliable retrieval" not in serialized
    assert "Built retrieval systems" not in serialized
    assert set(snapshot) <= {
        "user_id",
        "session_id",
        "route",
        "current_turn_id",
        "evaluated_turn_id",
        "follow_up_question",
        "follow_up_parent_turn_id",
        "answer",
        "evaluation",
        "completion_reason",
        "report_id",
        "__interrupt__",
    }


def test_status_projects_the_current_persisted_question(tmp_path: Path) -> None:
    graph, _, _, _ = _graph(tmp_path)
    started = graph.start(_request())

    status = graph.status(user_id="u1", session_id=started.session_id)

    assert isinstance(status, MockInterviewGraphResult)
    assert status.state == "awaiting_answer"
    assert status.turn_id == started.turn_id
    assert status.question == started.question


def test_evaluation_failure_retries_from_the_persisted_answer(tmp_path: Path) -> None:
    store = SQLiteMockInterviewStore(tmp_path / "mock.sqlite3")
    worker = FlakyEvaluationWorker()
    graph = MockInterviewGraph(store=store, worker=worker, sources=Sources())
    started = graph.start(_request())

    with pytest.raises(RuntimeError, match="temporary evaluation failure"):
        graph.resume(
            user_id="u1",
            session_id=started.session_id,
            answer="I owned the evaluation design.",
        )

    session = store.get_session(user_id="u1", session_id=started.session_id)
    turn = store.get_turn(user_id="u1", turn_id=started.turn_id)
    assert session.current_turn_id == turn.id
    assert turn.status == "answered"
    assert turn.answer == "I owned the evaluation design."

    resumed = graph.resume(
        user_id="u1",
        session_id=started.session_id,
        answer="I owned the evaluation design.",
    )
    assert resumed.state == "awaiting_answer"
    assert resumed.question == "How did you validate that decision?"
    assert worker.evaluate_calls == 1
