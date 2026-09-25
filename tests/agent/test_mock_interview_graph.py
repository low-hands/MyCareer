import sqlite3
from pathlib import Path

import pytest

from career_agent.agent.mock_interview_contracts import (
    MockInterviewFollowUpDecision,
    MockInterviewGraphResult,
    MockInterviewInputDecision,
    MockInterviewPlanDraft,
    MockInterviewQuestionDraft,
    MockInterviewReportDraft,
    MockInterviewStartRequest,
)
from career_agent.agent.interview_preparation_contracts import InterviewPreparationContext
from career_agent.agent.mock_interview_graph import (
    MockInterviewCheckpointMissingError,
    MockInterviewGraph,
    MockInterviewGraphVersionError,
    MockInterviewInputRoutingError,
    MockInterviewSources,
)
from career_agent.agent.openai_compatible_client import AgentWorkerError
from career_agent.domain.mock_interviews import (
    MockInterviewAnswerEvaluation,
    MockInterviewPlanItem,
    MockInterviewQuestionResult,
    MockInterviewScoreDimension,
)
from career_agent.storage.mock_interviews import SQLiteMockInterviewStore
from career_agent.storage.checkpoints import SQLiteCheckpointOwner
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
            context=InterviewPreparationContext(
                jd_text="Design reliable retrieval and explain technical trade-offs.",
                company_name="Example Corp",
                role_title="Retrieval Engineer",
            ),
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
        self.decide_calls = 0
        self.evaluate_calls = 0
        self.evaluated_chains: list[tuple[int, ...]] = []
        self.report_calls = 0
        self.input_action = "answer"
        self.plan_company: tuple[str, str] | None = None
        self.write_questions = True

    def route_input(self, **kwargs):
        self.route_calls = getattr(self, "route_calls", 0) + 1
        return MockInterviewInputDecision(action=self.input_action)

    def plan(self, **kwargs):
        context = kwargs["context"]
        self.plan_company = (context.company_name, context.role_title)
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
                    question=(
                        "What did you personally own in the retrieval system?"
                        if self.write_questions else None
                    ),
                ),
                MockInterviewPlanItem(
                    sequence_number=2,
                    question_type="system_design",
                    difficulty="advanced",
                    focus="Reliability trade-offs",
                    rationale="The JD requires reliable retrieval.",
                    jd_quotes=("Design reliable retrieval",),
                    question=(
                        "How would you design retrieval failure recovery?"
                        if self.write_questions else None
                    ),
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

    def decide_follow_up(self, *, plan_item, turns, **kwargs):
        self.decide_calls += 1
        assert {turn.plan_item_number for turn in turns} == {plan_item.sequence_number}
        if turns[-1].turn_type == "primary" and plan_item.sequence_number == 1:
            return MockInterviewFollowUpDecision(
                next_action="follow_up",
                follow_up_question="How did you validate that decision?",
            )
        return MockInterviewFollowUpDecision(next_action="next_question")

    def evaluate(self, *, turns, **kwargs):
        self.evaluate_calls += 1
        self.evaluated_chains.append(tuple(turn.plan_item_number for turn in turns))
        assert all(turn.status == "answered" for turn in turns)
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


class FlakyDecisionWorker(Worker):
    def __init__(self) -> None:
        super().__init__()
        self.failed_once = False

    def decide_follow_up(self, **kwargs):
        if not self.failed_once:
            self.failed_once = True
            raise RuntimeError("temporary decision failure")
        return super().decide_follow_up(**kwargs)


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
    # Nothing is scored mid-interview.
    assert follow_up.evaluation is None
    assert worker.evaluate_calls == 0

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
    # Questions come from the plan; a follow-up decision runs only while one is
    # still allowed; each question is scored once, on its own chain.
    assert worker.ask_calls == 0
    assert worker.decide_calls == 2
    assert worker.evaluate_calls == 2
    assert sorted(worker.evaluated_chains) == [(1, 1), (2,)]
    assert worker.report_calls == 1
    assert worker.plan_company == ("Example Corp", "Retrieval Engineer")

    session = store.get_session(user_id="u1", session_id=first.session_id)
    turns = store.list_turns(user_id="u1", session_id=first.session_id)
    assert session.status == "completed"
    assert [turn.turn_type for turn in turns] == ["primary", "follow_up", "primary"]
    assert [turn.status for turn in turns] == ["evaluated", "answered", "evaluated"]


def test_a_plan_saved_without_questions_falls_back_to_asking(tmp_path: Path) -> None:
    graph, _, worker, _ = _graph(tmp_path)
    worker.write_questions = False

    first = graph.start(_request())

    assert first.question == "What did you personally own in the retrieval system?"
    assert worker.ask_calls == 1


def test_one_failed_question_score_is_the_only_one_redone(tmp_path: Path) -> None:
    class SecondScoreFailsOnce(Worker):
        failed = False

        def evaluate(self, *, turns, **kwargs):
            if turns[0].plan_item_number == 2 and not self.failed:
                self.failed = True
                raise AgentWorkerError("MOCK_INTERVIEW_TRANSPORT_ERROR", "dropped", retryable=True)
            return super().evaluate(turns=turns, **kwargs)

    store = SQLiteMockInterviewStore(tmp_path / "mock.sqlite3")
    worker = SecondScoreFailsOnce()
    graph = MockInterviewGraph(store=store, worker=worker, sources=Sources())
    started = graph.start(_request(max_follow_ups_per_question=0))
    graph.resume(user_id="u1", session_id=started.session_id, answer="I owned evaluation.")

    with pytest.raises(AgentWorkerError):
        graph.resume(user_id="u1", session_id=started.session_id, answer="Degrade gracefully.")

    statuses = [turn.status for turn in store.list_turns(user_id="u1", session_id=started.session_id)]
    assert statuses == ["evaluated", "answered"]
    completed = graph.retry(user_id="u1", session_id=started.session_id)

    assert completed.state == "completed"
    assert sorted(worker.evaluated_chains) == [(1,), (2,)]


def test_handle_input_cancels_without_persisting_the_message_as_an_answer(
    tmp_path: Path,
) -> None:
    graph, store, worker, _ = _graph(tmp_path)
    started = graph.start(_request())
    worker.input_action = "cancel"

    result = graph.handle_input(
        user_id="u1",
        session_id=started.session_id,
        message="不想练了，结束面试",
    )

    turn = store.get_turn(user_id="u1", turn_id=started.turn_id)
    assert result.state == "cancelled"
    assert turn.status == "awaiting_answer"
    assert turn.answer is None
    assert graph.checkpointer.get_tuple(graph._config(started.session_id)) is None


def test_handle_input_persists_a_substantive_answer_even_when_it_says_end(
    tmp_path: Path,
) -> None:
    graph, store, _, _ = _graph(tmp_path)
    started = graph.start(_request())
    answer = "项目最后结束于灰度上线，我负责离线评测和回滚指标。"

    graph.handle_input(
        user_id="u1",
        session_id=started.session_id,
        message=answer,
    )

    turn = store.get_turn(user_id="u1", turn_id=started.turn_id)
    assert turn.answer == answer
    assert turn.status == "answered"


def test_an_answer_with_nothing_like_stopping_skips_the_router(tmp_path: Path) -> None:
    graph, store, worker, _ = _graph(tmp_path)
    started = graph.start(_request())

    graph.handle_input(
        user_id="u1", session_id=started.session_id, message="我负责离线评测和回滚指标。"
    )

    assert getattr(worker, "route_calls", 0) == 0
    assert store.get_turn(user_id="u1", turn_id=started.turn_id).status == "answered"


def test_input_routing_failure_does_not_consume_or_persist_the_message(
    tmp_path: Path,
) -> None:
    class RoutingFailure(Worker):
        def route_input(self, **kwargs):
            raise AgentWorkerError(
                "INPUT_ROUTE_UNAVAILABLE",
                "router unavailable",
                retryable=True,
            )

    store = SQLiteMockInterviewStore(tmp_path / "mock.sqlite3")
    graph = MockInterviewGraph(
        store=store,
        worker=RoutingFailure(),
        sources=Sources(),
    )
    started = graph.start(_request())

    with pytest.raises(MockInterviewInputRoutingError) as error:
        graph.handle_input(
            user_id="u1",
            session_id=started.session_id,
            # Contains a stop word, so it reaches the router that fails.
            message="这是不能被保存的回答，结束",
        )

    turn = store.get_turn(user_id="u1", turn_id=started.turn_id)
    assert error.value.code == "INPUT_ROUTE_UNAVAILABLE"
    assert error.value.retryable is True
    assert turn.status == "awaiting_answer"
    assert turn.answer is None


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
    # No follow-up allowed, so no decision is worth a model call.
    assert worker.decide_calls == 0


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
        "graph_version",
        "user_id",
        "session_id",
        "route",
        "current_turn_id",
        "decided_turn_id",
        "decision",
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


def test_a_failed_decision_retries_from_the_persisted_answer(tmp_path: Path) -> None:
    store = SQLiteMockInterviewStore(tmp_path / "mock.sqlite3")
    worker = FlakyDecisionWorker()
    graph = MockInterviewGraph(store=store, worker=worker, sources=Sources())
    started = graph.start(_request())

    with pytest.raises(RuntimeError, match="temporary decision failure"):
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
    assert worker.decide_calls == 1


def test_sqlite_checkpoint_resumes_after_connection_and_graph_reopen(
    tmp_path: Path,
) -> None:
    store = SQLiteMockInterviewStore(tmp_path / "mock.sqlite3")
    checkpoint_path = tmp_path / "checkpoints.sqlite3"
    first_owner = SQLiteCheckpointOwner(checkpoint_path)
    first_graph = MockInterviewGraph(
        store=store,
        worker=Worker(),
        sources=Sources(),
        checkpointer=first_owner.saver,
    )
    started = first_graph.start(_request(max_follow_ups_per_question=0))
    assert first_owner.saver.get_tuple(first_graph._config(started.session_id)) is not None
    first_owner.close()

    second_owner = SQLiteCheckpointOwner(checkpoint_path)
    second_graph = MockInterviewGraph(
        store=SQLiteMockInterviewStore(store.path),
        worker=Worker(),
        sources=Sources(),
        checkpointer=second_owner.saver,
    )
    next_question = second_graph.resume(
        user_id="u1",
        session_id=started.session_id,
        answer="I owned the offline evaluation design.",
    )

    assert next_question.state == "awaiting_answer"
    assert next_question.question == "How would you design retrieval failure recovery?"
    completed = second_graph.resume(
        user_id="u1",
        session_id=started.session_id,
        answer="I would degrade to lexical retrieval and monitor recovery.",
    )
    assert completed.state == "completed"
    assert second_owner.saver.get_tuple(
        second_graph._config(started.session_id)
    ) is None
    assert store.get_report(user_id="u1", session_id=started.session_id) is not None
    second_owner.close()


def test_resume_distinguishes_missing_checkpoint_from_business_state(
    tmp_path: Path,
) -> None:
    first_graph, store, _, _ = _graph(tmp_path)
    started = first_graph.start(_request())
    reopened_without_checkpoint = MockInterviewGraph(
        store=SQLiteMockInterviewStore(store.path),
        worker=Worker(),
        sources=Sources(),
    )

    with pytest.raises(MockInterviewCheckpointMissingError):
        reopened_without_checkpoint.resume(
            user_id="u1",
            session_id=started.session_id,
            answer="An answer that must not be written.",
        )

    turn = store.get_turn(user_id="u1", turn_id=started.turn_id)
    assert turn.status == "awaiting_answer"
    assert turn.answer is None


@pytest.mark.parametrize("finish", ["complete", "cancel"])
def test_resume_reports_a_finished_run_rather_than_a_lost_checkpoint(
    tmp_path: Path, finish
) -> None:
    """A finished run deletes its checkpoint, so its absence is not a fault.

    The conversation releases the workflow slot in a separate write from the
    one that stores the report. A crash in between leaves an answer arriving
    for a run that already finished, and calling that a lost checkpoint would
    bury a completed interview behind a failure message.
    """
    graph, store, _, _ = _graph(tmp_path)
    started = graph.start(_request(max_follow_ups_per_question=0))
    if finish == "complete":
        while graph.resume(
            user_id="u1", session_id=started.session_id, answer="An answer."
        ).state == "awaiting_answer":
            pass
    else:
        graph.cancel(user_id="u1", session_id=started.session_id)
    assert store.get_session(user_id="u1", session_id=started.session_id).status == (
        "completed" if finish == "complete" else "cancelled"
    )

    # The same answer arriving again, as it would after that crash.
    replayed = graph.resume(
        user_id="u1", session_id=started.session_id, answer="An answer."
    )

    assert replayed.state == ("completed" if finish == "complete" else "cancelled")
    if finish == "complete":
        assert replayed.report_id is not None


def test_a_completed_run_without_its_report_fails_instead_of_projecting(
    tmp_path: Path,
) -> None:
    """The same invariant the report node enforces, enforced where it is read.

    ``_report`` raises when a completed session has no report. ``_project`` used
    to answer the identical impossible state with ``report.id if report else
    None`` — a completed result carrying no reference. That shape is worse than
    the failure it avoids: the delivery layer treats ``mock_interview_completed``
    as card-backed, compresses the body, and fails open on the missing
    reference, so a finished interview arrives as a report nobody can name.

    Unreachable through the graph's own writes, which is why the row is deleted
    behind its back here. It is pinned because it was not a raise until now, and
    putting the conditional back breaks nothing else.
    """
    graph, store, _, _ = _graph(tmp_path)
    started = graph.start(_request(max_follow_ups_per_question=0))
    while graph.resume(
        user_id="u1", session_id=started.session_id, answer="An answer."
    ).state == "awaiting_answer":
        pass
    assert store.get_report(user_id="u1", session_id=started.session_id) is not None

    with sqlite3.connect(tmp_path / "mock.sqlite3") as connection:
        connection.execute(
            "DELETE FROM mock_interview_reports WHERE session_id = ?",
            (started.session_id,),
        )

    with pytest.raises(ValueError, match="Completed mock interview has no report"):
        graph.resume(user_id="u1", session_id=started.session_id, answer="An answer.")


def test_resume_rejects_an_incompatible_graph_version(tmp_path: Path) -> None:
    graph, store, _, _ = _graph(tmp_path)
    started = graph.start(_request())
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            "UPDATE mock_interview_sessions SET graph_version = 999 WHERE id = ?",
            (started.session_id,),
        )
    with pytest.raises(MockInterviewGraphVersionError, match="session=999"):
        graph.resume(
            user_id="u1", session_id=started.session_id,
            answer="Do not process this answer.",
        )


def test_free_practice_runs_on_exactly_the_chosen_resume_or_none(tmp_path: Path) -> None:
    class PinnedSources(Sources):
        def __init__(self):
            super().__init__()
            self.loaded_versions = []

        def load(self, *, session):
            self.loaded_versions.append(session.resume_version_id)
            if session.resume_version_id is None:
                return MockInterviewSources(
                    document=None,
                    context=InterviewPreparationContext(
                        jd_text="", company_name="", role_title="PM"
                    ),
                )
            return super().load(session=session)

    sources = PinnedSources()
    store = SQLiteMockInterviewStore(tmp_path / "mock-interviews.sqlite3")
    graph = MockInterviewGraph(store=store, worker=Worker(), sources=sources)
    chosen = graph.start(MockInterviewStartRequest(
        user_id="u1", interview_type="technical", max_primary_questions=2,
        max_follow_ups_per_question=0, resume_version_id="resume-v1",
    ))
    assert chosen.state == "awaiting_answer"
    assert set(sources.loaded_versions) == {"resume-v1"}

    graph.cancel(user_id="u1", session_id=chosen.session_id)
    sources.loaded_versions.clear()
    # No resume chosen means none: there is no "latest resume" fallback.
    without = graph.start(MockInterviewStartRequest(
        user_id="u1", interview_type="technical", max_primary_questions=2,
        max_follow_ups_per_question=0,
    ))
    assert without.state == "awaiting_answer"
    assert sources.loaded_versions and set(sources.loaded_versions) == {None}


def test_cancel_deletes_checkpoint_threads_without_accumulating_orphans(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "mock-interview-checkpoints.sqlite3"
    owner = SQLiteCheckpointOwner(checkpoint_path)
    store = SQLiteMockInterviewStore(tmp_path / "mock-interviews.sqlite3")
    graph = MockInterviewGraph(
        store=store,
        worker=Worker(),
        sources=Sources(),
        checkpointer=owner.saver,
    )

    session_ids = []
    for _ in range(4):
        started = graph.start(_request())
        session_ids.append(started.session_id)
        cancelled = graph.cancel(user_id="u1", session_id=started.session_id)

        assert cancelled.state == "cancelled"
        assert owner.saver.get_tuple(graph._config(started.session_id)) is None

    with sqlite3.connect(checkpoint_path) as connection:
        checkpoint_threads = connection.execute(
            "SELECT COUNT(DISTINCT thread_id) FROM checkpoints"
        ).fetchone()[0]
        pending_write_threads = connection.execute(
            "SELECT COUNT(DISTINCT thread_id) FROM writes"
        ).fetchone()[0]
    assert checkpoint_threads == 0
    assert pending_write_threads == 0

    # A repeated cancellation also acts as orphan-repair and remains stable.
    repeated = graph.cancel(user_id="u1", session_id=session_ids[-1])
    assert repeated.state == "cancelled"
    owner.close()


def test_retry_recovers_without_the_candidate_retyping_the_answer(
    tmp_path: Path,
) -> None:
    """A step failing after the answer is durable must not depend on recall.

    The turn rejects a different answer for the same question, so a candidate
    who rewords even slightly could otherwise never leave the failed phase.
    """

    class FailOnce(Worker):
        def __init__(self) -> None:
            super().__init__()
            self.fail = True

        def decide_follow_up(self, **kwargs):
            if self.fail:
                self.fail = False
                raise AgentWorkerError(
                    "worker_unavailable", "decision failed", retryable=True
                )
            return super().decide_follow_up(**kwargs)

    store = SQLiteMockInterviewStore(tmp_path / "mock-interviews.sqlite3")
    owner = SQLiteCheckpointOwner(tmp_path / "checkpoints.sqlite3")
    worker = FailOnce()
    graph = MockInterviewGraph(
        store=store, worker=worker, sources=Sources(), checkpointer=owner.saver
    )
    started = graph.start(_request(max_follow_ups_per_question=1))
    answer = "我主导了离线评测设计"
    with pytest.raises(AgentWorkerError):
        graph.resume(user_id="u1", session_id=started.session_id, answer=answer)

    session = store.get_session(user_id="u1", session_id=started.session_id)
    turn = store.get_turn(user_id="u1", turn_id=session.current_turn_id)
    assert turn.status == "answered"
    assert turn.answer == answer
    # The precondition that makes retry necessary rather than merely convenient.
    with pytest.raises(ValueError, match="already has a different answer"):
        graph.resume(
            user_id="u1", session_id=started.session_id, answer="换个说法的答案"
        )

    recovered = graph.retry(user_id="u1", session_id=started.session_id)
    assert recovered.state == "awaiting_answer"
    assert worker.decide_calls == 1
    owner.close()


@pytest.mark.parametrize(
    ("company", "inferred"), [("字节", False), ("北京字节跳动科技有限公司", True)]
)
def test_the_plan_records_whether_its_company_style_was_inferred(
    tmp_path: Path, company: str, inferred: bool
) -> None:
    class StyledWorker(Worker):
        def plan(self, **kwargs):
            return super().plan(**kwargs).model_copy(
                update={"company_style_profile": "ByteDance and related businesses"}
            )

    class CompanySources(Sources):
        def load(self, *, session):
            loaded = super().load(session=session)
            return MockInterviewSources(
                document=loaded.document,
                context=loaded.context.model_copy(update={"company_name": company}),
            )

    store = SQLiteMockInterviewStore(tmp_path / "mock-interviews.sqlite3")
    graph = MockInterviewGraph(store=store, worker=StyledWorker(), sources=CompanySources())
    started = graph.start(_request())

    plan = store.get_plan(user_id="u1", session_id=started.session_id)
    assert plan.company_style_profile == "ByteDance and related businesses"
    assert plan.company_style_inferred is inferred
