from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal, Protocol, TypedDict
from uuid import uuid4

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from career_agent.agent.mock_interview_contracts import (
    MockInterviewCompletionReason,
    MockInterviewGraphResult,
    MockInterviewStartRequest,
    MockInterviewWorker,
)
from career_agent.agent.resume_job_match_contracts import ConfirmedResumeFact
from career_agent.domain.mock_interviews import (
    MockInterviewAnswerEvaluation,
    MockInterviewPlan,
    MockInterviewReport,
    MockInterviewSession,
)
from career_agent.storage.jobs import JobPostingRepository
from career_agent.storage.career_history import CareerHistoryStore
from career_agent.storage.mock_interviews import SQLiteMockInterviewStore
from career_agent.storage.resumes import ResumeStore, StoredResumeDocument


@dataclass(frozen=True)
class MockInterviewSources:
    document: StoredResumeDocument
    jd_text: str
    confirmed_facts: tuple[ConfirmedResumeFact, ...] = ()


class MockInterviewCheckpointMissingError(RuntimeError):
    """Business state exists but LangGraph has no resumable checkpoint."""


class MockInterviewGraphVersionError(RuntimeError):
    """A persisted session belongs to an incompatible graph definition."""


class MockInterviewSourceProvider(Protocol):
    def load(self, *, session: MockInterviewSession) -> MockInterviewSources: ...


class StoredMockInterviewSourceProvider:
    """Loads the exact immutable JD and resume versions bound to a session."""

    def __init__(
        self,
        *,
        resumes: ResumeStore,
        jobs: JobPostingRepository,
        career_history: CareerHistoryStore | None = None,
    ) -> None:
        self._resumes = resumes
        self._jobs = jobs
        self._career_history = career_history

    def load(self, *, session: MockInterviewSession) -> MockInterviewSources:
        document = self._resumes.read_version_document(
            user_id=session.user_id,
            resume_version_id=session.resume_version_id,
        )
        if document is None:
            raise ValueError("Mock interview resume version no longer exists")
        snapshot = self._jobs.get_snapshot(
            user_id=session.user_id,
            jd_snapshot_id=session.jd_snapshot_id,
        )
        if snapshot is None or snapshot.job_posting_id != session.job_posting_id:
            raise ValueError("Mock interview JD snapshot no longer exists")
        confirmed_facts: tuple[ConfirmedResumeFact, ...] = ()
        if self._career_history is not None:
            confirmed_facts = tuple(
                ConfirmedResumeFact(
                    claim=item.claim,
                    source_locator=item.source_locator,
                    source_quote=item.source_quote,
                )
                for item in self._career_history.list_evidence(
                    user_id=session.user_id,
                    verification_status="confirmed",
                    source_resume_version_id=session.resume_version_id,
                )
                if item.source_locator is not None and item.source_quote is not None
            )
        return MockInterviewSources(
            document=document,
            jd_text=snapshot.content,
            confirmed_facts=confirmed_facts,
        )


class MockInterviewState(TypedDict, total=False):
    graph_version: int
    user_id: str
    session_id: str
    route: Literal["primary", "follow_up", "report"]
    current_turn_id: str
    evaluated_turn_id: str
    follow_up_question: str
    follow_up_parent_turn_id: str
    answer: str | None
    evaluation: dict[str, Any]
    completion_reason: MockInterviewCompletionReason
    report_id: str


class MockInterviewGraph:
    """Stateful mock interview loop with durable business writes per node.

    LangGraph owns execution and interrupt/resume. ``SQLiteMockInterviewStore``
    owns the queryable business record. Full JD and resume content are loaded by
    worker nodes and never copied into checkpoint state.
    """

    GRAPH_VERSION = 1

    def __init__(
        self,
        *,
        store: SQLiteMockInterviewStore,
        worker: MockInterviewWorker,
        sources: MockInterviewSourceProvider,
        checkpointer: Any | None = None,
    ) -> None:
        self._store = store
        self._worker = worker
        self._sources = sources
        self._checkpointer = checkpointer or InMemorySaver()

        graph = StateGraph(MockInterviewState)
        graph.add_node("initialize", self._initialize)
        graph.add_node("ask", self._ask)
        graph.add_node("await", self._await_answer)
        graph.add_node("evaluate", self._evaluate)
        graph.add_node("route", self._route)
        graph.add_node("report", self._report)
        graph.add_edge(START, "initialize")
        graph.add_edge("initialize", "ask")
        graph.add_edge("ask", "await")
        graph.add_edge("await", "evaluate")
        graph.add_edge("evaluate", "route")
        graph.add_conditional_edges(
            "route",
            lambda state: state["route"],
            {"primary": "ask", "follow_up": "ask", "report": "report"},
        )
        graph.add_edge("report", END)
        self._graph = graph.compile(checkpointer=self._checkpointer)

    @property
    def checkpointer(self) -> Any:
        return self._checkpointer

    def start(self, request: MockInterviewStartRequest) -> MockInterviewGraphResult:
        session = self._store.create_session(
            user_id=request.user_id,
            application_id=request.application_id,
            interview_round_id=request.interview_round_id,
            job_posting_id=request.job_posting_id,
            jd_snapshot_id=request.jd_snapshot_id,
            resume_version_id=request.resume_version_id,
            interview_type=request.interview_type,
            max_primary_questions=request.max_primary_questions,
            max_follow_ups_per_question=request.max_follow_ups_per_question,
            graph_version=self.GRAPH_VERSION,
        )
        try:
            state = self._graph.invoke(
                {
                    "user_id": session.user_id,
                    "session_id": session.id,
                    "route": "primary",
                    "graph_version": self.GRAPH_VERSION,
                },
                config=self._config(session.id),
            )
        except Exception:
            current = self._store.get_session(
                user_id=session.user_id,
                session_id=session.id,
            )
            if current is not None and current.status not in {"completed", "cancelled"}:
                self._store.cancel(session=current)
            self._delete_checkpoint_best_effort(session.id)
            raise
        result = self._project(session.user_id, session.id, state)
        self._cleanup_terminal_checkpoint(result)
        return result

    def resume(
        self,
        *,
        user_id: str,
        session_id: str,
        answer: str,
    ) -> MockInterviewGraphResult:
        normalized = answer.strip()
        if not normalized:
            raise ValueError("Mock interview answer must not be empty")
        session = self._require_session(user_id, session_id)
        self._require_resumable_checkpoint(session)
        if session.status != "active" or session.current_turn_id is None:
            raise ValueError("Mock interview is not awaiting an answer")
        turn = self._store.get_turn(user_id=user_id, turn_id=session.current_turn_id)
        if turn is None:
            raise ValueError("Mock interview current turn does not exist")
        if turn.status == "answered":
            if turn.answer != normalized:
                raise ValueError("Mock interview turn already has a different answer")
            state = self._graph.invoke(None, config=self._config(session_id))
        elif turn.status == "awaiting_answer":
            state = self._graph.invoke(
                Command(resume=normalized),
                config=self._config(session_id),
            )
        else:
            raise ValueError("Mock interview turn is no longer awaiting an answer")
        result = self._project(user_id, session_id, state)
        self._cleanup_terminal_checkpoint(result)
        return result

    def retry(self, *, user_id: str, session_id: str) -> MockInterviewGraphResult:
        session = self._require_session(user_id, session_id)
        if session.status != "active":
            return self._project(user_id, session_id, {})
        self._require_resumable_checkpoint(session)
        state = self._graph.invoke(None, config=self._config(session_id))
        result = self._project(user_id, session_id, state)
        self._cleanup_terminal_checkpoint(result)
        return result

    def cancel(
        self, *, user_id: str, session_id: str
    ) -> MockInterviewGraphResult:
        """Cancel one business session and remove its resumable graph thread.

        Cancellation is intentionally owned by the graph boundary: the store
        cannot clean execution state because it has no checkpointer. Deleting a
        terminal session's thread again makes this operation repair an orphan
        left by an interrupted earlier cleanup.
        """
        session = self._require_session(user_id, session_id)
        if session.status == "completed":
            raise ValueError("completed mock interviews cannot be cancelled")
        if session.status != "cancelled":
            self._store.cancel(session=session)
        # Do not hide cleanup failure here. The business transition is already
        # durable, and retrying cancel will enter the idempotent branch above
        # and attempt the deletion again.
        self._delete_checkpoint(session_id)
        return self._project(user_id, session_id, {})

    def status(self, *, user_id: str, session_id: str) -> MockInterviewGraphResult:
        self._require_session(user_id, session_id)
        values = self._graph.get_state(self._config(session_id)).values
        return self._project(user_id, session_id, values)

    def _initialize(self, state: MockInterviewState) -> dict[str, Any]:
        session = self._require_session(state["user_id"], state["session_id"])
        sources = self._sources.load(session=session)
        plan = self._store.get_plan(user_id=session.user_id, session_id=session.id)
        if plan is None:
            draft = self._worker.plan(
                session=session,
                document=sources.document,
                jd_text=sources.jd_text,
                confirmed_facts=sources.confirmed_facts,
            )
            plan = MockInterviewPlan(
                session_id=session.id,
                summary=draft.summary,
                items=draft.items,
                limitations=draft.limitations,
                created_at=datetime.now(timezone.utc),
            )
            self._store.save_plan(session=session, plan=plan)
        if session.status == "created":
            session = self._store.start(session=session)
        if session.status != "active":
            raise ValueError("Mock interview session cannot be initialized")
        return {"route": "primary"}

    def _ask(self, state: MockInterviewState) -> dict[str, Any]:
        session = self._require_session(state["user_id"], state["session_id"])
        if session.current_turn_id is not None:
            current = self._store.get_turn(
                user_id=session.user_id,
                turn_id=session.current_turn_id,
            )
            if current is None:
                raise ValueError("Mock interview current turn does not exist")
            return {"current_turn_id": current.id}
        plan = self._require_plan(session)
        turns = self._store.list_turns(user_id=session.user_id, session_id=session.id)
        route = state.get("route", "primary")
        if route == "follow_up":
            plan_item_number = session.current_plan_item
            question = state["follow_up_question"]
            parent_turn_id = state["follow_up_parent_turn_id"]
            turn_type = "follow_up"
        else:
            plan_item_number = session.current_plan_item + 1
            if plan_item_number > len(plan.items):
                raise ValueError("Mock interview plan has no remaining question")
            plan_item = plan.items[plan_item_number - 1]
            sources = self._sources.load(session=session)
            draft = self._worker.ask(
                session=session,
                plan=plan,
                plan_item=plan_item,
                prior_turns=turns,
                document=sources.document,
                jd_text=sources.jd_text,
                confirmed_facts=sources.confirmed_facts,
            )
            question = draft.question
            parent_turn_id = None
            turn_type = "primary"
        plan_item = plan.items[plan_item_number - 1]
        _, turn = self._store.ask(
            session=session,
            plan_item_number=plan_item_number,
            question_type=plan_item.question_type,
            question=question,
            turn_type=turn_type,
            parent_turn_id=parent_turn_id,
        )
        return {
            "current_turn_id": turn.id,
            "follow_up_question": None,
            "follow_up_parent_turn_id": None,
        }

    def _await_answer(self, state: MockInterviewState) -> dict[str, Any]:
        session = self._require_session(state["user_id"], state["session_id"])
        if session.current_turn_id is None:
            raise ValueError("Mock interview has no current question")
        turn = self._store.get_turn(
            user_id=session.user_id,
            turn_id=session.current_turn_id,
        )
        if turn is None or turn.status != "awaiting_answer":
            raise ValueError("Mock interview current question is not awaiting an answer")
        answer = interrupt(
            {
                "code": "MOCK_INTERVIEW_ANSWER_REQUIRED",
                "session_id": session.id,
                "turn_id": turn.id,
                "question": turn.question,
            }
        )
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("Mock interview answer must not be empty")
        return {"answer": answer.strip()}

    def _evaluate(self, state: MockInterviewState) -> dict[str, Any]:
        session = self._require_session(state["user_id"], state["session_id"])
        turn_id = state.get("current_turn_id") or session.current_turn_id
        if turn_id is None:
            raise ValueError("Mock interview has no turn to evaluate")
        turn = self._store.get_turn(user_id=session.user_id, turn_id=turn_id)
        if turn is None:
            raise ValueError("Mock interview turn does not exist")
        if turn.status == "evaluated":
            if turn.evaluation is None:
                raise ValueError("Evaluated mock interview turn has no evaluation")
            return {
                "answer": None,
                "evaluation": turn.evaluation.model_dump(mode="json"),
                "evaluated_turn_id": turn.id,
                "current_turn_id": None,
            }
        if turn.status == "awaiting_answer":
            session, turn = self._store.record_answer(
                session=session,
                turn=turn,
                answer=state.get("answer") or "",
            )
        plan = self._require_plan(session)
        plan_item = plan.items[turn.plan_item_number - 1]
        sources = self._sources.load(session=session)
        prior_turns = tuple(
            item
            for item in self._store.list_turns(
                user_id=session.user_id,
                session_id=session.id,
            )
            if item.id != turn.id and item.status == "evaluated"
        )
        evaluation = self._worker.evaluate(
            session=session,
            plan_item=plan_item,
            turn=turn,
            prior_turns=prior_turns,
            document=sources.document,
            jd_text=sources.jd_text,
            confirmed_facts=sources.confirmed_facts,
        )
        session, evaluated_turn = self._store.record_evaluation(
            session=session,
            turn=turn,
            evaluation=evaluation,
        )
        return {
            "answer": None,
            "evaluation": evaluated_turn.evaluation.model_dump(mode="json"),
            "evaluated_turn_id": evaluated_turn.id,
            "current_turn_id": None,
        }

    def _route(self, state: MockInterviewState) -> dict[str, Any]:
        session = self._require_session(state["user_id"], state["session_id"])
        plan = self._require_plan(session)
        turn = self._store.get_turn(
            user_id=session.user_id,
            turn_id=state["evaluated_turn_id"],
        )
        raw_evaluation = state.get("evaluation")
        if turn is None or raw_evaluation is None:
            raise ValueError("Mock interview evaluation is unavailable for routing")
        evaluation = MockInterviewAnswerEvaluation.model_validate(raw_evaluation)
        turns = self._store.list_turns(user_id=session.user_id, session_id=session.id)
        primary_id = turn.id if turn.turn_type == "primary" else turn.parent_turn_id
        follow_up_count = sum(
            item.turn_type == "follow_up" and item.parent_turn_id == primary_id
            for item in turns
        )
        if (
            evaluation.next_action == "follow_up"
            and evaluation.follow_up_question
            and follow_up_count < session.max_follow_ups_per_question
        ):
            return {
                "route": "follow_up",
                "follow_up_question": evaluation.follow_up_question,
                "follow_up_parent_turn_id": primary_id,
            }
        if evaluation.next_action == "finish":
            return {"route": "report", "completion_reason": "plan_completed"}
        if session.current_plan_item < len(plan.items):
            return {"route": "primary"}
        return {"route": "report", "completion_reason": "plan_completed"}

    def _report(self, state: MockInterviewState) -> dict[str, Any]:
        session = self._require_session(state["user_id"], state["session_id"])
        if session.status == "completed":
            existing = self._store.get_report(
                user_id=session.user_id,
                session_id=session.id,
            )
            if existing is None:
                raise ValueError("Completed mock interview has no report")
            return {"report_id": existing.id}
        plan = self._require_plan(session)
        turns = self._store.list_turns(user_id=session.user_id, session_id=session.id)
        sources = self._sources.load(session=session)
        completion_reason = state.get("completion_reason", "plan_completed")
        draft = self._worker.report(
            session=session,
            plan=plan,
            turns=turns,
            completion_reason=completion_reason,
            document=sources.document,
            jd_text=sources.jd_text,
            confirmed_facts=sources.confirmed_facts,
        )
        report = MockInterviewReport(
            id=f"mock_interview_report_{uuid4().hex}",
            session_id=session.id,
            completion_reason=completion_reason,
            summary=draft.summary,
            question_results=draft.question_results,
            strengths=draft.strengths,
            development_areas=draft.development_areas,
            practice_actions=draft.practice_actions,
            limitations=draft.limitations,
            created_at=datetime.now(timezone.utc),
        )
        _, stored = self._store.complete(session=session, report=report)
        return {"report_id": stored.id}

    def _project(
        self,
        user_id: str,
        session_id: str,
        state: MockInterviewState,
    ) -> MockInterviewGraphResult:
        session = self._require_session(user_id, session_id)
        raw_evaluation = state.get("evaluation")
        evaluation = (
            MockInterviewAnswerEvaluation.model_validate(raw_evaluation)
            if raw_evaluation is not None
            else None
        )
        if session.status == "completed":
            report = self._store.get_report(user_id=user_id, session_id=session_id)
            return MockInterviewGraphResult(
                session_id=session_id,
                state="completed",
                message="Mock interview completed.",
                evaluation=evaluation,
                report_id=report.id if report else None,
                report=report,
            )
        if session.status == "cancelled":
            return MockInterviewGraphResult(
                session_id=session_id,
                state="cancelled",
                message="Mock interview was cancelled.",
            )
        if session.current_turn_id is not None:
            turn = self._store.get_turn(
                user_id=user_id,
                turn_id=session.current_turn_id,
            )
            if turn is not None and turn.status == "awaiting_answer":
                return MockInterviewGraphResult(
                    session_id=session_id,
                    state="awaiting_answer",
                    message="Answer the current mock interview question.",
                    turn_id=turn.id,
                    question=turn.question,
                    evaluation=evaluation,
                )
        return MockInterviewGraphResult(
            session_id=session_id,
            state="running",
            message="Mock interview is processing the current turn.",
            evaluation=evaluation,
        )

    def _require_session(self, user_id: str, session_id: str) -> MockInterviewSession:
        session = self._store.get_session(user_id=user_id, session_id=session_id)
        if session is None:
            raise ValueError("Mock interview session does not exist")
        return session

    def _require_resumable_checkpoint(self, session: MockInterviewSession) -> None:
        if session.graph_version != self.GRAPH_VERSION:
            raise MockInterviewGraphVersionError(
                "Mock interview graph version is incompatible: "
                f"session={session.graph_version}, runtime={self.GRAPH_VERSION}"
            )
        checkpoint = self._checkpointer.get_tuple(self._config(session.id))
        if checkpoint is None:
            raise MockInterviewCheckpointMissingError(
                "Mock interview checkpoint is missing"
            )

    def _cleanup_terminal_checkpoint(
        self, result: MockInterviewGraphResult
    ) -> None:
        if result.state in {"completed", "cancelled"}:
            self._delete_checkpoint_best_effort(result.session_id)

    def _delete_checkpoint_best_effort(self, session_id: str) -> None:
        try:
            self._delete_checkpoint(session_id)
        except Exception:
            # Checkpoint retention must not turn an already persisted business
            # completion (or the original node failure) into a different result.
            # A later retention sweep may delete the orphaned thread.
            return

    def _delete_checkpoint(self, session_id: str) -> None:
        delete_thread = getattr(self._checkpointer, "delete_thread", None)
        if delete_thread is not None:
            delete_thread(session_id)

    def _require_plan(self, session: MockInterviewSession) -> MockInterviewPlan:
        plan = self._store.get_plan(user_id=session.user_id, session_id=session.id)
        if plan is None:
            raise ValueError("Mock interview session has no plan")
        return plan

    @staticmethod
    def _config(session_id: str) -> dict[str, dict[str, str]]:
        return {"configurable": {"thread_id": session_id}}
