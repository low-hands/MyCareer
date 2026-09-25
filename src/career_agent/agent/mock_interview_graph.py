from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import contextvars
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Literal, Protocol, TypedDict
from uuid import uuid4

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from career_agent.agent.mock_interview_contracts import (
    MockInterviewCompletionReason,
    MockInterviewFollowUpDecision,
    MockInterviewGraphResult,
    MockInterviewInputDecision,
    MockInterviewStartRequest,
    MockInterviewWorker,
)
from career_agent.agent.mock_interview_company_styles import match_company_style
from career_agent.agent.openai_compatible_client import AgentWorkerError
from career_agent.agent.interview_preparation_contracts import (
    CompanyResearchContext,
    CompanyResearchFindingContext,
    InterviewPreparationContext,
)
from career_agent.agent.resume_job_match_contracts import ConfirmedResumeFact
from career_agent.domain.mock_interviews import (
    MockInterviewAnswerEvaluation,
    MockInterviewPlan,
    MockInterviewReport,
    MockInterviewSession,
    MockInterviewTurn,
)
from career_agent.services.interview_context import InterviewPreparationContextFactory
from career_agent.storage.job_research import SQLiteJobResearchStore
from career_agent.storage.jobs import JobPostingRepository
from career_agent.storage.mock_interviews import SQLiteMockInterviewStore
from career_agent.storage.resumes import StoredResumeDocument



# Words a request to stop cannot avoid. A match only means "ask the model":
# "项目最后结束于灰度上线" contains 结束 and is an answer.
_STOP_REQUEST = re.compile(
    r"不练|不想练|不面了|不想面|结束|退出|停止|停一下|取消|算了|到此为止|就到这|先到这|"
    r"\b(?:quit|stop|cancel|exit|end)\b",
    re.IGNORECASE,
)


def _could_ask_to_stop(message: str) -> bool:
    return _STOP_REQUEST.search(message) is not None

@dataclass(frozen=True)
class MockInterviewSources:
    document: StoredResumeDocument | None
    context: InterviewPreparationContext

    @property
    def jd_text(self) -> str:
        return self.context.jd_text

    @property
    def company_name(self) -> str:
        return self.context.company_name

    @property
    def role_title(self) -> str:
        return self.context.role_title

    @property
    def confirmed_facts(self) -> tuple[ConfirmedResumeFact, ...]:
        return tuple(
            ConfirmedResumeFact.model_validate(fact.model_dump())
            for fact in self.context.confirmed_facts
        )


class MockInterviewCheckpointMissingError(RuntimeError):
    """Business state exists but LangGraph has no resumable checkpoint."""


class MockInterviewGraphVersionError(RuntimeError):
    """A persisted session belongs to an incompatible graph definition."""


class MockInterviewInputRoutingError(AgentWorkerError):
    """The local input router failed before the candidate answer was stored."""


class MockInterviewSourceProvider(Protocol):
    def load(self, *, session: MockInterviewSession) -> MockInterviewSources: ...


class StoredMockInterviewSourceProvider:
    """Loads the exact immutable JD and resume versions bound to a session."""

    def __init__(
        self,
        *,
        context_factory: InterviewPreparationContextFactory,
        jobs: JobPostingRepository | None = None,
        research: SQLiteJobResearchStore | None = None,
        research_freshness: timedelta = timedelta(days=7),
    ) -> None:
        self._context_factory = context_factory
        self._jobs = jobs
        self._research = research
        # Same window JobResearchService uses to call a report outdated.
        self._research_freshness = research_freshness

    def load(self, *, session: MockInterviewSession) -> MockInterviewSources:
        if session.application_id is None:
            free = self._context_factory.build_free(
                user_id=session.user_id,
                resume_version_id=session.resume_version_id,
                target_role=session.target_role or "",
            )
            context = free.context
            if session.jd_snapshot_id is not None:
                context = self._with_saved_job(session, context)
            elif session.target_company is not None:
                context = context.model_copy(update={"company_name": session.target_company})
            research = self._company_research(session)
            if research is not None:
                context = context.model_copy(update={"company_research": research})
            return MockInterviewSources(document=free.document, context=context)
        sources = self._context_factory.build(
            user_id=session.user_id,
            application_id=session.application_id,
            interview_round_id=session.interview_round_id,
        )
        expected = (
            session.application_id,
            session.job_posting_id,
            session.jd_snapshot_id,
            session.resume_version_id,
        )
        actual = (
            sources.application_id,
            sources.job_posting_id,
            sources.jd_snapshot_id,
            sources.resume_version_id,
        )
        if actual != expected or sources.document.resume_version_id != session.resume_version_id:
            raise ValueError("Mock interview sources no longer match the session")
        return MockInterviewSources(
            document=sources.document,
            context=sources.context,
        )


    def _with_saved_job(
        self, session: MockInterviewSession, context: InterviewPreparationContext
    ) -> InterviewPreparationContext:
        """The exact JD version pinned at start, with its company and title."""
        if self._jobs is None or session.job_posting_id is None:
            raise ValueError("Mock interview saved job cannot be loaded")
        job = self._jobs.get_job(user_id=session.user_id, job_posting_id=session.job_posting_id)
        snapshot = self._jobs.get_snapshot(
            user_id=session.user_id, jd_snapshot_id=session.jd_snapshot_id or ""
        )
        if job is None or snapshot is None or snapshot.job_posting_id != session.job_posting_id:
            raise ValueError("Mock interview saved job no longer exists")
        return context.model_copy(
            update={
                "company_name": job.posting.company_name,
                "role_title": session.target_role or job.posting.title,
                "jd_text": snapshot.content,
            }
        )

    def _company_research(
        self, session: MockInterviewSession
    ) -> CompanyResearchContext | None:
        """The report pinned at start, or none if it was never there or is gone."""
        if self._research is None or session.company_research_report_id is None:
            return None
        report = self._research.get_report(
            user_id=session.user_id,
            report_id=session.company_research_report_id,
            outdated_before=datetime.now(timezone.utc) - self._research_freshness,
        )
        if report is None:
            return None
        return CompanyResearchContext(
            researched_at=report.created_at,
            outdated=report.status != "current",
            summary=report.summary,
            findings=tuple(
                CompanyResearchFindingContext(
                    topic=finding.topic,
                    statement=finding.statement,
                    evidence_type=finding.evidence_type,
                    confidence=finding.confidence,
                )
                for finding in report.findings
            ),
        )


class MockInterviewState(TypedDict, total=False):
    graph_version: int
    user_id: str
    session_id: str
    route: Literal["primary", "follow_up", "report"]
    current_turn_id: str
    decided_turn_id: str
    decision: dict[str, Any]
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

    Primary questions are written with the plan, so moving to the next one
    costs no model call. Between answers the only model call is a short
    follow-up decision on the current question's own chain; every question is
    scored once, in parallel, when the interview ends. No request carries the
    whole interview, so none grows with its length.
    """

    # 2: questions come from the plan, answers are scored at the end. A
    # version-1 checkpoint cannot resume here; restart_mock_interview handles it.
    GRAPH_VERSION = 2
    _EVALUATION_CONCURRENCY = 4

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
        graph.add_node("decide", self._decide)
        graph.add_node("route", self._route)
        graph.add_node("report", self._report)
        graph.add_edge(START, "initialize")
        graph.add_edge("initialize", "ask")
        graph.add_edge("ask", "await")
        graph.add_edge("await", "decide")
        graph.add_edge("decide", "route")
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
        # Free practice runs on exactly the resume the user chose, or on none;
        # there is deliberately no "latest resume" fallback here.
        resume_version_id = request.resume_version_id
        session = self._store.create_session(
            user_id=request.user_id,
            application_id=request.application_id,
            interview_round_id=request.interview_round_id,
            job_posting_id=request.job_posting_id,
            jd_snapshot_id=request.jd_snapshot_id,
            resume_version_id=resume_version_id,
            target_role=request.target_role,
            target_company=request.target_company,
            company_research_report_id=request.company_research_report_id,
            interview_type=request.interview_type,
            max_primary_questions=request.max_primary_questions,
            max_follow_ups_per_question=request.max_follow_ups_per_question,
            conversation_id=request.conversation_id,
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
        # Business state first, checkpoint second. A finished run deletes its
        # checkpoint, so a crash between that deletion and the conversation
        # releasing the slot leaves an answer arriving for a run that already
        # has a report. Asking for the checkpoint first would call that a lost
        # checkpoint and bury a completed interview.
        if session.status in {"completed", "cancelled"}:
            return self._project(user_id, session_id, {})
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

    def handle_input(
        self,
        *,
        user_id: str,
        session_id: str,
        message: str,
    ) -> MockInterviewGraphResult:
        """Route one workflow-owned message before mutating the current turn."""
        normalized = message.strip()
        if not normalized:
            raise ValueError("Mock interview input must not be empty")
        session = self._require_session(user_id, session_id)
        if session.status in {"completed", "cancelled"}:
            return self._project(user_id, session_id, {})
        self._require_resumable_checkpoint(session)
        if session.status != "active" or session.current_turn_id is None:
            raise ValueError("Mock interview is not awaiting input")
        turn = self._store.get_turn(
            user_id=user_id,
            turn_id=session.current_turn_id,
        )
        if turn is None or turn.status != "awaiting_answer":
            raise ValueError("Mock interview current turn is not awaiting input")
        if not _could_ask_to_stop(normalized):
            # Nothing in it could mean stopping, so it is an answer. The model
            # call only earns its latency (15-35s measured on the relay) when
            # the message might be a request to stop.
            return self.resume(user_id=user_id, session_id=session_id, answer=normalized)
        try:
            decision = self._worker.route_input(
                session=session,
                turn=turn,
                user_message=normalized,
            )
        except AgentWorkerError as error:
            raise MockInterviewInputRoutingError(
                error.code,
                str(error),
                retryable=error.retryable,
                detail=error.detail,
            ) from error
        if not isinstance(decision, MockInterviewInputDecision):
            decision = MockInterviewInputDecision.model_validate(decision)
        if decision.action == "cancel":
            return self.cancel(user_id=user_id, session_id=session_id, message=normalized)
        return self.resume(
            user_id=user_id,
            session_id=session_id,
            answer=normalized,
        )

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
        self, *, user_id: str, session_id: str, message: str | None = None
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
            self._store.cancel(session=session, message=message)
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
                context=sources.context,
            )
            plan = MockInterviewPlan(
                session_id=session.id,
                summary=draft.summary,
                items=draft.items,
                limitations=draft.limitations,
                created_at=datetime.now(timezone.utc),
                company_style_profile=draft.company_style_profile,
                company_style_inferred=(
                    draft.company_style_profile is not None
                    and match_company_style(sources.context.company_name) is None
                ),
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
            if plan_item.question is not None:
                question = plan_item.question
            else:
                # A plan saved before questions were written with it.
                sources = self._sources.load(session=session)
                question = self._worker.ask(
                    session=session,
                    plan=plan,
                    plan_item=plan_item,
                    prior_turns=turns,
                    document=sources.document,
                    jd_text=sources.jd_text,
                    company_name=sources.company_name,
                    role_title=sources.role_title,
                    confirmed_facts=sources.confirmed_facts,
                ).question
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

    def _decide(self, state: MockInterviewState) -> dict[str, Any]:
        session = self._require_session(state["user_id"], state["session_id"])
        turn_id = state.get("current_turn_id") or session.current_turn_id
        if turn_id is None:
            raise ValueError("Mock interview has no answer to decide on")
        turn = self._store.get_turn(user_id=session.user_id, turn_id=turn_id)
        if turn is None:
            raise ValueError("Mock interview turn does not exist")
        if turn.status == "awaiting_answer":
            session, turn = self._store.record_answer(
                session=session,
                turn=turn,
                answer=state.get("answer") or "",
            )
        plan = self._require_plan(session)
        plan_item = plan.items[turn.plan_item_number - 1]
        chain = tuple(
            item
            for item in self._store.list_turns(user_id=session.user_id, session_id=session.id)
            if item.plan_item_number == turn.plan_item_number and item.answer is not None
        )
        remaining = session.max_follow_ups_per_question - sum(
            item.turn_type == "follow_up" for item in chain
        )
        if remaining > 0:
            sources = self._sources.load(session=session)
            decision = self._worker.decide_follow_up(
                session=session,
                plan_item=plan_item,
                turns=chain,
                follow_ups_remaining=remaining,
                document=sources.document,
                jd_text=sources.jd_text,
            )
        else:
            decision = MockInterviewFollowUpDecision(next_action="next_question")
        self._store.settle_answer(session=session, turn=turn)
        return {
            "answer": None,
            "decision": decision.model_dump(mode="json"),
            "decided_turn_id": turn.id,
            "current_turn_id": None,
        }

    def _route(self, state: MockInterviewState) -> dict[str, Any]:
        session = self._require_session(state["user_id"], state["session_id"])
        plan = self._require_plan(session)
        turn = self._store.get_turn(
            user_id=session.user_id,
            turn_id=state["decided_turn_id"],
        )
        raw_decision = state.get("decision")
        if turn is None or raw_decision is None:
            raise ValueError("Mock interview decision is unavailable for routing")
        decision = MockInterviewFollowUpDecision.model_validate(raw_decision)
        turns = self._store.list_turns(user_id=session.user_id, session_id=session.id)
        primary_id = turn.id if turn.turn_type == "primary" else turn.parent_turn_id
        follow_up_count = sum(
            item.turn_type == "follow_up" and item.parent_turn_id == primary_id
            for item in turns
        )
        if (
            decision.next_action == "follow_up"
            and decision.follow_up_question
            and follow_up_count < session.max_follow_ups_per_question
        ):
            return {
                "route": "follow_up",
                "follow_up_question": decision.follow_up_question,
                "follow_up_parent_turn_id": primary_id,
            }
        remaining = session.current_plan_item < len(plan.items)
        if decision.next_action == "finish" and remaining:
            return {"route": "report", "completion_reason": "user_ended"}
        if remaining:
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
        sources = self._sources.load(session=session)
        self._evaluate_questions(session, plan, sources)
        turns = self._store.list_turns(user_id=session.user_id, session_id=session.id)
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

    def _evaluate_questions(
        self,
        session: MockInterviewSession,
        plan: MockInterviewPlan,
        sources: MockInterviewSources,
    ) -> None:
        """Score every answered question not yet scored, one request per question.

        Each request carries only its own question's chain. Successes are
        stored before any failure is raised, so a retry scores only what is
        left.
        """
        turns = self._store.list_turns(user_id=session.user_id, session_id=session.id)
        pending: list[tuple[MockInterviewTurn, tuple[MockInterviewTurn, ...]]] = [
            (
                primary,
                tuple(
                    item
                    for item in turns
                    if item.plan_item_number == primary.plan_item_number
                    and item.answer is not None
                ),
            )
            for primary in turns
            if primary.turn_type == "primary" and primary.status == "answered"
        ]
        if not pending:
            return

        def score(primary: MockInterviewTurn, chain: tuple[MockInterviewTurn, ...]):
            return self._worker.evaluate(
                session=session,
                plan_item=plan.items[primary.plan_item_number - 1],
                turns=chain,
                document=sources.document,
                jd_text=sources.jd_text,
                confirmed_facts=sources.confirmed_facts,
            )

        first_error: Exception | None = None
        with ThreadPoolExecutor(
            max_workers=min(self._EVALUATION_CONCURRENCY, len(pending))
        ) as pool:
            # Each task runs in a copy of this turn's context so its model call
            # is still traced under the turn.
            futures = [
                (primary, pool.submit(contextvars.copy_context().run, score, primary, chain))
                for primary, chain in pending
            ]
            for primary, future in futures:
                try:
                    evaluation = future.result()
                except Exception as error:
                    first_error = first_error or error
                    continue
                self._store.record_question_evaluation(
                    session=session, turn=primary, evaluation=evaluation
                )
        if first_error is not None:
            raise first_error

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
            if report is None:
                # The same invariant ``_report`` enforces, answered the same way.
                # Reading it as ``report.id if report else None`` degraded an
                # impossible state into a card-shaped result with no reference,
                # which ``_conversation_content`` then fails open on — a finished
                # interview delivered as a report nobody can name. Unreachable
                # either way; this way it says so.
                raise ValueError("Completed mock interview has no report")
            return MockInterviewGraphResult(
                session_id=session_id,
                state="completed",
                message="Mock interview completed.",
                evaluation=evaluation,
                report_id=report.id,
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
