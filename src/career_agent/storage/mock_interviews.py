from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import sqlite3
from typing import Literal
from uuid import uuid4

from career_agent.domain.mock_interviews import (
    MockInterviewAnswerEvaluation,
    MockInterviewPlan,
    MockInterviewQuestionType,
    MockInterviewReport,
    MockInterviewSession,
    MockInterviewStatus,
    MockInterviewTurn,
    MockInterviewType,
)
from career_agent.storage.schema import apply_schema


class SQLiteMockInterviewStore:
    """Durable state for multi-turn mock interview sessions.

    The store owns lifecycle and turn invariants so the graph never has to
    reconstruct them from model context: one in-progress turn per session, one
    active session per user, follow-ups bounded per primary question, and
    reports grounded in evaluated turns.
    """

    def __init__(self, path: Path) -> None:
        self.path = path.expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        os.chmod(self.path.parent, 0o700)
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            apply_schema(
                connection,
                "mock_interviews",
                3,
                self._migrate,
                {2: self._upgrade_v2, 3: self._upgrade_v3},
            )
        os.chmod(self.path, 0o600)

    def create_session(
        self,
        *,
        user_id: str,
        application_id: str,
        job_posting_id: str,
        jd_snapshot_id: str,
        resume_version_id: str,
        interview_type: MockInterviewType,
        interview_round_id: str | None = None,
        max_primary_questions: int = 6,
        max_follow_ups_per_question: int = 2,
        graph_version: int = 1,
    ) -> MockInterviewSession:
        now = datetime.now(timezone.utc)
        session = MockInterviewSession(
            id=f"mock_interview_{uuid4().hex}",
            user_id=user_id,
            application_id=application_id,
            interview_round_id=interview_round_id,
            job_posting_id=job_posting_id,
            jd_snapshot_id=jd_snapshot_id,
            resume_version_id=resume_version_id,
            interview_type=interview_type,
            graph_version=graph_version,
            status="created",
            max_primary_questions=max_primary_questions,
            max_follow_ups_per_question=max_follow_ups_per_question,
            created_at=now,
            updated_at=now,
        )
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._require_no_resumable_session(connection, user_id)
            connection.execute(
                """
                INSERT INTO mock_interview_sessions(
                    id, user_id, application_id, interview_round_id, job_posting_id,
                    jd_snapshot_id, resume_version_id, interview_type, graph_version, status,
                    max_primary_questions, max_follow_ups_per_question,
                    current_plan_item, current_turn_id,
                    created_at, started_at, paused_at, completed_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                self._session_values(session),
            )
        return session

    def save_plan(
        self, *, session: MockInterviewSession, plan: MockInterviewPlan
    ) -> MockInterviewPlan:
        if plan.session_id != session.id:
            raise ValueError("plan does not belong to this session")
        if session.status != "created":
            raise ValueError("plan can only be saved before the session starts")
        if len(plan.items) > session.max_primary_questions:
            raise ValueError("plan has more items than max_primary_questions")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT 1 FROM mock_interview_plans WHERE session_id = ?",
                (session.id,),
            ).fetchone()
            if existing is not None:
                raise ValueError("session already has a plan")
            connection.execute(
                """
                INSERT INTO mock_interview_plans(session_id, user_id, plan_json, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (
                    session.id,
                    session.user_id,
                    plan.model_dump_json(),
                    plan.created_at.isoformat(),
                ),
            )
        return plan

    def get_plan(self, *, user_id: str, session_id: str) -> MockInterviewPlan | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT plan_json FROM mock_interview_plans WHERE session_id = ? AND user_id = ?",
                (session_id, user_id),
            ).fetchone()
        return MockInterviewPlan.model_validate_json(row[0]) if row else None

    def start(self, *, session: MockInterviewSession) -> MockInterviewSession:
        if session.status != "created":
            raise ValueError("only a created session can be started")
        if self.get_plan(user_id=session.user_id, session_id=session.id) is None:
            raise ValueError("session requires a plan before starting")
        now = datetime.now(timezone.utc)
        return self._transition(
            session,
            {"status": "active", "started_at": now, "updated_at": now},
        )

    def pause(self, *, session: MockInterviewSession) -> MockInterviewSession:
        if session.status != "active":
            raise ValueError("only an active session can be paused")
        if session.current_turn_id is not None:
            raise ValueError("cannot pause while a turn is in progress")
        now = datetime.now(timezone.utc)
        return self._transition(
            session,
            {"status": "paused", "paused_at": now, "updated_at": now},
        )

    def resume(self, *, session: MockInterviewSession) -> MockInterviewSession:
        if session.status != "paused":
            raise ValueError("only a paused session can be resumed")
        now = datetime.now(timezone.utc)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._require_no_active_session(connection, session.user_id)
            return self._apply(
                connection,
                session,
                {"status": "active", "paused_at": None, "updated_at": now},
            )

    def cancel(self, *, session: MockInterviewSession) -> MockInterviewSession:
        if session.status in {"completed", "cancelled"}:
            raise ValueError("finished sessions cannot be cancelled")
        now = datetime.now(timezone.utc)
        return self._transition(
            session,
            {
                "status": "cancelled",
                "current_turn_id": None,
                "paused_at": None,
                "updated_at": now,
            },
        )

    def ask(
        self,
        *,
        session: MockInterviewSession,
        plan_item_number: int,
        question_type: MockInterviewQuestionType,
        question: str,
        turn_type: Literal["primary", "follow_up"] = "primary",
        parent_turn_id: str | None = None,
        asked_at: datetime | None = None,
    ) -> tuple[MockInterviewSession, MockInterviewTurn]:
        if session.status != "active":
            raise ValueError("only an active session can ask a question")
        if session.current_turn_id is not None:
            raise ValueError("a previous turn is still in progress")
        plan = self.get_plan(user_id=session.user_id, session_id=session.id)
        if plan is None:
            raise ValueError("session has no plan")
        if not 1 <= plan_item_number <= len(plan.items):
            raise ValueError("plan_item_number is outside the saved plan")
        plan_item = plan.items[plan_item_number - 1]
        if question_type != plan_item.question_type:
            raise ValueError("question_type must match the saved plan item")
        asked = asked_at or datetime.now(timezone.utc)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if turn_type == "follow_up":
                parent = self._turn_row(connection, session.id, parent_turn_id)
                if parent is None or parent["plan_item_number"] != plan_item_number:
                    raise ValueError("follow-up must reference a parent turn on the same plan item")
                if parent["turn_type"] != "primary":
                    raise ValueError("follow-up must reference the primary turn")
                follow_ups = int(
                    connection.execute(
                        """
                        SELECT COUNT(*) FROM mock_interview_turns
                        WHERE session_id = ? AND parent_turn_id = ?
                        """,
                        (session.id, parent_turn_id),
                    ).fetchone()[0]
                )
                if follow_ups >= session.max_follow_ups_per_question:
                    raise ValueError("follow-up limit reached for this question")
            elif plan_item_number <= session.current_plan_item:
                raise ValueError("primary questions must advance the plan item")
            sequence_number = int(
                connection.execute(
                    """
                    SELECT COALESCE(MAX(sequence_number), 0) + 1
                    FROM mock_interview_turns WHERE session_id = ?
                    """,
                    (session.id,),
                ).fetchone()[0]
            )
            turn = MockInterviewTurn(
                id=f"mock_interview_turn_{uuid4().hex}",
                session_id=session.id,
                sequence_number=sequence_number,
                plan_item_number=plan_item_number,
                turn_type=turn_type,
                parent_turn_id=parent_turn_id,
                question_type=question_type,
                question=question,
                status="awaiting_answer",
                asked_at=asked,
            )
            connection.execute(
                """
                INSERT INTO mock_interview_turns(
                    id, session_id, user_id, sequence_number, plan_item_number,
                    turn_type, parent_turn_id, question_type, question, answer,
                    evaluation_json, status, asked_at, answered_at, evaluated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                self._turn_values(turn, session.user_id),
            )
            updates: dict[str, object] = {
                "current_turn_id": turn.id,
                "updated_at": datetime.now(timezone.utc),
            }
            if turn_type == "primary":
                updates["current_plan_item"] = plan_item_number
            updated = self._apply(connection, session, updates)
        return updated, turn

    def record_answer(
        self,
        *,
        session: MockInterviewSession,
        turn: MockInterviewTurn,
        answer: str,
        answered_at: datetime | None = None,
    ) -> tuple[MockInterviewSession, MockInterviewTurn]:
        answered = answered_at or datetime.now(timezone.utc)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = self._load_turn(connection, session.id, turn.id)
            if current is None:
                raise ValueError("mock interview turn does not exist")
            candidate = current.model_copy(
                update={
                    "answer": answer,
                    "evaluation": None,
                    "status": "answered",
                    "answered_at": answered,
                    "evaluated_at": None,
                }
            )
            answered_turn = MockInterviewTurn.model_validate(candidate.model_dump())
            if current.status in {"answered", "evaluated"}:
                if current.answer != answered_turn.answer:
                    raise ValueError("turn already has a different answer")
                stored_session = self._load_session(
                    connection, session.user_id, session.id
                )
                if stored_session is None:
                    raise ValueError("mock interview session does not exist")
                return stored_session, current
            if session.current_turn_id != turn.id:
                raise ValueError("turn is not the session's current turn")
            if current.status != "awaiting_answer":
                raise ValueError("turn cannot accept an answer in its current state")
            changed = connection.execute(
                """
                UPDATE mock_interview_turns SET
                    answer = ?, status = 'answered', answered_at = ?
                WHERE id = ? AND session_id = ? AND status = 'awaiting_answer'
                """,
                (
                    answered_turn.answer,
                    answered.isoformat(),
                    turn.id,
                    session.id,
                ),
            ).rowcount
            if not changed:
                raise RuntimeError("mock interview turn changed concurrently")
            updated = self._apply(
                connection,
                session,
                {
                    "updated_at": datetime.now(timezone.utc),
                },
            )
        return updated, answered_turn

    def record_evaluation(
        self,
        *,
        session: MockInterviewSession,
        turn: MockInterviewTurn,
        evaluation: MockInterviewAnswerEvaluation,
        evaluated_at: datetime | None = None,
    ) -> tuple[MockInterviewSession, MockInterviewTurn]:
        evaluated = evaluated_at or datetime.now(timezone.utc)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = self._load_turn(connection, session.id, turn.id)
            if current is None:
                raise ValueError("mock interview turn does not exist")
            if current.status == "evaluated":
                stored_session = self._load_session(
                    connection, session.user_id, session.id
                )
                if stored_session is None:
                    raise ValueError("mock interview session does not exist")
                return stored_session, current
            if session.current_turn_id != turn.id:
                raise ValueError("turn is not the session's current turn")
            if current.status != "answered":
                raise ValueError("turn requires a persisted answer before evaluation")
            evaluated_turn = current.model_copy(
                update={
                    "evaluation": evaluation,
                    "status": "evaluated",
                    "evaluated_at": evaluated,
                }
            )
            evaluated_turn = MockInterviewTurn.model_validate(
                evaluated_turn.model_dump()
            )
            changed = connection.execute(
                """
                UPDATE mock_interview_turns SET
                    evaluation_json = ?, status = 'evaluated', evaluated_at = ?
                WHERE id = ? AND session_id = ? AND status = 'answered'
                """,
                (
                    evaluation.model_dump_json(),
                    evaluated.isoformat(),
                    turn.id,
                    session.id,
                ),
            ).rowcount
            if not changed:
                raise RuntimeError("mock interview turn changed concurrently")
            updated = self._apply(
                connection,
                session,
                {
                    "current_turn_id": None,
                    "updated_at": datetime.now(timezone.utc),
                },
            )
        return updated, evaluated_turn

    def complete(
        self, *, session: MockInterviewSession, report: MockInterviewReport
    ) -> tuple[MockInterviewSession, MockInterviewReport]:
        if session.status not in {"active", "paused"}:
            raise ValueError("only a started session can be completed")
        if session.current_turn_id is not None:
            raise ValueError("cannot complete while a turn is in progress")
        if report.session_id != session.id:
            raise ValueError("report does not belong to this session")
        evaluated = {
            turn.plan_item_number
            for turn in self.list_turns(user_id=session.user_id, session_id=session.id)
            if turn.status == "evaluated" and turn.turn_type == "primary"
        }
        reported = {
            result.plan_item_number
            for result in report.question_results
        }
        if reported != evaluated:
            raise ValueError(
                "report must cover exactly the plan items with evaluated primary answers"
            )
        now = datetime.now(timezone.utc)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                INSERT INTO mock_interview_reports(
                    id, session_id, user_id, report_json, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    report.id,
                    report.session_id,
                    session.user_id,
                    report.model_dump_json(),
                    report.created_at.isoformat(),
                ),
            )
            updated = self._apply(
                connection,
                session,
                {
                    "status": "completed",
                    "completed_at": now,
                    "paused_at": None,
                    "updated_at": now,
                },
            )
        return updated, report

    def get_session(
        self, *, user_id: str, session_id: str
    ) -> MockInterviewSession | None:
        with self._connect() as connection:
            row = connection.execute(
                self._SESSION_SELECT + " WHERE id = ? AND user_id = ?",
                (session_id, user_id),
            ).fetchone()
        return self._session(row) if row else None

    def find_resumable(self, *, user_id: str) -> MockInterviewSession | None:
        with self._connect() as connection:
            row = connection.execute(
                self._SESSION_SELECT
                + """
                  WHERE user_id = ? AND status IN ('created', 'active', 'paused')
                  ORDER BY updated_at DESC LIMIT 1
                  """,
                (user_id,),
            ).fetchone()
        return self._session(row) if row else None

    def list_sessions(
        self,
        *,
        user_id: str,
        application_id: str | None = None,
        statuses: tuple[MockInterviewStatus, ...] = (),
        limit: int = 50,
    ) -> tuple[MockInterviewSession, ...]:
        query = self._SESSION_SELECT + " WHERE user_id = ?"
        params: list[object] = [user_id]
        if application_id is not None:
            query += " AND application_id = ?"
            params.append(application_id)
        if statuses:
            query += f" AND status IN ({','.join('?' for _ in statuses)})"
            params.extend(statuses)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        with self._connect() as connection:
            rows = connection.execute(query, tuple(params)).fetchall()
        return tuple(self._session(row) for row in rows)

    def get_turn(
        self, *, user_id: str, turn_id: str
    ) -> MockInterviewTurn | None:
        with self._connect() as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute(
                self._TURN_SELECT + " WHERE id = ? AND user_id = ?",
                (turn_id, user_id),
            ).fetchone()
        return self._turn(row) if row else None

    def list_turns(
        self, *, user_id: str, session_id: str
    ) -> tuple[MockInterviewTurn, ...]:
        with self._connect() as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                self._TURN_SELECT
                + " WHERE session_id = ? AND user_id = ? ORDER BY sequence_number",
                (session_id, user_id),
            ).fetchall()
        return tuple(self._turn(row) for row in rows)

    def get_report(
        self, *, user_id: str, session_id: str
    ) -> MockInterviewReport | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT report_json FROM mock_interview_reports WHERE session_id = ? AND user_id = ?",
                (session_id, user_id),
            ).fetchone()
        return MockInterviewReport.model_validate_json(row[0]) if row else None

    def find_report_session_id(
        self, *, user_id: str, report_id: str
    ) -> str | None:
        """Which run produced this report, scoped to its owner.

        Reading a report back needs the run's turns too, and those hang off the
        session rather than the report. The user filter is the ownership check:
        a report id from another user's conversation resolves to nothing rather
        than to their interview.
        """
        with self._connect() as connection:
            row = connection.execute(
                "SELECT session_id FROM mock_interview_reports "
                "WHERE id = ? AND user_id = ?",
                (report_id, user_id),
            ).fetchone()
        return str(row[0]) if row else None

    _SESSION_SELECT = (
        "SELECT id, user_id, application_id, interview_round_id, job_posting_id, "
        "jd_snapshot_id, resume_version_id, interview_type, graph_version, status, "
        "max_primary_questions, max_follow_ups_per_question, current_plan_item, "
        "current_turn_id, created_at, started_at, paused_at, completed_at, updated_at "
        "FROM mock_interview_sessions"
    )
    _TURN_SELECT = (
        "SELECT id, session_id, sequence_number, plan_item_number, turn_type, "
        "parent_turn_id, question_type, question, answer, evaluation_json, status, "
        "asked_at, answered_at, evaluated_at FROM mock_interview_turns"
    )

    def _transition(
        self, session: MockInterviewSession, updates: dict[str, object]
    ) -> MockInterviewSession:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            return self._apply(connection, session, updates)

    def _apply(
        self,
        connection: sqlite3.Connection,
        session: MockInterviewSession,
        updates: dict[str, object],
    ) -> MockInterviewSession:
        updated = session.model_copy(update=updates)
        MockInterviewSession.model_validate(updated.model_dump())
        changed = connection.execute(
            """
            UPDATE mock_interview_sessions SET
                status = ?, current_plan_item = ?, current_turn_id = ?,
                started_at = ?, paused_at = ?, completed_at = ?, updated_at = ?
            WHERE id = ? AND user_id = ? AND updated_at = ?
            """,
            (
                updated.status,
                updated.current_plan_item,
                updated.current_turn_id,
                self._iso(updated.started_at),
                self._iso(updated.paused_at),
                self._iso(updated.completed_at),
                updated.updated_at.isoformat(),
                updated.id,
                updated.user_id,
                session.updated_at.isoformat(),
            ),
        ).rowcount
        if not changed:
            raise RuntimeError("mock interview session changed concurrently")
        return updated

    def _load_session(
        self,
        connection: sqlite3.Connection,
        user_id: str,
        session_id: str,
    ) -> MockInterviewSession | None:
        row = connection.execute(
            self._SESSION_SELECT + " WHERE id = ? AND user_id = ?",
            (session_id, user_id),
        ).fetchone()
        return self._session(row) if row else None

    def _load_turn(
        self,
        connection: sqlite3.Connection,
        session_id: str,
        turn_id: str,
    ) -> MockInterviewTurn | None:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            self._TURN_SELECT + " WHERE id = ? AND session_id = ?",
            (turn_id, session_id),
        ).fetchone()
        return self._turn(row) if row else None

    @staticmethod
    def _require_no_resumable_session(
        connection: sqlite3.Connection, user_id: str
    ) -> None:
        row = connection.execute(
            """
            SELECT id FROM mock_interview_sessions
            WHERE user_id = ? AND status IN ('created', 'active', 'paused') LIMIT 1
            """,
            (user_id,),
        ).fetchone()
        if row is not None:
            raise ValueError(f"user already has an unfinished mock interview: {row[0]}")

    @staticmethod
    def _require_no_active_session(
        connection: sqlite3.Connection, user_id: str
    ) -> None:
        row = connection.execute(
            "SELECT id FROM mock_interview_sessions WHERE user_id = ? AND status = 'active' LIMIT 1",
            (user_id,),
        ).fetchone()
        if row is not None:
            raise ValueError(f"user already has an active mock interview: {row[0]}")

    @staticmethod
    def _turn_row(
        connection: sqlite3.Connection, session_id: str, turn_id: str | None
    ) -> sqlite3.Row | None:
        if turn_id is None:
            return None
        connection.row_factory = sqlite3.Row
        return connection.execute(
            """
            SELECT id, plan_item_number, turn_type FROM mock_interview_turns
            WHERE id = ? AND session_id = ?
            """,
            (turn_id, session_id),
        ).fetchone()

    def _migrate(self, connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS mock_interview_sessions (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                application_id TEXT NOT NULL,
                interview_round_id TEXT,
                job_posting_id TEXT NOT NULL,
                jd_snapshot_id TEXT NOT NULL,
                resume_version_id TEXT NOT NULL,
                interview_type TEXT NOT NULL,
                graph_version INTEGER NOT NULL DEFAULT 1,
                status TEXT NOT NULL,
                max_primary_questions INTEGER NOT NULL,
                max_follow_ups_per_question INTEGER NOT NULL,
                current_plan_item INTEGER NOT NULL,
                current_turn_id TEXT,
                created_at TEXT NOT NULL,
                started_at TEXT,
                paused_at TEXT,
                completed_at TEXT,
                updated_at TEXT NOT NULL
            )
            """
        )

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS mock_interview_plans (
                session_id TEXT PRIMARY KEY
                    REFERENCES mock_interview_sessions(id),
                user_id TEXT NOT NULL,
                plan_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS mock_interview_turns (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL REFERENCES mock_interview_sessions(id),
                user_id TEXT NOT NULL,
                sequence_number INTEGER NOT NULL,
                plan_item_number INTEGER NOT NULL,
                turn_type TEXT NOT NULL,
                parent_turn_id TEXT REFERENCES mock_interview_turns(id),
                question_type TEXT NOT NULL,
                question TEXT NOT NULL,
                answer TEXT,
                evaluation_json TEXT,
                status TEXT NOT NULL,
                asked_at TEXT NOT NULL,
                answered_at TEXT,
                evaluated_at TEXT,
                UNIQUE(session_id, sequence_number)
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS mock_interview_reports (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL UNIQUE
                    REFERENCES mock_interview_sessions(id),
                user_id TEXT NOT NULL,
                report_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS mock_interview_sessions_active_idx
            ON mock_interview_sessions(user_id) WHERE status = 'active'
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS mock_interview_sessions_user_idx
            ON mock_interview_sessions(user_id, status, updated_at DESC)
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS mock_interview_turns_session_idx
            ON mock_interview_turns(session_id, sequence_number)
            """
        )

    @staticmethod
    def _upgrade_v2(connection: sqlite3.Connection) -> None:
        # Version 2 adds the domain-level ``answered`` status. The status column
        # intentionally has no SQLite CHECK constraint, so existing rows need no
        # physical rewrite; the explicit step prevents a silent semantic bump.
        return None

    @staticmethod
    def _upgrade_v3(connection: sqlite3.Connection) -> None:
        columns = {
            row[1]
            for row in connection.execute(
                "PRAGMA table_info(mock_interview_sessions)"
            ).fetchall()
        }
        if "graph_version" not in columns:
            connection.execute(
                "ALTER TABLE mock_interview_sessions "
                "ADD COLUMN graph_version INTEGER NOT NULL DEFAULT 1"
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    @classmethod
    def _session_values(cls, session: MockInterviewSession) -> tuple[object, ...]:
        return (
            session.id, session.user_id, session.application_id,
            session.interview_round_id, session.job_posting_id,
            session.jd_snapshot_id, session.resume_version_id,
            session.interview_type, session.graph_version, session.status,
            session.max_primary_questions,
            session.max_follow_ups_per_question, session.current_plan_item,
            session.current_turn_id, session.created_at.isoformat(),
            cls._iso(session.started_at), cls._iso(session.paused_at),
            cls._iso(session.completed_at), session.updated_at.isoformat(),
        )

    @classmethod
    def _turn_values(
        cls, turn: MockInterviewTurn, user_id: str
    ) -> tuple[object, ...]:
        return (
            turn.id, turn.session_id, user_id, turn.sequence_number,
            turn.plan_item_number, turn.turn_type, turn.parent_turn_id,
            turn.question_type, turn.question, turn.answer,
            turn.evaluation.model_dump_json() if turn.evaluation else None,
            turn.status, turn.asked_at.isoformat(), cls._iso(turn.answered_at),
            cls._iso(turn.evaluated_at),
        )

    @staticmethod
    def _session(row: tuple[object, ...]) -> MockInterviewSession:
        return MockInterviewSession(
            id=row[0], user_id=row[1], application_id=row[2],
            interview_round_id=row[3], job_posting_id=row[4], jd_snapshot_id=row[5],
            resume_version_id=row[6], interview_type=row[7], graph_version=row[8],
            status=row[9], max_primary_questions=row[10],
            max_follow_ups_per_question=row[11], current_plan_item=row[12],
            current_turn_id=row[13], created_at=row[14], started_at=row[15],
            paused_at=row[16], completed_at=row[17], updated_at=row[18],
        )

    @staticmethod
    def _turn(row: sqlite3.Row) -> MockInterviewTurn:
        evaluation = row["evaluation_json"]
        return MockInterviewTurn(
            id=row["id"], session_id=row["session_id"],
            sequence_number=row["sequence_number"],
            plan_item_number=row["plan_item_number"], turn_type=row["turn_type"],
            parent_turn_id=row["parent_turn_id"],
            question_type=row["question_type"], question=row["question"],
            answer=row["answer"],
            evaluation=(
                MockInterviewAnswerEvaluation.model_validate_json(evaluation)
                if evaluation
                else None
            ),
            status=row["status"], asked_at=row["asked_at"],
            answered_at=row["answered_at"], evaluated_at=row["evaluated_at"],
        )

    @staticmethod
    def _iso(value: datetime | None) -> str | None:
        return value.isoformat() if value is not None else None
