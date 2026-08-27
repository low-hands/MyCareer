from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import sqlite3

from pydantic import BaseModel, ConfigDict

from career_agent.agent.job_discovery_contracts import JobDiscoveryRequest
from career_agent.domain.job_discovery import SearchResult
from career_agent.harness.observability import RunTrace
from career_agent.storage.schema import apply_schema


class StoredJobDiscoveryRun(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    request: JobDiscoveryRequest
    results: tuple[SearchResult, ...]
    phase: str
    trace: RunTrace
    selected_result_ref: str | None = None
    selected_result_refs: tuple[str, ...] = ()
    error_code: str | None = None
    error_stage: str | None = None
    error_detail: str | None = None
    recoverable: bool | None = None


class JobDiscoveryRunStore:
    def __init__(self, path: Path) -> None:
        self.path = path.expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            os.chmod(self.path, 0o600)
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            apply_schema(connection, "job_discovery_runs", 1, self._migrate)
        os.chmod(self.path, 0o600)

    @staticmethod
    def _migrate(connection: sqlite3.Connection) -> None:
        connection.execute("CREATE TABLE IF NOT EXISTS job_discovery_runs (run_id TEXT PRIMARY KEY, payload TEXT NOT NULL, updated_at TEXT NOT NULL)")

    def save(
        self,
        *,
        run_id: str,
        request: JobDiscoveryRequest,
        results: tuple[SearchResult, ...],
        phase: str,
        trace: RunTrace,
        selected_result_ref: str | None = None,
        selected_result_refs: tuple[str, ...] = (),
        error_code: str | None = None,
        error_stage: str | None = None,
        error_detail: str | None = None,
        recoverable: bool | None = None,
    ) -> StoredJobDiscoveryRun:
        safe_request = request.model_copy(update={"resume_text": None})
        record = StoredJobDiscoveryRun(
            run_id=run_id,
            request=safe_request,
            results=results,
            phase=phase,
            trace=trace,
            selected_result_ref=selected_result_ref,
            selected_result_refs=selected_result_refs,
            error_code=error_code,
            error_stage=error_stage,
            error_detail=error_detail,
            recoverable=recoverable,
        )
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO job_discovery_runs(run_id, payload, updated_at) VALUES (?, ?, ?) ON CONFLICT(run_id) DO UPDATE SET payload=excluded.payload, updated_at=excluded.updated_at",
                (run_id, record.model_dump_json(), datetime.now(timezone.utc).isoformat()),
            )
        os.chmod(self.path, 0o600)
        return record

    def get(self, run_id: str) -> StoredJobDiscoveryRun | None:
        with self._connect() as connection:
            row = connection.execute("SELECT payload FROM job_discovery_runs WHERE run_id = ?", (run_id,)).fetchone()
        return StoredJobDiscoveryRun.model_validate_json(row[0]) if row else None

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.execute("PRAGMA foreign_keys=ON")
        return connection
