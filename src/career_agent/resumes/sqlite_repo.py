"""SQLite implementation of the resume pool repository."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from career_agent.resumes.models import ResumeData, ResumeVersion
from career_agent.resumes.repository import DuplicateResumeError, ResumeNotFoundError


class SQLiteResumeVersionRepository:
    """Store all tenants in one SQLite database with mandatory row scoping."""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path).expanduser().resolve()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize_schema(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS resume_versions (
                    version_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    role_type TEXT NOT NULL,
                    version_number INTEGER NOT NULL,
                    note TEXT NOT NULL DEFAULT '',
                    original_filename TEXT NOT NULL,
                    stored_file_path TEXT NOT NULL,
                    file_hash TEXT NOT NULL,
                    parsed_data_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE (tenant_id, role_type, version_number),
                    UNIQUE (tenant_id, role_type, file_hash)
                );

                CREATE INDEX IF NOT EXISTS idx_resume_versions_tenant_role
                ON resume_versions (tenant_id, role_type, version_number);

                CREATE TABLE IF NOT EXISTS resume_tenant_state (
                    tenant_id TEXT PRIMARY KEY,
                    active_version_id TEXT
                );
                """
            )

    @staticmethod
    def _row_to_version(row: sqlite3.Row) -> ResumeVersion:
        return ResumeVersion(
            version_id=row["version_id"],
            tenant_id=row["tenant_id"],
            role_type=row["role_type"],
            version_number=row["version_number"],
            note=row["note"],
            original_filename=row["original_filename"],
            stored_file_path=row["stored_file_path"],
            file_hash=row["file_hash"],
            parsed_data=ResumeData.model_validate_json(row["parsed_data_json"]),
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    def create_version(
        self,
        *,
        version_id: str,
        tenant_id: str,
        role_type: str,
        note: str,
        original_filename: str,
        stored_file_path: str,
        file_hash: str,
        parsed_data: ResumeData,
        created_at: datetime | None = None,
    ) -> ResumeVersion:
        created_at = created_at or datetime.now(UTC)
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            duplicate = connection.execute(
                """
                SELECT version_id
                FROM resume_versions
                WHERE tenant_id = ? AND role_type = ? AND file_hash = ?
                """,
                (tenant_id, role_type, file_hash),
            ).fetchone()
            if duplicate is not None:
                raise DuplicateResumeError(
                    f"Resume file already exists in role pool {role_type!r}: {duplicate['version_id']}"
                )

            row = connection.execute(
                """
                SELECT COALESCE(MAX(version_number), 0) + 1 AS next_version
                FROM resume_versions
                WHERE tenant_id = ? AND role_type = ?
                """,
                (tenant_id, role_type),
            ).fetchone()
            version_number = int(row["next_version"])
            connection.execute(
                """
                INSERT INTO resume_versions (
                    version_id,
                    tenant_id,
                    role_type,
                    version_number,
                    note,
                    original_filename,
                    stored_file_path,
                    file_hash,
                    parsed_data_json,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    version_id,
                    tenant_id,
                    role_type,
                    version_number,
                    note,
                    original_filename,
                    stored_file_path,
                    file_hash,
                    parsed_data.model_dump_json(),
                    created_at.isoformat(),
                ),
            )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

        created = self.get_version(tenant_id, version_id)
        if created is None:  # pragma: no cover - defensive check after commit
            raise RuntimeError(f"Created resume version cannot be reloaded: {version_id}")
        return created

    def find_by_hash(
        self,
        tenant_id: str,
        role_type: str,
        file_hash: str,
    ) -> ResumeVersion | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT *
                FROM resume_versions
                WHERE tenant_id = ? AND role_type = ? AND file_hash = ?
                """,
                (tenant_id, role_type, file_hash),
            ).fetchone()
        return self._row_to_version(row) if row is not None else None

    def list_versions(
        self,
        tenant_id: str,
        role_type: str | None = None,
    ) -> list[ResumeVersion]:
        query = "SELECT * FROM resume_versions WHERE tenant_id = ?"
        params: list[str] = [tenant_id]
        if role_type is not None:
            query += " AND role_type = ?"
            params.append(role_type)
        query += " ORDER BY role_type, version_number"

        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [self._row_to_version(row) for row in rows]

    def list_role_types(self, tenant_id: str) -> list[str]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT DISTINCT role_type
                FROM resume_versions
                WHERE tenant_id = ?
                ORDER BY role_type
                """,
                (tenant_id,),
            ).fetchall()
        return [str(row["role_type"]) for row in rows]

    def get_version(
        self,
        tenant_id: str,
        version_id: str,
    ) -> ResumeVersion | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT *
                FROM resume_versions
                WHERE tenant_id = ? AND version_id = ?
                """,
                (tenant_id, version_id),
            ).fetchone()
        return self._row_to_version(row) if row is not None else None

    def set_active_version(
        self,
        tenant_id: str,
        version_id: str,
    ) -> None:
        if self.get_version(tenant_id, version_id) is None:
            raise ResumeNotFoundError(f"Resume version not found for tenant {tenant_id!r}: {version_id}")

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO resume_tenant_state (tenant_id, active_version_id)
                VALUES (?, ?)
                ON CONFLICT(tenant_id)
                DO UPDATE SET active_version_id = excluded.active_version_id
                """,
                (tenant_id, version_id),
            )

    def get_active_version(self, tenant_id: str) -> ResumeVersion | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT versions.*
                FROM resume_tenant_state AS state
                JOIN resume_versions AS versions
                  ON versions.version_id = state.active_version_id
                 AND versions.tenant_id = state.tenant_id
                WHERE state.tenant_id = ?
                """,
                (tenant_id,),
            ).fetchone()
        return self._row_to_version(row) if row is not None else None
