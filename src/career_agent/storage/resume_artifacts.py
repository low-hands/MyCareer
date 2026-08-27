from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import sqlite3
from uuid import uuid4

from career_agent.domain.resume import ResumeArtifactReference
from career_agent.storage.schema import apply_schema


class SQLiteResumeArtifactStore:
    """Stores delivery references; document bytes remain in ResumeStore."""

    def __init__(self, path: Path) -> None:
        self.path = path.expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        os.chmod(self.path.parent, 0o700)
        with self._connect() as connection:
            apply_schema(connection, "resume_artifacts", 1, self._migrate)
        os.chmod(self.path, 0o600)

    def create(
        self,
        *,
        user_id: str,
        resume_version_id: str,
        filename: str,
        media_type: str,
        byte_size: int,
    ) -> ResumeArtifactReference:
        if not all((user_id.strip(), resume_version_id.strip(), filename.strip(), media_type.strip())):
            raise ValueError("Artifact owner, version, filename, and media type are required")
        if byte_size < 1:
            raise ValueError("Artifact byte size must be positive")
        now = datetime.now(timezone.utc)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                """
                SELECT id, user_id, resume_version_id, filename, media_type,
                       byte_size, created_at
                FROM resume_artifacts
                WHERE user_id = ? AND resume_version_id = ?
                """,
                (user_id, resume_version_id),
            ).fetchone()
            if existing is not None:
                return self._record(existing)
            owned = connection.execute(
                """
                SELECT 1
                FROM resume_versions AS version
                JOIN resumes AS resume ON resume.id = version.resume_id
                WHERE version.id = ? AND resume.user_id = ?
                """,
                (resume_version_id, user_id),
            ).fetchone()
            if owned is None:
                raise ValueError("Resume version not found for artifact owner")
            artifact = ResumeArtifactReference(
                id=f"resume_artifact_{uuid4().hex}",
                user_id=user_id,
                resume_version_id=resume_version_id,
                filename=filename.strip(),
                media_type=media_type.strip(),
                byte_size=byte_size,
                created_at=now,
            )
            connection.execute(
                """
                INSERT INTO resume_artifacts(
                    id, user_id, resume_version_id, filename,
                    media_type, byte_size, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    artifact.id,
                    artifact.user_id,
                    artifact.resume_version_id,
                    artifact.filename,
                    artifact.media_type,
                    artifact.byte_size,
                    artifact.created_at.isoformat(),
                ),
            )
        return artifact

    def get(
        self, *, user_id: str, artifact_id: str
    ) -> ResumeArtifactReference | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT id, user_id, resume_version_id, filename, media_type,
                       byte_size, created_at
                FROM resume_artifacts
                WHERE id = ? AND user_id = ?
                """,
                (artifact_id, user_id),
            ).fetchone()
        return self._record(row) if row else None

    def _migrate(self, connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS resume_artifacts (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                resume_version_id TEXT NOT NULL,
                filename TEXT NOT NULL,
                media_type TEXT NOT NULL,
                byte_size INTEGER NOT NULL CHECK(byte_size > 0),
                created_at TEXT NOT NULL,
                UNIQUE(user_id, resume_version_id),
                FOREIGN KEY(resume_version_id) REFERENCES resume_versions(id)
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS resume_artifacts_user_created_idx
            ON resume_artifacts(user_id, created_at DESC)
            """
        )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    @staticmethod
    def _record(row: tuple[object, ...]) -> ResumeArtifactReference:
        return ResumeArtifactReference(
            id=row[0],
            user_id=row[1],
            resume_version_id=row[2],
            filename=row[3],
            media_type=row[4],
            byte_size=row[5],
            created_at=row[6],
        )
