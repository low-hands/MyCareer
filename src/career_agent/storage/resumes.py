from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import sqlite3
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict

from career_agent.domain.resume import Resume, ResumeVersion, TargetRole
from career_agent.storage.schema import apply_schema


class StoredResumeDocument(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    resume_version_id: str
    document_format: Literal["pdf", "text", "markdown"]
    raw_bytes: bytes


class ResumeStore:
    def __init__(self, path: Path) -> None:
        self.path = path.expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        os.chmod(self.path.parent, 0o700)
        with self._connect() as connection:
            apply_schema(
                connection,
                "resumes",
                4,
                self._migrate,
                upgrades={4: self._add_target_role_intent_columns},
            )
        os.chmod(self.path, 0o600)

    def create_target_role(self, *, user_id: str, title: str, priority: int) -> TargetRole:
        if not user_id.strip() or not title.strip():
            raise ValueError("user_id and target role title are required.")
        now = datetime.now(timezone.utc)
        role = TargetRole(id=f"target_role_{uuid4().hex}", user_id=user_id, title=title.strip(), priority=priority, created_at=now, updated_at=now)
        with self._connect() as connection:
            connection.execute("INSERT INTO target_roles(id, user_id, title, priority, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)", (role.id, role.user_id, role.title, role.priority, role.status, role.created_at.isoformat(), role.updated_at.isoformat()))
        return role

    def list_target_roles(self, *, user_id: str) -> tuple[TargetRole, ...]:
        with self._connect() as connection:
            rows = connection.execute("SELECT id, user_id, title, priority, status, city, salary_expectation, experience, education, created_at, updated_at FROM target_roles WHERE user_id = ? ORDER BY priority, created_at", (user_id,)).fetchall()
        return tuple(self._role(row) for row in rows)

    def get_target_role(self, *, user_id: str, target_role_id: str) -> TargetRole | None:
        with self._connect() as connection:
            row = connection.execute("SELECT id, user_id, title, priority, status, city, salary_expectation, experience, education, created_at, updated_at FROM target_roles WHERE id = ? AND user_id = ?", (target_role_id, user_id)).fetchone()
        return self._role(row) if row else None

    def update_target_role_intent(
        self,
        *,
        user_id: str,
        target_role_id: str,
        city: str | None = None,
        salary_expectation: str | None = None,
        experience: str | None = None,
        education: str | None = None,
    ) -> TargetRole:
        """Overwrite only the intent fields that were given.

        A user naming a salary this turn has not withdrawn the city they named
        last week, so None means "leave alone" rather than "clear".
        """
        role = self.get_target_role(user_id=user_id, target_role_id=target_role_id)
        if role is None:
            raise ValueError("target role not found or does not belong to the user")
        changes = {
            key: value
            for key, value in (
                ("city", city),
                ("salary_expectation", salary_expectation),
                ("experience", experience),
                ("education", education),
            )
            if value is not None
        }
        if not changes:
            return role
        updated = role.model_copy(
            update={**changes, "updated_at": datetime.now(timezone.utc)}
        )
        assignments = ", ".join(f"{key} = ?" for key in changes)
        with self._connect() as connection:
            connection.execute(
                f"UPDATE target_roles SET {assignments}, updated_at = ? WHERE id = ? AND user_id = ?",
                (*changes.values(), updated.updated_at.isoformat(), target_role_id, user_id),
            )
        return updated

    def import_document(self, *, user_id: str, content: bytes, document_format: str, name: str | None = None, resume_id: str | None = None, target_role_id: str | None = None) -> tuple[Resume, ResumeVersion]:
        if bool(name) == bool(resume_id):
            raise ValueError("Provide exactly one of name or resume_id.")
        if document_format not in {"pdf", "text", "markdown"}:
            raise ValueError("Unsupported resume document format.")
        if not user_id.strip():
            raise ValueError("user_id is required.")
        if name and not target_role_id:
            raise ValueError("A target_role_id is required for a new resume.")
        if resume_id and target_role_id:
            raise ValueError("target_role_id cannot change when appending a version.")
        now = datetime.now(timezone.utc)
        digest = hashlib.sha256(content).hexdigest()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if resume_id:
                row = connection.execute("SELECT id, user_id, target_role_id, name, status, latest_version_id, created_at, updated_at FROM resumes WHERE id = ? AND user_id = ?", (resume_id, user_id)).fetchone()
                if row is None:
                    raise ValueError("Resume not found.")
                resume = self._resume(row)
                version_number = connection.execute("SELECT COALESCE(MAX(version_number), 0) + 1 FROM resume_versions WHERE resume_id = ?", (resume_id,)).fetchone()[0]
            else:
                target = connection.execute("SELECT id FROM target_roles WHERE id = ? AND user_id = ?", (target_role_id, user_id)).fetchone()
                if target is None:
                    raise ValueError("Target role not found.")
                resume_id = f"resume_{uuid4().hex}"
                resume = Resume(id=resume_id, user_id=user_id, target_role_id=target_role_id, name=name.strip(), latest_version_id="pending", created_at=now, updated_at=now)
                version_number = 1
            version = ResumeVersion(id=f"resume_version_{uuid4().hex}", resume_id=resume_id, version_number=version_number, document_format=document_format, content_sha256=digest, byte_size=len(content), created_at=now)
            resume = resume.model_copy(update={"latest_version_id": version.id, "updated_at": now})
            if version_number == 1:
                connection.execute("INSERT INTO resumes(id, user_id, target_role_id, name, status, latest_version_id, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (resume.id, resume.user_id, resume.target_role_id, resume.name, resume.status, resume.latest_version_id, resume.created_at.isoformat(), resume.updated_at.isoformat()))
            else:
                connection.execute("UPDATE resumes SET latest_version_id = ?, updated_at = ? WHERE id = ? AND user_id = ?", (version.id, now.isoformat(), resume.id, user_id))
            connection.execute("INSERT INTO resume_versions(id, resume_id, version_number, source_type, document_format, content_sha256, byte_size, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (version.id, version.resume_id, version.version_number, version.source_type, version.document_format, version.content_sha256, version.byte_size, version.created_at.isoformat()))
            connection.execute("INSERT INTO resume_version_documents(resume_version_id, content) VALUES (?, ?)", (version.id, content))
        os.chmod(self.path, 0o600)
        return resume, version

    def list_resumes(self, *, user_id: str, target_role_id: str | None = None) -> tuple[Resume, ...]:
        query = "SELECT id, user_id, target_role_id, name, status, latest_version_id, created_at, updated_at FROM resumes WHERE user_id = ?"
        params: tuple[str, ...] = (user_id,)
        if target_role_id:
            query += " AND target_role_id = ?"
            params += (target_role_id,)
        query += " ORDER BY updated_at DESC"
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return tuple(self._resume(row) for row in rows)

    def get_resume(self, *, user_id: str, resume_id: str) -> Resume | None:
        with self._connect() as connection:
            row = connection.execute("SELECT id, user_id, target_role_id, name, status, latest_version_id, created_at, updated_at FROM resumes WHERE id = ? AND user_id = ?", (resume_id, user_id)).fetchone()
        return self._resume(row) if row else None

    def list_versions(self, *, user_id: str, resume_id: str) -> tuple[ResumeVersion, ...]:
        with self._connect() as connection:
            rows = connection.execute("SELECT v.id, v.resume_id, v.version_number, v.source_type, v.document_format, v.content_sha256, v.byte_size, v.created_at FROM resume_versions v JOIN resumes r ON r.id = v.resume_id WHERE v.resume_id = ? AND r.user_id = ? ORDER BY v.version_number DESC", (resume_id, user_id)).fetchall()
        return tuple(self._version(row) for row in rows)

    def get_version(
        self, *, user_id: str, resume_version_id: str
    ) -> tuple[Resume, ResumeVersion] | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT resume.id, resume.user_id, resume.target_role_id,
                       resume.name, resume.status, resume.latest_version_id,
                       resume.created_at, resume.updated_at,
                       version.id, version.resume_id, version.version_number,
                       version.source_type, version.document_format,
                       version.content_sha256, version.byte_size, version.created_at
                FROM resume_versions AS version
                JOIN resumes AS resume ON resume.id = version.resume_id
                WHERE version.id = ? AND resume.user_id = ?
                """,
                (resume_version_id, user_id),
            ).fetchone()
        if row is None:
            return None
        return self._resume(row[:8]), self._version(row[8:])

    def get_tailored_version(
        self, *, user_id: str, tailoring_draft_id: str
    ) -> tuple[Resume, ResumeVersion] | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT resume.id, resume.user_id, resume.target_role_id,
                       resume.name, resume.status, resume.latest_version_id,
                       resume.created_at, resume.updated_at,
                       version.id, version.resume_id, version.version_number,
                       version.source_type, version.document_format,
                       version.content_sha256, version.byte_size, version.created_at
                FROM resume_tailoring_version_links AS link
                JOIN resume_versions AS version
                  ON version.id = link.new_resume_version_id
                JOIN resumes AS resume ON resume.id = version.resume_id
                WHERE link.tailoring_draft_id = ? AND link.user_id = ?
                """,
                (tailoring_draft_id, user_id),
            ).fetchone()
        if row is None:
            return None
        return self._resume(row[:8]), self._version(row[8:])

    def create_tailored_version(
        self,
        *,
        user_id: str,
        source_resume_version_id: str,
        tailoring_draft_id: str,
        markdown: str,
    ) -> tuple[Resume, ResumeVersion]:
        if not user_id.strip() or not source_resume_version_id.strip() or not tailoring_draft_id.strip():
            raise ValueError(
                "user_id, source_resume_version_id, and tailoring_draft_id are required"
            )
        normalized = markdown.strip()
        if not normalized:
            raise ValueError("Tailored resume Markdown must not be empty")
        content = normalized.encode("utf-8")
        now = datetime.now(timezone.utc)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                """
                SELECT new_resume_version_id
                FROM resume_tailoring_version_links
                WHERE tailoring_draft_id = ? AND user_id = ?
                """,
                (tailoring_draft_id, user_id),
            ).fetchone()
            if existing is not None:
                version_row = connection.execute(
                    """
                    SELECT id, resume_id, version_number, source_type,
                           document_format, content_sha256, byte_size, created_at
                    FROM resume_versions WHERE id = ?
                    """,
                    (existing[0],),
                ).fetchone()
                resume_row = connection.execute(
                    """
                    SELECT id, user_id, target_role_id, name, status,
                           latest_version_id, created_at, updated_at
                    FROM resumes WHERE id = ? AND user_id = ?
                    """,
                    (version_row[1], user_id),
                ).fetchone()
                return self._resume(resume_row), self._version(version_row)

            source = connection.execute(
                """
                SELECT version.resume_id
                FROM resume_versions AS version
                JOIN resumes AS resume ON resume.id = version.resume_id
                WHERE version.id = ? AND resume.user_id = ?
                """,
                (source_resume_version_id, user_id),
            ).fetchone()
            if source is None:
                raise ValueError("Source resume version not found")
            resume_id = source[0]
            version_number = connection.execute(
                """
                SELECT COALESCE(MAX(version_number), 0) + 1
                FROM resume_versions WHERE resume_id = ?
                """,
                (resume_id,),
            ).fetchone()[0]
            version = ResumeVersion(
                id=f"resume_version_{uuid4().hex}",
                resume_id=resume_id,
                version_number=version_number,
                source_type="agent_tailoring",
                document_format="markdown",
                content_sha256=hashlib.sha256(content).hexdigest(),
                byte_size=len(content),
                created_at=now,
            )
            connection.execute(
                """
                INSERT INTO resume_versions(
                    id, resume_id, version_number, source_type, document_format,
                    content_sha256, byte_size, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    version.id,
                    version.resume_id,
                    version.version_number,
                    version.source_type,
                    version.document_format,
                    version.content_sha256,
                    version.byte_size,
                    version.created_at.isoformat(),
                ),
            )
            connection.execute(
                """
                INSERT INTO resume_version_documents(resume_version_id, content)
                VALUES (?, ?)
                """,
                (version.id, content),
            )
            connection.execute(
                """
                INSERT INTO resume_tailoring_version_links(
                    tailoring_draft_id, user_id, source_resume_version_id,
                    new_resume_version_id, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    tailoring_draft_id,
                    user_id,
                    source_resume_version_id,
                    version.id,
                    now.isoformat(),
                ),
            )
            connection.execute(
                """
                UPDATE resumes SET latest_version_id = ?, updated_at = ?
                WHERE id = ? AND user_id = ?
                """,
                (version.id, now.isoformat(), resume_id, user_id),
            )
            resume_row = connection.execute(
                """
                SELECT id, user_id, target_role_id, name, status,
                       latest_version_id, created_at, updated_at
                FROM resumes WHERE id = ? AND user_id = ?
                """,
                (resume_id, user_id),
            ).fetchone()
        return self._resume(resume_row), version

    def read_version_document(
        self, *, user_id: str, resume_version_id: str
    ) -> StoredResumeDocument | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT version.id, version.document_format, document.content
                FROM resume_version_documents AS document
                JOIN resume_versions AS version
                  ON version.id = document.resume_version_id
                JOIN resumes AS resume
                  ON resume.id = version.resume_id
                WHERE version.id = ? AND resume.user_id = ?
                """,
                (resume_version_id, user_id),
            ).fetchone()
        if row is None:
            return None
        return StoredResumeDocument(
            resume_version_id=row[0],
            document_format=row[1],
            raw_bytes=bytes(row[2]),
        )

    @staticmethod
    def _add_target_role_intent_columns(connection: sqlite3.Connection) -> None:
        """Version 4: job intent moved from the person onto each target role.

        The baseline creates these columns for a fresh file; a database already
        registered at version 3 has the table without them and needs the ALTER.
        """
        existing = {
            row[1]
            for row in connection.execute("PRAGMA table_info(target_roles)").fetchall()
        }
        for column in ("city", "salary_expectation", "experience", "education"):
            if column not in existing:
                connection.execute(
                    f"ALTER TABLE target_roles ADD COLUMN {column} TEXT"
                )

    def _migrate(self, connection: sqlite3.Connection) -> None:
        has_resumes = connection.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'resumes'").fetchone() is not None
        if not has_resumes:
            connection.execute("CREATE TABLE resumes (id TEXT PRIMARY KEY, user_id TEXT NOT NULL, target_role_id TEXT NOT NULL, name TEXT NOT NULL, status TEXT NOT NULL, latest_version_id TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL)")
            connection.execute("CREATE TABLE resume_versions (id TEXT PRIMARY KEY, resume_id TEXT NOT NULL REFERENCES resumes(id), version_number INTEGER NOT NULL, source_type TEXT NOT NULL, document_format TEXT NOT NULL, content_sha256 TEXT NOT NULL, byte_size INTEGER NOT NULL, created_at TEXT NOT NULL, UNIQUE(resume_id, version_number))")
            connection.execute("CREATE TABLE resume_version_documents (resume_version_id TEXT PRIMARY KEY REFERENCES resume_versions(id), content BLOB NOT NULL)")
        connection.execute("CREATE TABLE IF NOT EXISTS target_roles (id TEXT PRIMARY KEY, user_id TEXT NOT NULL, title TEXT NOT NULL, priority INTEGER NOT NULL, status TEXT NOT NULL, city TEXT, salary_expectation TEXT, experience TEXT, education TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL, UNIQUE(user_id, title))")
        columns = {row[1] for row in connection.execute("PRAGMA table_info(resumes)").fetchall()}
        if "target_role_id" not in columns:
            connection.execute("ALTER TABLE resumes ADD COLUMN target_role_id TEXT")
        users = connection.execute("SELECT DISTINCT user_id FROM resumes WHERE target_role_id IS NULL").fetchall()
        now = datetime.now(timezone.utc).isoformat()
        for (user_id,) in users:
            role_id = f"target_role_unassigned_{hashlib.sha256(user_id.encode()).hexdigest()[:24]}"
            connection.execute("INSERT OR IGNORE INTO target_roles(id, user_id, title, priority, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)", (role_id, user_id, "Unassigned", 9999, "active", now, now))
            connection.execute("UPDATE resumes SET target_role_id = ? WHERE user_id = ? AND target_role_id IS NULL", (role_id, user_id))
        connection.execute("CREATE INDEX IF NOT EXISTS resumes_user_role_updated_idx ON resumes(user_id, target_role_id, updated_at DESC)")
        connection.execute("CREATE INDEX IF NOT EXISTS target_roles_user_priority_idx ON target_roles(user_id, priority, created_at)")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS resume_tailoring_version_links (
                tailoring_draft_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                source_resume_version_id TEXT NOT NULL,
                new_resume_version_id TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL,
                FOREIGN KEY(source_resume_version_id) REFERENCES resume_versions(id),
                FOREIGN KEY(new_resume_version_id) REFERENCES resume_versions(id)
            )
            """
        )
        # user_version is per-file and this file has seven owners, so it cannot
        # describe any one of them. Keep writing it for backward compatibility
        # with files created before the registry existed.
        connection.execute("PRAGMA user_version = 3")

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    @staticmethod
    def _role(row: tuple) -> TargetRole:
        return TargetRole(
            id=row[0], user_id=row[1], title=row[2], priority=row[3], status=row[4],
            city=row[5], salary_expectation=row[6], experience=row[7],
            education=row[8], created_at=row[9], updated_at=row[10],
        )

    @staticmethod
    def _resume(row: tuple) -> Resume:
        return Resume(id=row[0], user_id=row[1], target_role_id=row[2], name=row[3], status=row[4], latest_version_id=row[5], created_at=row[6], updated_at=row[7])

    @staticmethod
    def _version(row: tuple) -> ResumeVersion:
        return ResumeVersion(id=row[0], resume_id=row[1], version_number=row[2], source_type=row[3], document_format=row[4], content_sha256=row[5], byte_size=row[6], created_at=row[7])
