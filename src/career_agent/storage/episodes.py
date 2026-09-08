from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
from uuid import uuid4

from career_agent.domain.episodes import (
    CareerEpisode,
    CareerEpisodeDraft,
    EpisodeResourceRef,
)
from career_agent.storage.schema import apply_schema

_OPAQUE_LINEAGE_MARKER = re.compile(
    r"^(?:detail|evidence|lineage)_[a-f0-9]{24}$"
)


def _usable_lineage_markers(markers: tuple[str, ...]) -> tuple[str, ...]:
    usable: list[str] = []
    for marker in markers:
        text = marker.strip()
        if not text:
            continue
        if _OPAQUE_LINEAGE_MARKER.fullmatch(text):
            usable.append(text)
    return tuple(dict.fromkeys(usable))


def apply_episode_schema(connection: sqlite3.Connection) -> None:
    """Adopt the episodic projection in a context database."""

    apply_schema(
        connection,
        "career_episodes",
        6,
        SQLiteCareerEpisodeStore._baseline,
        {
            2: SQLiteCareerEpisodeStore._upgrade_to_v2,
            3: SQLiteCareerEpisodeStore._upgrade_to_v3,
            4: SQLiteCareerEpisodeStore._upgrade_to_v4,
            5: SQLiteCareerEpisodeStore._upgrade_to_v5,
            6: SQLiteCareerEpisodeStore._upgrade_to_v6,
        },
    )


class SQLiteCareerEpisodeStore:
    """Time-ordered L1 retrieval records backed by the context database."""

    def __init__(self, path: Path) -> None:
        self.path = path.expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        os.chmod(self.path.parent, 0o700)
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            apply_episode_schema(connection)
        os.chmod(self.path, 0o600)

    def upsert(
        self,
        draft: CareerEpisodeDraft,
        *,
        memory_scope_keys: tuple[str, ...] = (),
    ) -> CareerEpisode | None:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            episode = self.upsert_on(
                connection,
                draft,
                memory_scope_keys=memory_scope_keys,
            )
        os.chmod(self.path, 0o600)
        return episode

    def upsert_many(
        self,
        drafts: tuple[CareerEpisodeDraft, ...],
        *,
        memory_scope_keys: tuple[str, ...] = (),
    ) -> tuple[CareerEpisode, ...]:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            episodes = tuple(
                episode
                for draft in drafts
                if (
                    episode := self.upsert_on(
                        connection,
                        draft,
                        memory_scope_keys=memory_scope_keys,
                    )
                )
                is not None
            )
        os.chmod(self.path, 0o600)
        return episodes

    @staticmethod
    def upsert_on(
        connection: sqlite3.Connection,
        draft: CareerEpisodeDraft,
        *,
        memory_scope_keys: tuple[str, ...] = (),
    ) -> CareerEpisode | None:
        """Upsert on a caller-owned transaction.

        This is used by ``CareerContextStore`` so the episode and its visible
        conversation seam commit together.
        """

        content_digest = SQLiteCareerEpisodeStore._content_digest(draft)
        deleted_scope_keys = tuple(dict.fromkeys(memory_scope_keys))
        scope_placeholders = ",".join("?" for _ in deleted_scope_keys)
        stale_scope = bool(deleted_scope_keys) and connection.execute(
            f"""
            SELECT 1 FROM career_episode_deleted_scopes
            WHERE user_id = ? AND scope_key IN ({scope_placeholders})
            LIMIT 1
            """,
            (draft.user_id, *deleted_scope_keys),
        ).fetchone()
        if stale_scope or connection.execute(
            """
            SELECT 1
            FROM career_episode_content_suppressions
            WHERE user_id = ? AND kind = ? AND source_run_id = ?
              AND content_digest = ?
            LIMIT 1
            """,
            (
                draft.user_id,
                draft.kind,
                draft.source_run_id,
                content_digest,
            ),
        ).fetchone():
            return None
        now = datetime.now(timezone.utc).isoformat()
        episode_id = f"career_episode_{uuid4().hex}"
        refs_json = json.dumps(
            [reference.model_dump(mode="json") for reference in draft.resource_refs],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        connection.execute(
            """
            INSERT INTO career_episodes(
                id, user_id, kind, source_run_id, occurred_at, title, summary,
                conversation_id, resource_refs_json, salience,
                last_accessed_at, access_count, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1.0, NULL, 0, ?, ?)
            ON CONFLICT(user_id, kind, source_run_id) DO UPDATE SET
                occurred_at = excluded.occurred_at,
                title = excluded.title,
                summary = excluded.summary,
                conversation_id = COALESCE(
                    excluded.conversation_id, career_episodes.conversation_id
                ),
                resource_refs_json = excluded.resource_refs_json,
                updated_at = excluded.updated_at
            """,
            (
                episode_id,
                draft.user_id,
                draft.kind,
                draft.source_run_id,
                draft.occurred_at.isoformat(),
                draft.title,
                draft.summary,
                draft.conversation_id,
                refs_json,
                now,
                now,
            ),
        )
        row = connection.execute(
            SQLiteCareerEpisodeStore._SELECT
            + " WHERE user_id = ? AND kind = ? AND source_run_id = ?",
            (draft.user_id, draft.kind, draft.source_run_id),
        ).fetchone()
        if row is None:
            raise RuntimeError("episode upsert lookup failed")
        episode = SQLiteCareerEpisodeStore._episode(row)
        connection.executemany(
            """
            INSERT OR IGNORE INTO career_episode_memory_bindings(
                episode_id, user_id, scope_key, created_at
            ) VALUES (?, ?, ?, ?)
            """,
            (
                (episode.id, episode.user_id, scope_key, now)
                for scope_key in dict.fromkeys(memory_scope_keys)
            ),
        )
        connection.execute(
            "DELETE FROM career_episodes_fts WHERE episode_id = ?",
            (episode.id,),
        )
        connection.execute(
            """
            INSERT INTO career_episodes_fts(
                episode_id, user_id, title, summary
            ) VALUES (?, ?, ?, ?)
            """,
            (episode.id, episode.user_id, episode.title, episode.summary),
        )
        SQLiteCareerEpisodeStore._replace_short_terms(connection, episode)
        return episode

    def get_by_source(
        self, *, user_id: str, kind: str, source_run_id: str
    ) -> CareerEpisode | None:
        with self._connect() as connection:
            row = connection.execute(
                self._SELECT
                + " WHERE user_id = ? AND kind = ? AND source_run_id = ?",
                (user_id, kind, source_run_id),
            ).fetchone()
        return self._episode(row) if row else None

    def has_source(
        self, *, user_id: str, kind: str, source_run_id: str
    ) -> bool:
        return (
            self.get_by_source(
                user_id=user_id,
                kind=kind,
                source_run_id=source_run_id,
            )
            is not None
        )

    def list_source_keys(self, *, user_id: str) -> frozenset[tuple[str, str]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT kind, source_run_id FROM career_episodes WHERE user_id = ?",
                (user_id,),
            ).fetchall()
        return frozenset((str(row[0]), str(row[1])) for row in rows)

    @staticmethod
    def delete_for_scope_on(
        connection: sqlite3.Connection,
        *,
        user_id: str,
        scope_key: str,
        lineage_markers: tuple[str, ...] = (),
    ) -> int:
        """Delete only episodes whose derivation observed one memory lineage."""

        now = datetime.now(timezone.utc).isoformat()
        connection.execute(
            """
            INSERT INTO career_episode_deleted_scopes(
                user_id, scope_key, deleted_at
            ) VALUES (?, ?, ?)
            ON CONFLICT(user_id, scope_key) DO NOTHING
            """,
            (user_id, scope_key, now),
        )
        markers = _usable_lineage_markers(lineage_markers)
        marker_clause = ""
        marker_parameters: tuple[str, ...] = ()
        if markers:
            marker_predicate = " OR ".join(
                "(instr(COALESCE(e.title, ''), ?) > 0 "
                "OR instr(COALESCE(e.summary, ''), ?) > 0)"
                for _ in markers
            )
            marker_clause = f"""
                    OR (
                        NOT EXISTS (
                            SELECT 1
                            FROM career_episode_memory_bindings AS any_binding
                            WHERE any_binding.user_id = e.user_id
                              AND any_binding.episode_id = e.id
                        )
                        AND ({marker_predicate})
                    )
            """
            marker_parameters = tuple(
                marker for marker in markers for _ in range(2)
            )
        rows = connection.execute(
            f"""
            SELECT e.id, e.kind, e.source_run_id, e.occurred_at, e.title,
                   e.summary, e.conversation_id, e.resource_refs_json
            FROM career_episodes AS e
            WHERE e.user_id = ?
              AND (
                    EXISTS (
                        SELECT 1
                        FROM career_episode_memory_bindings AS binding
                        WHERE binding.user_id = e.user_id
                          AND binding.episode_id = e.id
                          AND binding.scope_key = ?
                    )
                    {marker_clause}
                  )
            """,
            (user_id, scope_key, *marker_parameters),
        ).fetchall()
        if not rows:
            return 0
        ids = tuple(str(row[0]) for row in rows)
        connection.executemany(
            """
            INSERT OR IGNORE INTO career_episode_content_suppressions(
                user_id, kind, source_run_id, content_digest,
                scope_key, deleted_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                (
                    user_id,
                    str(row[1]),
                    str(row[2]),
                    SQLiteCareerEpisodeStore._stored_content_digest(row),
                    scope_key,
                    now,
                )
                for row in rows
            ),
        )
        placeholders = ",".join("?" for _ in ids)
        connection.execute(
            f"DELETE FROM career_episodes_fts WHERE episode_id IN ({placeholders})",
            ids,
        )
        connection.execute(
            f"DELETE FROM career_episode_short_terms WHERE episode_id IN ({placeholders})",
            ids,
        )
        connection.execute(
            f"DELETE FROM career_episode_memory_bindings WHERE episode_id IN ({placeholders})",
            ids,
        )
        connection.execute(
            f"DELETE FROM career_episodes WHERE id IN ({placeholders})",
            ids,
        )
        return len(ids)

    def search(
        self, *, user_id: str, query: str, limit: int = 20
    ) -> tuple[CareerEpisode, ...]:
        if limit < 1:
            raise ValueError("limit must be positive")
        normalized_query = query.strip()
        tokens = tuple(
            dict.fromkeys(
                token.casefold()
                for token in re.findall(
                    r"[\w+#.-]+",
                    normalized_query,
                    flags=re.UNICODE,
                )
            )
        )
        short_tokens = tuple(token for token in tokens if len(token) < 3)
        long_tokens = tuple(token for token in tokens if len(token) >= 3)
        with self._connect() as connection:
            if not normalized_query:
                rows = connection.execute(
                    self._SELECT
                    + " WHERE user_id = ? ORDER BY occurred_at DESC, id DESC LIMIT ?",
                    (user_id, limit),
                ).fetchall()
            elif tokens:
                ctes: list[str] = []
                hit_selects: list[str] = []
                parameters: list[object] = []
                if long_tokens:
                    ctes.append(
                        """
                        long_hits AS MATERIALIZED (
                            SELECT episode_id,
                                   bm25(
                                       career_episodes_fts,
                                       0.0, 0.0, 4.0, 1.0
                                   ) AS relevance
                            FROM career_episodes_fts
                            WHERE user_id = ?
                              AND career_episodes_fts MATCH ?
                        )
                        """
                    )
                    parameters.extend(
                        (
                            user_id,
                            " OR ".join(
                                json.dumps(token, ensure_ascii=False)
                                for token in long_tokens
                            ),
                        )
                    )
                    hit_selects.append(
                        "SELECT episode_id, relevance AS long_relevance, "
                        "0 AS short_score FROM long_hits"
                    )
                if short_tokens:
                    placeholders = ",".join("?" for _ in short_tokens)
                    ctes.append(
                        f"""
                        short_hits AS (
                            SELECT episode_id,
                                   SUM(title_count * 4 + summary_count)
                                       AS short_score
                            FROM career_episode_short_terms
                            WHERE user_id = ? AND term IN ({placeholders})
                            GROUP BY episode_id
                        )
                        """
                    )
                    parameters.extend((user_id, *short_tokens))
                    hit_selects.append(
                        "SELECT episode_id, NULL AS long_relevance, "
                        "short_score FROM short_hits"
                    )
                ctes.append(
                    """
                    hits AS (
                        SELECT episode_id,
                               MIN(long_relevance) AS long_relevance,
                               SUM(short_score) AS short_score
                        FROM (
                    """
                    + " UNION ALL ".join(hit_selects)
                    + """
                        )
                        GROUP BY episode_id
                    )
                    """
                )
                rows = connection.execute(
                    "WITH "
                    + ", ".join(ctes)
                    + " "
                    + self._SELECT
                    + """
                    JOIN hits ON hits.episode_id = e.id
                    WHERE e.user_id = ?
                    ORDER BY
                        CASE
                            WHEN lower(e.title) = lower(?) THEN 0
                            WHEN e.title LIKE ? THEN 1
                            ELSE 2
                        END,
                        COALESCE(hits.short_score, 0) DESC,
                        COALESCE(hits.long_relevance, 1000000.0),
                        e.occurred_at DESC
                    LIMIT ?
                    """,
                    (
                        *parameters,
                        user_id,
                        normalized_query,
                        f"{normalized_query}%",
                        limit,
                    ),
                ).fetchall()
            else:
                rows = []
        return tuple(self._episode(row) for row in rows)

    @staticmethod
    def _baseline(connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS career_episodes (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                kind TEXT NOT NULL CHECK(kind IN (
                    'mock_interview', 'job_research',
                    'application', 'interview_round'
                )),
                source_run_id TEXT NOT NULL CHECK(length(source_run_id) > 0),
                occurred_at TEXT NOT NULL,
                title TEXT NOT NULL,
                summary TEXT NOT NULL,
                conversation_id TEXT,
                resource_refs_json TEXT NOT NULL DEFAULT '[]',
                salience REAL NOT NULL DEFAULT 1.0,
                last_accessed_at TEXT,
                access_count INTEGER NOT NULL DEFAULT 0 CHECK(access_count >= 0),
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(user_id, kind, source_run_id)
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS career_episodes_user_time_idx
            ON career_episodes(user_id, occurred_at DESC)
            """
        )
        SQLiteCareerEpisodeStore._create_fts(connection)
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS career_episode_short_terms (
                episode_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                term TEXT NOT NULL,
                title_count INTEGER NOT NULL CHECK(title_count >= 0),
                summary_count INTEGER NOT NULL CHECK(summary_count >= 0),
                PRIMARY KEY(episode_id, term)
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS career_episode_short_terms_lookup_idx
            ON career_episode_short_terms(user_id, term)
            """
        )
        SQLiteCareerEpisodeStore._ensure_deletion_schema(connection)

    @staticmethod
    def _create_fts(connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS career_episodes_fts USING fts5(
                episode_id UNINDEXED,
                user_id UNINDEXED,
                title,
                summary,
                tokenize='trigram'
            )
            """
        )

    @staticmethod
    def _upgrade_to_v2(connection: sqlite3.Connection) -> None:
        """Rebuild the index so Chinese substrings participate in BM25."""

        connection.execute("DROP TABLE IF EXISTS career_episodes_fts")
        SQLiteCareerEpisodeStore._create_fts(connection)
        connection.execute(
            """
            INSERT INTO career_episodes_fts(
                episode_id, user_id, title, summary
            )
            SELECT id, user_id, title, summary
            FROM career_episodes
            """
        )

    @staticmethod
    def _upgrade_to_v3(connection: sqlite3.Connection) -> None:
        rows = connection.execute(
            SQLiteCareerEpisodeStore._SELECT
        ).fetchall()
        for row in rows:
            SQLiteCareerEpisodeStore._replace_short_terms(
                connection,
                SQLiteCareerEpisodeStore._episode(row),
            )

    @staticmethod
    def _upgrade_to_v4(connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS career_episode_deletion_cutoffs (
                user_id TEXT PRIMARY KEY,
                through_occurred_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS career_episode_deletion_suppressions (
                user_id TEXT NOT NULL,
                kind TEXT NOT NULL,
                source_run_id TEXT NOT NULL,
                deleted_at TEXT NOT NULL,
                PRIMARY KEY(user_id, kind, source_run_id)
            )
            """
        )

    @staticmethod
    def _upgrade_to_v5(connection: sqlite3.Connection) -> None:
        SQLiteCareerEpisodeStore._ensure_deletion_schema(connection)

    @staticmethod
    def _upgrade_to_v6(connection: sqlite3.Connection) -> None:
        connection.execute("DROP TABLE IF EXISTS career_episode_deletion_cutoffs")
        connection.execute(
            "DROP TABLE IF EXISTS career_episode_deletion_suppressions"
        )

    @staticmethod
    def _ensure_deletion_schema(connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS career_episode_memory_bindings (
                episode_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                scope_key TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY(episode_id, scope_key)
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS career_episode_memory_scope_idx
            ON career_episode_memory_bindings(user_id, scope_key)
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS career_episode_content_suppressions (
                user_id TEXT NOT NULL,
                kind TEXT NOT NULL,
                source_run_id TEXT NOT NULL,
                content_digest TEXT NOT NULL,
                scope_key TEXT NOT NULL,
                deleted_at TEXT NOT NULL,
                PRIMARY KEY(
                    user_id, kind, source_run_id, content_digest, scope_key
                )
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS career_episode_deleted_scopes (
                user_id TEXT NOT NULL,
                scope_key TEXT NOT NULL,
                deleted_at TEXT NOT NULL,
                PRIMARY KEY(user_id, scope_key)
            )
            """
        )

    @staticmethod
    def _replace_short_terms(
        connection: sqlite3.Connection, episode: CareerEpisode
    ) -> None:
        connection.execute(
            "DELETE FROM career_episode_short_terms WHERE episode_id = ?",
            (episode.id,),
        )
        title_terms = SQLiteCareerEpisodeStore._short_term_counts(episode.title)
        summary_terms = SQLiteCareerEpisodeStore._short_term_counts(
            episode.summary
        )
        terms = tuple(dict.fromkeys((*title_terms, *summary_terms)))
        connection.executemany(
            """
            INSERT INTO career_episode_short_terms(
                episode_id, user_id, term, title_count, summary_count
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                (
                    episode.id,
                    episode.user_id,
                    term,
                    title_terms.get(term, 0),
                    summary_terms.get(term, 0),
                )
                for term in terms
            ),
        )

    @staticmethod
    def _content_digest(draft: CareerEpisodeDraft) -> str:
        payload = {
            "title": draft.title,
            "summary": draft.summary,
        }
        canonical = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @staticmethod
    def _stored_content_digest(row: sqlite3.Row | tuple[object, ...]) -> str:
        payload = {
            "title": str(row[4]),
            "summary": str(row[5]),
        }
        canonical = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @staticmethod
    def _short_term_counts(value: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        for token in re.findall(
            r"[\w+#.-]+",
            value.casefold(),
            flags=re.UNICODE,
        ):
            for width in (1, 2):
                for offset in range(0, len(token) - width + 1):
                    term = token[offset : offset + width]
                    counts[term] = counts.get(term, 0) + 1
        return counts

    _SELECT = (
        "SELECT e.id, e.user_id, e.kind, e.source_run_id, e.occurred_at, "
        "e.title, e.summary, e.conversation_id, e.resource_refs_json, "
        "e.salience, e.last_accessed_at, e.access_count FROM career_episodes e"
    )

    @staticmethod
    def _episode(row) -> CareerEpisode:
        return CareerEpisode(
            id=row[0],
            user_id=row[1],
            kind=row[2],
            source_run_id=row[3],
            occurred_at=row[4],
            title=row[5],
            summary=row[6],
            conversation_id=row[7],
            resource_refs=tuple(
                EpisodeResourceRef.model_validate(reference)
                for reference in json.loads(row[8])
            ),
            salience=row[9],
            last_accessed_at=row[10],
            access_count=row[11],
        )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.execute("PRAGMA busy_timeout=30000")
        return connection
