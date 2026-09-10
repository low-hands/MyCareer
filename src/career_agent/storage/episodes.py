from __future__ import annotations

from datetime import datetime, timezone
import json
import math
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

_SHORT_QUERY_CANDIDATE_LIMIT = 200


def _like_fragment(value: str) -> str:
    return (
        value.replace("\\", "\\\\")
        .replace("%", "\\%")
        .replace("_", "\\_")
    )


def _utc_bound(value: datetime, *, name: str) -> datetime:
    if value.utcoffset() is None:
        raise ValueError(f"{name} must include a timezone offset")
    return value.astimezone(timezone.utc)


def apply_episode_schema(connection: sqlite3.Connection) -> None:
    """Adopt the episodic projection in a context database."""

    apply_schema(
        connection,
        "career_episodes",
        8,
        SQLiteCareerEpisodeStore._baseline,
        {
            2: SQLiteCareerEpisodeStore._upgrade_to_v2,
            3: SQLiteCareerEpisodeStore._upgrade_to_v3,
            4: SQLiteCareerEpisodeStore._upgrade_to_v4,
            5: SQLiteCareerEpisodeStore._upgrade_to_v5,
            6: SQLiteCareerEpisodeStore._upgrade_to_v6,
            7: SQLiteCareerEpisodeStore._upgrade_to_v7,
            8: SQLiteCareerEpisodeStore._upgrade_to_v8,
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

        scope_keys = tuple(dict.fromkeys(memory_scope_keys))
        deleted_scope_table_exists = connection.execute(
            """
            SELECT 1 FROM sqlite_master
            WHERE type = 'table' AND name = 'memory_deleted_scopes'
            """
        ).fetchone()
        if scope_keys and deleted_scope_table_exists is not None:
            placeholders = ",".join("?" for _ in scope_keys)
            deleted = connection.execute(
                f"""
                SELECT 1 FROM memory_deleted_scopes
                WHERE user_id = ? AND scope_key IN ({placeholders})
                LIMIT 1
                """,
                (draft.user_id, *scope_keys),
            ).fetchone()
            if deleted is not None:
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
                for scope_key in scope_keys
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

    def get(self, *, user_id: str, episode_id: str) -> CareerEpisode | None:
        with self._connect() as connection:
            row = connection.execute(
                self._SELECT + " WHERE e.user_id = ? AND e.id = ?",
                (user_id, episode_id),
            ).fetchone()
        return self._episode(row) if row else None

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
    ) -> int:
        """Remove one association, then delete only orphaned episodes."""

        candidate_ids = tuple(
            str(row[0])
            for row in connection.execute(
                """
                SELECT episode_id
                FROM career_episode_memory_bindings
                WHERE user_id = ? AND scope_key = ?
                """,
                (user_id, scope_key),
            ).fetchall()
        )
        if not candidate_ids:
            return 0
        connection.execute(
            """
            DELETE FROM career_episode_memory_bindings
            WHERE user_id = ? AND scope_key = ?
            """,
            (user_id, scope_key),
        )
        placeholders = ",".join("?" for _ in candidate_ids)
        ids = tuple(
            str(row[0])
            for row in connection.execute(
                f"""
                SELECT e.id
                FROM career_episodes AS e
                WHERE e.user_id = ?
                  AND e.id IN ({placeholders})
                  AND NOT EXISTS (
                        SELECT 1
                        FROM career_episode_memory_bindings AS binding
                        WHERE binding.user_id = e.user_id
                          AND binding.episode_id = e.id
                      )
                """,
                (user_id, *candidate_ids),
            ).fetchall()
        )
        if not ids:
            return 0
        orphan_placeholders = ",".join("?" for _ in ids)
        connection.execute(
            f"DELETE FROM career_episodes_fts WHERE episode_id IN ({orphan_placeholders})",
            ids,
        )
        connection.execute(
            f"DELETE FROM career_episodes WHERE id IN ({orphan_placeholders})",
            ids,
        )
        return len(ids)

    def search(
        self,
        *,
        user_id: str,
        query: str,
        limit: int = 20,
        start_datetime: datetime | None = None,
        end_datetime: datetime | None = None,
        kinds: tuple[str, ...] = (),
    ) -> tuple[CareerEpisode, ...]:
        if limit < 1:
            raise ValueError("limit must be positive")
        if start_datetime is not None:
            start_datetime = _utc_bound(
                start_datetime, name="start_datetime"
            )
        if end_datetime is not None:
            end_datetime = _utc_bound(end_datetime, name="end_datetime")
        if start_datetime is not None and end_datetime is not None:
            if start_datetime > end_datetime:
                raise ValueError("start_datetime cannot exceed end_datetime")
        allowed_kinds = {
            "mock_interview",
            "job_research",
            "application",
            "interview_round",
        }
        selected_kinds = tuple(dict.fromkeys(kinds))
        if any(kind not in allowed_kinds for kind in selected_kinds):
            raise ValueError("unknown episode kind")
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
        filters = ["e.user_id = ?"]
        filter_parameters: list[object] = [user_id]
        if start_datetime is not None:
            filters.append("julianday(e.occurred_at) >= julianday(?)")
            filter_parameters.append(start_datetime.isoformat())
        if end_datetime is not None:
            filters.append("julianday(e.occurred_at) <= julianday(?)")
            filter_parameters.append(end_datetime.isoformat())
        if selected_kinds:
            filters.append(
                "e.kind IN (" + ",".join("?" for _ in selected_kinds) + ")"
            )
            filter_parameters.extend(selected_kinds)
        where = " AND ".join(filters)
        with self._connect() as connection:
            if not normalized_query:
                rows = connection.execute(
                    self._SELECT
                    + f" WHERE {where} "
                    "ORDER BY e.occurred_at DESC, e.id DESC LIMIT ?",
                    (*filter_parameters, limit),
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
                    predicates = " OR ".join(
                        "(lower(e.title) LIKE ? ESCAPE '\\' "
                        "OR lower(e.summary) LIKE ? ESCAPE '\\')"
                        for _ in short_tokens
                    )
                    title_score = " + ".join(
                        "CASE WHEN lower(e.title) LIKE ? ESCAPE '\\' "
                        "THEN 4 ELSE 0 END"
                        for _ in short_tokens
                    )
                    ctes.append(
                        f"""
                        short_hits AS (
                            SELECT e.id AS episode_id,
                                   ({title_score}) AS short_score
                            FROM career_episodes AS e
                            WHERE {where} AND ({predicates})
                            ORDER BY e.occurred_at DESC, e.id DESC
                            LIMIT ?
                        )
                        """
                    )
                    escaped_tokens = tuple(
                        _like_fragment(token) for token in short_tokens
                    )
                    title_patterns = tuple(
                        f"%{token}%" for token in escaped_tokens
                    )
                    like_patterns = tuple(
                        pattern
                        for token in escaped_tokens
                        for pattern in (f"%{token}%", f"%{token}%")
                    )
                    parameters.extend(
                        (
                            *title_patterns,
                            *filter_parameters,
                            *like_patterns,
                            _SHORT_QUERY_CANDIDATE_LIMIT,
                        )
                    )
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
                    WHERE """
                    + where
                    + """
                    ORDER BY
                        CASE
                            WHEN lower(e.title) = lower(?) THEN 0
                            WHEN lower(e.title) LIKE lower(?) ESCAPE '\\' THEN 1
                            ELSE 2
                        END,
                        COALESCE(hits.short_score, 0) DESC,
                        COALESCE(hits.long_relevance, 1000000.0),
                        e.occurred_at DESC
                    LIMIT ?
                    """,
                    (
                        *parameters,
                        *filter_parameters,
                        normalized_query,
                        f"{_like_fragment(normalized_query)}%",
                        limit,
                    ),
                ).fetchall()
            else:
                rows = []
        return tuple(self._episode(row) for row in rows)

    def project_relevant(
        self,
        *,
        user_id: str,
        query: str,
        limit: int = 5,
        exclude_conversation_id: str | None = None,
    ) -> tuple[CareerEpisode, ...]:
        """Rank lexical hits by relevance, salience, and recency."""

        if not 1 <= limit <= 5:
            raise ValueError("episode projection limit must be between 1 and 5")
        terms = self._projection_terms(query)
        if not terms:
            return ()
        candidates = self.search(
            user_id=user_id,
            query=" ".join(terms),
            limit=max(30, limit * 10),
        )
        if exclude_conversation_id is not None:
            candidates = tuple(
                item
                for item in candidates
                if item.conversation_id != exclude_conversation_id
            )
        now = datetime.now(timezone.utc)

        def score(indexed: tuple[int, CareerEpisode]) -> tuple[float, datetime]:
            rank, episode = indexed
            age_days = max(
                0.0,
                (now - episode.occurred_at.astimezone(timezone.utc)).total_seconds()
                / 86_400,
            )
            decayed_salience = max(0.0, episode.salience) * math.exp(
                -age_days / 180.0
            )
            lexical_rank = 1.0 / (rank + 1)
            return (
                0.65 * lexical_rank + 0.35 * decayed_salience,
                episode.occurred_at,
            )

        selected = tuple(
            episode
            for _, episode in sorted(
                enumerate(candidates),
                key=score,
                reverse=True,
            )[:limit]
        )
        self.mark_accessed(
            user_id=user_id,
            episode_ids=tuple(item.id for item in selected),
        )
        return selected

    def mark_accessed(
        self,
        *,
        user_id: str,
        episode_ids: tuple[str, ...],
    ) -> None:
        selected = tuple(dict.fromkeys(episode_ids))
        if not selected:
            return
        placeholders = ",".join("?" for _ in selected)
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                f"""
                UPDATE career_episodes
                SET last_accessed_at = ?,
                    access_count = access_count + 1,
                    updated_at = ?
                WHERE user_id = ? AND id IN ({placeholders})
                """,
                (
                    now,
                    now,
                    user_id,
                    *selected,
                ),
            )

    @staticmethod
    def _projection_terms(query: str) -> tuple[str, ...]:
        normalized = query.casefold()
        terms: list[str] = re.findall(
            r"[a-z0-9][a-z0-9+#.-]{1,}",
            normalized,
        )
        for run in re.findall(r"[\u4e00-\u9fff]+", normalized):
            width = 3 if len(run) >= 3 else len(run)
            if width:
                terms.extend(
                    run[index : index + width]
                    for index in range(len(run) - width + 1)
                )
        return tuple(dict.fromkeys(terms))

    @staticmethod
    def _baseline(connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS career_episodes (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                kind TEXT NOT NULL CHECK(kind IN (
                    'mock_interview', 'job_research',
                    'application', 'interview_round',
                    'resume_analysis', 'intent_confirmation',
                    'resume_tailoring'
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
        SQLiteCareerEpisodeStore._ensure_binding_schema(connection)

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
        # v3 previously added a short-term side index. v7 removes it in favor
        # of a bounded LIKE fallback for one- and two-character queries.
        pass

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
        SQLiteCareerEpisodeStore._ensure_binding_schema(connection)

    @staticmethod
    def _upgrade_to_v6(connection: sqlite3.Connection) -> None:
        connection.execute("DROP TABLE IF EXISTS career_episode_deletion_cutoffs")
        connection.execute(
            "DROP TABLE IF EXISTS career_episode_deletion_suppressions"
        )

    @staticmethod
    def _upgrade_to_v7(connection: sqlite3.Connection) -> None:
        connection.execute("DROP TABLE IF EXISTS career_episode_short_terms")
        connection.execute(
            "DROP TABLE IF EXISTS career_episode_content_suppressions"
        )
        connection.execute("DROP TABLE IF EXISTS career_episode_deleted_scopes")
        SQLiteCareerEpisodeStore._ensure_binding_schema(connection)

    @staticmethod
    def _upgrade_to_v8(connection: sqlite3.Connection) -> None:
        """Expand the immutable episode-kind constraint without losing rows."""

        connection.execute("DROP TABLE IF EXISTS career_episodes_fts")
        connection.execute(
            "ALTER TABLE career_episodes RENAME TO career_episodes_v7"
        )
        connection.execute(
            "DROP INDEX IF EXISTS career_episodes_user_time_idx"
        )
        connection.execute(
            """
            CREATE TABLE career_episodes (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                kind TEXT NOT NULL CHECK(kind IN (
                    'mock_interview', 'job_research',
                    'application', 'interview_round',
                    'resume_analysis', 'intent_confirmation',
                    'resume_tailoring'
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
            INSERT INTO career_episodes(
                id, user_id, kind, source_run_id, occurred_at, title, summary,
                conversation_id, resource_refs_json, salience,
                last_accessed_at, access_count, created_at, updated_at
            )
            SELECT
                id, user_id, kind, source_run_id, occurred_at, title, summary,
                conversation_id, resource_refs_json, salience,
                last_accessed_at, access_count, created_at, updated_at
            FROM career_episodes_v7
            """
        )
        connection.execute("DROP TABLE career_episodes_v7")
        connection.execute(
            """
            CREATE INDEX career_episodes_user_time_idx
            ON career_episodes(user_id, occurred_at DESC)
            """
        )
        SQLiteCareerEpisodeStore._create_fts(connection)
        connection.execute(
            """
            INSERT INTO career_episodes_fts(
                episode_id, user_id, title, summary
            )
            SELECT id, user_id, title, summary FROM career_episodes
            """
        )
        SQLiteCareerEpisodeStore._ensure_binding_schema(connection)

    @staticmethod
    def _ensure_binding_schema(connection: sqlite3.Connection) -> None:
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
