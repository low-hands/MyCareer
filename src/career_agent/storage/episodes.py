from __future__ import annotations

from datetime import datetime, timezone
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


def apply_episode_schema(connection: sqlite3.Connection) -> None:
    """Adopt the episodic projection in a context database."""

    apply_schema(
        connection,
        "career_episodes",
        3,
        SQLiteCareerEpisodeStore._baseline,
        {
            2: SQLiteCareerEpisodeStore._upgrade_to_v2,
            3: SQLiteCareerEpisodeStore._upgrade_to_v3,
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

    def upsert(self, draft: CareerEpisodeDraft) -> CareerEpisode:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            episode = self.upsert_on(connection, draft)
        os.chmod(self.path, 0o600)
        return episode

    def upsert_many(
        self, drafts: tuple[CareerEpisodeDraft, ...]
    ) -> tuple[CareerEpisode, ...]:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            episodes = tuple(self.upsert_on(connection, draft) for draft in drafts)
        os.chmod(self.path, 0o600)
        return episodes

    @staticmethod
    def upsert_on(
        connection: sqlite3.Connection, draft: CareerEpisodeDraft
    ) -> CareerEpisode:
        """Upsert on a caller-owned transaction.

        This is used by ``CareerContextStore`` so the episode and its visible
        conversation seam commit together.
        """

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
