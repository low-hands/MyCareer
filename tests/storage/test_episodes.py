from datetime import datetime, timedelta, timezone
import sqlite3

import pytest

from career_agent.domain.episodes import CareerEpisodeDraft, EpisodeResourceRef
from career_agent.storage.episodes import SQLiteCareerEpisodeStore


def _draft(
    *,
    summary: str = "讨论了上一次投递没有继续的原因。",
) -> CareerEpisodeDraft:
    return CareerEpisodeDraft(
        user_id="u1",
        kind="application",
        source_run_id="application-1",
        occurred_at=datetime(2026, 9, 1, tzinfo=timezone.utc),
        title="Example · ML Engineer",
        summary=summary,
        conversation_id="conversation-1",
        resource_refs=(
            EpisodeResourceRef(
                kind="application",
                resource_id="application-1",
                title="ML Engineer",
            ),
        ),
    )


def test_upsert_is_idempotent_by_owner_kind_and_source(tmp_path) -> None:
    store = SQLiteCareerEpisodeStore(tmp_path / "context.sqlite3")

    first = store.upsert(_draft())
    second = store.upsert(
        _draft(summary="投递状态后来更新为 interviewing。").model_copy(
            update={
                "occurred_at": datetime(2026, 9, 2, tzinfo=timezone.utc),
            }
        )
    )

    assert second.id == first.id
    assert second.summary == "投递状态后来更新为 interviewing。"
    assert store.list_source_keys(user_id="u1") == {
        ("application", "application-1")
    }


def test_fts_tracks_the_latest_synopsis_without_authorizing_facts(tmp_path) -> None:
    store = SQLiteCareerEpisodeStore(tmp_path / "context.sqlite3")
    store.upsert(_draft(summary="STAR 回答的结果部分不够清晰。"))

    assert [item.source_run_id for item in store.search(
        user_id="u1", query="STAR"
    )] == ["application-1"]

    store.upsert(_draft(summary="系统设计里需要补充容量估算。"))

    assert store.search(user_id="u1", query="STAR") == ()
    assert [item.source_run_id for item in store.search(
        user_id="u1", query="容量估算"
    )] == ["application-1"]
    with sqlite3.connect(store.path) as connection:
        fts_hits = connection.execute(
            "SELECT count(*) FROM career_episodes_fts "
            "WHERE career_episodes_fts MATCH ?",
            ("容量估算",),
        ).fetchone()[0]
    assert fts_hits == 1


def test_delete_for_scope_removes_episode_and_both_indexes(tmp_path) -> None:
    store = SQLiteCareerEpisodeStore(tmp_path / "context.sqlite3")
    scope_key = "career_evidence/record-1/claim"
    stored = store.upsert(
        _draft(summary="Contains deleted career evidence."),
        memory_scope_keys=(scope_key,),
    )

    with store._connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        assert store.delete_for_scope_on(
            connection, user_id="u1", scope_key=scope_key
        ) == 1

    assert store.get_by_source(
        user_id="u1",
        kind="application",
        source_run_id="application-1",
    ) is None
    assert store.search(user_id="u1", query="deleted") == ()
    with sqlite3.connect(store.path) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM career_episodes_fts WHERE episode_id = ?",
            (stored.id,),
        ).fetchone()[0] == 0
        assert connection.execute(
            "SELECT COUNT(*) FROM career_episode_short_terms WHERE episode_id = ?",
            (stored.id,),
        ).fetchone()[0] == 0


def test_scope_delete_blocks_exact_replay_but_allows_clean_rederivation(
    tmp_path,
) -> None:
    store = SQLiteCareerEpisodeStore(tmp_path / "context.sqlite3")
    scope_key = "career_evidence/record-1/claim"
    assert store.upsert(_draft(), memory_scope_keys=(scope_key,)) is not None
    with store._connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        assert store.delete_for_scope_on(
            connection, user_id="u1", scope_key=scope_key
        ) == 1

    assert store.upsert(_draft()) is None
    assert store.upsert(_draft(summary="Cleanly re-derived source.")) is not None
    updated = store.upsert(
        _draft(summary="Old source updated after deletion.").model_copy(
            update={"occurred_at": datetime.now(timezone.utc) + timedelta(seconds=1)}
        )
    )
    assert updated is not None
    assert store.list_source_keys(user_id="u1") == frozenset(
        {("application", "application-1")}
    )

    fresh = store.upsert(
        _draft(summary="New event after deletion.").model_copy(
            update={
                "source_run_id": "application-2",
                "occurred_at": datetime.now(timezone.utc) + timedelta(seconds=1),
            }
        )
    )
    assert fresh is not None
    assert fresh.summary == "New event after deletion."


def test_upsert_many_forwards_memory_scope_bindings(tmp_path) -> None:
    store = SQLiteCareerEpisodeStore(tmp_path / "context.sqlite3")
    scope_key = "career_evidence/record-1/claim"
    drafts = (
        _draft(),
        _draft().model_copy(update={"source_run_id": "application-2"}),
    )

    assert len(store.upsert_many(drafts, memory_scope_keys=(scope_key,))) == 2

    with store._connect() as connection:
        assert connection.execute(
            """
            SELECT COUNT(*) FROM career_episode_memory_bindings
            WHERE user_id = ? AND scope_key = ?
            """,
            ("u1", scope_key),
        ).fetchone()[0] == 2


def test_short_queries_use_the_auxiliary_index_and_rank_title_hits(tmp_path) -> None:
    store = SQLiteCareerEpisodeStore(tmp_path / "context.sqlite3")
    store.upsert(
        _draft(summary="复盘记录。").model_copy(
            update={
                "source_run_id": "mock-1",
                "kind": "mock_interview",
                "title": "模拟面试",
            }
        )
    )
    store.upsert(
        _draft(summary="面试安排已更新，AI 方向。").model_copy(
            update={
                "source_run_id": "interview-1",
                "kind": "interview_round",
                "title": "日程记录",
                "occurred_at": datetime(2026, 9, 2, tzinfo=timezone.utc),
            }
        )
    )

    assert [episode.source_run_id for episode in store.search(
        user_id="u1", query="面试"
    )] == ["mock-1", "interview-1"]
    assert [episode.source_run_id for episode in store.search(
        user_id="u1", query="AI"
    )] == ["interview-1"]
    assert len(store.search(user_id="u1", query="面")) == 2
    with sqlite3.connect(store.path) as connection:
        indexed = connection.execute(
            """
            SELECT episode_id
            FROM career_episode_short_terms
            WHERE user_id = ? AND term = ?
            ORDER BY episode_id
            """,
            ("u1", "面试"),
        ).fetchall()
    assert len(indexed) == 2


def test_source_identity_is_non_null_at_the_database_boundary(tmp_path) -> None:
    path = tmp_path / "context.sqlite3"
    SQLiteCareerEpisodeStore(path)

    with sqlite3.connect(path) as connection:
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """
                INSERT INTO career_episodes(
                    id, user_id, kind, source_run_id, occurred_at, title,
                    summary, resource_refs_json, created_at, updated_at
                ) VALUES (?, ?, ?, NULL, ?, ?, ?, '[]', ?, ?)
                """,
                (
                    "episode-invalid",
                    "u1",
                    "application",
                    datetime.now(timezone.utc).isoformat(),
                    "Invalid",
                    "Invalid",
                    datetime.now(timezone.utc).isoformat(),
                    (datetime.now(timezone.utc) + timedelta(seconds=1)).isoformat(),
                ),
            )


def test_v1_unicode_index_is_rebuilt_and_backfilled_as_trigram(tmp_path) -> None:
    path = tmp_path / "context.sqlite3"
    store = SQLiteCareerEpisodeStore(path)
    stored = store.upsert(_draft(summary="模拟面试反馈需要补充量化结果。"))
    with sqlite3.connect(path) as connection:
        connection.execute("DROP TABLE career_episodes_fts")
        connection.execute(
            """
            CREATE VIRTUAL TABLE career_episodes_fts USING fts5(
                episode_id UNINDEXED,
                user_id UNINDEXED,
                title,
                summary,
                tokenize='unicode61'
            )
            """
        )
        connection.execute(
            "UPDATE schema_versions SET version = 1 "
            "WHERE component = 'career_episodes'"
        )

    reopened = SQLiteCareerEpisodeStore(path)

    with sqlite3.connect(path) as connection:
        version = connection.execute(
            "SELECT version FROM schema_versions "
            "WHERE component = 'career_episodes'"
        ).fetchone()[0]
        indexed_ids = connection.execute(
            "SELECT episode_id FROM career_episodes_fts "
            "WHERE career_episodes_fts MATCH ?",
            ("模拟面试",),
        ).fetchall()
    assert version == 6
    assert indexed_ids == [(stored.id,)]
    assert reopened.get_by_source(
        user_id="u1",
        kind="application",
        source_run_id="application-1",
    ) is not None


def test_v6_migration_drops_unread_legacy_deletion_tables(tmp_path) -> None:
    path = tmp_path / "context.sqlite3"
    store = SQLiteCareerEpisodeStore(path)
    stored = store.upsert(_draft())
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE career_episode_deletion_cutoffs (
                user_id TEXT PRIMARY KEY,
                through_occurred_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE career_episode_deletion_suppressions (
                user_id TEXT NOT NULL,
                kind TEXT NOT NULL,
                source_run_id TEXT NOT NULL,
                deleted_at TEXT NOT NULL,
                PRIMARY KEY(user_id, kind, source_run_id)
            )
            """
        )
        connection.execute(
            "UPDATE schema_versions SET version = 5 "
            "WHERE component = 'career_episodes'"
        )

    reopened = SQLiteCareerEpisodeStore(path)

    with sqlite3.connect(path) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                """
                SELECT name FROM sqlite_master
                WHERE type = 'table' AND name IN (
                    'career_episode_deletion_cutoffs',
                    'career_episode_deletion_suppressions'
                )
                """
            )
        }
    assert tables == set()
    assert reopened.get_by_source(
        user_id="u1",
        kind="application",
        source_run_id="application-1",
    ) == stored
