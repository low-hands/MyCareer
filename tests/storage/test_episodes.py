from datetime import datetime, timedelta, timezone
import sqlite3

import pytest

from career_agent.domain.episodes import CareerEpisodeDraft, EpisodeResourceRef
from career_agent.storage.episodes import DecayPolicy, SQLiteCareerEpisodeStore


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


def test_delete_for_scope_removes_episode_and_fts_index(tmp_path) -> None:
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
            """
            SELECT 1 FROM sqlite_master
            WHERE type = 'table' AND name = 'career_episode_short_terms'
            """
        ).fetchone() is None


def test_delete_for_scope_keeps_episode_with_another_live_binding(tmp_path) -> None:
    store = SQLiteCareerEpisodeStore(tmp_path / "context.sqlite3")
    first_scope = "career_evidence/record-1/claim"
    second_scope = "career_evidence/record-2/claim"
    stored = store.upsert(
        _draft(summary="Derived from two live memories."),
        memory_scope_keys=(first_scope, second_scope),
    )
    assert stored is not None

    with store._connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        assert store.delete_for_scope_on(
            connection,
            user_id="u1",
            scope_key=first_scope,
        ) == 0

    assert store.get_by_source(
        user_id="u1",
        kind="application",
        source_run_id="application-1",
    ) is not None
    with store._connect() as connection:
        assert connection.execute(
            """
            SELECT scope_key
            FROM career_episode_memory_bindings
            WHERE episode_id = ?
            """,
            (stored.id,),
        ).fetchall() == [(second_scope,)]

    with store._connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        assert store.delete_for_scope_on(
            connection,
            user_id="u1",
            scope_key=second_scope,
        ) == 1

    assert store.get_by_source(
        user_id="u1",
        kind="application",
        source_run_id="application-1",
    ) is None


def test_scope_delete_does_not_install_a_content_suppression_registry(
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

    assert store.upsert(_draft()) is not None
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


def test_short_queries_use_bounded_like_fallback_and_rank_title_hits(tmp_path) -> None:
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
        assert connection.execute(
            """
            SELECT 1 FROM sqlite_master
            WHERE type = 'table' AND name = 'career_episode_short_terms'
            """
        ).fetchone() is None


def test_short_query_like_wildcards_are_literal(tmp_path) -> None:
    store = SQLiteCareerEpisodeStore(tmp_path / "context.sqlite3")
    store.upsert(_draft(summary="ordinary text"))

    assert store.search(user_id="u1", query="_") == ()
    assert store.search(user_id="u1", query="%") == ()


def test_projection_returns_only_relevant_events_without_recording_access(
    tmp_path,
) -> None:
    store = SQLiteCareerEpisodeStore(tmp_path / "context.sqlite3")
    relevant = store.upsert(
        _draft(summary="系统设计里需要补充容量估算。").model_copy(
            update={"source_run_id": "mock-system-design"}
        )
    )
    store.upsert(
        _draft(summary="讨论了薪资谈判和入职日期。").model_copy(
            update={"source_run_id": "application-salary"}
        )
    )

    projected = store.project_relevant(
        user_id="u1",
        query="容量估算怎么准备？",
    )

    assert [item.id for item in projected] == [relevant.id]
    # Projection is a pure read: the runtime builds a context several times per
    # turn, and stamping access here counted its bookkeeping instead of the
    # episode being seen. Exposure is marked once, by the caller.
    unchanged = store.get(user_id="u1", episode_id=relevant.id)
    assert unchanged is not None
    assert unchanged.access_count == 0
    assert unchanged.last_accessed_at is None
    store.mark_accessed(user_id="u1", episode_ids=(relevant.id,))
    accessed = store.get(user_id="u1", episode_id=relevant.id)
    assert accessed is not None
    assert accessed.access_count == 1
    assert accessed.last_accessed_at is not None
    assert store.project_relevant(
        user_id="u1",
        query="股权归属期怎么谈？",
    ) == ()
    assert store.get(user_id="other", episode_id=relevant.id) is None


def test_projection_excludes_events_from_the_current_conversation(tmp_path) -> None:
    store = SQLiteCareerEpisodeStore(tmp_path / "context.sqlite3")
    store.upsert(_draft(summary="系统设计容量估算复盘。"))

    assert store.project_relevant(
        user_id="u1",
        query="系统设计容量估算",
        exclude_conversation_id="conversation-1",
    ) == ()


def test_projection_combines_fts_rank_with_decayed_salience(tmp_path) -> None:
    store = SQLiteCareerEpisodeStore(tmp_path / "context.sqlite3")
    recent = store.upsert(
        _draft(summary="容量估算复盘。").model_copy(
            update={
                "source_run_id": "recent",
                "occurred_at": datetime.now(timezone.utc),
            }
        )
    )
    salient = store.upsert(
        _draft(summary="容量估算复盘。").model_copy(
            update={
                "source_run_id": "salient",
                "occurred_at": datetime.now(timezone.utc) - timedelta(days=30),
            }
        )
    )
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            "UPDATE career_episodes SET salience = 10 WHERE id = ?",
            (salient.id,),
        )

    projected = store.project_relevant(
        user_id="u1",
        query="容量估算",
        limit=2,
    )

    assert [item.id for item in projected] == [salient.id, recent.id]


def test_projection_freshness_uses_access_or_creation_and_search_keeps_faded(
    tmp_path,
) -> None:
    store = SQLiteCareerEpisodeStore(tmp_path / "context.sqlite3")
    now = datetime.now(timezone.utc)
    frequently_accessed = store.upsert(
        _draft(summary="容量估算复盘。").model_copy(
            update={"source_run_id": "frequent"}
        )
    )
    never_accessed = store.upsert(
        _draft(summary="容量估算复盘。").model_copy(
            update={"source_run_id": "forgotten"}
        )
    )
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            """
            UPDATE career_episodes
            SET created_at = ?, last_accessed_at = ?, access_count = 20
            WHERE id = ?
            """,
            (
                (now - timedelta(days=730)).isoformat(),
                (now - timedelta(days=7)).isoformat(),
                frequently_accessed.id,
            ),
        )
        connection.execute(
            """
            UPDATE career_episodes
            SET created_at = ?, last_accessed_at = NULL, access_count = 0
            WHERE id = ?
            """,
            ((now - timedelta(days=730)).isoformat(), never_accessed.id),
        )

    projected = store.project_relevant(
        user_id="u1",
        query="容量估算",
        limit=2,
    )

    assert [item.id for item in projected] == [frequently_accessed.id]
    assert {item.id for item in store.search(
        user_id="u1", query="容量估算"
    )} == {frequently_accessed.id, never_accessed.id}


def test_projection_threshold_is_inclusive(tmp_path) -> None:
    policy = DecayPolicy(
        half_life_days=180,
        access_boost=0,
        projection_threshold=0.4,
    )
    store = SQLiteCareerEpisodeStore(
        tmp_path / "context.sqlite3",
        decay_policy=policy,
    )
    boundary = store.upsert(_draft(summary="容量估算边界测试。"))
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            "UPDATE career_episodes SET salience = ?, created_at = ? WHERE id = ?",
            (
                policy.projection_threshold,
                (datetime.now(timezone.utc) + timedelta(days=1)).isoformat(),
                boundary.id,
            ),
        )

    assert store.project_relevant(
        user_id="u1", query="容量估算"
    )[0].id == boundary.id


def test_access_boost_is_bounded() -> None:
    policy = DecayPolicy(access_boost=100)
    now = datetime.now(timezone.utc)

    assert policy.effective_salience(
        salience=1,
        accessed_or_created_at=now,
        access_count=10**100,
        now=now,
    ) == 2.0


def test_half_life_days_really_halves_salience() -> None:
    policy = DecayPolicy(
        half_life_days=180,
        access_boost=0,
        projection_threshold=0,
    )
    now = datetime.now(timezone.utc)

    assert policy.effective_salience(
        salience=1,
        accessed_or_created_at=now - timedelta(days=180),
        access_count=0,
        now=now,
    ) == pytest.approx(0.5)


def test_time_filters_compare_instants_and_require_offsets(tmp_path) -> None:
    store = SQLiteCareerEpisodeStore(tmp_path / "context.sqlite3")
    store.upsert(_draft())

    assert store.search(
        user_id="u1",
        query="",
        start_datetime=datetime.fromisoformat("2026-09-01T08:00:00+08:00"),
        end_datetime=datetime.fromisoformat("2026-08-31T20:00:00-04:00"),
    )
    with pytest.raises(ValueError, match="timezone offset"):
        store.search(
            user_id="u1",
            query="",
            start_datetime=datetime(2026, 9, 1),
        )


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
    assert version == 8
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
