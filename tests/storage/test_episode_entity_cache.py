from datetime import datetime, timezone
import sqlite3

from career_agent.domain.episodes import CareerEpisodeDraft
from career_agent.storage import episodes as episode_storage
from career_agent.storage.episodes import SQLiteCareerEpisodeStore


def _draft(title="Pinnacle Robotics · Engineer", *, user_id="u1", source="app-1", summary="已投递"):
    return CareerEpisodeDraft(
        user_id=user_id, kind="application", source_run_id=source,
        occurred_at=datetime(2026, 9, 1, tzinfo=timezone.utc),
        title=title, summary=summary,
    )


def _count_title_reads(store, monkeypatch):
    reads = []
    connect = store._connect

    def traced_connect():
        connection = connect()
        connection.set_trace_callback(lambda sql: reads.append(sql) if sql.startswith("SELECT title FROM career_episodes") else None)
        return connection

    monkeypatch.setattr(store, "_connect", traced_connect)
    return reads


def test_title_cache_is_per_user_and_ignores_access_and_summary_changes(tmp_path, monkeypatch):
    store = SQLiteCareerEpisodeStore(tmp_path / "context.sqlite3")
    writer = SQLiteCareerEpisodeStore(store.path)
    reads = _count_title_reads(store, monkeypatch)
    # Empty users are cached too.
    assert store.prepare_query(user_id="u1", query="Pinnacle")[1] == ()
    store.prepare_query(user_id="u1", query="面试")
    assert len(reads) == 1
    episode = writer.upsert(_draft())
    assert "pinnacle robotics" in store.prepare_query(user_id="u1", query="Pinnacle Robotics")[1]
    assert len(reads) == 2
    writer.upsert(_draft("Other Co · Developer", user_id="u2"))
    assert "other co" in store.prepare_query(user_id="u2", query="Other Co")[1]
    assert len(reads) == 3
    store.mark_accessed(user_id="u1", episode_ids=(episode.id,))
    writer.upsert(_draft(summary="新摘要"))
    assert "pinnacle robotics" in store.prepare_query(user_id="u1", query="Pinnacle Robotics")[1]
    assert len(reads) == 3
    writer.upsert(_draft("New Co · Developer"))
    assert "new co" in store.prepare_query(user_id="u1", query="New Co")[1]
    store.prepare_query(user_id="u2", query="Other Co")
    assert len(reads) == 4  # u2's cache was not invalidated by u1's title change.


def test_caller_owned_transactions_invalidate_only_on_commit_and_delete(tmp_path, monkeypatch):
    store = SQLiteCareerEpisodeStore(tmp_path / "context.sqlite3")
    reads = _count_title_reads(store, monkeypatch)
    store.prepare_query(user_id="u1", query="Pinnacle Robotics")
    with sqlite3.connect(store.path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        store.upsert_on(connection, _draft(), memory_scope_keys=("scope-1",))
        assert store.prepare_query(user_id="u1", query="Pinnacle Robotics")[1] == ()
        connection.rollback()
    assert store.prepare_query(user_id="u1", query="Pinnacle Robotics")[1] == ()
    assert len(reads) == 1
    with sqlite3.connect(store.path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        store.upsert_on(connection, _draft(), memory_scope_keys=("scope-1",))
    assert "pinnacle robotics" in store.prepare_query(user_id="u1", query="Pinnacle Robotics")[1]
    with sqlite3.connect(store.path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        assert store.delete_for_scope_on(connection, user_id="u1", scope_key="scope-1") == 1
    assert store.prepare_query(user_id="u1", query="Pinnacle Robotics")[1] == ()
    assert len(reads) == 3


def test_cache_revision_and_titles_use_the_same_read_snapshot(tmp_path, monkeypatch):
    store = SQLiteCareerEpisodeStore(tmp_path / "context.sqlite3")
    writer = SQLiteCareerEpisodeStore(store.path)
    writer.upsert(_draft())
    connect = store._connect
    changes = []

    def traced_connect():
        connection = connect()

        def change_before_title_read(sql):
            if sql.startswith("SELECT title FROM career_episodes") and not changes:
                changes.append(True)
                writer.upsert(_draft("New Co · Developer"))

        connection.set_trace_callback(change_before_title_read)
        return connection

    monkeypatch.setattr(store, "_connect", traced_connect)
    first = store.prepare_query(user_id="u1", query="Pinnacle Robotics")[1]
    assert "pinnacle robotics" in first and "new co" not in first
    second = store.prepare_query(user_id="u1", query="New Co")[1]
    assert "new co" in second and "pinnacle robotics" not in second


def test_v8_database_adopts_revision_tracking_for_existing_titles(tmp_path):
    store = SQLiteCareerEpisodeStore(tmp_path / "context.sqlite3")
    store.upsert(_draft())
    with sqlite3.connect(store.path) as connection:
        for operation in ("insert", "update", "delete"):
            connection.execute(f"DROP TRIGGER career_episode_entities_{operation}")
        connection.execute("DROP TABLE career_episode_entity_revisions")
        connection.execute("UPDATE schema_versions SET version = 8 WHERE component = 'career_episodes'")
    migrated = SQLiteCareerEpisodeStore(store.path)
    assert "pinnacle robotics" in migrated.prepare_query(user_id="u1", query="Pinnacle Robotics")[1]
    migrated.upsert(_draft("New Co · Developer"))
    assert "new co" in migrated.prepare_query(user_id="u1", query="New Co")[1]
    with sqlite3.connect(store.path) as connection:
        assert connection.execute("SELECT version FROM schema_versions WHERE component = 'career_episodes'").fetchone() == (9,)


def test_title_cache_has_a_bounded_user_count(tmp_path, monkeypatch):
    monkeypatch.setattr(episode_storage, "_ENTITY_CACHE_USERS", 2)
    store = SQLiteCareerEpisodeStore(tmp_path / "context.sqlite3")
    reads = _count_title_reads(store, monkeypatch)
    for user in ("u1", "u2", "u1", "u3", "u1", "u2"):
        store.prepare_query(user_id=user, query="面试")
    assert len(reads) == 4
    assert len(store._entity_cache) == 2
