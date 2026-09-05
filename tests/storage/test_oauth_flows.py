from datetime import datetime, timedelta, timezone
import sqlite3

from career_agent.storage.oauth_flows import OAuthFlow, SQLiteOAuthFlowStore


def test_oauth_flow_is_single_use_and_expires(tmp_path) -> None:
    store = SQLiteOAuthFlowStore(tmp_path / "context.sqlite3")
    now = datetime.now(timezone.utc)
    store.create(OAuthFlow("valid", "alice", "verifier", now + timedelta(minutes=5)))
    store.create(OAuthFlow("expired", "alice", "old", now - timedelta(seconds=1)))

    assert store.peek("valid", now=now).connection_kind == "gmail"
    assert store.peek("valid", now=now).user_id == "alice"
    assert store.consume("valid", now=now).user_id == "alice"
    assert store.consume("valid", now=now) is None
    assert store.consume("expired", now=now) is None


def test_v1_oauth_flow_store_upgrades_with_a_default_connection_kind(tmp_path) -> None:
    path = tmp_path / "context.sqlite3"
    expires_at = datetime.now(timezone.utc) + timedelta(minutes=5)
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE schema_versions (component TEXT PRIMARY KEY, version INTEGER NOT NULL, updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)"
        )
        connection.execute(
            "INSERT INTO schema_versions(component, version) VALUES ('oauth_flows', 1)"
        )
        connection.execute(
            "CREATE TABLE oauth_flows (state TEXT PRIMARY KEY, user_id TEXT NOT NULL, code_verifier TEXT NOT NULL, expires_at TEXT NOT NULL, used_at TEXT)"
        )
        connection.execute(
            "INSERT INTO oauth_flows VALUES (?, ?, ?, ?, NULL)",
            ("legacy", "alice", "verifier", expires_at.isoformat()),
        )

    flow = SQLiteOAuthFlowStore(path).consume("legacy")

    assert flow is not None
    assert flow.connection_kind == "gmail"
