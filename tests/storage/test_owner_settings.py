import sqlite3

import pytest

from career_agent.agent.main_agent_contracts import OwnerSettingsContext
from career_agent.storage.context import CareerContextStore, OwnerSettingsConflictError


def _desired(current, *, boss_search=None, application_confirmation=None):
    return current.model_copy(
        update={
            "preferences": current.preferences.model_copy(
                update={"boss_search": boss_search or current.preferences.boss_search}
            ),
            "behavior_policy": current.behavior_policy.model_copy(
                update={
                    "application_confirmation": application_confirmation
                    or current.behavior_policy.application_confirmation
                }
            ),
        }
    )


def test_soft_preferences_and_behavior_policy_have_independent_revisions(tmp_path):
    store = CareerContextStore(tmp_path / "context.sqlite3")
    initial = OwnerSettingsContext()

    soft = store.update_owner_settings(
        user_id="u1",
        desired=_desired(initial, boss_search="allowed"),
        expected_revision=0,
        actor_type="cli",
        actor_id="local-cli",
    )
    hard = store.update_owner_settings(
        user_id="u1",
        desired=_desired(soft, application_confirmation="always_ask"),
        expected_revision=1,
        actor_type="api_key",
        actor_id="key-1",
    )

    assert (soft.revision, soft.behavior_policy.revision) == (1, 0)
    assert (hard.revision, hard.behavior_policy.revision) == (2, 1)
    events = store.list_owner_settings_events(user_id="u1")
    assert [event.changed_fields for event in events] == [
        ("behavior_policy",),
        ("preferences",),
    ]
    assert events[0].actor_id == "key-1"


def test_stale_settings_update_is_rejected_without_losing_the_winner(tmp_path):
    store = CareerContextStore(tmp_path / "context.sqlite3")
    current = OwnerSettingsContext()
    winner = store.update_owner_settings(
        user_id="u1",
        desired=_desired(current, boss_search="allowed"),
        expected_revision=0,
        actor_type="cli",
        actor_id="one",
    )

    with pytest.raises(OwnerSettingsConflictError):
        store.update_owner_settings(
            user_id="u1",
            desired=_desired(current, application_confirmation="always_ask"),
            expected_revision=0,
            actor_type="cli",
            actor_id="two",
        )

    assert store.get_owner_settings("u1") == winner
    assert len(store.list_owner_settings_events(user_id="u1")) == 1


def test_v2_flat_preferences_are_migrated_once_to_owner_settings(tmp_path):
    path = tmp_path / "context.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE schema_versions(component TEXT PRIMARY KEY, version INTEGER, updated_at TEXT)"
        )
        connection.execute(
            "INSERT INTO schema_versions VALUES ('agent_context', 2, datetime('now'))"
        )
        connection.execute(
            "CREATE TABLE agent_preferences_context(user_id TEXT PRIMARY KEY, payload TEXT, updated_at TEXT)"
        )
        connection.execute(
            "INSERT INTO agent_preferences_context VALUES "
            "('u1', '{\"boss_search\":\"allowed\",\"application_confirmation\":\"always_ask\"}', datetime('now'))"
        )

    store = CareerContextStore(path)
    migrated = store.get_owner_settings("u1")

    assert migrated.preferences.boss_search == "allowed"
    assert migrated.behavior_policy.application_confirmation == "always_ask"
    with sqlite3.connect(path) as connection:
        assert connection.execute(
            "SELECT 1 FROM sqlite_master WHERE name='agent_preferences_context'"
        ).fetchone() is None

