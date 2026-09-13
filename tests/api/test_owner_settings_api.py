from fastapi.testclient import TestClient

from career_agent.api.app import create_app
from career_agent.storage.api_keys import SETTINGS_WRITE, WORKSPACE_READ
from career_agent.storage.context import CareerContextStore


def _app(api_keys, store):
    return create_app(
        api_key_store_factory=lambda: api_keys,
        runtime_factory=lambda: None,
        capture_repository_factory=lambda: None,
        action_center_factory=lambda: None,
        workspace_reader_factory=lambda: None,
        owner_settings_store_factory=lambda: store,
    )


def test_authenticated_owner_can_update_and_read_versioned_settings(
    tmp_path, api_keys, issue_key
):
    store = CareerContextStore(tmp_path / "context.sqlite3")
    writer = issue_key("u1", SETTINGS_WRITE)
    reader = issue_key("u1", WORKSPACE_READ)

    with TestClient(_app(api_keys, store)) as client:
        changed = client.put(
            "/v1/settings",
            headers=writer,
            json={"expected_revision": 0, "application_confirmation": "always_ask"},
        )
        read = client.get("/v1/settings", headers=reader)
        history = client.get("/v1/settings/history", headers=reader)

    assert changed.status_code == 200
    settings = changed.json()["owner_settings"]
    assert settings["revision"] == 1
    assert settings["behavior_policy"] == {
        "revision": 1,
        "application_confirmation": "always_ask",
        "confirm_before": [],
    }
    assert read.json() == changed.json()
    assert history.json()["events"][0]["changed_fields"] == ["behavior_policy"]
    assert store.list_owner_settings_events(user_id="u1")[0].actor_type == "api_key"


def test_confirm_before_is_a_controlled_write_capability_set(
    tmp_path, api_keys, issue_key
):
    """The owner names WRITE capabilities; anything else is refused at the edge."""

    store = CareerContextStore(tmp_path / "context.sqlite3")
    writer = issue_key("u1", SETTINGS_WRITE)

    with TestClient(_app(api_keys, store)) as client:
        listed = client.put(
            "/v1/settings",
            headers=writer,
            json={
                "expected_revision": 0,
                "confirm_before": [
                    "create_interview",
                    "create_application",
                    "create_application",
                ],
            },
        )
        read_only = client.put(
            "/v1/settings",
            headers=writer,
            json={"expected_revision": 1, "confirm_before": ["search_career_history"]},
        )
        unknown = client.put(
            "/v1/settings",
            headers=writer,
            json={"expected_revision": 1, "confirm_before": ["delete_everything"]},
        )
        untouched = client.put(
            "/v1/settings",
            headers=writer,
            json={"expected_revision": 1, "boss_search": "allowed"},
        )
        cleared = client.put(
            "/v1/settings",
            headers=writer,
            json={"expected_revision": 2, "confirm_before": []},
        )

    assert listed.status_code == 200
    # Canonical: de-duplicated and sorted, so two owners naming the same rule
    # store the same document.
    assert listed.json()["owner_settings"]["behavior_policy"]["confirm_before"] == [
        "create_application",
        "create_interview",
    ]
    assert read_only.status_code == 422
    assert unknown.status_code == 422
    assert "search_career_history" in read_only.text
    # An omitted field keeps the list; only an explicit list replaces it.
    assert untouched.json()["owner_settings"]["behavior_policy"]["confirm_before"] == [
        "create_application",
        "create_interview",
    ]
    assert cleared.status_code == 200
    assert cleared.json()["owner_settings"]["behavior_policy"]["confirm_before"] == []
    assert cleared.json()["owner_settings"]["revision"] == 3


def test_chat_credential_cannot_relax_behavior_policy(tmp_path, api_keys, issue_key):
    store = CareerContextStore(tmp_path / "context.sqlite3")
    chat_only = issue_key("u1", "chat:write")

    with TestClient(_app(api_keys, store)) as client:
        response = client.put(
            "/v1/settings",
            headers=chat_only,
            json={"expected_revision": 0, "application_confirmation": "on_user_report"},
        )

    assert response.status_code == 403
    assert store.get_owner_settings("u1") is None


def test_stale_api_update_returns_current_revision(tmp_path, api_keys, issue_key):
    store = CareerContextStore(tmp_path / "context.sqlite3")
    writer = issue_key("u1", SETTINGS_WRITE)

    with TestClient(_app(api_keys, store)) as client:
        first = client.put(
            "/v1/settings",
            headers=writer,
            json={"expected_revision": 0, "boss_search": "allowed"},
        )
        stale = client.put(
            "/v1/settings",
            headers=writer,
            json={
                "expected_revision": 0,
                "application_confirmation": "always_ask",
            },
        )

    assert first.status_code == 200
    assert stale.status_code == 409
    assert stale.json()["detail"]["current_revision"] == 1
