import json
from urllib.parse import parse_qs, urlparse

import pytest

from career_agent.services.integrations import IntegrationConnectionService
from career_agent.storage.calendar import SQLiteCalendarStore
from career_agent.storage.email_tracking import SQLiteEmailTrackingStore
from career_agent.storage.oauth_flows import SQLiteOAuthFlowStore


class MemorySecrets:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}
        self.deleted: list[str] = []

    def put(self, secret: str) -> str:
        reference = f"keyring:{len(self.values) + 1}"
        self.values[reference] = secret
        return reference

    def get(self, reference: str) -> str:
        return self.values[reference]

    def delete(self, reference: str) -> None:
        self.deleted.append(reference)
        self.values.pop(reference, None)


class JsonResponse:
    def __init__(self, payload: str) -> None:
        self.payload = payload.encode()

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        return None

    def read(self) -> bytes:
        return self.payload


def test_google_connection_uses_pkce_and_only_creates_requested_account(tmp_path) -> None:
    email_store = SQLiteEmailTrackingStore(tmp_path / "email.sqlite3")
    calendar_store = SQLiteCalendarStore(tmp_path / "calendar.sqlite3")
    secrets = MemorySecrets()

    def opener(request, timeout):
        assert timeout == 30.0
        if request.full_url.endswith("/token"):
            return JsonResponse('{"access_token":"access","refresh_token":"refresh"}')
        return JsonResponse('{"email":"alice@example.com","email_verified":true}')

    service = IntegrationConnectionService(
        email_store=email_store,
        calendar_store=calendar_store,
        flow_store=SQLiteOAuthFlowStore(tmp_path / "context.sqlite3"),
        secret_store=secrets,
        client_id="client",
        client_secret="secret",
        callback_url="http://localhost/callback",
        frontend_url="http://localhost:5173",
        opener=opener,
    )

    authorization_url = service.google_authorization_url(
        user_id="alice", kind="gmail"
    )
    query = parse_qs(urlparse(authorization_url).query)
    assert query["code_challenge_method"] == ["S256"]
    assert "gmail.readonly" in query["scope"][0]
    assert "calendar.events" not in query["scope"][0]
    assert service.google_connection_kind(state=query["state"][0]) == "gmail"

    connected = service.complete_google(state=query["state"][0], code="code")

    assert connected.kind == "gmail"
    email = email_store.get_account(user_id="alice", account_id=connected.account_id)
    assert json.loads(secrets.get(email.credential_ref)) == {
        "refresh_token": "refresh"
    }
    assert calendar_store.list_accounts(user_id="alice") == ()
    assert service.disconnect(
        user_id="alice", kind="email", account_id=email.id
    )
    assert secrets.deleted == [email.credential_ref]


def test_calendar_authorization_does_not_request_gmail_scope(tmp_path) -> None:
    service = IntegrationConnectionService(
        email_store=SQLiteEmailTrackingStore(tmp_path / "email.sqlite3"),
        calendar_store=SQLiteCalendarStore(tmp_path / "calendar.sqlite3"),
        flow_store=SQLiteOAuthFlowStore(tmp_path / "context.sqlite3"),
        secret_store=MemorySecrets(),
        client_id="client",
        client_secret="secret",
        callback_url="http://localhost/callback",
        frontend_url="http://localhost:5173",
    )

    query = parse_qs(
        urlparse(
            service.google_authorization_url(user_id="alice", kind="calendar")
        ).query
    )

    assert "calendar.events" in query["scope"][0]
    assert "gmail.readonly" not in query["scope"][0]
    assert service.google_connection_kind(
        state=query["state"][0], consume=True
    ) == "calendar"
    assert service.google_connection_kind(state=query["state"][0]) is None


def test_google_connection_rejects_an_unverified_email(tmp_path) -> None:
    secrets = MemorySecrets()

    def opener(request, timeout):
        if request.full_url.endswith("/token"):
            return JsonResponse('{"access_token":"access","refresh_token":"refresh"}')
        return JsonResponse(
            '{"email":"unverified@example.com","email_verified":false}'
        )

    service = IntegrationConnectionService(
        email_store=SQLiteEmailTrackingStore(tmp_path / "email.sqlite3"),
        calendar_store=SQLiteCalendarStore(tmp_path / "calendar.sqlite3"),
        flow_store=SQLiteOAuthFlowStore(tmp_path / "context.sqlite3"),
        secret_store=secrets,
        client_id="client",
        client_secret="secret",
        callback_url="http://localhost/callback",
        frontend_url="http://localhost:5173",
        opener=opener,
    )
    query = parse_qs(
        urlparse(service.google_authorization_url(user_id="alice", kind="gmail")).query
    )

    with pytest.raises(ValueError, match="not verified"):
        service.complete_google(state=query["state"][0], code="code")

    assert secrets.values == {}
