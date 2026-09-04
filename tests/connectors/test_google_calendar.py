import json
from datetime import datetime, timedelta, timezone
from io import BytesIO
from urllib.error import HTTPError

import pytest

from career_agent.connectors.calendar import (
    CalendarConnectorError,
    GoogleCalendarConnector,
)
from career_agent.domain.calendar import CalendarEventPayload


class Response:
    def __init__(self, body):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self.body


def test_google_calendar_create_uses_fixed_event_id_and_no_guest_updates(monkeypatch) -> None:
    requests = []

    def fake_urlopen(request, timeout):
        requests.append((request, timeout))
        return Response(json.dumps({
            "id": "ca12345", "etag": "etag-1",
            "htmlLink": "https://calendar.google.com/event",
        }).encode())

    monkeypatch.setattr("career_agent.connectors.calendar.urlopen", fake_urlopen)
    start = datetime(2026, 8, 28, tzinfo=timezone.utc)
    payload = CalendarEventPayload(
        title="面试 · Acme", description="岗位面试", start_at=start,
        end_at=start + timedelta(hours=1), timezone="Asia/Shanghai",
        location="Online",
    )
    connector = GoogleCalendarConnector(lambda: "access-token")

    result = connector.apply(
        operation="create", calendar_id="primary",
        external_event_id="ca12345", payload=payload,
        idempotency_key="proposal-1", payload_hash="a" * 64,
    )

    request, timeout = requests[0]
    body = json.loads(request.data)
    assert request.method == "POST"
    assert request.full_url.endswith("/calendars/primary/events?sendUpdates=none")
    assert request.headers["Authorization"] == "Bearer access-token"
    assert body["id"] == "ca12345"
    assert body["start"]["dateTime"] == start.isoformat()
    assert body["extendedProperties"]["private"] == {
        "careerAgentExecutionKey": "proposal-1",
        "careerAgentPayloadHash": "a" * 64,
    }
    assert "attendees" not in body
    assert timeout == 30.0
    assert result.external_event_id == "ca12345"


def test_google_calendar_reconciles_applied_write_by_private_payload_hash(monkeypatch) -> None:
    methods = []

    def fake_urlopen(request, timeout):
        methods.append(request.method)
        return Response(json.dumps({
            "id": "ca12345",
            "etag": "etag-1",
            "extendedProperties": {
                "private": {"careerAgentPayloadHash": "b" * 64}
            },
        }).encode())

    monkeypatch.setattr("career_agent.connectors.calendar.urlopen", fake_urlopen)
    start = datetime(2026, 8, 28, tzinfo=timezone.utc)
    payload = CalendarEventPayload(
        title="面试 · Acme", description="岗位面试", start_at=start,
        end_at=start + timedelta(hours=1), timezone="Asia/Shanghai",
    )

    result = GoogleCalendarConnector(lambda: "access-token").reconcile(
        operation="create", calendar_id="primary",
        external_event_id="ca12345", payload_hash="b" * 64,
    )

    assert methods == ["GET"]
    assert result.outcome == "applied"
    assert result.write_result.external_event_id == "ca12345"


def test_google_calendar_reconciliation_refuses_an_unmarked_existing_event(monkeypatch) -> None:
    def fake_urlopen(request, timeout):
        return Response(json.dumps({"id": "ca12345"}).encode())

    monkeypatch.setattr("career_agent.connectors.calendar.urlopen", fake_urlopen)

    result = GoogleCalendarConnector(lambda: "access-token").reconcile(
        operation="create", calendar_id="primary",
        external_event_id="ca12345", payload_hash="b" * 64,
    )

    assert result.outcome == "conflict"


def test_google_calendar_reconciliation_recognizes_update_not_applied(monkeypatch) -> None:
    def fake_urlopen(request, timeout):
        return Response(json.dumps({
            "id": "ca12345",
            "extendedProperties": {
                "private": {"careerAgentPayloadHash": "a" * 64}
            },
        }).encode())

    monkeypatch.setattr("career_agent.connectors.calendar.urlopen", fake_urlopen)

    result = GoogleCalendarConnector(lambda: "access-token").reconcile(
        operation="update", calendar_id="primary",
        external_event_id="ca12345", payload_hash="b" * 64,
        prior_payload_hash="a" * 64,
    )

    assert result.outcome == "not_applied"


def test_google_calendar_missing_event_is_not_applied_for_create(monkeypatch) -> None:
    def fake_urlopen(request, timeout):
        raise HTTPError(request.full_url, 404, "Not Found", {}, BytesIO(b"missing"))

    monkeypatch.setattr("career_agent.connectors.calendar.urlopen", fake_urlopen)

    result = GoogleCalendarConnector(lambda: "access-token").reconcile(
        operation="create", calendar_id="primary",
        external_event_id="ca12345", payload_hash="b" * 64,
    )

    assert result.outcome == "not_applied"


def test_google_calendar_missing_event_confirms_cancel(monkeypatch) -> None:
    def fake_urlopen(request, timeout):
        raise HTTPError(request.full_url, 410, "Gone", {}, BytesIO(b"deleted"))

    monkeypatch.setattr("career_agent.connectors.calendar.urlopen", fake_urlopen)

    result = GoogleCalendarConnector(lambda: "access-token").reconcile(
        operation="cancel", calendar_id="primary",
        external_event_id="ca12345", payload_hash="b" * 64,
    )

    assert result.outcome == "applied"
    assert result.write_result.external_event_id == "ca12345"


def test_google_calendar_create_conflict_never_blindly_patches(monkeypatch) -> None:
    methods = []

    def fake_urlopen(request, timeout):
        methods.append(request.method)
        raise HTTPError(
            request.full_url, 409, "Conflict", {}, BytesIO(b"already exists")
        )

    monkeypatch.setattr("career_agent.connectors.calendar.urlopen", fake_urlopen)
    start = datetime(2026, 8, 28, tzinfo=timezone.utc)
    payload = CalendarEventPayload(
        title="面试 · Acme", description="岗位面试", start_at=start,
        end_at=start + timedelta(hours=1), timezone="Asia/Shanghai",
    )

    with pytest.raises(CalendarConnectorError) as raised:
        GoogleCalendarConnector(lambda: "access-token").apply(
            operation="create", calendar_id="primary",
            external_event_id="ca12345", payload=payload,
            idempotency_key="proposal-1", payload_hash="b" * 64,
        )

    assert raised.value.outcome_unknown is True
    assert methods == ["POST"]
