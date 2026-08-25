import json
from datetime import datetime, timedelta, timezone
from io import BytesIO
from urllib.error import HTTPError

from career_agent.connectors.calendar import GoogleCalendarConnector
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
    )

    request, timeout = requests[0]
    body = json.loads(request.data)
    assert request.method == "POST"
    assert request.full_url.endswith("/calendars/primary/events?sendUpdates=none")
    assert request.headers["Authorization"] == "Bearer access-token"
    assert body["id"] == "ca12345"
    assert body["start"]["dateTime"] == start.isoformat()
    assert "attendees" not in body
    assert timeout == 30.0
    assert result.external_event_id == "ca12345"


def test_google_calendar_reconciles_same_fixed_id_after_ambiguous_create(monkeypatch) -> None:
    methods = []

    def fake_urlopen(request, timeout):
        methods.append(request.method)
        if request.method == "POST":
            raise HTTPError(
                request.full_url, 409, "Conflict", {}, BytesIO(b"already exists")
            )
        return Response(json.dumps({"id": "ca12345"}).encode())

    monkeypatch.setattr("career_agent.connectors.calendar.urlopen", fake_urlopen)
    start = datetime(2026, 8, 28, tzinfo=timezone.utc)
    payload = CalendarEventPayload(
        title="面试 · Acme", description="岗位面试", start_at=start,
        end_at=start + timedelta(hours=1), timezone="Asia/Shanghai",
    )

    result = GoogleCalendarConnector(lambda: "access-token").apply(
        operation="create", calendar_id="primary",
        external_event_id="ca12345", payload=payload,
    )

    assert methods == ["POST", "PATCH"]
    assert result.external_event_id == "ca12345"
