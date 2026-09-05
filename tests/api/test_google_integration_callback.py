from urllib.parse import parse_qs, urlparse
from urllib.error import URLError

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from career_agent.api.integrations import build_integration_router
from career_agent.services.integrations import ConnectedGoogleAccount


class IntegrationService:
    frontend_url = "http://localhost:5173"

    def google_connection_kind(self, *, state: str, consume: bool = False):
        return "calendar"

    def complete_google(self, *, state: str, code: str) -> ConnectedGoogleAccount:
        return ConnectedGoogleAccount(
            kind="gmail",
            account_id="account-1",
            email_address="alice@example.com",
        )


def test_google_callback_redirect_contains_no_email_address() -> None:
    app = FastAPI()
    app.include_router(build_integration_router(IntegrationService))

    with TestClient(app) as client:
        response = client.get(
            "/v1/connections/google/callback",
            params={"state": "state-long-enough-for-validation", "code": "code"},
            follow_redirects=False,
        )

    assert response.status_code == 307
    assert parse_qs(urlparse(response.headers["location"]).query) == {
        "status": ["connected"],
        "kind": ["gmail"],
    }


@pytest.mark.parametrize(
    ("failure", "status"),
    (
        (ValueError("invalid state"), "invalid"),
        (URLError("offline"), "network_error"),
    ),
)
def test_google_callback_distinguishes_validation_and_network_failures(
    failure, status, caplog
) -> None:
    class FailingService(IntegrationService):
        def complete_google(self, *, state: str, code: str):
            raise failure

    app = FastAPI()
    app.include_router(build_integration_router(FailingService))

    with TestClient(app) as client:
        response = client.get(
            "/v1/connections/google/callback",
            params={"state": "state-long-enough-for-validation", "code": "code"},
            follow_redirects=False,
        )

    query = parse_qs(urlparse(response.headers["location"]).query)
    assert query["status"] == [status]
    assert query["kind"] == ["calendar"]
    assert caplog.records


def test_cancelled_calendar_oauth_returns_to_calendar_panel() -> None:
    app = FastAPI()
    app.include_router(build_integration_router(IntegrationService))

    with TestClient(app) as client:
        response = client.get(
            "/v1/connections/google/callback",
            params={
                "state": "state-long-enough-for-validation",
                "error": "access_denied",
            },
            follow_redirects=False,
        )

    assert parse_qs(urlparse(response.headers["location"]).query) == {
        "status": ["denied"],
        "kind": ["calendar"],
    }
