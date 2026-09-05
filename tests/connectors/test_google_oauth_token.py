import json
from urllib.parse import parse_qs

from career_agent.connectors.email_accounts import GoogleOAuthTokenProvider


class JsonResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = json.dumps(payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        return None

    def read(self) -> bytes:
        return self.payload


def test_google_token_refresh_reads_current_client_secret_from_environment(
    monkeypatch,
) -> None:
    environment = {
        "GOOGLE_OAUTH_CLIENT_ID": "current-client",
        "GOOGLE_OAUTH_CLIENT_SECRET": "first-secret",
    }
    submitted: list[dict[str, list[str]]] = []

    def fake_urlopen(request, timeout):
        assert timeout == 30.0
        submitted.append(parse_qs(request.data.decode()))
        return JsonResponse({"access_token": f"access-{len(submitted)}", "expires_in": 0})

    monkeypatch.setattr("career_agent.connectors.email_accounts.urlopen", fake_urlopen)
    provider = GoogleOAuthTokenProvider(
        json.dumps({"refresh_token": "refresh"}), environ=environment
    )

    assert provider() == "access-1"
    environment["GOOGLE_OAUTH_CLIENT_SECRET"] = "rotated-secret"
    assert provider() == "access-2"

    assert submitted[0]["client_secret"] == ["first-secret"]
    assert submitted[1]["client_secret"] == ["rotated-secret"]
    assert submitted[1]["client_id"] == ["current-client"]
    assert submitted[1]["refresh_token"] == ["refresh"]
