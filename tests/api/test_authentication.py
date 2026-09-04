"""Every route needs a credential, and none of them takes a user's name.

Written as a sweep over the app's own route table rather than as one test per
endpoint. The defect being prevented is an *omission* — a new route that forgets
the dependency, or reintroduces ``user_id`` as a parameter — and a per-endpoint
test cannot fail for a route nobody wrote a test for.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from career_agent.api.app import create_app
from career_agent.storage.api_keys import (
    CAPTURE_WRITE,
    CHAT_WRITE,
    SQLiteApiKeyStore,
    WORKSPACE_READ,
)


PUBLIC_PATHS = {
    "/health",
    "/ready",
    "/openapi.json",
    "/docs",
    "/docs/oauth2-redirect",
    "/redoc",
}
"""Routes that answer before a credential exists.

Liveness and readiness are checked by whatever runs the process, which holds no
API key, and neither reveals anything about a user.
"""


def _app(api_keys: SQLiteApiKeyStore):
    return create_app(
        api_key_store_factory=lambda: api_keys,
        runtime_factory=lambda: None,
        capture_repository_factory=lambda: None,
        action_center_factory=lambda: None,
        workspace_reader_factory=lambda: None,
    )


def _guarded_routes(app):
    for route in app.routes:
        path = getattr(route, "path", "")
        methods = getattr(route, "methods", set()) - {"HEAD", "OPTIONS"}
        if not methods or path in PUBLIC_PATHS:
            continue
        for method in sorted(methods):
            yield method, path


def test_no_route_accepts_a_caller_supplied_user_id(api_keys) -> None:
    """The structural half of the fix, and the half a 401 test cannot cover.

    Requiring a credential while still reading ``user_id`` from the request
    would let any authenticated caller act as anyone. The field has to be
    absent, not merely cross-checked — there is then nothing to cross-check.
    """
    app = _app(api_keys)
    schema = app.openapi()

    offenders = []
    for path, operations in schema["paths"].items():
        for method, operation in operations.items():
            for parameter in operation.get("parameters", ()):
                if parameter.get("name") == "user_id":
                    offenders.append(f"{method.upper()} {path} (query/path)")
            body = (
                operation.get("requestBody", {})
                .get("content", {})
                .get("application/json", {})
                .get("schema", {})
            )
            reference = body.get("$ref", "").rsplit("/", 1)[-1]
            model = schema.get("components", {}).get("schemas", {}).get(reference, {})
            if "user_id" in model.get("properties", {}):
                offenders.append(f"{method.upper()} {path} (body)")

    assert offenders == []


def test_every_non_public_route_refuses_an_anonymous_request(api_keys) -> None:
    app = _app(api_keys)

    with TestClient(app) as client:
        for method, path in _guarded_routes(app):
            response = client.request(method, path.replace("{kind}", "x").replace(
                "{resource_id}", "y"
            ).replace("{conversation_id}", "c1"), json={})
            assert response.status_code == 401, f"{method} {path}"


def test_a_server_without_a_credential_store_refuses_rather_than_opens(
    tmp_path: Path,
) -> None:
    """Failing closed turns a misconfiguration into an outage, not an open door.

    The alternative — treating a missing store as "no auth configured, allow
    everything" — is how a deployment silently reverts to the state this work
    removed.
    """
    app = create_app(
        api_key_store_factory=lambda: None,
        runtime_factory=lambda: None,
        capture_repository_factory=lambda: None,
        action_center_factory=lambda: None,
        workspace_reader_factory=lambda: None,
    )

    with TestClient(app) as client:
        assert client.get("/v1/jobs").status_code == 503
        assert client.get("/health").status_code == 200


def test_a_scope_is_required_per_route_not_merely_a_valid_key(api_keys) -> None:
    """Authentication is not authorization, and the browser key proves why.

    A capture key lives in a browser extension, where no secret is safe. If a
    valid key were enough, that key would be a key to every resume and
    application on the machine.
    """
    app = _app(api_keys)
    capture = api_keys.issue(
        user_id="u1", name="extension", scopes=frozenset({CAPTURE_WRITE})
    )
    reader = api_keys.issue(
        user_id="u1", name="dashboard", scopes=frozenset({WORKSPACE_READ})
    )

    with TestClient(app) as client:
        as_capture = {"Authorization": f"Bearer {capture.secret}"}
        as_reader = {"Authorization": f"Bearer {reader.secret}"}
        assert client.get("/v1/jobs", headers=as_capture).status_code == 403
        assert client.post(
            "/v1/chat/stream",
            headers=as_reader,
            json={"conversation_id": "c1", "message": "hi"},
        ).status_code == 403
        assert client.post(
            "/v1/browser-captures/jobs",
            headers={**as_reader, "X-Career-Agent-Capture": "v1"},
            json={
                "source_url": "https://www.zhipin.com/job_detail/x.html",
                "title": "t",
                "company_name": "c",
                "description": "d",
            },
        ).status_code == 403


@pytest.mark.parametrize(
    "header",
    ("", "Bearer", "Bearer   ", "bearer", "Basic abc", "Token abc", "abc"),
)
def test_a_malformed_authorization_header_is_never_accepted(api_keys, header) -> None:
    """A lenient parse would make a broken client into a security event."""
    app = _app(api_keys)

    with TestClient(app) as client:
        response = client.get("/v1/jobs", headers={"Authorization": header})

    assert response.status_code == 401
