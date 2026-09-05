"""The only path by which a job stops being ``active``."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from career_agent.api.app import create_app
from career_agent.storage.api_keys import CAPTURE_WRITE


class _Repository:
    def __init__(self, known: str | None = "job-1") -> None:
        self._known = known
        self.marked: list[tuple[str, str, str]] = []
        self.looked_up: list[tuple[str, str | None, str | None]] = []

    def find_by_source(self, *, user_id, source_name, source_job_id, source_url):
        self.looked_up.append((source_name, source_job_id, source_url))
        return self._known

    def mark_availability(self, *, user_id, job_posting_id, status, checked_at=None):
        self.marked.append((user_id, job_posting_id, status))
        return True


def _app(api_keys, repository):
    return create_app(
        api_key_store_factory=lambda: api_keys,
        runtime_factory=lambda: None,
        capture_repository_factory=lambda: repository,
        action_center_factory=lambda: None,
        workspace_reader_factory=lambda: None,
    )


def _headers(api_keys):
    key = api_keys.issue(
        user_id="u1", name="extension", scopes=frozenset({CAPTURE_WRITE})
    )
    return {
        "Authorization": f"Bearer {key.secret}",
        "X-Career-Agent-Capture": "v1",
    }


def test_a_closed_page_marks_the_saved_job_and_nothing_else(api_keys) -> None:
    """Observed, never polled.

    Re-fetching every saved posting on a schedule is what got the earlier API
    search blocked, and it would also make this field a claim about the site
    rather than a report of a page the user opened. So the extension is the
    only writer, and it writes one field.
    """
    repository = _Repository()

    with TestClient(_app(api_keys, repository)) as client:
        response = client.post(
            "/v1/browser-captures/job-closures",
            headers=_headers(api_keys),
            json={"source_url": "https://www.zhipin.com/job_detail/abc.html?lid=x"},
        )

    assert response.json() == {"matched": True, "job_posting_id": "job-1"}
    assert repository.marked == [("u1", "job-1", "closed")]
    # Resolved by the identity capture writes, and by the canonical URL: the
    # tracking parameters that differ between two visits to the same posting
    # must not make it a different job.
    assert repository.looked_up == [
        ("boss", "abc", "https://www.zhipin.com/job_detail/abc.html")
    ]


def test_a_page_that_is_not_in_the_library_is_not_an_error(api_keys) -> None:
    """A closed posting the user never saved is simply not about anything here.

    Reporting it as a failure would put a red banner on their screen for a page
    they had no stake in.
    """
    repository = _Repository(known=None)

    with TestClient(_app(api_keys, repository)) as client:
        response = client.post(
            "/v1/browser-captures/job-closures",
            headers=_headers(api_keys),
            json={"source_url": "https://www.zhipin.com/job_detail/zzz.html"},
        )

    assert response.status_code == 200
    assert response.json()["matched"] is False
    assert repository.marked == []


@pytest.mark.parametrize(
    "url",
    (
        "http://www.zhipin.com/job_detail/abc.html",
        "https://evil.example/job_detail/abc.html",
        "not-a-url",
    ),
)
def test_only_https_boss_urls_may_close_a_job(api_keys, url) -> None:
    repository = _Repository()

    with TestClient(_app(api_keys, repository)) as client:
        response = client.post(
            "/v1/browser-captures/job-closures",
            headers=_headers(api_keys),
            json={"source_url": url},
        )

    assert response.status_code == 422
    assert repository.marked == []


def test_the_closure_report_needs_the_same_credential_as_a_capture(api_keys) -> None:
    """Same key, same weak storage, same class of statement about the same page.

    A separate scope would suggest this carries more authority than a capture,
    and it carries less: one status field on a job the user already saved.
    """
    from career_agent.storage.api_keys import WORKSPACE_READ

    repository = _Repository()
    reader = api_keys.issue(
        user_id="u1", name="dashboard", scopes=frozenset({WORKSPACE_READ})
    )

    with TestClient(_app(api_keys, repository)) as client:
        response = client.post(
            "/v1/browser-captures/job-closures",
            headers={
                "Authorization": f"Bearer {reader.secret}",
                "X-Career-Agent-Capture": "v1",
            },
            json={"source_url": "https://www.zhipin.com/job_detail/abc.html"},
        )

    assert response.status_code == 403
    assert repository.marked == []
