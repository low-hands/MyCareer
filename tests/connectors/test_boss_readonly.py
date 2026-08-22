import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pytest

from career_agent.connectors.boss_readonly import (
    BossAdapterError,
    BossReadOnlyAdapter,
    SearchQuery,
    parse_envelope,
)


NOW = datetime(2026, 8, 17, tzinfo=timezone.utc)


def envelope(data, *, ok=True, command="detail", error=None):
    return json.dumps({
        "ok": ok,
        "schema_version": "1.0",
        "command": command,
        "data": data,
        "pagination": None,
        "error": error,
        "hints": {},
    })


def test_search_maps_only_structured_safe_fields() -> None:
    calls = []

    def transport(args):
        calls.append(tuple(args))
        return envelope([{"security_id": "s1", "job_id": "j1", "title": "Backend", "company": "Acme", "salary": "20-30K", "skills": ["Python"], "url": "https://www.zhipin.com/job_detail/abc?token=secret#section"}], command="search")

    adapter = BossReadOnlyAdapter(transport, clock=lambda: NOW)
    results = adapter.search(SearchQuery(query="Python", city="Shanghai"))

    assert calls == [("search", "Python", "--page", "1", "--city", "Shanghai")]
    assert results[0].source_job_id == "j1"
    assert results[0].source_url == "https://www.zhipin.com/job_detail/abc"
    assert results[0].provenance.source_url == "https://www.zhipin.com/job_detail/abc"
    assert results[0].labels == ("Python",)


def test_search_discards_untrusted_url_without_failing() -> None:
    adapter = BossReadOnlyAdapter(
        lambda args: envelope([{"security_id": "s1", "title": "Backend", "company": "Acme", "url": "http://zhipin.com@evil.example/path"}], command="search"),
        clock=lambda: NOW,
    )

    assert adapter.search(SearchQuery(query="Python"))[0].source_url is None


def test_detail_preserves_non_empty_description_without_synthesizing_url() -> None:
    adapter = BossReadOnlyAdapter(
        lambda args: envelope({"security_id": "s1", "job_id": "j1", "title": "Backend", "company": "Acme", "description": "Full JD text", "url": "https://unexpected"}),
        clock=lambda: NOW,
    )

    result = adapter.detail("s1", "j1")

    assert result.description == "Full JD text"
    assert result.source_url is None
    assert result.provenance.source_job_id == "j1"


def test_detail_retries_without_job_id_after_invalid_param() -> None:
    calls = []

    def transport(args):
        calls.append(tuple(args))
        if len(calls) == 1:
            return envelope(None, ok=False, error={"code": "INVALID_PARAM", "message": "invalid job id"})
        return envelope({"security_id": "s1", "job_id": "j1", "title": "Backend", "company": "Acme", "description": "Full JD text"})

    result = BossReadOnlyAdapter(transport, clock=lambda: NOW).detail("s1", "j1")

    assert result.description == "Full JD text"
    assert calls == [("detail", "s1", "--job-id", "j1"), ("detail", "s1")]

    with pytest.raises(BossAdapterError) as error:
        parse_envelope("not-json")

    assert error.value.code == "MALFORMED_RESPONSE"
    assert error.value.recoverable is False


def test_auth_error_becomes_user_recoverable() -> None:
    raw = envelope(None, ok=False, error={"code": "AUTH_EXPIRED", "message": "expired"})

    with pytest.raises(BossAdapterError) as error:
        parse_envelope(raw)

    assert error.value.code == "AUTH_EXPIRED"
    assert error.value.recoverable is True
    assert "login" in error.value.recovery_action.lower()


def test_risk_error_never_retries() -> None:
    raw = envelope(None, ok=False, error={"code": "ACCOUNT_RISK", "message": "risk"})

    with pytest.raises(BossAdapterError) as error:
        parse_envelope(raw)

    assert error.value.recoverable is False


def test_transport_rejects_non_allowlisted_command() -> None:
    from career_agent.connectors.boss_readonly import SubprocessBossTransport

    transport = SubprocessBossTransport(Path("/isolated/boss"))
    with pytest.raises(BossAdapterError) as error:
        transport(("apply", "s1", "j1"))

    assert error.value.code == "OPERATION_NOT_ALLOWED"


def test_transport_maps_timeout_to_recoverable_error(monkeypatch) -> None:
    from career_agent.connectors.boss_readonly import SubprocessBossTransport

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("boss", 30)

    monkeypatch.setattr("career_agent.connectors.boss_readonly.subprocess.run", timeout)
    with pytest.raises(BossAdapterError) as error:
        SubprocessBossTransport(Path("/isolated/boss"))(("status",))

    assert error.value.code == "TIMEOUT"
    assert error.value.recoverable is True
    assert "retry" in error.value.recovery_action.lower()


def test_unknown_detail_error_preserves_safe_diagnostics() -> None:
    raw = envelope(None, ok=False, command="detail", error={"code": "UNKNOWN", "message": "detail unavailable"})
    with pytest.raises(BossAdapterError) as error:
        parse_envelope(raw)

    assert error.value.code == "UNKNOWN"
    assert error.value.recoverable is True
    assert error.value.operation == "detail"
    assert str(error.value) == "detail unavailable"
