import sqlite3
from datetime import datetime, timezone

from career_agent.agent.job_discovery_contracts import JobDiscoveryRequest
from career_agent.domain.job_discovery import Provenance, SearchResult
from career_agent.harness.observability import InMemoryTraceRecorder
from career_agent.storage.runs import JobDiscoveryRunStore

NOW = datetime(2026, 8, 18, tzinfo=timezone.utc)


def search_result() -> SearchResult:
    return SearchResult(
        result_ref="boss:r1",
        source_name="boss",
        source_job_id="job-1",
        security_id="security-1",
        title="AI Engineer",
        company_name="Acme",
        captured_at=NOW,
        provenance=Provenance(source_name="boss", captured_at=NOW, operation="search", adapter_version="test"),
    )


def test_run_store_persists_search_state_without_resume_text(tmp_path) -> None:
    path = tmp_path / "runs.sqlite3"
    store = JobDiscoveryRunStore(path)
    request = JobDiscoveryRequest(user_id="u", conversation_id="c", target_role="AI Engineer", resume_text="private resume")
    trace = InMemoryTraceRecorder()
    trace.record("run-1", "run_started", "job_discovery", outcome="started")

    store.save(run_id="run-1", request=request, results=(search_result(),), phase="selection_required", trace=trace.snapshot("run-1"))

    loaded = JobDiscoveryRunStore(path).get("run-1")
    raw = sqlite3.connect(path).execute("SELECT payload FROM job_discovery_runs WHERE run_id = ?", ("run-1",)).fetchone()[0]
    assert loaded is not None
    assert loaded.request.resume_text is None
    assert loaded.results[0].security_id == "security-1"
    assert "private resume" not in raw
    assert path.stat().st_mode & 0o077 == 0
