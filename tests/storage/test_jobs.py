from datetime import datetime, timezone

from career_agent.domain.job_discovery import JobDetail, Provenance
from career_agent.storage.jobs import SQLiteJobPostingRepository


NOW = datetime(2026, 8, 23, 8, 0, tzinfo=timezone.utc)


def detail(*, source_job_id: str = "boss-1", description: str = "Build reliable RAG and agent systems.") -> JobDetail:
    return JobDetail(
        source_name="boss",
        source_job_id=source_job_id,
        source_url=f"https://example.test/jobs/{source_job_id}",
        title="AI Engineer",
        company_name="Acme",
        description=description,
        city="Shanghai",
        salary="25-35K",
        captured_at=NOW,
        provenance=Provenance(source_name="boss", source_job_id=source_job_id, source_url=f"https://example.test/jobs/{source_job_id}", captured_at=NOW, operation="detail", adapter_version="test-v1"),
    )


def test_persists_full_jd_and_restores_after_repository_rebuild(tmp_path) -> None:
    path = tmp_path / "jobs.sqlite3"
    first = SQLiteJobPostingRepository(path)
    saved = first.save_detail(user_id="u1", run_id="run-1", result_ref="ref-1", selection_index=1, detail=detail())

    rebuilt = SQLiteJobPostingRepository(path)
    restored = rebuilt.get_for_run(user_id="u1", run_id="run-1", selection_index=1)

    assert restored is not None
    assert restored.posting.id == saved.posting.id
    assert restored.snapshot.content == "Build reliable RAG and agent systems."
    assert restored.snapshot.content_hash == saved.snapshot.content_hash


def test_same_jd_is_idempotent_and_changed_content_creates_new_snapshot(tmp_path) -> None:
    repository = SQLiteJobPostingRepository(tmp_path / "jobs.sqlite3")

    first = repository.save_detail(user_id="u1", run_id="run-1", result_ref="ref-1", selection_index=1, detail=detail())
    same = repository.save_detail(user_id="u1", run_id="run-2", result_ref="ref-2", selection_index=1, detail=detail())
    changed = repository.save_detail(user_id="u1", run_id="run-3", result_ref="ref-3", selection_index=1, detail=detail(description="Build production RAG, agent, and evaluation systems."))

    assert same.posting.id == first.posting.id
    assert same.snapshot.id == first.snapshot.id
    assert changed.posting.id == first.posting.id
    assert changed.snapshot.id != first.snapshot.id
    assert changed.snapshot.version == 2


def test_search_uses_metadata_and_full_jd_with_user_isolation(tmp_path) -> None:
    repository = SQLiteJobPostingRepository(tmp_path / "jobs.sqlite3")
    repository.save_detail(user_id="u1", run_id="run-1", result_ref="ref-1", selection_index=1, detail=detail())
    repository.save_detail(user_id="u2", run_id="run-2", result_ref="ref-2", selection_index=1, detail=detail(source_job_id="boss-2"))

    assert [item.company_name for item in repository.search_saved_jobs(user_id="u1", query="Acme")] == ["Acme"]
    assert [item.title for item in repository.search_saved_jobs(user_id="u1", query="RAG")] == ["AI Engineer"]
    assert [item.city for item in repository.search_saved_jobs(user_id="u1", query="Shanghai")] == ["Shanghai"]
    assert repository.search_saved_jobs(user_id="other", query="RAG") == ()
    assert repository.get_job(user_id="u2", job_posting_id=repository.list_jobs(user_id="u1")[0].job_posting_id) is None


def test_availability_distinguishes_closed_from_unknown(tmp_path) -> None:
    repository = SQLiteJobPostingRepository(tmp_path / "jobs.sqlite3")
    saved = repository.save_detail(user_id="u1", run_id="run-1", result_ref="ref-1", selection_index=1, detail=detail())

    assert repository.mark_availability(user_id="u1", job_posting_id=saved.posting.id, status="unknown", checked_at=NOW)
    assert repository.list_jobs(user_id="u1")[0].availability_status == "unknown"
    assert repository.mark_availability(user_id="u1", job_posting_id=saved.posting.id, status="closed", checked_at=NOW)
    assert repository.list_jobs(user_id="u1")[0].availability_status == "closed"
    assert not repository.mark_availability(user_id="other", job_posting_id=saved.posting.id, status="closed", checked_at=NOW)
