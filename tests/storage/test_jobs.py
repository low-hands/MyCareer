from datetime import datetime, timezone

import pytest
import sqlite3

from career_agent.agent.job_discovery_contracts import JDAnalysis
from career_agent.domain.job_discovery import JobDetail, Provenance
from career_agent.storage.jobs import JDAnalysisPayload, SQLiteJobPostingRepository


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


def analysis_payload() -> JDAnalysisPayload:
    analysis = JDAnalysis(
        result_ref="run-specific-ref",
        job_summary="构建可靠的 RAG 系统。",
        responsibilities=("建设 RAG 平台",),
        required_skills=("Python",),
        preferred_qualifications=("Agent 经验",),
        clarification_questions=("团队规模未说明",),
    )
    return JDAnalysisPayload.model_validate(analysis.model_dump(exclude={"result_ref"}))


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


def test_browser_capture_persists_without_creating_a_discovery_run_link(tmp_path) -> None:
    path = tmp_path / "jobs.sqlite3"
    repository = SQLiteJobPostingRepository(path)

    saved = repository.save_captured_detail(user_id="u1", detail=detail())

    assert repository.get_job(
        user_id="u1",
        job_posting_id=saved.posting.id,
    ) == saved
    assert repository.count_jobs(user_id="u1") == 1
    assert repository.count_jobs(user_id="u2") == 0
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM job_run_links").fetchone()[0] == 0


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
    assert repository.get_snapshot(
        user_id="u1", jd_snapshot_id=first.snapshot.id
    ) == first.snapshot
    assert repository.get_snapshot(
        user_id="u1", jd_snapshot_id=changed.snapshot.id
    ) == changed.snapshot
    assert repository.get_snapshot(
        user_id="other", jd_snapshot_id=first.snapshot.id
    ) is None


def test_analysis_is_persisted_per_snapshot_and_idempotent_across_rebuild(tmp_path) -> None:
    path = tmp_path / "jobs.sqlite3"
    repository = SQLiteJobPostingRepository(path)
    saved = repository.save_detail(user_id="u1", run_id="run-1", result_ref="ref-1", selection_index=1, detail=detail())

    first = repository.save_analysis(
        user_id="u1",
        jd_snapshot_id=saved.snapshot.id,
        analyzer_version="jd-analysis-v1",
        analysis=analysis_payload(),
    )
    same = repository.save_analysis(
        user_id="u1",
        jd_snapshot_id=saved.snapshot.id,
        analyzer_version="jd-analysis-v1",
        analysis=analysis_payload().model_copy(update={"job_summary": "不会覆盖不可变分析"}),
    )

    rebuilt = SQLiteJobPostingRepository(path)
    restored = rebuilt.get_job(user_id="u1", job_posting_id=saved.posting.id)

    assert same.id == first.id
    assert same.analysis.job_summary == first.analysis.job_summary
    assert restored is not None
    assert restored.analysis is not None
    assert restored.analysis.id == first.id
    assert restored.analysis.analysis.required_skills == ("Python",)
    assert "run-specific-ref" not in restored.analysis.model_dump_json()


def test_changed_jd_requires_analysis_for_the_new_snapshot(tmp_path) -> None:
    repository = SQLiteJobPostingRepository(tmp_path / "jobs.sqlite3")
    first = repository.save_detail(user_id="u1", run_id="run-1", result_ref="ref-1", selection_index=1, detail=detail())
    repository.save_analysis(user_id="u1", jd_snapshot_id=first.snapshot.id, analyzer_version="jd-analysis-v1", analysis=analysis_payload())
    changed = repository.save_detail(user_id="u1", run_id="run-2", result_ref="ref-2", selection_index=1, detail=detail(description="A completely changed JD requiring Go."))

    current = repository.get_job(user_id="u1", job_posting_id=changed.posting.id)
    first_run = repository.get_for_run(user_id="u1", run_id="run-1", selection_index=1)
    changed_run = repository.get_for_run(user_id="u1", run_id="run-2", selection_index=1)

    assert current is not None
    assert current.snapshot.id == changed.snapshot.id
    assert current.analysis is None
    assert first_run is not None
    assert first_run.snapshot.id == first.snapshot.id
    assert first_run.analysis is not None
    assert first_run.analysis.analysis.job_summary == "构建可靠的 RAG 系统。"
    assert changed_run is not None
    assert changed_run.snapshot.id == changed.snapshot.id
    assert changed_run.analysis is None
    assert repository.get_latest_analysis(user_id="u1", job_posting_id=changed.posting.id) is None


def test_analysis_write_and_read_are_user_scoped(tmp_path) -> None:
    repository = SQLiteJobPostingRepository(tmp_path / "jobs.sqlite3")
    saved = repository.save_detail(user_id="u1", run_id="run-1", result_ref="ref-1", selection_index=1, detail=detail())

    with pytest.raises(ValueError, match="not found for this user"):
        repository.save_analysis(user_id="other", jd_snapshot_id=saved.snapshot.id, analyzer_version="jd-analysis-v1", analysis=analysis_payload())

    assert repository.get_latest_analysis(user_id="other", job_posting_id=saved.posting.id) is None


def test_existing_job_store_migrates_run_links_to_snapshot_refs(tmp_path) -> None:
    path = tmp_path / "jobs.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE job_run_links (user_id TEXT NOT NULL, run_id TEXT NOT NULL, result_ref TEXT NOT NULL, selection_index INTEGER NOT NULL, job_posting_id TEXT NOT NULL, PRIMARY KEY(user_id, run_id, result_ref), UNIQUE(user_id, run_id, selection_index))"
        )

    repository = SQLiteJobPostingRepository(path)
    saved = repository.save_detail(user_id="u1", run_id="run-1", result_ref="ref-1", selection_index=1, detail=detail())

    with sqlite3.connect(path) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(job_run_links)")}
        linked_snapshot = connection.execute("SELECT jd_snapshot_id FROM job_run_links").fetchone()[0]
    assert "jd_snapshot_id" in columns
    assert linked_snapshot == saved.snapshot.id


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
