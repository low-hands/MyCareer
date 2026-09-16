from datetime import datetime, timezone

import pytest

from career_agent.agent.job_analysis_contracts import JobAnalysisResult, TieredRequirement
from career_agent.domain.job_discovery import JobDetail, Provenance
from career_agent.services.job_analysis import (
    JobAnalysisInputNotFoundError,
    JobAnalysisService,
)
from career_agent.storage.jobs import SQLiteJobPostingRepository


NOW = datetime(2026, 8, 23, 8, 0, tzinfo=timezone.utc)


def detail(*, description: str = "Build reliable RAG and agent systems. 3+ years Python.") -> JobDetail:
    return JobDetail(
        source_name="boss",
        source_job_id="boss-1",
        source_url="https://example.test/jobs/boss-1",
        title="AI Engineer",
        company_name="Acme",
        description=description,
        city="Shanghai",
        salary="25-35K",
        captured_at=NOW,
        provenance=Provenance(source_name="boss", source_job_id="boss-1", source_url="https://example.test/jobs/boss-1", captured_at=NOW, operation="detail", adapter_version="test-v1"),
    )


class FakeWorker:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def analyze(self, *, jd_text: str) -> JobAnalysisResult:
        self.calls.append(jd_text)
        return JobAnalysisResult(
            core_objective="建设可靠的 RAG 与 Agent 系统。",
            seniority="mid",
            requirements=(
                TieredRequirement(text="Python", tier="S", kind="fact", jd_quote="3+ years Python"),
                TieredRequirement(text="RAG 系统经验", tier="A", kind="inference", jd_quote="reliable RAG"),
            ),
            core_competencies=("Python", "RAG"),
            ats_keywords=("Python", "RAG", "agent"),
            summary=f"中级 AI 工程师。{len(self.calls)}",
        )


def test_analyze_uses_the_snapshot_text_only_and_caches_per_snapshot(tmp_path) -> None:
    repository = SQLiteJobPostingRepository(tmp_path / "jobs.sqlite3")
    saved = repository.save_captured_detail(user_id="u1", detail=detail())
    worker = FakeWorker()
    service = JobAnalysisService(repository, worker)

    first = service.analyze(user_id="u1", job_posting_id=saved.posting.id)
    again = service.analyze(user_id="u1", job_posting_id=saved.posting.id)

    assert worker.calls == [saved.snapshot.content]
    assert again.id == first.id
    assert first.jd_snapshot_id == saved.snapshot.id
    assert first.analyzer_version == service.analyzer_version
    assert first.content_fingerprint
    result = first.analysis.to_result()
    assert result is not None
    assert result.seniority == "mid"
    assert [item.tier for item in result.requirements] == ["S", "A"]
    current = repository.get_job(user_id="u1", job_posting_id=saved.posting.id)
    assert current is not None and current.analysis is not None and current.analysis.id == first.id


def test_new_snapshot_keeps_old_analysis_as_history_and_leaves_current_pending(tmp_path) -> None:
    repository = SQLiteJobPostingRepository(tmp_path / "jobs.sqlite3")
    first_saved = repository.save_captured_detail(user_id="u1", detail=detail())
    worker = FakeWorker()
    service = JobAnalysisService(repository, worker)
    first = service.analyze(user_id="u1", job_posting_id=first_saved.posting.id)

    changed = repository.save_captured_detail(user_id="u1", detail=detail(description="Completely new JD requiring Go and Kubernetes."))
    current = repository.get_job(user_id="u1", job_posting_id=changed.posting.id)
    assert current is not None
    assert current.snapshot.id != first_saved.snapshot.id
    assert current.analysis is None

    pinned = service.analyze(user_id="u1", job_posting_id=changed.posting.id, jd_snapshot_id=first_saved.snapshot.id)
    assert pinned.id == first.id
    assert worker.calls == [first_saved.snapshot.content]

    latest = service.analyze(user_id="u1", job_posting_id=changed.posting.id)
    assert latest.id != first.id
    assert latest.jd_snapshot_id == changed.snapshot.id
    assert worker.calls[-1] == changed.snapshot.content


def test_rejects_missing_job_and_foreign_snapshot(tmp_path) -> None:
    repository = SQLiteJobPostingRepository(tmp_path / "jobs.sqlite3")
    mine = repository.save_captured_detail(user_id="u1", detail=detail())
    theirs = repository.save_captured_detail(user_id="u2", detail=detail())
    service = JobAnalysisService(repository, FakeWorker())

    with pytest.raises(JobAnalysisInputNotFoundError) as missing:
        service.analyze(user_id="u1", job_posting_id="job_missing")
    assert missing.value.input_kind == "job_posting"
    with pytest.raises(JobAnalysisInputNotFoundError) as foreign:
        service.analyze(user_id="u1", job_posting_id=mine.posting.id, jd_snapshot_id=theirs.snapshot.id)
    assert foreign.value.input_kind == "jd_snapshot"
    with pytest.raises(ValueError):
        service.analyze(user_id=" ", job_posting_id=mine.posting.id)
