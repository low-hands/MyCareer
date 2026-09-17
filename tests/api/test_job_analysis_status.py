from argparse import Namespace
from asyncio import CancelledError
from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from career_agent.agent.job_analysis_contracts import JobAnalysisResult
from career_agent.agent.openai_compatible_client import AgentWorkerError
from career_agent.agent.resume_job_match_contracts import ResumeJobMatchResult
from career_agent.api.app import create_app
from career_agent.api.reads import WorkspaceReader
from career_agent.domain.job_discovery import JobDetail, Provenance
from career_agent.services.job_analysis import JobAnalysisService
from career_agent.storage.jobs import SQLiteJobPostingRepository
from career_agent.storage.resume_job_matches import SQLiteResumeJobMatchStore


class _Analyzer:
    def __init__(self, error: BaseException | None = None) -> None:
        self.error = error

    def analyze(self, *, jd_text: str) -> JobAnalysisResult:
        if self.error is not None:
            raise self.error
        return JobAnalysisResult(
            core_objective="Build reliable software.",
            seniority="mid",
            summary=f"Analysis of: {jd_text}",
        )


class _Runtime:
    def close(self) -> None:
        pass


@pytest.fixture
def paths(tmp_path: Path) -> Namespace:
    return Namespace(**{
        name: str(tmp_path / f"{name}.sqlite3")
        for name in (
            "context_store", "resume_store", "application_store", "job_store",
            "calendar_store", "email_store", "mock_interview_store",
            "job_research_store",
        )
    })


def _capture(
    paths: Namespace, text: str, *, user_id: str = "u1", source_id: str = "job-1"
):
    now = datetime.now(timezone.utc)
    url = f"https://example.test/jobs/{source_id}"
    return SQLiteJobPostingRepository(Path(paths.job_store)).save_captured_detail(
        user_id=user_id,
        detail=JobDetail(
            source_name="test",
            source_job_id=source_id,
            source_url=url,
            title="Software Engineer",
            company_name="Example",
            description=text,
            captured_at=now,
            provenance=Provenance(
                source_name="test", source_job_id=source_id, source_url=url,
                captured_at=now, operation="detail", adapter_version="test-v1",
            ),
        ),
    )


def _match(paths: Namespace, saved, *, user_id: str = "u1", summary: str = "Match"):
    return SQLiteResumeJobMatchStore(Path(paths.resume_store)).save(
        user_id=user_id,
        resume_version_id="resume-v1",
        job_posting_id=saved.posting.id,
        jd_snapshot_id=saved.snapshot.id,
        matcher_version="test-v1",
        evidence_fingerprint="test",
        result=ResumeJobMatchResult(overall_fit="moderate", summary=summary),
    )


def test_analysis_is_visible_after_reopening_the_workspace_and_api(
    paths, api_keys, auth
):
    saved = _capture(paths, "Python required")
    before = WorkspaceReader(paths).jobs(user_id="u1")[0]
    assert before.jd_analysis_status == "none"
    assert before.resume_match_status == "none"
    repository = SQLiteJobPostingRepository(Path(paths.job_store))
    analysis = JobAnalysisService(repository, _Analyzer()).analyze(
        user_id="u1", job_posting_id=saved.posting.id,
        jd_snapshot_id=saved.snapshot.id,
    )

    app = create_app(
        runtime_factory=_Runtime,
        workspace_reader_factory=lambda: WorkspaceReader(paths),
        api_key_store_factory=lambda: api_keys,
        action_center_factory=lambda: None,
    )
    with TestClient(app) as client:
        for _ in range(2):
            response = client.get("/v1/jobs", headers=auth)
            assert response.status_code == 200
            [job] = response.json()
            assert job["jd_analysis_status"] == "ready"
            assert job["jd_snapshot_id"] == analysis.jd_snapshot_id
            assert job["analysis_summary"] == "Analysis of: Python required"
            assert job["analyzed_at"] is not None
            assert job["resume_match_status"] == "none"


def test_new_snapshot_keeps_the_analysis_history_until_that_snapshot_is_analyzed(paths):
    first = _capture(paths, "Python required")
    repository = SQLiteJobPostingRepository(Path(paths.job_store))
    service = JobAnalysisService(repository, _Analyzer())
    original = service.analyze(user_id="u1", job_posting_id=first.posting.id)
    current = _capture(paths, "Go required")

    service.analyze(
        user_id="u1", job_posting_id=current.posting.id,
        jd_snapshot_id=first.snapshot.id,
    )
    [pending] = WorkspaceReader(paths).jobs(user_id="u1")
    assert pending.jd_snapshot_id == current.snapshot.id
    assert pending.jd_analysis_status == "stale"
    assert pending.jd_analysis_version == 1
    assert pending.analysis_summary == original.analysis.job_summary

    service.analyze(user_id="u1", job_posting_id=current.posting.id)
    [ready] = WorkspaceReader(paths).jobs(user_id="u1")
    assert ready.jd_analysis_status == "ready"
    assert ready.jd_analysis_version == 2
    assert ready.analysis_summary == "Analysis of: Go required"
    assert repository.get_analysis(user_id="u1", analysis_id=original.id) is not None


def test_match_without_jd_analysis_has_its_own_state_and_becomes_historical(paths):
    first = _capture(paths, "Python required")
    _match(paths, first)
    [matched] = WorkspaceReader(paths).jobs(user_id="u1")
    assert matched.jd_analysis_status == "none"
    assert matched.analysis_summary is None
    assert matched.resume_match_status == "ready"
    assert matched.resume_match_fit == "moderate"

    _capture(paths, "Go required")
    [changed] = WorkspaceReader(paths).jobs(user_id="u1")
    assert changed.jd_analysis_status == "none"
    assert changed.resume_match_status == "stale"
    assert changed.resume_match_at == matched.resume_match_at


def test_later_historical_match_does_not_hide_an_existing_current_match(paths):
    first = _capture(paths, "Python required")
    current = _capture(paths, "Go required")
    current_match = _match(paths, current, summary="Current JD")
    historical_match = _match(paths, first, summary="Historical JD")
    store = SQLiteResumeJobMatchStore(Path(paths.resume_store))
    assert store.find_latest_for_job(
        user_id="u1", job_posting_id=current.posting.id
    ).id == historical_match.id

    [job] = WorkspaceReader(paths).jobs(user_id="u1")
    assert job.jd_snapshot_id == current.snapshot.id
    assert job.jd_analysis_status == "none"
    assert job.resume_match_status == "ready"
    assert job.resume_match_at == current_match.created_at


@pytest.mark.parametrize("error", [
    AgentWorkerError("TEST_FAILURE", "Analysis failed", retryable=True),
    CancelledError(),
])
@pytest.mark.parametrize("has_history", [False, True])
def test_failed_or_cancelled_analysis_never_completes_the_current_snapshot(
    paths, error, has_history
):
    first = _capture(paths, "Python required")
    repository = SQLiteJobPostingRepository(Path(paths.job_store))
    if has_history:
        JobAnalysisService(repository, _Analyzer()).analyze(
            user_id="u1", job_posting_id=first.posting.id
        )
    current = _capture(paths, "Go required")

    with pytest.raises(type(error)):
        JobAnalysisService(repository, _Analyzer(error)).analyze(
            user_id="u1", job_posting_id=current.posting.id
        )

    [job] = WorkspaceReader(paths).jobs(user_id="u1")
    assert job.jd_analysis_status == ("stale" if has_history else "none")
    assert job.resume_match_status == "none"
    assert repository.get_latest_analysis(
        user_id="u1", job_posting_id=current.posting.id
    ) is None


def test_match_query_filters_snapshot_job_and_user(paths):
    saved = _capture(paths, "Python required")
    match = _match(paths, saved)
    store = SQLiteResumeJobMatchStore(Path(paths.resume_store))

    assert store.find_latest_for_job(
        user_id="u1", job_posting_id=saved.posting.id,
        jd_snapshot_id=saved.snapshot.id,
    ).id == match.id
    for user_id, job_id, snapshot_id in (
        ("u2", saved.posting.id, saved.snapshot.id),
        ("u1", "other-job", saved.snapshot.id),
        ("u1", saved.posting.id, "other-snapshot"),
    ):
        assert store.find_latest_for_job(
            user_id=user_id, job_posting_id=job_id, jd_snapshot_id=snapshot_id
        ) is None
    assert WorkspaceReader(paths).jobs(user_id="u2") == ()
