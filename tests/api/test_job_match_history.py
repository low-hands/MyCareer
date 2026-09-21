from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path
import sqlite3

import pytest
from fastapi.testclient import TestClient

from career_agent.agent.job_analysis_contracts import JobAnalysisResult, TieredRequirement
from career_agent.agent.resume_job_match_contracts import ResumeJobMatchResult
from career_agent.api.app import create_app
from career_agent.api.reads import WorkspaceReader
from career_agent.domain.job_discovery import JobDetail, Provenance
from career_agent.services.job_analysis import JobAnalysisService
from career_agent.services.resume_job_match import ResumeJobMatchService
from career_agent.storage.career_history import CareerHistoryStore
from career_agent.storage.jobs import SQLiteJobPostingRepository
from career_agent.storage.resumes import ResumeStore
from career_agent.storage.resume_job_matches import SQLiteResumeJobMatchStore
from career_agent.storage.api_keys import CHAT_WRITE


class MatchWorker:
    def __init__(self):
        self.calls = 0

    def match(self, **kwargs: object) -> ResumeJobMatchResult:
        self.calls += 1
        requirements = kwargs["tiered_requirements"]
        return ResumeJobMatchResult(
            overall_fit="moderate", summary="Python fits; Go needs evidence.",
            requirements=[{
                "requirement_id": requirements[0].requirement_id,
                "requirement": requirements[0].text,
                "jd_quote": requirements[0].jd_quote,
                "status": "missing", "rationale": "No Go evidence in the resume.",
                "resume_evidence": [],
            }, {
                "requirement_id": requirements[1].requirement_id,
                "requirement": requirements[1].text,
                "jd_quote": requirements[1].jd_quote,
                "status": "matched", "rationale": "Python project evidence.",
                "resume_evidence": [{"source_locator": "Experience", "source_quote": "Built Python services"}],
            }],
            recommendations=["Describe a Go project."],
            clarification_questions=["Any Go experience?"],
            limitations=["Only the supplied documents were reviewed."],
        )


class AnalysisWorker:
    def analyze(self, *, jd_text: str) -> JobAnalysisResult:
        return JobAnalysisResult(
            core_objective="Hire an engineer.",
            seniority="mid",
            requirements=(
                TieredRequirement(
                    text="Go experience",
                    tier="A",
                    kind="fact",
                    jd_quote="Go required",
                ),
                TieredRequirement(
                    text="Python",
                    tier="A",
                    kind="fact",
                    jd_quote="Python required",
                ),
            ),
            summary="Python and Go engineering role.",
        )


def analyze(paths: Namespace, saved) -> None:
    JobAnalysisService(
        SQLiteJobPostingRepository(Path(paths.job_store)),
        AnalysisWorker(),
    ).analyze(
        user_id="u1",
        job_posting_id=saved.posting.id,
        jd_snapshot_id=saved.snapshot.id,
    )


class Runtime:
    def close(self) -> None:
        pass


@pytest.fixture
def paths(tmp_path: Path) -> Namespace:
    return Namespace(**{
        name: str(tmp_path / f"{name}.sqlite3")
        for name in (
            "context_store", "resume_store", "application_store", "job_store",
            "calendar_store", "email_store", "mock_interview_store", "job_research_store",
        )
    })


def capture(paths: Namespace, text: str, *, user_id: str = "u1"):
    now = datetime.now(timezone.utc)
    return SQLiteJobPostingRepository(Path(paths.job_store)).save_captured_detail(
        user_id=user_id,
        detail=JobDetail(
            source_name="test", source_job_id="one",
            source_url="https://example.test/jobs/one",
            title="Engineer", company_name="Example", description=text,
            captured_at=now,
            provenance=Provenance(
                source_name="test", source_job_id="one",
                source_url="https://example.test/jobs/one",
                captured_at=now, operation="detail", adapter_version="test",
            ),
        ),
    )


def inputs(paths: Namespace):
    resumes = ResumeStore(Path(paths.resume_store))
    role = resumes.create_target_role(user_id="u1", title="Engineer", priority=1)
    resume, first = resumes.import_document(
        user_id="u1", target_role_id=role.id, name="Engineering",
        content=b"Built Python services", document_format="text",
    )
    _, second = resumes.import_document(
        user_id="u1", resume_id=resume.id,
        content=b"Built Python and Go services", document_format="text",
    )
    worker = MatchWorker()
    service = ResumeJobMatchService(
        resumes, SQLiteJobPostingRepository(Path(paths.job_store)),
        CareerHistoryStore(Path(paths.resume_store)), worker,
        SQLiteResumeJobMatchStore(Path(paths.resume_store)),
    )
    return resume, first, second, service, worker


def client_for(paths: Namespace, api_keys) -> TestClient:
    return TestClient(create_app(
        runtime_factory=Runtime,
        workspace_reader_factory=lambda: WorkspaceReader(paths),
        api_key_store_factory=lambda: api_keys,
        action_center_factory=lambda: None,
    ))


def test_history_reads_exact_versions_and_all_report_sections_without_matching(
    paths, api_keys, auth
):
    resume, v1, v2, service, worker = inputs(paths)
    jd1 = capture(paths, "Python required\nGo required")
    analyze(paths, jd1)
    first = service.match(user_id="u1", resume_version_id=v1.id, job_posting_id=jd1.posting.id)
    second = service.match(user_id="u1", resume_version_id=v2.id, job_posting_id=jd1.posting.id)
    jd2 = capture(paths, "Python required\nGo required\nKubernetes required")
    analyze(paths, jd2)
    third = service.match(user_id="u1", resume_version_id=v1.id, job_posting_id=jd2.posting.id)
    cached = service.match(
        user_id="u1", resume_version_id=v1.id, job_posting_id=jd2.posting.id,
        jd_snapshot_id=jd1.snapshot.id,
    )
    assert cached.id == first.id
    assert worker.calls == 3

    with client_for(paths, api_keys) as client:
        for _ in range(2):
            history = client.get(f"/v1/jobs/{jd1.posting.id}/matches", headers=auth).json()
            assert history["total"] == 3
            assert [row["report_id"] for row in history["items"]] == [third.id, second.id, first.id]
            assert [(row["resume_version_number"], row["jd_version"]) for row in history["items"]] == [(1, 2), (2, 1), (1, 1)]
            assert [row["current_jd"] for row in history["items"]] == [True, False, False]
            page = client.get(f"/v1/jobs/{jd1.posting.id}/matches?limit=1&offset=1", headers=auth).json()
            assert page["total"] == 3
            assert [row["report_id"] for row in page["items"]] == [second.id]
            report = client.get(f"/v1/reports/resume_job_match/{first.id}", headers=auth).json()
            meta = report["resume_job_match"]
            assert meta["resume_id"] == resume.id
            assert meta["resume_version_id"] == v1.id
            assert meta["jd_snapshot_id"] == jd1.snapshot.id
            assert meta["resume_created_at"] == v1.created_at.isoformat().replace("+00:00", "Z")
            assert meta["jd_captured_at"] == jd1.snapshot.captured_at.isoformat().replace("+00:00", "Z")
            for text in (
                "整体判断", "要求逐项核对", "Go required", "No Go evidence",
                "Built Python services", "Describe a Go project", "Any Go experience",
                "Only the supplied documents",
            ):
                assert text in report["body"]
        [job] = client.get("/v1/jobs", headers=auth).json()
        assert job["resume_match_count"] == 3
        assert job["jd_analysis_status"] == "ready"
    assert worker.calls == 3


def test_owned_report_keeps_identity_after_original_inputs_disappear(paths, api_keys, auth, issue_key):
    resume, v1, _, service, _ = inputs(paths)
    old = capture(paths, "Python required\nGo required")
    current = capture(paths, "Python required\nGo required\nLeadership")
    analyze(paths, old)
    report = service.match(
        user_id="u1", resume_version_id=v1.id, job_posting_id=current.posting.id,
        jd_snapshot_id=old.snapshot.id,
    )
    assert report.inputs.jd_version == 1
    assert report.inputs.resume_version_number == 1
    SQLiteJobPostingRepository(Path(paths.job_store)).delete_job(user_id="u1", job_posting_id=old.posting.id)
    with sqlite3.connect(paths.resume_store) as connection:
        connection.execute("DELETE FROM resume_version_documents WHERE resume_version_id = ?", (v1.id,))
        connection.execute("DELETE FROM resume_versions WHERE id = ?", (v1.id,))
    foreign = issue_key("u2")
    with client_for(paths, api_keys) as client:
        response = client.get(f"/v1/reports/resume_job_match/{report.id}", headers=auth)
        assert response.status_code == 200
        meta = response.json()["resume_job_match"]
        assert meta["resume_id"] == resume.id
        assert meta["resume_version_number"] == 1
        assert meta["jd_version"] == 1
        assert meta["jd_snapshot_id"] == old.snapshot.id
        assert meta["company_name"] == "Example"
        assert meta["resume_available"] is False
        assert meta["jd_available"] is False
        assert client.get(f"/v1/jobs/{old.posting.id}/matches", headers=auth).json()["total"] == 1
        assert client.get(f"/v1/jobs/{old.posting.id}/matches", headers=foreign).status_code == 404
        assert client.get(f"/v1/reports/resume_job_match/{report.id}", headers=foreign).status_code == 404


def test_empty_history_permissions_and_pagination_validation(paths, api_keys, auth, issue_key):
    own = capture(paths, "Python required")
    foreign = capture(paths, "Confidential", user_id="u2")
    no_read_scope = issue_key("u1", CHAT_WRITE)
    with client_for(paths, api_keys) as client:
        url = f"/v1/jobs/{own.posting.id}/matches"
        assert client.get(url, headers=auth).json() == {"items": [], "total": 0, "limit": 20, "offset": 0}
        assert client.get(url).status_code == 401
        assert client.get(url, headers=no_read_scope).status_code == 403
        assert client.get(f"/v1/jobs/{foreign.posting.id}/matches", headers=auth).status_code == 404
        for query in ("limit=0", "limit=101", "offset=-1"):
            assert client.get(f"{url}?{query}", headers=auth).status_code == 422


def test_legacy_report_uses_only_exact_versions_and_survives_missing_inputs(paths):
    _, v1, _, _, _ = inputs(paths)
    old = capture(paths, "Old JD")
    capture(paths, "New JD")
    store = SQLiteResumeJobMatchStore(Path(paths.resume_store))
    report = store.save(
        user_id="u1", resume_version_id=v1.id, job_posting_id=old.posting.id,
        jd_snapshot_id=old.snapshot.id, matcher_version="legacy",
        evidence_fingerprint="legacy", result=ResumeJobMatchResult(overall_fit="weak", summary="Legacy"),
    )
    with sqlite3.connect(paths.resume_store) as connection:
        connection.execute("ALTER TABLE resume_job_matches DROP COLUMN inputs_json")
        connection.execute("UPDATE schema_versions SET version = 1 WHERE component = 'resume_job_matches'")
    reader = WorkspaceReader(paths)
    meta = reader.report(user_id="u1", kind="resume_job_match", resource_id=report.id).resume_job_match
    assert (meta.resume_version_number, meta.jd_version) == (1, 1)
    assert meta.matcher_version == "legacy"
    SQLiteJobPostingRepository(Path(paths.job_store)).delete_job(user_id="u1", job_posting_id=old.posting.id)
    meta = WorkspaceReader(paths).report(user_id="u1", kind="resume_job_match", resource_id=report.id).resume_job_match
    assert meta.jd_snapshot_id == old.snapshot.id
    assert meta.jd_version is None
    assert meta.jd_available is False
