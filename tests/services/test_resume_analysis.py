from __future__ import annotations

import pytest

from career_agent.agent.resume_analysis_contracts import (
    ExtractedCareerRecord,
    ResumeAnalysisResult,
)
from career_agent.services.resume_analysis import (
    ResumeAnalysisNotFoundError,
    ResumeAnalysisNotPendingError,
    ResumeAnalysisService,
    ResumeVersionNotFoundError,
)
from career_agent.storage.resume_analysis import SQLiteResumeAnalysisDraftStore
from career_agent.storage.career_history import CareerHistoryStore
from career_agent.storage.resumes import ResumeStore, StoredResumeDocument


class RecordingWorker:
    def __init__(self, result: ResumeAnalysisResult | None = None) -> None:
        self.documents: list[StoredResumeDocument] = []
        self.result = result or ResumeAnalysisResult(warnings=("test warning",))

    def analyze(self, document: StoredResumeDocument) -> ResumeAnalysisResult:
        self.documents.append(document)
        return self.result


def _seed_version(store: ResumeStore, *, user_id: str, content: bytes) -> str:
    role = store.create_target_role(user_id=user_id, title="AI Engineer", priority=1)
    _, version = store.import_document(
        user_id=user_id,
        target_role_id=role.id,
        name="Base Resume",
        content=content,
        document_format="pdf",
    )
    return version.id


def test_service_reads_owned_version_and_passes_exact_document_to_worker(tmp_path) -> None:
    store = ResumeStore(tmp_path / "resumes.sqlite3")
    raw_pdf = b"%PDF-private-resume"
    version_id = _seed_version(store, user_id="u1", content=raw_pdf)
    worker = RecordingWorker()
    draft_store = SQLiteResumeAnalysisDraftStore(tmp_path / "drafts.sqlite3")
    service = ResumeAnalysisService(
        store,
        worker,
        draft_store,
        CareerHistoryStore(tmp_path / "resumes.sqlite3"),
    )

    draft = service.analyze_version(user_id="u1", resume_version_id=version_id)

    assert draft.status == "pending"
    assert draft.result.warnings == ("test warning",)
    assert draft_store.get(user_id="u1", analysis_id=draft.id) == draft
    assert worker.documents == [
        StoredResumeDocument(
            resume_version_id=version_id,
            document_format="pdf",
            raw_bytes=raw_pdf,
        )
    ]


def test_service_hides_foreign_version_and_does_not_call_worker(tmp_path) -> None:
    store = ResumeStore(tmp_path / "resumes.sqlite3")
    version_id = _seed_version(store, user_id="other", content=b"private")
    worker = RecordingWorker()
    service = ResumeAnalysisService(
        store,
        worker,
        SQLiteResumeAnalysisDraftStore(tmp_path / "drafts.sqlite3"),
        CareerHistoryStore(tmp_path / "resumes.sqlite3"),
    )

    with pytest.raises(ResumeVersionNotFoundError):
        service.analyze_version(user_id="u1", resume_version_id=version_id)

    assert worker.documents == []


def test_service_rejects_missing_version(tmp_path) -> None:
    worker = RecordingWorker()
    service = ResumeAnalysisService(
        ResumeStore(tmp_path / "resumes.sqlite3"),
        worker,
        SQLiteResumeAnalysisDraftStore(tmp_path / "drafts.sqlite3"),
        CareerHistoryStore(tmp_path / "resumes.sqlite3"),
    )

    with pytest.raises(ResumeVersionNotFoundError):
        service.analyze_version(user_id="u1", resume_version_id="missing")

    assert worker.documents == []


def test_service_gets_and_idempotently_confirms_analysis(tmp_path) -> None:
    path = tmp_path / "resumes.sqlite3"
    store = ResumeStore(path)
    version_id = _seed_version(store, user_id="u1", content=b"%PDF-resume")
    result = ResumeAnalysisResult(
        records=(
            ExtractedCareerRecord(
                record_type="project",
                title="Career Agent",
                source_locator="page 1, Projects",
                source_quote="Career Agent",
            ),
        )
    )
    draft_store = SQLiteResumeAnalysisDraftStore(path)
    career_store = CareerHistoryStore(path)
    service = ResumeAnalysisService(
        store,
        RecordingWorker(result),
        draft_store,
        career_store,
    )
    draft = service.analyze_version(user_id="u1", resume_version_id=version_id)

    assert service.get_analysis(user_id="u1", analysis_id=draft.id) == draft
    with pytest.raises(ResumeAnalysisNotFoundError):
        service.get_analysis(user_id="other", analysis_id=draft.id)
    confirmed = service.confirm_analysis(user_id="u1", analysis_id=draft.id)
    with pytest.raises(ResumeAnalysisNotPendingError):
        service.confirm_analysis(user_id="u1", analysis_id=draft.id)

    assert len(confirmed.records) == 1
    assert len(confirmed.evidence) == 1
    assert draft_store.get(user_id="u1", analysis_id=draft.id).status == "confirmed"
    assert career_store.list_records(user_id="u1") == confirmed.records
