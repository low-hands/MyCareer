from __future__ import annotations

import pytest

from career_agent.agent.resume_analysis_contracts import ResumeAnalysisResult
from career_agent.services.resume_analysis import (
    ResumeAnalysisService,
    ResumeVersionNotFoundError,
)
from career_agent.storage.resumes import ResumeStore, StoredResumeDocument


class RecordingWorker:
    def __init__(self) -> None:
        self.documents: list[StoredResumeDocument] = []

    def analyze(self, document: StoredResumeDocument) -> ResumeAnalysisResult:
        self.documents.append(document)
        return ResumeAnalysisResult(warnings=("test warning",))


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
    service = ResumeAnalysisService(store, worker)

    result = service.analyze_version(user_id="u1", resume_version_id=version_id)

    assert result.warnings == ("test warning",)
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
    service = ResumeAnalysisService(store, worker)

    with pytest.raises(ResumeVersionNotFoundError):
        service.analyze_version(user_id="u1", resume_version_id=version_id)

    assert worker.documents == []


def test_service_rejects_missing_version(tmp_path) -> None:
    worker = RecordingWorker()
    service = ResumeAnalysisService(ResumeStore(tmp_path / "resumes.sqlite3"), worker)

    with pytest.raises(ResumeVersionNotFoundError):
        service.analyze_version(user_id="u1", resume_version_id="missing")

    assert worker.documents == []
