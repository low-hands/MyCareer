from __future__ import annotations

import sqlite3

import pytest

from career_agent.services.resume_export import (
    ResumeExportIntegrityError,
    ResumeExportNotFoundError,
    ResumeExportService,
)
from career_agent.storage.resume_artifacts import SQLiteResumeArtifactStore
from career_agent.storage.resumes import ResumeStore


def seed_export_service(tmp_path):
    path = tmp_path / "resumes.sqlite3"
    resumes = ResumeStore(path)
    role = resumes.create_target_role(user_id="u1", title="AI Engineer", priority=1)
    _, version = resumes.import_document(
        user_id="u1",
        target_role_id=role.id,
        name="AI/平台 简历",
        content=b"# Candidate\n\nPrivate resume content",
        document_format="markdown",
    )
    service = ResumeExportService(resumes, SQLiteResumeArtifactStore(path))
    return service, version


def test_resume_export_creates_idempotent_reference_without_copying_document(
    tmp_path,
) -> None:
    service, version = seed_export_service(tmp_path)

    first = service.prepare_export(user_id="u1", resume_version_id=version.id)
    second = service.prepare_export(user_id="u1", resume_version_id=version.id)

    assert second == first
    assert first.id.startswith("resume_artifact_")
    assert first.filename == "AI_平台 简历-v1.md"
    assert first.media_type == "text/markdown; charset=utf-8"
    delivery = service.read_artifact(user_id="u1", artifact_id=first.id)
    assert delivery.reference == first
    assert delivery.content == b"# Candidate\n\nPrivate resume content"


def test_resume_export_is_scoped_to_current_user(tmp_path) -> None:
    service, version = seed_export_service(tmp_path)
    artifact = service.prepare_export(user_id="u1", resume_version_id=version.id)

    with pytest.raises(ResumeExportNotFoundError):
        service.prepare_export(user_id="other", resume_version_id=version.id)
    with pytest.raises(ResumeExportNotFoundError):
        service.read_artifact(user_id="other", artifact_id=artifact.id)


def test_resume_export_detects_document_corruption_before_delivery(tmp_path) -> None:
    service, version = seed_export_service(tmp_path)
    artifact = service.prepare_export(user_id="u1", resume_version_id=version.id)
    with sqlite3.connect(tmp_path / "resumes.sqlite3") as connection:
        connection.execute(
            "UPDATE resume_version_documents SET content = ? WHERE resume_version_id = ?",
            (b"X" * version.byte_size, version.id),
        )

    with pytest.raises(ResumeExportIntegrityError, match="content does not match"):
        service.read_artifact(user_id="u1", artifact_id=artifact.id)
