from __future__ import annotations

import hashlib
import re

from career_agent.domain.resume import ResumeArtifactDelivery, ResumeArtifactReference
from career_agent.storage.resume_artifacts import SQLiteResumeArtifactStore
from career_agent.storage.resumes import ResumeStore


class ResumeExportNotFoundError(ValueError):
    """Raised when a version or artifact is missing for the current user."""


class ResumeExportIntegrityError(ValueError):
    """Raised when immutable artifact metadata no longer matches its document."""


class ResumeExportService:
    _FORMAT = {
        "pdf": ("pdf", "application/pdf"),
        "text": ("txt", "text/plain; charset=utf-8"),
        "markdown": ("md", "text/markdown; charset=utf-8"),
    }

    def __init__(
        self,
        resume_store: ResumeStore,
        artifact_store: SQLiteResumeArtifactStore,
    ) -> None:
        self._resume_store = resume_store
        self._artifact_store = artifact_store

    def prepare_export(
        self, *, user_id: str, resume_version_id: str
    ) -> ResumeArtifactReference:
        stored = self._resume_store.get_version(
            user_id=user_id,
            resume_version_id=resume_version_id,
        )
        document = self._resume_store.read_version_document(
            user_id=user_id,
            resume_version_id=resume_version_id,
        )
        if stored is None or document is None:
            raise ResumeExportNotFoundError("Resume version not found for current user")
        resume, version = stored
        self._verify_document(
            content=document.raw_bytes,
            expected_size=version.byte_size,
            expected_sha256=version.content_sha256,
        )
        extension, media_type = self._FORMAT[document.document_format]
        filename = f"{self._safe_name(resume.name)}-v{version.version_number}.{extension}"
        return self._artifact_store.create(
            user_id=user_id,
            resume_version_id=resume_version_id,
            filename=filename,
            media_type=media_type,
            byte_size=len(document.raw_bytes),
        )

    def read_artifact(
        self, *, user_id: str, artifact_id: str
    ) -> ResumeArtifactDelivery:
        reference = self._artifact_store.get(user_id=user_id, artifact_id=artifact_id)
        if reference is None:
            raise ResumeExportNotFoundError("Resume artifact not found for current user")
        document = self._resume_store.read_version_document(
            user_id=user_id,
            resume_version_id=reference.resume_version_id,
        )
        if document is None:
            raise ResumeExportIntegrityError("Resume artifact document is missing")
        stored = self._resume_store.get_version(
            user_id=user_id,
            resume_version_id=reference.resume_version_id,
        )
        if stored is None:
            raise ResumeExportIntegrityError("Resume artifact version metadata is missing")
        self._verify_document(
            content=document.raw_bytes,
            expected_size=reference.byte_size,
            expected_sha256=stored[1].content_sha256,
        )
        return ResumeArtifactDelivery(reference=reference, content=document.raw_bytes)

    @staticmethod
    def _safe_name(name: str) -> str:
        sanitized = re.sub(r"[\\/\x00-\x1f\x7f]+", "_", name.strip())
        sanitized = re.sub(r"\s+", " ", sanitized).strip(" ._")
        return (sanitized or "resume")[:140]

    @staticmethod
    def _verify_document(
        *, content: bytes, expected_size: int, expected_sha256: str
    ) -> None:
        if len(content) != expected_size:
            raise ResumeExportIntegrityError(
                "Resume artifact size does not match immutable metadata"
            )
        if hashlib.sha256(content).hexdigest() != expected_sha256:
            raise ResumeExportIntegrityError(
                "Resume artifact content does not match immutable metadata"
            )
