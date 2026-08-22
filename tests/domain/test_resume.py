from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from career_agent.domain.resume import Resume, ResumeVersion, TargetRole


NOW = datetime(2026, 8, 20, tzinfo=timezone.utc)


def test_target_role_and_resume_metadata_exclude_document_content() -> None:
    role = TargetRole(id="target_role_1", user_id="u1", title="AI Engineer", priority=1, created_at=NOW, updated_at=NOW)
    resume = Resume(id="resume_1", user_id="u1", target_role_id=role.id, name="Base resume", latest_version_id="resume_version_1", created_at=NOW, updated_at=NOW)
    version = ResumeVersion(id="resume_version_1", resume_id=resume.id, version_number=1, document_format="pdf", content_sha256="a" * 64, byte_size=42, created_at=NOW)

    assert "content" not in resume.model_dump()
    assert "content" not in version.model_dump()
    assert resume.target_role_id == role.id
    assert version.source_type == "user_import"


def test_resume_version_rejects_invalid_hash_and_version() -> None:
    with pytest.raises(ValidationError):
        ResumeVersion(id="v", resume_id="r", version_number=0, document_format="text", content_sha256="not-a-hash", byte_size=1, created_at=NOW)
