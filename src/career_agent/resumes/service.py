"""Application service for uploading and selecting resume versions."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from uuid import uuid4

from langchain_core.language_models import BaseChatModel

from career_agent.resumes.models import ResumeVersion
from career_agent.resumes.parser import load_resume
from career_agent.resumes.repository import (
    DuplicateResumeError,
    ResumeNotFoundError,
    ResumeVersionRepository,
)


class ResumePoolService:
    """Manage tenant-scoped role pools without exposing persistence details."""

    def __init__(
        self,
        *,
        repository: ResumeVersionRepository,
        parser_model: BaseChatModel,
        storage_dir: str | Path,
    ) -> None:
        self.repository = repository
        self.parser_model = parser_model
        self.storage_dir = Path(storage_dir).expanduser().resolve()

    @staticmethod
    def _validate_scope(tenant_id: str, role_type: str) -> tuple[str, str]:
        normalized_tenant = tenant_id.strip()
        normalized_role = role_type.strip()
        if not normalized_tenant:
            raise ValueError("tenant_id must not be empty")
        if not normalized_role:
            raise ValueError("role_type must not be empty")
        return normalized_tenant, normalized_role

    @staticmethod
    def _hash_file(file_path: Path) -> str:
        digest = hashlib.sha256()
        with file_path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _tenant_storage_key(tenant_id: str) -> str:
        return hashlib.sha256(tenant_id.encode("utf-8")).hexdigest()[:16]

    async def add_version(
        self,
        *,
        tenant_id: str,
        role_type: str,
        file_path: str | Path,
        note: str = "",
    ) -> ResumeVersion:
        tenant_id, role_type = self._validate_scope(tenant_id, role_type)
        source = Path(file_path).expanduser().resolve()
        if not source.exists() or not source.is_file():
            raise FileNotFoundError(f"Resume file not found: {source}")

        file_hash = self._hash_file(source)
        duplicate = self.repository.find_by_hash(tenant_id, role_type, file_hash)
        if duplicate is not None:
            raise DuplicateResumeError(
                f"{duplicate.display_name} already contains the same file ({duplicate.version_id})"
            )

        parsed_data = await load_resume(source, self.parser_model)
        version_id = uuid4().hex
        tenant_directory = self.storage_dir / self._tenant_storage_key(tenant_id)
        destination = tenant_directory / f"{version_id}{source.suffix.lower()}"
        tenant_directory.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

        try:
            return self.repository.create_version(
                version_id=version_id,
                tenant_id=tenant_id,
                role_type=role_type,
                note=note.strip(),
                original_filename=source.name,
                stored_file_path=str(destination),
                file_hash=file_hash,
                parsed_data=parsed_data,
            )
        except Exception:
            destination.unlink(missing_ok=True)
            raise

    def list_versions(
        self,
        tenant_id: str,
        role_type: str | None = None,
    ) -> list[ResumeVersion]:
        normalized_tenant = tenant_id.strip()
        if not normalized_tenant:
            raise ValueError("tenant_id must not be empty")
        normalized_role = role_type.strip() if role_type is not None else None
        return self.repository.list_versions(normalized_tenant, normalized_role)

    def get_version(self, tenant_id: str, version_id: str) -> ResumeVersion:
        version = self.repository.get_version(tenant_id.strip(), version_id.strip())
        if version is None:
            raise ResumeNotFoundError(f"Resume version not found: {version_id}")
        return version

    def resolve_version(self, tenant_id: str, selector: str) -> ResumeVersion:
        """Resolve a full ID or an unambiguous short ID shown by the CLI."""

        selector = selector.strip()
        if not selector:
            raise ResumeNotFoundError("Resume version selector must not be empty")

        exact = self.repository.get_version(tenant_id.strip(), selector)
        if exact is not None:
            return exact

        matches = [
            version
            for version in self.repository.list_versions(tenant_id.strip())
            if version.version_id.startswith(selector)
        ]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ResumeNotFoundError(f"Resume selector is ambiguous: {selector}. Use a longer ID.")
        raise ResumeNotFoundError(f"Resume version not found: {selector}")

    def use_version(self, tenant_id: str, selector: str) -> ResumeVersion:
        version = self.resolve_version(tenant_id, selector)
        self.repository.set_active_version(tenant_id.strip(), version.version_id)
        return version

    def get_active_version(self, tenant_id: str) -> ResumeVersion | None:
        return self.repository.get_active_version(tenant_id.strip())
