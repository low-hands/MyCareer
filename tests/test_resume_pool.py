"""Tests for the persistent, tenant-aware resume pool."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from career_agent.resumes.models import ResumeData
from career_agent.resumes.repository import DuplicateResumeError, ResumeNotFoundError
from career_agent.resumes.service import ResumePoolService
from career_agent.resumes.sqlite_repo import SQLiteResumeVersionRepository


def _parsed_resume(name: str = "张三") -> ResumeData:
    return ResumeData(
        name=name,
        skills=["提示词工程"],
        technologies=["Python", "LangGraph"],
    )


def _create_version(
    repository: SQLiteResumeVersionRepository,
    *,
    tenant_id: str = "tenant-a",
    role_type: str = "Agent开发",
    version_id: str = "version-1",
    file_hash: str = "hash-1",
):
    return repository.create_version(
        version_id=version_id,
        tenant_id=tenant_id,
        role_type=role_type,
        note="通用版",
        original_filename="resume.pdf",
        stored_file_path="/tmp/resume.pdf",
        file_hash=file_hash,
        parsed_data=_parsed_resume(),
    )


class TestSQLiteResumeVersionRepository:
    def test_versions_increment_within_each_tenant_and_role_pool(
        self,
        tmp_path: Path,
    ) -> None:
        repository = SQLiteResumeVersionRepository(tmp_path / "career-agent.db")

        first = _create_version(repository, version_id="a1", file_hash="hash-a1")
        second = _create_version(repository, version_id="a2", file_hash="hash-a2")
        other_role = _create_version(
            repository,
            role_type="后端开发",
            version_id="a3",
            file_hash="hash-a3",
        )
        other_tenant = _create_version(
            repository,
            tenant_id="tenant-b",
            version_id="b1",
            file_hash="hash-b1",
        )

        assert first.version_number == 1
        assert second.version_number == 2
        assert other_role.version_number == 1
        assert other_tenant.version_number == 1

    def test_list_versions_is_tenant_isolated(self, tmp_path: Path) -> None:
        repository = SQLiteResumeVersionRepository(tmp_path / "career-agent.db")
        _create_version(repository, tenant_id="tenant-a", version_id="a1", file_hash="a")
        _create_version(repository, tenant_id="tenant-b", version_id="b1", file_hash="b")

        tenant_a_versions = repository.list_versions("tenant-a")

        assert [version.version_id for version in tenant_a_versions] == ["a1"]

    def test_duplicate_file_is_rejected_only_within_same_pool(
        self,
        tmp_path: Path,
    ) -> None:
        repository = SQLiteResumeVersionRepository(tmp_path / "career-agent.db")
        _create_version(repository, version_id="a1", file_hash="same-hash")

        with pytest.raises(DuplicateResumeError):
            _create_version(repository, version_id="a2", file_hash="same-hash")

        other_pool = _create_version(
            repository,
            role_type="大模型算法",
            version_id="a3",
            file_hash="same-hash",
        )
        assert other_pool.version_number == 1

    def test_active_version_is_persisted_and_tenant_isolated(
        self,
        tmp_path: Path,
    ) -> None:
        database_path = tmp_path / "career-agent.db"
        repository = SQLiteResumeVersionRepository(database_path)
        created = _create_version(repository)
        repository.set_active_version("tenant-a", created.version_id)

        reopened = SQLiteResumeVersionRepository(database_path)

        assert reopened.get_active_version("tenant-a") == created
        assert reopened.get_active_version("tenant-b") is None

        with pytest.raises(ResumeNotFoundError):
            reopened.set_active_version("tenant-b", created.version_id)


class TestResumePoolService:
    @pytest.mark.asyncio
    async def test_add_version_parses_copies_and_persists_resume(
        self,
        tmp_path: Path,
    ) -> None:
        source = tmp_path / "agent-resume.txt"
        source.write_text("张三\nAgent开发\nPython", encoding="utf-8")
        expected = _parsed_resume()
        model = MagicMock()
        model.with_structured_output.return_value = model
        model.ainvoke = AsyncMock(return_value=expected)
        repository = SQLiteResumeVersionRepository(tmp_path / "career-agent.db")
        service = ResumePoolService(
            repository=repository,
            parser_model=model,
            storage_dir=tmp_path / "stored-resumes",
        )

        created = await service.add_version(
            tenant_id="tenant-a",
            role_type="Agent开发",
            file_path=source,
            note="通用版",
        )

        assert created.role_type == "Agent开发"
        assert created.version_number == 1
        assert created.parsed_data.name == "张三"
        assert Path(created.stored_file_path).read_text(encoding="utf-8") == source.read_text(encoding="utf-8")
        assert repository.get_version("tenant-a", created.version_id) == created

    @pytest.mark.asyncio
    async def test_duplicate_upload_does_not_call_llm_twice(
        self,
        tmp_path: Path,
    ) -> None:
        source = tmp_path / "resume.txt"
        source.write_text("same resume", encoding="utf-8")
        model = MagicMock()
        model.with_structured_output.return_value = model
        model.ainvoke = AsyncMock(return_value=_parsed_resume())
        repository = SQLiteResumeVersionRepository(tmp_path / "career-agent.db")
        service = ResumePoolService(
            repository=repository,
            parser_model=model,
            storage_dir=tmp_path / "stored-resumes",
        )

        await service.add_version(
            tenant_id="tenant-a",
            role_type="Agent开发",
            file_path=source,
        )
        model.ainvoke.reset_mock()

        with pytest.raises(DuplicateResumeError):
            await service.add_version(
                tenant_id="tenant-a",
                role_type="Agent开发",
                file_path=source,
            )

        model.ainvoke.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_parse_failure_leaves_no_database_or_file_record(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        source = tmp_path / "resume.txt"
        source.write_text("broken parse", encoding="utf-8")
        repository = SQLiteResumeVersionRepository(tmp_path / "career-agent.db")
        service = ResumePoolService(
            repository=repository,
            parser_model=MagicMock(),
            storage_dir=tmp_path / "stored-resumes",
        )
        parse = AsyncMock(side_effect=ValueError("parse failed"))
        monkeypatch.setattr("career_agent.resumes.service.load_resume", parse)

        with pytest.raises(ValueError, match="parse failed"):
            await service.add_version(
                tenant_id="tenant-a",
                role_type="Agent开发",
                file_path=source,
            )

        assert repository.list_versions("tenant-a") == []
        assert not list((tmp_path / "stored-resumes").rglob("*.txt"))

    @pytest.mark.asyncio
    async def test_role_and_tenant_are_required(self, tmp_path: Path) -> None:
        service = ResumePoolService(
            repository=SQLiteResumeVersionRepository(tmp_path / "career-agent.db"),
            parser_model=MagicMock(),
            storage_dir=tmp_path / "stored-resumes",
        )

        with pytest.raises(ValueError, match="tenant_id"):
            await service.add_version(tenant_id=" ", role_type="Agent开发", file_path="resume.pdf")
        with pytest.raises(ValueError, match="role_type"):
            await service.add_version(tenant_id="tenant-a", role_type=" ", file_path="resume.pdf")
