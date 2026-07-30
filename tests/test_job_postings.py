"""Tests for source-grounded, tenant-aware job posting persistence."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from career_agent.jobs.models import JobPosting
from career_agent.jobs.repository import JobPostingNotFoundError
from career_agent.jobs.sqlite_repo import SQLiteJobPostingRepository


def _posting(
    *,
    job_id: str = "job-1",
    tenant_id: str = "tenant-a",
    source: str = "boss",
    external_id: str = "boss-123",
    detail_locator: str = "security-123",
    source_url: str = "https://example.com/jobs/123",
    title: str = "大模型开发工程师",
    description: str = "",
    detail_level: str = "summary",
    fetched_at: datetime | None = None,
) -> JobPosting:
    now = fetched_at or datetime(2026, 7, 30, tzinfo=UTC)
    return JobPosting(
        job_id=job_id,
        tenant_id=tenant_id,
        source=source,
        external_id=external_id,
        detail_locator=detail_locator,
        source_url=source_url,
        title=title,
        company_name="示例科技",
        location="上海",
        employment_type="全职",
        salary="25-45K",
        experience_required="3-5年",
        education_required="本科",
        company_industry="人工智能",
        company_size="100-499人",
        company_stage="B轮",
        description=description,
        detail_level=detail_level,
        fetched_at=now,
        created_at=now,
        updated_at=now,
    )


class TestSQLiteJobPostingRepository:
    def test_upsert_and_get_are_tenant_isolated(self, tmp_path: Path) -> None:
        repository = SQLiteJobPostingRepository(tmp_path / "career-agent.db")
        tenant_a = repository.upsert(_posting())
        tenant_b = repository.upsert(
            _posting(
                job_id="job-2",
                tenant_id="tenant-b",
            )
        )

        assert repository.get("tenant-a", tenant_a.job_id) == tenant_a
        assert repository.get("tenant-a", tenant_b.job_id) is None
        assert tenant_a.detail_locator == "security-123"
        assert repository.get("tenant-b", tenant_a.job_id) is None

    def test_same_source_external_id_updates_existing_posting(self, tmp_path: Path) -> None:
        repository = SQLiteJobPostingRepository(tmp_path / "career-agent.db")
        first = repository.upsert(_posting())
        later = datetime(2026, 7, 31, tzinfo=UTC)

        refreshed = repository.upsert(
            _posting(
                job_id="different-proposed-id",
                title="大模型应用开发工程师",
                fetched_at=later,
            )
        )

        assert refreshed.job_id == first.job_id
        assert refreshed.title == "大模型应用开发工程师"
        assert refreshed.created_at == first.created_at
        assert refreshed.fetched_at == later

    def test_normalized_url_is_used_when_external_id_is_missing(self, tmp_path: Path) -> None:
        repository = SQLiteJobPostingRepository(tmp_path / "career-agent.db")
        first = repository.upsert(
            _posting(
                external_id="",
                source_url="HTTPS://Example.com/jobs/123/?utm_source=test#details",
            )
        )

        duplicate = repository.upsert(
            _posting(
                job_id="different-proposed-id",
                external_id="",
                source_url="https://example.com/jobs/123",
            )
        )

        assert duplicate.job_id == first.job_id

    def test_content_fingerprint_is_used_for_manual_posting(self, tmp_path: Path) -> None:
        repository = SQLiteJobPostingRepository(tmp_path / "career-agent.db")
        first = repository.upsert(
            _posting(
                source="manual",
                external_id="",
                source_url="",
                description="负责 Agent 平台开发。",
            )
        )

        duplicate = repository.upsert(
            _posting(
                job_id="different-proposed-id",
                source="manual",
                external_id="",
                source_url="",
                description="负责 Agent 平台开发。",
            )
        )

        assert duplicate.job_id == first.job_id

    def test_summary_refresh_does_not_downgrade_full_detail(self, tmp_path: Path) -> None:
        repository = SQLiteJobPostingRepository(tmp_path / "career-agent.db")
        full = repository.upsert(
            _posting(
                description="完整岗位职责和要求",
                detail_level="full",
            )
        )
        summary_refresh = _posting(
            job_id="different-proposed-id",
            title="更新后的岗位标题",
            fetched_at=full.fetched_at + timedelta(days=1),
        ).model_copy(
            update={
                "experience_required": "",
                "company_size": "",
            }
        )

        refreshed = repository.upsert(summary_refresh)

        assert refreshed.title == "更新后的岗位标题"
        assert refreshed.detail_level == "full"
        assert refreshed.description == "完整岗位职责和要求"
        assert refreshed.experience_required == "3-5年"
        assert refreshed.company_size == "100-499人"

    def test_save_full_detail_and_get_many_preserve_requested_order(self, tmp_path: Path) -> None:
        repository = SQLiteJobPostingRepository(tmp_path / "career-agent.db")
        first = repository.upsert(_posting(job_id="job-1", external_id="external-1"))
        second = repository.upsert(_posting(job_id="job-2", external_id="external-2"))

        detailed = repository.save_full_detail(
            tenant_id="tenant-a",
            job_id=second.job_id,
            description="完整 JD",
            fetched_at=datetime(2026, 7, 31, tzinfo=UTC),
        )
        selected = repository.get_many(
            "tenant-a",
            [second.job_id, first.job_id, "missing"],
        )

        assert detailed.detail_level == "full"
        assert detailed.description == "完整 JD"
        assert [posting.job_id for posting in selected] == ["job-2", "job-1"]

    def test_save_full_detail_rejects_cross_tenant_access(self, tmp_path: Path) -> None:
        repository = SQLiteJobPostingRepository(tmp_path / "career-agent.db")
        created = repository.upsert(_posting())

        with pytest.raises(JobPostingNotFoundError):
            repository.save_full_detail(
                tenant_id="tenant-b",
                job_id=created.job_id,
                description="不应写入",
            )
