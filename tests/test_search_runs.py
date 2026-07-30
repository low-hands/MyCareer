"""Tests for immutable search results and deterministic follow-up resolution."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from career_agent.jobs.models import JobPosting
from career_agent.jobs.search_models import (
    AmbiguousSearchResult,
    ResolvedSearchResult,
    SearchCriteria,
    SearchResultNotFound,
    SearchResultSelector,
    SearchRun,
)
from career_agent.jobs.search_repository import DuplicateSearchRunError
from career_agent.jobs.search_service import SearchRunService
from career_agent.jobs.search_sqlite_repo import SQLiteSearchRunRepository
from career_agent.jobs.sqlite_repo import SQLiteJobPostingRepository


def _posting(
    *,
    job_id: str,
    tenant_id: str = "tenant-a",
    external_id: str | None = None,
    title: str = "Agent 开发工程师",
    company_name: str = "腾讯科技",
    location: str = "深圳",
    salary: str = "25-45K",
) -> JobPosting:
    now = datetime(2026, 7, 30, tzinfo=UTC)
    return JobPosting(
        job_id=job_id,
        tenant_id=tenant_id,
        source="boss",
        external_id=external_id or f"external-{job_id}",
        title=title,
        company_name=company_name,
        location=location,
        salary=salary,
        fetched_at=now,
        created_at=now,
        updated_at=now,
    )


def _run(
    *,
    run_id: str = "run-1",
    tenant_id: str = "tenant-a",
    thread_id: str = "thread-1",
    job_ids: tuple[str, ...] = ("job-1", "job-2"),
    candidate_count: int = 2,
    created_at: datetime | None = None,
) -> SearchRun:
    return SearchRun(
        run_id=run_id,
        tenant_id=tenant_id,
        thread_id=thread_id,
        user_query="帮我找 Agent 开发岗位",
        criteria=SearchCriteria(
            query="Agent 开发",
            cities=["深圳", "北京"],
        ),
        job_ids=job_ids,
        candidate_count=candidate_count,
        created_at=created_at or datetime(2026, 7, 30, tzinfo=UTC),
    )


def _repositories(
    tmp_path: Path,
) -> tuple[SQLiteJobPostingRepository, SQLiteSearchRunRepository]:
    database_path = tmp_path / "career-agent.db"
    postings = SQLiteJobPostingRepository(database_path)
    runs = SQLiteSearchRunRepository(database_path)
    return postings, runs


class TestSearchModels:
    def test_search_criteria_normalizes_lists_without_inventing_values(self) -> None:
        criteria = SearchCriteria(
            query="  Agent 开发 ",
            cities=[" 上海 ", "上海", "", "北京"],
        )

        assert criteria.query == "Agent 开发"
        assert criteria.cities == ["上海", "北京"]

    def test_search_criteria_rejects_invalid_salary_range(self) -> None:
        with pytest.raises(ValidationError):
            SearchCriteria(salary_min_k=-1)

        with pytest.raises(ValidationError):
            SearchCriteria(salary_min_k=40, salary_max_k=20)

    def test_search_run_requires_unique_results_covered_by_candidate_count(self) -> None:
        with pytest.raises(ValidationError):
            _run(job_ids=())

        with pytest.raises(ValidationError):
            _run(job_ids=("job-1", "job-1"))

        with pytest.raises(ValidationError):
            _run(job_ids=("job-1", "job-2"), candidate_count=1)

    def test_selector_requires_at_least_one_reference_field(self) -> None:
        with pytest.raises(ValidationError):
            SearchResultSelector()


class TestSQLiteSearchRunRepository:
    def test_create_round_trips_ordered_job_references(self, tmp_path: Path) -> None:
        postings, runs = _repositories(tmp_path)
        postings.upsert(_posting(job_id="job-1"))
        postings.upsert(_posting(job_id="job-2", external_id="external-2"))
        expected = _run()

        created = runs.create(expected)
        reloaded = runs.get("tenant-a", expected.run_id)

        assert created == expected
        assert reloaded == expected
        assert reloaded is not None
        assert reloaded.job_ids == ("job-1", "job-2")

    def test_create_rejects_missing_or_cross_tenant_jobs(self, tmp_path: Path) -> None:
        postings, runs = _repositories(tmp_path)
        postings.upsert(_posting(job_id="job-1", tenant_id="tenant-b"))

        with pytest.raises(ValueError, match="not found for tenant"):
            runs.create(_run(job_ids=("job-1",), candidate_count=1))

    def test_duplicate_run_id_is_reported_as_domain_error(self, tmp_path: Path) -> None:
        postings, runs = _repositories(tmp_path)
        postings.upsert(_posting(job_id="job-1"))
        first = _run(job_ids=("job-1",), candidate_count=1)
        runs.create(first)

        with pytest.raises(DuplicateSearchRunError):
            runs.create(first)

    def test_latest_and_list_are_scoped_by_tenant_and_thread(self, tmp_path: Path) -> None:
        postings, runs = _repositories(tmp_path)
        postings.upsert(_posting(job_id="a1"))
        postings.upsert(_posting(job_id="a2", external_id="external-a2"))
        postings.upsert(_posting(job_id="b1", tenant_id="tenant-b"))
        start = datetime(2026, 7, 30, tzinfo=UTC)
        older = _run(
            run_id="older",
            job_ids=("a1",),
            candidate_count=1,
            created_at=start,
        )
        newer = _run(
            run_id="newer",
            job_ids=("a2",),
            candidate_count=1,
            created_at=start + timedelta(minutes=1),
        )
        other_tenant = _run(
            run_id="other",
            tenant_id="tenant-b",
            job_ids=("b1",),
            candidate_count=1,
            created_at=start + timedelta(minutes=2),
        )
        runs.create(older)
        runs.create(newer)
        runs.create(other_tenant)

        assert runs.get_latest("tenant-a", "thread-1") == newer
        assert runs.get_latest("tenant-b", "thread-1") == other_tenant
        assert runs.get("tenant-a", other_tenant.run_id) is None
        assert runs.list_by_thread("tenant-a", "thread-1") == [newer, older]


class TestSearchRunService:
    def _service(
        self,
        tmp_path: Path,
    ) -> tuple[SQLiteJobPostingRepository, SearchRunService]:
        postings, runs = _repositories(tmp_path)
        postings.upsert(_posting(job_id="job-sz"))
        postings.upsert(
            _posting(
                job_id="job-bj",
                external_id="external-bj",
                location="北京",
                salary="30-50K",
            )
        )
        postings.upsert(
            _posting(
                job_id="job-ali",
                external_id="external-ali",
                title="后端开发工程师",
                company_name="阿里巴巴",
                location="杭州",
            )
        )
        return postings, SearchRunService(
            run_repository=runs,
            posting_repository=postings,
        )

    def test_publish_results_creates_immutable_snapshot(self, tmp_path: Path) -> None:
        _, service = self._service(tmp_path)

        published = service.publish_results(
            tenant_id="tenant-a",
            thread_id="thread-1",
            user_query="看看 Agent 岗位",
            criteria=SearchCriteria(query="Agent"),
            job_ids=["job-sz", "job-bj"],
            candidate_count=8,
            run_id="published",
        )

        assert published.run_id == "published"
        assert published.job_ids == ("job-sz", "job-bj")
        assert published.candidate_count == 8

    def test_resolve_by_position_or_unique_descriptors(self, tmp_path: Path) -> None:
        _, service = self._service(tmp_path)
        service.publish_results(
            tenant_id="tenant-a",
            thread_id="thread-1",
            user_query="岗位列表",
            criteria=SearchCriteria(),
            job_ids=["job-sz", "job-bj", "job-ali"],
            candidate_count=3,
            run_id="published",
        )

        by_position = service.resolve_reference(
            tenant_id="tenant-a",
            thread_id="thread-1",
            selector=SearchResultSelector(position=3),
        )
        by_description = service.resolve_reference(
            tenant_id="tenant-a",
            thread_id="thread-1",
            selector=SearchResultSelector(
                company_name="腾讯",
                title="Agent开发",
                location="北京",
            ),
        )

        assert isinstance(by_position, ResolvedSearchResult)
        assert by_position.posting.job_id == "job-ali"
        assert isinstance(by_description, ResolvedSearchResult)
        assert by_description.posting.job_id == "job-bj"

    def test_ambiguous_reference_returns_display_candidates_without_job_ids(
        self,
        tmp_path: Path,
    ) -> None:
        _, service = self._service(tmp_path)
        service.publish_results(
            tenant_id="tenant-a",
            thread_id="thread-1",
            user_query="岗位列表",
            criteria=SearchCriteria(),
            job_ids=["job-sz", "job-bj", "job-ali"],
            candidate_count=3,
            run_id="published",
        )

        result = service.resolve_reference(
            tenant_id="tenant-a",
            thread_id="thread-1",
            selector=SearchResultSelector(company_name="腾讯"),
        )

        assert isinstance(result, AmbiguousSearchResult)
        assert [candidate.position for candidate in result.candidates] == [1, 2]
        assert [candidate.location for candidate in result.candidates] == ["深圳", "北京"]
        assert all("job_id" not in candidate.model_dump() for candidate in result.candidates)

    def test_missing_run_or_reference_returns_not_found(self, tmp_path: Path) -> None:
        _, service = self._service(tmp_path)

        no_run = service.resolve_reference(
            tenant_id="tenant-a",
            thread_id="thread-1",
            selector=SearchResultSelector(position=1),
        )
        service.publish_results(
            tenant_id="tenant-a",
            thread_id="thread-1",
            user_query="岗位列表",
            criteria=SearchCriteria(),
            job_ids=["job-sz"],
            candidate_count=1,
            run_id="published",
        )
        out_of_range = service.resolve_reference(
            tenant_id="tenant-a",
            thread_id="thread-1",
            selector=SearchResultSelector(position=4),
        )

        assert isinstance(no_run, SearchResultNotFound)
        assert isinstance(out_of_range, SearchResultNotFound)

    def test_empty_posting_field_never_matches_non_empty_selector(
        self,
        tmp_path: Path,
    ) -> None:
        postings, service = self._service(tmp_path)
        postings.upsert(
            _posting(
                job_id="job-no-salary",
                external_id="external-no-salary",
                salary="",
            )
        )
        service.publish_results(
            tenant_id="tenant-a",
            thread_id="thread-1",
            user_query="岗位列表",
            criteria=SearchCriteria(),
            job_ids=["job-no-salary"],
            candidate_count=1,
            run_id="published",
        )

        result = service.resolve_reference(
            tenant_id="tenant-a",
            thread_id="thread-1",
            selector=SearchResultSelector(salary="30-50K"),
        )

        assert isinstance(result, SearchResultNotFound)
