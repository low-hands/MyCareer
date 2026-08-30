from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from career_agent.agent.job_comparison_presenter import render_job_comparison
from career_agent.agent.resume_job_match_contracts import (
    RequirementAssessment,
    ResumeJobMatchResult,
    ResumeMatchEvidence,
)
from career_agent.domain.job_comparison import DIMENSION_ORDER, ComparisonCell
from career_agent.services.job_comparison import (
    JobComparisonInputNotFoundError,
    JobComparisonService,
)
from career_agent.storage.resume_job_matches import StoredResumeJobMatch


NOW = datetime(2026, 8, 31, tzinfo=timezone.utc)


def requirement(status: str) -> RequirementAssessment:
    evidence = (
        (ResumeMatchEvidence(source_locator="p1", source_quote="built it"),)
        if status in {"matched", "partial"}
        else ()
    )
    return RequirementAssessment(
        requirement="Python",
        jd_quote="需要 Python",
        status=status,
        rationale="because",
        resume_evidence=evidence,
    )


def match(job_posting_id: str, *, overall_fit: str, statuses: tuple[str, ...]):
    return StoredResumeJobMatch(
        id=f"match-{job_posting_id}",
        user_id="u1",
        resume_version_id="rv1",
        job_posting_id=job_posting_id,
        jd_snapshot_id=f"snap-{job_posting_id}",
        matcher_version="resume-job-match-v1",
        evidence_fingerprint="fp",
        result=ResumeJobMatchResult(
            overall_fit=overall_fit,
            summary="summary",
            requirements=tuple(requirement(status) for status in statuses),
        ),
        created_at=NOW - timedelta(days=1),
    )


class Jobs:
    def __init__(self, records: dict[str, object]) -> None:
        self.records = records

    def get_job(self, *, user_id, job_posting_id):
        return self.records.get(job_posting_id)


class Matches:
    def __init__(self, matches: dict[str, StoredResumeJobMatch]) -> None:
        self.matches = matches
        self.lookups: list[str] = []

    def find_latest_for_job(self, *, user_id, job_posting_id):
        self.lookups.append(job_posting_id)
        return self.matches.get(job_posting_id)


def job(
    job_posting_id: str,
    *,
    city: str | None = "上海",
    salary: str | None = "30-50K",
    availability: str = "active",
):
    from types import SimpleNamespace

    return SimpleNamespace(
        posting=SimpleNamespace(
            id=job_posting_id, title=f"岗位{job_posting_id}", company_name="Acme"
        ),
        city=city,
        salary=salary,
        availability_status=availability,
    )


def build(jobs: dict[str, object], matches: dict[str, StoredResumeJobMatch]):
    return JobComparisonService(Jobs(jobs), Matches(matches))


def cell(comparison, row_index: int, dimension: str) -> ComparisonCell:
    position = DIMENSION_ORDER.index(dimension)
    return comparison.rows[row_index].cells[position]


def test_a_job_without_a_match_is_unknown_rather_than_matched_on_the_spot() -> None:
    """Comparison must stay cheap and side-effect free.

    Quietly matching an unmatched job would make the tool expensive, and would
    also mean the user's comparison silently spent a worker call they never
    asked for.
    """
    matches = Matches({})
    service = JobComparisonService(Jobs({"a": job("a"), "b": job("b")}), matches)

    comparison = service.compare(user_id="u1", job_posting_ids=("a", "b"))

    assert cell(comparison, 0, "resume_fit").value == "unknown"
    assert cell(comparison, 0, "requirement_coverage").value == "unknown"
    assert comparison.jobs_without_match == ("岗位a", "岗位b")
    assert matches.lookups == ["a", "b"]


@pytest.mark.parametrize(
    "statuses, expected",
    [
        (("matched", "matched", "matched"), "full"),
        (("matched", "matched", "partial"), "most"),
        (("matched", "missing"), "partial"),
        (("missing", "missing", "partial"), "little"),
    ],
)
def test_requirement_coverage_buckets_reflect_the_graded_requirements(
    statuses, expected
) -> None:
    service = build(
        {"a": job("a"), "b": job("b")},
        {"a": match("a", overall_fit="moderate", statuses=statuses)},
    )

    comparison = service.compare(user_id="u1", job_posting_ids=("a", "b"))

    assert cell(comparison, 0, "requirement_coverage").value == expected


def test_unclear_requirements_stay_out_of_the_denominator() -> None:
    """"The JD did not let us tell" is not a grade and must not become one."""
    service = build(
        {"a": job("a"), "b": job("b")},
        {
            "a": match(
                "a",
                overall_fit="moderate",
                statuses=("matched", "unclear", "unclear"),
            )
        },
    )

    comparison = service.compare(user_id="u1", job_posting_ids=("a", "b"))
    coverage = cell(comparison, 0, "requirement_coverage")

    assert coverage.value == "full"
    assert "2 无法判断" in coverage.basis


def test_a_job_whose_requirements_are_all_unclear_is_unknown_not_a_middle_grade() -> None:
    service = build(
        {"a": job("a"), "b": job("b")},
        {"a": match("a", overall_fit="insufficient_evidence", statuses=("unclear",))},
    )

    comparison = service.compare(user_id="u1", job_posting_ids=("a", "b"))

    assert cell(comparison, 0, "requirement_coverage").value == "unknown"


def test_an_undisclosed_salary_is_recorded_as_information_not_as_unknown() -> None:
    """A JD that says nothing about pay has told us something about the employer."""
    service = build({"a": job("a", salary=None), "b": job("b")}, {})

    comparison = service.compare(user_id="u1", job_posting_ids=("a", "b"))

    assert cell(comparison, 0, "salary_disclosure").value == "undisclosed"
    assert cell(comparison, 1, "salary_disclosure").value == "disclosed"


def test_location_needs_both_a_job_city_and_a_preference() -> None:
    service = build({"a": job("a", city=None), "b": job("b", city="杭州")}, {})

    without_preference = service.compare(user_id="u1", job_posting_ids=("a", "b"))
    with_preference = service.compare(
        user_id="u1", job_posting_ids=("a", "b"), preferred_city="上海"
    )

    assert cell(without_preference, 1, "location").value == "unknown"
    assert cell(with_preference, 0, "location").value == "unknown"
    assert cell(with_preference, 1, "location").value == "differs"


def test_a_dimension_unknown_for_every_job_is_reported_as_carrying_no_signal() -> None:
    """An all-blank column reads like "no difference here" unless it is named."""
    service = build(
        {"a": job("a", city=None), "b": job("b", city=None)},
        {},
    )

    comparison = service.compare(user_id="u1", job_posting_ids=("a", "b"))

    assert set(comparison.uninformative_dimensions) == {
        "resume_fit",
        "requirement_coverage",
        "location",
    }
    assert "salary_disclosure" not in comparison.uninformative_dimensions


def test_rows_keep_the_caller_order_and_carry_no_ranking_field() -> None:
    service = build({"a": job("a"), "b": job("b"), "c": job("c")}, {})

    comparison = service.compare(user_id="u1", job_posting_ids=("c", "a", "b"))
    serialized = comparison.model_dump()

    assert [row.job_posting_id for row in comparison.rows] == ["c", "a", "b"]
    assert not {"score", "total", "rank", "weight", "best"} & set(serialized)
    assert all(
        not {"score", "total", "rank", "weight"} & set(row)
        for row in serialized["rows"]
    )


def test_a_repeated_job_counts_once() -> None:
    """Otherwise one job read as two independent data points in the matrix."""
    service = build({"a": job("a"), "b": job("b")}, {})

    comparison = service.compare(user_id="u1", job_posting_ids=("a", "b", "a"))

    assert [row.job_posting_id for row in comparison.rows] == ["a", "b"]


def test_comparing_needs_at_least_two_distinct_jobs() -> None:
    service = build({"a": job("a")}, {})

    with pytest.raises(ValueError, match="at least two distinct"):
        service.compare(user_id="u1", job_posting_ids=("a", "a"))


def test_a_foreign_job_is_rejected_rather_than_skipped(tmp_path) -> None:
    service = build({"a": job("a")}, {})

    with pytest.raises(JobComparisonInputNotFoundError):
        service.compare(user_id="u1", job_posting_ids=("a", "not-mine"))


def test_the_rendered_matrix_shows_no_total_and_names_the_blank_columns() -> None:
    service = build(
        {"a": job("a", city=None), "b": job("b", city=None)},
        {"b": match("b", overall_fit="strong", statuses=("matched",))},
    )

    rendered = render_job_comparison(
        service.compare(user_id="u1", job_posting_ids=("a", "b"))
    )

    table = [line for line in rendered.splitlines() if line.startswith("|")]
    header = table[0]

    assert "不是排名" in rendered
    assert "这些维度分不出差别" in rendered
    # The blank column is named rather than left for the reader to interpret.
    assert "城市" in rendered
    # No total or ranking column can sneak into the matrix itself; the only
    # mention of a total anywhere is the caveat explaining its absence.
    assert not {"总分", "综合", "排名", "权重"} & set(header.split("|"))
    assert len(header.split("|")) == len(DIMENSION_ORDER) + 3
