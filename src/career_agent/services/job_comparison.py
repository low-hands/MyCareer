from __future__ import annotations

from career_agent.domain.job_comparison import (
    DIMENSION_ORDER,
    ComparisonCell,
    ComparisonRow,
    DimensionId,
    JobComparison,
)
from career_agent.storage.jobs import JobPostingRepository, StoredJobRecord
from career_agent.storage.resume_job_matches import (
    SQLiteResumeJobMatchStore,
    StoredResumeJobMatch,
)


class JobComparisonInputNotFoundError(ValueError):
    def __init__(self, job_posting_id: str) -> None:
        self.job_posting_id = job_posting_id
        super().__init__("saved job not found or does not belong to the current user")


class JobComparisonService:
    """Lays saved jobs out on fixed axes without scoring or ranking them.

    Every dimension is derived from data already on record, so the comparison
    costs nothing to recompute and, more importantly, has no opportunity to
    invent a grade for something the JD never stated.
    """

    _MAX_JOBS = 10

    def __init__(
        self,
        job_repository: JobPostingRepository,
        match_store: SQLiteResumeJobMatchStore,
    ) -> None:
        self._job_repository = job_repository
        self._match_store = match_store

    def compare(
        self,
        *,
        user_id: str,
        job_posting_ids: tuple[str, ...],
        preferred_city: str | None = None,
    ) -> JobComparison:
        if not user_id.strip():
            raise ValueError("user_id is required")
        # Order is the caller's, and duplicates collapse to their first mention
        # so a repeated job cannot look like two independent data points.
        ordered: list[str] = []
        for job_posting_id in job_posting_ids:
            if job_posting_id not in ordered:
                ordered.append(job_posting_id)
        if len(ordered) < 2:
            raise ValueError("comparing needs at least two distinct saved jobs")
        if len(ordered) > self._MAX_JOBS:
            raise ValueError(f"at most {self._MAX_JOBS} saved jobs can be compared")

        rows: list[ComparisonRow] = []
        without_match: list[str] = []
        for job_posting_id in ordered:
            job = self._job_repository.get_job(
                user_id=user_id,
                job_posting_id=job_posting_id,
            )
            if job is None:
                raise JobComparisonInputNotFoundError(job_posting_id)
            match = self._match_store.find_latest_for_job(
                user_id=user_id,
                job_posting_id=job_posting_id,
            )
            if match is None:
                without_match.append(job.posting.title)
            rows.append(
                ComparisonRow(
                    job_posting_id=job_posting_id,
                    title=job.posting.title,
                    company_name=job.posting.company_name,
                    cells=(
                        self._resume_fit(match),
                        self._requirement_coverage(match),
                        self._salary_disclosure(job),
                        self._location(job, preferred_city),
                        self._availability(job),
                    ),
                )
            )

        notes: list[str] = []
        if without_match:
            notes.append(
                "以下岗位还没有简历匹配记录，因此匹配相关维度为未知，可以先对它们"
                f"运行简历匹配：{'、'.join(without_match)}。"
            )
        return JobComparison(
            rows=tuple(rows),
            uninformative_dimensions=self._uninformative(tuple(rows)),
            jobs_without_match=tuple(without_match),
            notes=tuple(notes),
        )

    @staticmethod
    def _uninformative(rows: tuple[ComparisonRow, ...]) -> tuple[DimensionId, ...]:
        blank: list[DimensionId] = []
        for position, dimension in enumerate(DIMENSION_ORDER):
            if all(row.cells[position].value == "unknown" for row in rows):
                blank.append(dimension)
        return tuple(blank)

    @staticmethod
    def _resume_fit(match: StoredResumeJobMatch | None) -> ComparisonCell:
        if match is None:
            return ComparisonCell(
                dimension="resume_fit",
                value="unknown",
                basis="没有简历匹配记录。",
            )
        return ComparisonCell(
            dimension="resume_fit",
            value=match.result.overall_fit,
            basis=f"来自 {match.created_at.date().isoformat()} 的简历匹配记录。",
        )

    @staticmethod
    def _requirement_coverage(match: StoredResumeJobMatch | None) -> ComparisonCell:
        if match is None:
            return ComparisonCell(
                dimension="requirement_coverage",
                value="unknown",
                basis="没有简历匹配记录。",
            )
        requirements = match.result.requirements
        matched = sum(1 for item in requirements if item.status == "matched")
        partial = sum(1 for item in requirements if item.status == "partial")
        missing = sum(1 for item in requirements if item.status == "missing")
        unclear = sum(1 for item in requirements if item.status == "unclear")
        # Unclear requirements stay out of the denominator. Counting them as
        # anything at all would turn "the JD did not let us tell" into a grade.
        assessable = matched + partial + missing
        detail = (
            f"{len(requirements)} 项要求："
            f"{matched} 满足 / {partial} 部分 / {missing} 缺失 / {unclear} 无法判断。"
        )
        if assessable == 0:
            return ComparisonCell(
                dimension="requirement_coverage",
                value="unknown",
                basis=detail,
            )
        ratio = (matched + 0.5 * partial) / assessable
        if ratio >= 0.9:
            value = "full"
        elif ratio >= 0.7:
            value = "most"
        elif ratio >= 0.4:
            value = "partial"
        else:
            value = "little"
        return ComparisonCell(
            dimension="requirement_coverage",
            value=value,
            basis=detail,
        )

    @staticmethod
    def _salary_disclosure(job: StoredJobRecord) -> ComparisonCell:
        salary = (job.salary or "").strip()
        if not salary:
            return ComparisonCell(
                dimension="salary_disclosure",
                value="undisclosed",
                basis="JD 未写薪资。",
            )
        return ComparisonCell(
            dimension="salary_disclosure",
            value="disclosed",
            basis=f"JD 标注 {salary}。",
        )

    @staticmethod
    def _location(job: StoredJobRecord, preferred_city: str | None) -> ComparisonCell:
        city = (job.city or "").strip()
        preferred = (preferred_city or "").strip()
        if not city:
            return ComparisonCell(
                dimension="location",
                value="unknown",
                basis="JD 未写城市。",
            )
        if not preferred:
            return ComparisonCell(
                dimension="location",
                value="unknown",
                basis=f"岗位城市为 {city}，但你没有设置默认城市，无从比较。",
            )
        if preferred in city or city in preferred:
            return ComparisonCell(
                dimension="location",
                value="matches_preference",
                basis=f"岗位城市 {city} 与默认城市 {preferred} 一致。",
            )
        return ComparisonCell(
            dimension="location",
            value="differs",
            basis=f"岗位城市 {city}，默认城市 {preferred}。",
        )

    @staticmethod
    def _availability(job: StoredJobRecord) -> ComparisonCell:
        status = job.availability_status
        basis = {
            "active": "岗位记录标记为在招。",
            "closed": "岗位记录标记为已关闭。",
            "unknown": "岗位记录没有在招状态，保存后也没有再确认过。",
        }[status]
        return ComparisonCell(
            dimension="availability",
            value=status,
            basis=basis,
        )
