"""Application service for publishing and resolving visible search results."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from career_agent.jobs.models import JobPosting
from career_agent.jobs.repository import JobPostingRepository
from career_agent.jobs.search_models import (
    AmbiguousSearchResult,
    ResolvedSearchResult,
    SearchCriteria,
    SearchResolution,
    SearchResultCandidate,
    SearchResultNotFound,
    SearchResultSelector,
    SearchRun,
)
from career_agent.jobs.search_repository import SearchRunRepository


class SearchRunService:
    """Coordinate immutable result snapshots with persisted job postings."""

    def __init__(
        self,
        *,
        run_repository: SearchRunRepository,
        posting_repository: JobPostingRepository,
    ) -> None:
        self._runs = run_repository
        self._postings = posting_repository

    def publish_results(
        self,
        *,
        tenant_id: str,
        thread_id: str,
        user_query: str,
        criteria: SearchCriteria,
        job_ids: list[str] | tuple[str, ...],
        candidate_count: int | None = None,
        run_id: str | None = None,
        created_at: datetime | None = None,
    ) -> SearchRun:
        """Persist the final ordered list that will be shown to the user."""

        ordered_ids = tuple(job_ids)
        run = SearchRun(
            run_id=run_id or uuid4().hex,
            tenant_id=tenant_id,
            thread_id=thread_id,
            user_query=user_query,
            criteria=criteria,
            job_ids=ordered_ids,
            candidate_count=(len(ordered_ids) if candidate_count is None else candidate_count),
            created_at=created_at or datetime.now(UTC),
        )
        return self._runs.create(run)

    def resolve_reference(
        self,
        *,
        tenant_id: str,
        thread_id: str,
        selector: SearchResultSelector,
        run_id: str | None = None,
    ) -> SearchResolution:
        """Resolve a natural-language selector within one visible result set."""

        run = self._runs.get(tenant_id, run_id) if run_id is not None else self._runs.get_latest(tenant_id, thread_id)
        if run is None:
            return SearchResultNotFound(
                run_id=run_id,
                message="No visible search results were found for this conversation.",
            )

        postings = self._postings.get_many(tenant_id, run.job_ids)
        postings_by_id = {posting.job_id: posting for posting in postings}
        visible: list[tuple[int, JobPosting]] = [
            (position, postings_by_id[job_id])
            for position, job_id in enumerate(run.job_ids, start=1)
            if job_id in postings_by_id
        ]
        if len(visible) != len(run.job_ids):
            return SearchResultNotFound(
                run_id=run.run_id,
                message="One or more postings in this search result are unavailable.",
            )

        if selector.position is not None:
            selected = next(
                (
                    item
                    for item in visible
                    if item[0] == selector.position and _posting_matches(item[1], selector, allow_contains=True)
                ),
                None,
            )
            if selected is None:
                return SearchResultNotFound(
                    run_id=run.run_id,
                    message=f"No visible posting matches position {selector.position}.",
                )
            return _resolved(run.run_id, selected)

        exact_matches = [item for item in visible if _posting_matches(item[1], selector, allow_contains=False)]
        matches = exact_matches or [
            item for item in visible if _posting_matches(item[1], selector, allow_contains=True)
        ]

        if not matches:
            return SearchResultNotFound(
                run_id=run.run_id,
                message="No visible posting matches the supplied description.",
            )
        if len(matches) == 1:
            return _resolved(run.run_id, matches[0])

        return AmbiguousSearchResult(
            run_id=run.run_id,
            candidates=tuple(
                SearchResultCandidate(
                    position=position,
                    title=posting.title,
                    company_name=posting.company_name,
                    location=posting.location,
                    salary=posting.salary,
                )
                for position, posting in matches
            ),
        )


def _resolved(
    run_id: str,
    item: tuple[int, JobPosting],
) -> ResolvedSearchResult:
    position, posting = item
    return ResolvedSearchResult(
        run_id=run_id,
        position=position,
        posting=posting,
    )


def _posting_matches(
    posting: JobPosting,
    selector: SearchResultSelector,
    *,
    allow_contains: bool,
) -> bool:
    pairs = (
        (selector.company_name, posting.company_name),
        (selector.title, posting.title),
        (selector.location, posting.location),
        (selector.salary, posting.salary),
    )
    return all(_text_matches(expected, actual, allow_contains=allow_contains) for expected, actual in pairs if expected)


def _text_matches(
    expected: str,
    actual: str,
    *,
    allow_contains: bool,
) -> bool:
    normalized_expected = _normalize_reference_text(expected)
    normalized_actual = _normalize_reference_text(actual)
    if not normalized_expected or not normalized_actual:
        return False
    if normalized_expected == normalized_actual:
        return True
    return allow_contains and (normalized_expected in normalized_actual or normalized_actual in normalized_expected)


def _normalize_reference_text(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalnum())
