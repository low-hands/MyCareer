"""The library reminder: one row for the whole shortlist, and no invented facts."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from career_agent.services.action_center import ActionCenterService
from career_agent.storage.action_center import SQLiteActionItemStore


NOW = datetime(2026, 9, 5, 2, 0, tzinfo=timezone.utc)


class _Empty:
    def __init__(self, applied_job_ids: frozenset[str] = frozenset()) -> None:
        self._applied_job_ids = applied_job_ids

    def list_applications(self, **_):
        return ()

    def list_job_posting_ids(self, **_):
        return self._applied_job_ids

    def list_interviews(self, **_):
        return ()

    def list_events(self, **_):
        return ()

    def list_email_events(self, **_):
        return ()


class _Jobs:
    def __init__(self, *ages_in_days: int) -> None:
        self._jobs = tuple(
            type(
                "Summary",
                (),
                {
                    "job_posting_id": f"job-{index}",
                    "last_checked_at": NOW - timedelta(days=age),
                },
            )()
            for index, age in enumerate(ages_in_days)
        )

    def stale_open_job_stats(
        self, *, user_id, checked_before, excluded_job_posting_ids=frozenset()
    ):
        eligible = tuple(
            job.last_checked_at
            for job in self._jobs
            if job.job_posting_id not in excluded_job_posting_ids
            and job.last_checked_at < checked_before
        )
        return len(eligible), min(eligible) if eligible else None


def _service(tmp_path, jobs, *, applied_job_ids=frozenset()) -> ActionCenterService:
    empty = _Empty(applied_job_ids)
    return ActionCenterService(
        SQLiteActionItemStore(tmp_path / "actions.sqlite3"),
        empty,
        empty,
        empty,
        job_repository=jobs,
    )


def test_twenty_untouched_jobs_produce_one_row_not_twenty(tmp_path) -> None:
    """The bound is the feature.

    A brief with twenty "have another look at this job" rows buries the things
    that actually have to happen today under a list the reader can already see
    in the library. Same lesson the follow-up cadence caps already encode.
    """
    service = _service(tmp_path, _Jobs(*([40] * 20)))

    items = service.refresh(user_id="u1", now=NOW)

    assert [item.action_type for item in items] == ["saved_job_review"]
    assert "20 个岗位" in items[0].title


def test_user_resolved_review_stays_resolved_while_count_is_unchanged(tmp_path) -> None:
    service = _service(tmp_path, _Jobs(40, 45))
    item = service.refresh(user_id="u1", now=NOW)[0]
    service.complete_action(user_id="u1", action_item_id=item.id)

    refreshed = service.refresh(user_id="u1", now=NOW + timedelta(days=1))

    assert refreshed == ()


def test_the_reminder_says_what_was_observed_and_not_that_a_job_closed(
    tmp_path,
) -> None:
    """Nothing here can see whether a posting is still live.

    A job is only re-checked when the user happens to open its page again, and
    polling the site is what got the earlier API search blocked. So the item
    reports time since the page was last read and leaves the conclusion to the
    reader; claiming a closure would be inventing an employer decision.
    """
    service = _service(tmp_path, _Jobs(45))

    item = service.refresh(user_id="u1", now=NOW)[0]

    assert "45 天" in item.summary
    for invented in ("下架", "已关闭", "失效"):
        assert invented not in item.title + item.summary


def test_a_recently_checked_library_raises_nothing(tmp_path) -> None:
    service = _service(tmp_path, _Jobs(3, 10, 29))

    assert service.refresh(user_id="u1", now=NOW) == ()


def test_an_applied_job_is_not_presented_as_still_awaiting_a_decision(
    tmp_path,
) -> None:
    service = _service(
        tmp_path,
        _Jobs(45, 45),
        applied_job_ids=frozenset({"job-0"}),
    )

    item = service.refresh(user_id="u1", now=NOW)[0]

    assert "1 个岗位" in item.title


def test_the_library_condition_is_not_bounded_by_the_first_hundred_rows(
    tmp_path,
) -> None:
    # The old implementation read a newest-first display page of 100. With a
    # larger library, the stale tail was exactly what that page omitted.
    service = _service(tmp_path, _Jobs(*([1] * 100), 45))

    item = service.refresh(user_id="u1", now=NOW)[0]

    assert "1 个岗位" in item.title


def test_tidying_the_library_resolves_the_reminder_without_being_completed(
    tmp_path,
) -> None:
    """Derived, so it goes away on its own — and says it went away by itself.

    ``obsolete`` rather than ``completed``: the condition stopped holding, and
    recording that as work the user did would let the history claim credit for
    something nobody performed.
    """
    store = SQLiteActionItemStore(tmp_path / "actions.sqlite3")
    empty = _Empty()
    stale = ActionCenterService(store, empty, empty, empty, job_repository=_Jobs(40))
    raised = stale.refresh(user_id="u1", now=NOW)[0]

    fresh = ActionCenterService(store, empty, empty, empty, job_repository=_Jobs(1))
    assert fresh.refresh(user_id="u1", now=NOW) == ()

    assert store.get(user_id="u1", action_item_id=raised.id).status == "obsolete"


def test_a_deployment_without_a_job_library_still_produces_a_brief(tmp_path) -> None:
    empty = _Empty()
    service = ActionCenterService(
        SQLiteActionItemStore(tmp_path / "actions.sqlite3"), empty, empty, empty
    )

    assert service.refresh(user_id="u1", now=NOW) == ()
