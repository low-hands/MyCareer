"""The one thing a shortlist cannot derive: that the user ruled a job out."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from career_agent.domain.job_discovery import JobDetail, Provenance
from career_agent.storage.jobs import SQLiteJobPostingRepository


NOW = datetime(2026, 9, 5, 2, 0, tzinfo=timezone.utc)


def _store(tmp_path: Path) -> SQLiteJobPostingRepository:
    return SQLiteJobPostingRepository(tmp_path / "jobs.sqlite3")


@pytest.fixture
def saved_job():
    def _save(store, *, user_id: str, identity: str = "job-a") -> str:
        detail = JobDetail(
            source_name="boss",
            source_job_id=identity,
            source_url=f"https://example.test/jobs/{identity}",
            title="AI Engineer",
            company_name="Acme",
            description="负责检索系统的端到端优化。",
            city="Shanghai",
            salary="25-35K",
            captured_at=NOW,
            provenance=Provenance(
                source_name="boss",
                source_job_id=identity,
                source_url=f"https://example.test/jobs/{identity}",
                captured_at=NOW,
                operation="detail",
                adapter_version="test-v1",
            ),
        )
        return store.save_captured_detail(user_id=user_id, detail=detail).posting.id

    return _save


def test_a_saved_job_is_on_the_shortlist_without_being_marked(tmp_path, saved_job):
    """Saving is the opt-in. Asking for a second one would be asking twice.

    Everything else about the shortlist is derived — saved, applied to, closed
    by the employer. Requiring an explicit "I might apply" would make the user
    re-state what saving already said, and every job they forgot to mark would
    silently vanish from the list they rely on.
    """
    store = _store(tmp_path)
    job = saved_job(store, user_id="u1")

    listed = store.list_jobs(user_id="u1", include_dismissed=False)

    assert [item.job_posting_id for item in listed] == [job]
    assert listed[0].pursuit_status == "open"


def test_a_dismissed_job_leaves_the_shortlist_but_not_the_database(
    tmp_path, saved_job
):
    """Not shown, still there — the same posture as a settled confirmation.

    Deleting would make "why is this not in my list?" unanswerable and turn a
    misclick into a re-capture. The row stays, and an explicit listing can
    still reach it.
    """
    store = _store(tmp_path)
    job = saved_job(store, user_id="u1")

    assert store.set_pursuit_status(
        user_id="u1", job_posting_id=job, status="dismissed"
    ) is True

    assert store.list_jobs(user_id="u1", include_dismissed=False) == ()
    revealed = store.list_jobs(user_id="u1", include_dismissed=True)
    assert [item.pursuit_status for item in revealed] == ["dismissed"]


def test_ruling_a_job_back_in_costs_one_click(tmp_path, saved_job):
    store = _store(tmp_path)
    job = saved_job(store, user_id="u1")
    store.set_pursuit_status(user_id="u1", job_posting_id=job, status="dismissed")

    assert store.set_pursuit_status(
        user_id="u1", job_posting_id=job, status="open"
    ) is True

    assert [item.job_posting_id for item in store.list_jobs(user_id="u1", include_dismissed=False)] == [job]


def test_setting_the_status_it_already_has_changes_nothing(tmp_path, saved_job):
    """A double-click on a card is a double-click, not an error."""
    store = _store(tmp_path)
    job = saved_job(store, user_id="u1")
    assert store.set_pursuit_status(
        user_id="u1", job_posting_id=job, status="dismissed"
    ) is True

    assert store.set_pursuit_status(
        user_id="u1", job_posting_id=job, status="dismissed"
    ) is False


def test_one_users_decision_never_reaches_another_users_shortlist(
    tmp_path, saved_job
):
    store = _store(tmp_path)
    mine = saved_job(store, user_id="u1")
    saved_job(store, user_id="u2", identity="job-b")

    assert store.set_pursuit_status(
        user_id="u2", job_posting_id=mine, status="dismissed"
    ) is False
    assert [item.job_posting_id for item in store.list_jobs(user_id="u1", include_dismissed=False)] == [mine]


def test_the_employers_state_and_the_users_decision_are_different_fields(
    tmp_path, saved_job
):
    """A live job the user ruled out, and a closed job they still want to see.

    Both are real. Folding them into one status would make one of them
    unrepresentable, and it is not obvious in advance which one.
    """
    store = _store(tmp_path)
    job = saved_job(store, user_id="u1")

    store.set_pursuit_status(user_id="u1", job_posting_id=job, status="dismissed")

    only = store.list_jobs(user_id="u1", include_dismissed=True)[0]
    assert only.availability_status == "active"
    assert only.pursuit_status == "dismissed"


def test_every_read_path_must_answer_whether_ignored_jobs_belong(tmp_path) -> None:
    """The omission this whole column already caused once.

    The filter was first added to ``list_jobs`` alone, and four other read
    paths kept returning ignored jobs: the agent's own search, the dashboard
    count, and the CLI listings. Each looked right on its own, which is exactly
    why a default would have hidden all four.

    Read from the signature rather than by exercising every path, because the
    defect is an omission — a new read method that forgets the parameter has no
    test of its own to fail.
    """
    import inspect

    from career_agent.storage.jobs import SQLiteJobPostingRepository

    for name in ("list_jobs", "search_saved_jobs", "count_jobs"):
        parameter = inspect.signature(
            getattr(SQLiteJobPostingRepository, name)
        ).parameters.get("include_dismissed")
        assert parameter is not None, name
        # No default: forgetting is a TypeError, not a silently wrong list.
        assert parameter.default is inspect.Parameter.empty, name


def test_an_ignored_job_disappears_from_every_read_path(tmp_path, saved_job) -> None:
    """One removal, and it holds wherever the user looks next.

    Removing a job in the library and then having the agent hand it back is
    worse than not being able to remove it at all: it is the system disagreeing
    with itself about a decision the user just made.
    """
    store = _store(tmp_path)
    job = saved_job(store, user_id="u1")
    store.set_pursuit_status(user_id="u1", job_posting_id=job, status="dismissed")

    assert store.list_jobs(user_id="u1", include_dismissed=False) == ()
    assert store.search_saved_jobs(
        user_id="u1", query="AI Engineer", include_dismissed=False
    ) == ()
    assert store.count_jobs(user_id="u1", include_dismissed=False) == 0
    # And all of it is still reachable when something explicitly asks for it.
    assert store.count_jobs(user_id="u1", include_dismissed=True) == 1


def test_looking_one_up_by_id_still_works_after_it_is_ignored(tmp_path, saved_job):
    """Listings are filtered; lookups are not, and that asymmetry is deliberate.

    A listing answers "what am I choosing from", so an ignored job does not
    belong in it. A lookup answers "show me this one", and the caller already
    named it — filtering there would make "what did that job I just ignored
    actually say?" unanswerable, and would break reading back a comparison the
    user is looking at.
    """
    store = _store(tmp_path)
    job = saved_job(store, user_id="u1")
    store.set_pursuit_status(user_id="u1", job_posting_id=job, status="dismissed")

    record = store.get_job(user_id="u1", job_posting_id=job)

    assert record is not None
    assert record.pursuit_status == "dismissed"
    assert record.snapshot.content.startswith("负责检索系统")
