"""Clearing applications takes everything that hangs off them with it.

Clearing used to delete only applications and their events. The interview
round of a cleared application stayed, and the action center kept generating a
retro reminder for it in the daily brief.
"""

from __future__ import annotations

from argparse import Namespace
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sqlite3

from fastapi.testclient import TestClient

from career_agent.api.app import create_app
from career_agent.api.reads import WorkspaceReader
from career_agent.domain.action_center import ActionCandidate
from career_agent.domain.interviews import InterviewDetails
from career_agent.domain.job_discovery import JobDetail, Provenance
from career_agent.services.applications import ApplicationService
from career_agent.services.action_center import ActionCenterService
from career_agent.services.interviews import InterviewService
from career_agent.storage.action_center import SQLiteActionItemStore
from career_agent.storage.applications import SQLiteApplicationStore
from career_agent.storage.calendar import SQLiteCalendarStore
from career_agent.storage.interview_preparations import SQLiteInterviewPreparationStore
from career_agent.storage.interviews import SQLiteInterviewStore
from career_agent.storage.jobs import SQLiteJobPostingRepository
from career_agent.storage.mock_interviews import SQLiteMockInterviewStore
from career_agent.storage.resumes import ResumeStore
from career_agent.storage.api_keys import WORKSPACE_READ, WORKSPACE_WRITE


NOW = datetime.now(timezone.utc)


def _paths(tmp_path) -> Namespace:
    return Namespace(**{
        name: str(tmp_path / f"{name}.sqlite3")
        for name in (
            "context_store", "resume_store", "application_store", "job_store",
            "calendar_store", "email_store", "mock_interview_store",
            "job_research_store", "action_store",
        )
    })


def _seed_application(
    paths: Namespace, *, user_id: str, source_job_id: str, mock: bool = True
):
    jobs = SQLiteJobPostingRepository(Path(paths.job_store))
    resumes = ResumeStore(Path(paths.resume_store))
    role = resumes.create_target_role(
        user_id=user_id, title=f"role-{source_job_id}", priority=1
    )
    _, version = resumes.import_document(
        user_id=user_id, content=b"Built production RAG systems",
        document_format="text", name=f"resume-{source_job_id}",
        target_role_id=role.id,
    )
    saved = jobs.save_detail(
        user_id=user_id, run_id=f"run-{source_job_id}", result_ref=source_job_id,
        selection_index=1,
        detail=JobDetail(
            source_name="test", source_job_id=source_job_id,
            title="RAG Engineer", company_name="Acme",
            description="Build reliable retrieval systems.", captured_at=NOW,
            provenance=Provenance(
                source_name="test", source_job_id=source_job_id, captured_at=NOW,
                operation="detail", adapter_version="test-v1",
            ),
        ),
    )
    applications = ApplicationService(
        SQLiteApplicationStore(Path(paths.application_store)), jobs, resumes
    )
    application = applications.create_application(
        user_id=user_id, job_posting_id=saved.posting.id,
        resume_version_id=version.id,
    ).application
    interview = InterviewService(
        SQLiteInterviewStore(Path(paths.application_store)), applications
    ).create_manual(
        user_id=user_id, application_id=application.id,
        details=InterviewDetails(scheduled_start=NOW + timedelta(days=1)),
    )
    # A conversation may hold multiple mock interview sessions.
    if mock:
        SQLiteMockInterviewStore(Path(paths.mock_interview_store)).create_session(
            user_id=user_id, application_id=application.id,
            job_posting_id=saved.posting.id, jd_snapshot_id=saved.snapshot.id,
            resume_version_id=version.id, interview_type="technical",
            interview_round_id=interview.id,
        )
    # Written directly: generating one needs a model worker, and what is under
    # test is only that the row goes with its application.
    SQLiteInterviewPreparationStore(Path(paths.resume_store))
    with sqlite3.connect(paths.resume_store) as connection:
        connection.execute(
            "INSERT INTO interview_preparations(id, user_id, interview_round_id, "
            "application_id, job_posting_id, jd_snapshot_id, resume_version_id, "
            "input_fingerprint, worker_version, result_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                f"prep-{source_job_id}", user_id, interview.id, application.id,
                saved.posting.id, saved.snapshot.id, version.id, "fp", "v1", "{}",
                NOW.isoformat(),
            ),
        )
    actions = SQLiteActionItemStore(Path(paths.action_store))
    for action_type, source_type, source_id in (
        ("interview_retro", "interview_round", interview.id),
        ("application_follow_up", "application", application.id),
    ):
        actions.upsert_candidate(
            user_id=user_id, now=NOW,
            candidate=ActionCandidate(
                stable_key=f"{action_type}:{source_id}",
                action_type=action_type, source_type=source_type,
                source_id=source_id, application_id=application.id,
                title=action_type, summary=action_type,
            ),
        )
    return application, interview


def _count(path: str, table: str, user_id: str) -> int:
    with sqlite3.connect(path) as connection:
        return connection.execute(
            f"SELECT COUNT(*) FROM {table} WHERE user_id = ?", (user_id,)
        ).fetchone()[0]


def test_clearing_applications_takes_their_dependents_and_earlier_orphans(
    tmp_path,
) -> None:
    paths = _paths(tmp_path)
    _seed_application(paths, user_id="u1", source_job_id="orphaned")
    # The state an application-only clear left behind: interview, mock
    # interview and reminders pointing at an application that is gone.
    SQLiteApplicationStore(Path(paths.application_store)).clear_user(user_id="u1")
    _seed_application(paths, user_id="u1", source_job_id="current", mock=False)
    _seed_application(paths, user_id="u2", source_job_id="someone-else")
    unrelated = SQLiteActionItemStore(Path(paths.action_store)).upsert_candidate(
        user_id="u1", now=NOW,
        candidate=ActionCandidate(
            stable_key="saved_job_review", action_type="saved_job_review",
            source_type="saved_job_library", source_id="saved_job_library",
            title="整理岗位库", summary="整理岗位库",
        ),
    )

    cleared = WorkspaceReader(paths).clear_applications(user_id="u1")

    assert cleared == 1
    for path, table in (
        (paths.application_store, "applications"),
        (paths.application_store, "interview_rounds"),
        (paths.application_store, "interview_round_events"),
        (paths.mock_interview_store, "mock_interview_sessions"),
        (paths.resume_store, "interview_preparations"),
    ):
        assert _count(path, table, "u1") == 0, table
        assert _count(path, table, "u2") > 0, table
    remaining = SQLiteActionItemStore(Path(paths.action_store)).list(
        user_id="u1", statuses=("open", "snoozed", "completed", "dismissed", "obsolete")
    )
    assert [item.id for item in remaining] == [unrelated.id]
    assert _count(paths.action_store, "action_items", "u2") == 2


def test_a_mistaken_interview_and_application_can_be_removed_individually(tmp_path) -> None:
    paths = _paths(tmp_path)
    application, interview = _seed_application(
        paths, user_id="u1", source_job_id="mistake", mock=False,
    )
    other, other_interview = _seed_application(
        paths, user_id="u2", source_job_id="other", mock=False,
    )
    reader = WorkspaceReader(paths)

    assert reader.delete_application(user_id="u1", application_id=application.id) == "has_dependents"
    assert reader.delete_interview(user_id="u2", interview_round_id=interview.id) == "not_found"
    assert reader.delete_interview(user_id="u1", interview_round_id=interview.id) == "deleted"
    assert SQLiteInterviewStore(Path(paths.application_store)).get(
        user_id="u1", interview_round_id=interview.id,
    ) is None
    assert _count(paths.resume_store, "interview_preparations", "u1") == 0
    assert len(SQLiteActionItemStore(Path(paths.action_store)).list(
        user_id="u1", statuses=("open",),
    )) == 1  # application follow-up remains until its parent is deleted
    assert reader.delete_application(user_id="u1", application_id=application.id) == "deleted"
    assert SQLiteApplicationStore(Path(paths.application_store)).get(
        user_id="u1", application_id=application.id,
    ) is None
    assert SQLiteApplicationStore(Path(paths.application_store)).get(
        user_id="u2", application_id=other.id,
    ) is not None
    assert SQLiteInterviewStore(Path(paths.application_store)).get(
        user_id="u2", interview_round_id=other_interview.id,
    ) is not None


def test_a_dismissed_action_can_be_restored(tmp_path) -> None:
    store = SQLiteActionItemStore(tmp_path / "actions.sqlite3")
    item = store.upsert_candidate(
        user_id="u1", now=NOW,
        candidate=ActionCandidate(
            stable_key="follow-up", action_type="application_follow_up",
            source_type="application", source_id="app-1", application_id="app-1",
            title="跟进投递", summary="询问进展",
        ),
    )
    service = ActionCenterService(store, None, None, None)
    service.dismiss_action(user_id="u1", action_item_id=item.id)
    assert store.get(user_id="u1", action_item_id=item.id).status == "dismissed"
    restored = service.restore_dismissed_action(user_id="u1", action_item_id=item.id)
    assert restored.status == "open"
    assert restored.resolved_at is None
    assert [event.event_type for event in store.list_events(
        user_id="u1", action_item_id=item.id,
    )] == ["created", "dismissed", "reopened"]


def test_interview_removal_restores_only_its_automatic_application_status(tmp_path) -> None:
    paths = _paths(tmp_path)
    application, interview = _seed_application(
        paths, user_id="u1", source_job_id="auto-status", mock=False,
    )
    store = SQLiteApplicationStore(Path(paths.application_store))
    store.update(
        user_id="u1", application_id=application.id,
        expected_status="submitted", new_status="interviewing",
        submitted_at=application.submitted_at,
        note="文案可以修改，不影响恢复逻辑。", reason="interview_created",
    )
    assert WorkspaceReader(paths).delete_interview(
        user_id="u1", interview_round_id=interview.id,
    ) == "deleted"
    assert store.get(user_id="u1", application_id=application.id).status == "submitted"
    assert store.list_events(user_id="u1", application_id=application.id)[-1].source == "system"


def test_interview_removal_does_not_clear_dependents_when_round_delete_fails(
    tmp_path, monkeypatch,
) -> None:
    paths = _paths(tmp_path)
    application, interview = _seed_application(
        paths, user_id="u1", source_job_id="concurrent-delete", mock=False,
    )
    monkeypatch.setattr(SQLiteInterviewStore, "delete", lambda self, **kwargs: False)

    assert WorkspaceReader(paths).delete_interview(
        user_id="u1", interview_round_id=interview.id,
    ) == "not_found"
    assert _count(paths.resume_store, "interview_preparations", "u1") == 1
    assert len(SQLiteActionItemStore(Path(paths.action_store)).list(
        user_id="u1", statuses=("open",),
    )) == 2
    assert SQLiteApplicationStore(Path(paths.application_store)).get(
        user_id="u1", application_id=application.id,
    ) is not None


def test_application_event_reason_is_backfilled_from_v2(tmp_path) -> None:
    path = tmp_path / "applications.sqlite3"
    store = SQLiteApplicationStore(path)
    # Recreate the old column shape while retaining an event to migrate.
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA foreign_keys=OFF")
        connection.execute("ALTER TABLE application_events DROP COLUMN source_interview_round_id")
        connection.execute("ALTER TABLE application_events DROP COLUMN reason")
        connection.execute(
            "UPDATE schema_versions SET version = 2 WHERE component = 'applications'"
        )
        connection.execute(
            "INSERT INTO application_events(id, application_id, user_id, source, "
            "event_type, previous_status, new_status, note, occurred_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("old-event", "old-application", "u1", "user_reported", "status_changed",
             "submitted", "interviewing", "用户已报告收到面试安排。", NOW.isoformat()),
        )
    store = SQLiteApplicationStore(path)
    with sqlite3.connect(path) as connection:
        assert connection.execute(
            "SELECT reason FROM application_events WHERE id = 'old-event'"
        ).fetchone()[0] == "interview_created"
        assert connection.execute(
            "SELECT version FROM schema_versions WHERE component = 'applications'"
        ).fetchone()[0] == 4


def test_status_correction_retires_stale_follow_up_on_refresh(tmp_path) -> None:
    paths = _paths(tmp_path)
    application, _ = _seed_application(
        paths, user_id="u1", source_job_id="follow-up-correction", mock=False,
    )
    reader = WorkspaceReader(paths)
    with sqlite3.connect(paths.application_store) as connection:
        connection.execute(
            "UPDATE applications SET updated_at = ? WHERE id = ?",
            ((NOW - timedelta(days=8)).isoformat(), application.id),
        )

    class EmptyEmail:
        def list_events(self, **kwargs):
            return ()

    class EmptyInterviews:
        def list_interviews(self, **kwargs):
            return ()

    actions = ActionCenterService(
        SQLiteActionItemStore(Path(paths.action_store)),
        reader._applications, EmptyEmail(), EmptyInterviews(),
    )
    assert any(item.action_type == "application_follow_up" for item in actions.refresh(
        user_id="u1", now=NOW, timezone_name="UTC",
    ))
    reader.correct_application_status(
        user_id="u1", application_id=application.id, status="acknowledged",
    )
    assert not any(item.action_type == "application_follow_up" for item in actions.refresh(
        user_id="u1", now=NOW + timedelta(minutes=1), timezone_name="UTC",
    ))
    assert any(item.status == "obsolete" for item in actions._store.list(
        user_id="u1", statuses=("obsolete",),
    ))


def test_mistaken_record_removal_rejects_linked_mock_practice(tmp_path) -> None:
    paths = _paths(tmp_path)
    application, interview = _seed_application(
        paths, user_id="u1", source_job_id="linked", mock=True,
    )
    reader = WorkspaceReader(paths)
    assert reader.delete_interview(user_id="u1", interview_round_id=interview.id) == "has_dependents"
    assert reader.delete_application(user_id="u1", application_id=application.id) == "has_dependents"
    assert SQLiteInterviewStore(Path(paths.application_store)).get(
        user_id="u1", interview_round_id=interview.id,
    ) is not None


def test_workspace_restore_and_delete_routes_are_user_scoped(
    tmp_path, api_keys, issue_key,
) -> None:
    paths = _paths(tmp_path)
    application, interview = _seed_application(
        paths, user_id="u1", source_job_id="route", mock=False,
    )
    item = SQLiteActionItemStore(Path(paths.action_store)).list(
        user_id="u1", statuses=("open",),
    )[0]
    actions = ActionCenterService(
        SQLiteActionItemStore(Path(paths.action_store)), None, None, None,
    )
    actions.dismiss_action(user_id="u1", action_item_id=item.id)
    app = create_app(
        runtime_factory=lambda: None,
        workspace_reader_factory=lambda: WorkspaceReader(paths),
        action_center_factory=lambda: actions,
        api_key_store_factory=lambda: api_keys,
    )
    owner = issue_key("u1", WORKSPACE_READ, WORKSPACE_WRITE)
    other = issue_key("u2", WORKSPACE_READ, WORKSPACE_WRITE)
    with TestClient(app) as client:
        assert client.get("/v1/action-items/dismissed", headers=owner).json()[0]["id"] == item.id
        assert client.get("/v1/interviews", headers=owner).json()[0]["id"] == interview.id
        assert client.get("/v1/interviews", headers=other).json() == []
        assert client.patch(
            f"/v1/applications/{application.id}/status",
            headers=other, json={"status": "acknowledged"},
        ).status_code == 404
        assert client.patch(
            f"/v1/applications/{application.id}/status",
            headers=owner, json={"status": "interviewing"},
        ).json()["status"] == "interviewing"
        assert client.patch(
            f"/v1/applications/{application.id}/status",
            headers=owner, json={"status": "submitted"},
        ).json()["status"] == "submitted"
        assert client.post(f"/v1/action-items/{item.id}/restore", headers=other).status_code == 404
        assert client.post(f"/v1/action-items/{item.id}/restore", headers=owner).json()["status"] == "open"
        assert client.delete(f"/v1/applications/{application.id}", headers=owner).status_code == 409
        assert client.delete(f"/v1/interviews/{interview.id}", headers=other).status_code == 404
        assert client.delete(f"/v1/interviews/{interview.id}", headers=owner).status_code == 204
        assert client.get("/v1/interviews", headers=owner).json() == []
        assert client.delete(f"/v1/applications/{application.id}", headers=owner).status_code == 204


def test_unscheduled_interviews_are_visible_in_the_correction_list(tmp_path) -> None:
    paths = _paths(tmp_path)
    application, _ = _seed_application(
        paths, user_id="u1", source_job_id="unscheduled", mock=False,
    )
    service = InterviewService(
        SQLiteInterviewStore(Path(paths.application_store)),
        WorkspaceReader(paths)._applications,
    )
    unscheduled = service.create_manual(
        user_id="u1", application_id=application.id,
        details=InterviewDetails(employer_label="招聘沟通"),
    )
    records = WorkspaceReader(paths).interviews(user_id="u1")
    assert any(item.id == unscheduled.id and item.scheduled_start is None for item in records)


def test_external_calendar_link_blocks_local_interview_and_bulk_deletion(tmp_path) -> None:
    paths = _paths(tmp_path)
    application, interview = _seed_application(
        paths, user_id="u1", source_job_id="calendar-linked", mock=False,
    )
    calendar = SQLiteCalendarStore(Path(paths.calendar_store))
    account = calendar.add_account(
        user_id="u1", email_address="u1@example.com", calendar_id="primary",
        credential_ref="test-credential",
    )
    with sqlite3.connect(paths.calendar_store) as connection:
        connection.execute(
            "INSERT INTO calendar_event_links(id, user_id, calendar_account_id, "
            "interview_round_id, external_event_id, status, last_payload_hash, "
            "created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("link-1", "u1", account.id, interview.id, "external-1", "active",
             "hash", NOW.isoformat(), NOW.isoformat()),
        )
    reader = WorkspaceReader(paths)
    assert reader.delete_interview(user_id="u1", interview_round_id=interview.id) == "has_dependents"
    assert reader.delete_application(user_id="u1", application_id=application.id) == "has_dependents"
    try:
        reader.clear_applications(user_id="u1")
    except ValueError:
        pass
    else:
        raise AssertionError("bulk clear bypassed the calendar-link guard")


def test_completed_and_snoozed_actions_have_scoped_restore_routes(tmp_path, api_keys, issue_key):
    paths = _paths(tmp_path)
    _seed_application(paths, user_id="u1", source_job_id="restore", mock=False)
    store = SQLiteActionItemStore(Path(paths.action_store))
    actions = ActionCenterService(store, None, None, None)
    item = store.list(user_id="u1", statuses=("open",))[0]
    owner = issue_key("u1", WORKSPACE_READ, WORKSPACE_WRITE)
    other = issue_key("u2", WORKSPACE_READ, WORKSPACE_WRITE)
    read_only = issue_key("u1", WORKSPACE_READ)
    app = create_app(runtime_factory=lambda: None, workspace_reader_factory=lambda: WorkspaceReader(paths),
                     action_center_factory=lambda: actions, api_key_store_factory=lambda: api_keys)
    with TestClient(app) as client:
        for status in ("completed", "snoozed"):
            if status == "completed":
                actions.complete_action(user_id="u1", action_item_id=item.id)
            else:
                actions.snooze_action(user_id="u1", action_item_id=item.id, snoozed_until=datetime.now(timezone.utc) + timedelta(days=1))
            assert client.get("/v1/action-items/restorable", headers=owner).json()[0]["status"] == status
            assert client.get("/v1/action-items/restorable", headers=other).json() == []
            assert client.post(f"/v1/action-items/{item.id}/restore", headers=read_only).status_code == 403
            assert client.post(f"/v1/action-items/{item.id}/restore", headers=other).status_code == 404
            response = client.post(f"/v1/action-items/{item.id}/restore", headers=owner)
            assert response.status_code == 200
            assert response.json()["status"] == "open"
            assert response.json()["snoozed_until"] is None


def test_interview_undo_rolls_back_only_its_unchanged_application_transition(tmp_path, api_keys, issue_key):
    paths = _paths(tmp_path)
    application, interview = _seed_application(paths, user_id="u1", source_job_id="undo", mock=False)
    reader = WorkspaceReader(paths)
    apps = reader._applications
    apps.correct_status(user_id="u1", application_id=application.id, status="interviewing")
    service = InterviewService(reader._interviews, apps)
    service.complete_interview(user_id="u1", interview_round_id=interview.id)
    assert apps.get_record(user_id="u1", application_id=application.id).status == "interview_completed"
    event = apps.list_events(user_id="u1", application_id=application.id)[-1]
    assert event.source_interview_round_id == interview.id
    assert apps.restore_status_after_completion(
        user_id="u1", application_id=application.id, interview_round_id="unrelated-round",
    ) is None
    # Display wording is mutable and is no longer part of the rollback contract.
    with sqlite3.connect(paths.application_store) as connection:
        connection.execute("UPDATE application_events SET note = ? WHERE id = ?", ("新的完成提示文案", event.id))
    app = create_app(runtime_factory=lambda: None, workspace_reader_factory=lambda: reader, api_key_store_factory=lambda: api_keys)
    owner = issue_key("u1", WORKSPACE_READ, WORKSPACE_WRITE)
    other = issue_key("u2", WORKSPACE_READ, WORKSPACE_WRITE)
    read_only = issue_key("u1", WORKSPACE_READ)
    with TestClient(app) as client:
        url = f"/v1/interviews/{interview.id}/restore"
        assert client.post(url, headers=other).status_code == 404
        assert client.post(url, headers=read_only).status_code == 403
        restored = client.post(url, headers=owner)
        assert restored.status_code == 200
        assert restored.json()["status"] == "scheduled"
        assert apps.get_record(user_id="u1", application_id=application.id).status == "interviewing"
        # Even a later correction to the same status must not be overwritten.
        service.complete_interview(user_id="u1", interview_round_id=interview.id)
        apps.correct_status(user_id="u1", application_id=application.id, status="offer")
        apps.correct_status(user_id="u1", application_id=application.id, status="interview_completed")
        assert client.post(url, headers=owner).status_code == 200
        assert apps.get_record(user_id="u1", application_id=application.id).status == "interview_completed"
        service.complete_interview(user_id="u1", interview_round_id=interview.id)
        service.record_retro(user_id="u1", interview_round_id=interview.id, source_notes="已经复盘", summary="总结")
        assert client.post(url, headers=owner).status_code == 409
        assert reader._interviews.get(user_id="u1", interview_round_id=interview.id).status == "completed"
