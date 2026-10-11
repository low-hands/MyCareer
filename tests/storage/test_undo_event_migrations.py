from datetime import datetime, timezone
import json
import sqlite3

from career_agent.domain.interviews import InterviewDetails
from career_agent.storage.applications import SQLiteApplicationStore
from career_agent.storage.interviews import SQLiteInterviewStore


NOW = datetime.now(timezone.utc)
ROUND_ID = "interview_round_" + "a" * 32


def test_v3_application_events_migrate_only_known_completion_provenance(tmp_path):
    path = tmp_path / "applications.sqlite3"
    store = SQLiteApplicationStore(path)
    cases = [
        ("known", "interview_completed", f"面试已完成，等待招聘方结果。（面试记录：{ROUND_ID}）", ROUND_ID),
        ("edited", "interview_completed", "未知旧文案，不应猜来源", None),
        ("manual", None, f"面试已完成，等待招聘方结果。（面试记录：{ROUND_ID}）", None),
    ]
    applications = []
    for name, reason, note, expected in cases:
        app = store.create(user_id="u1", job_posting_id=name, jd_snapshot_id=name, resume_version_id=None, submitted_at=NOW)
        store.update(user_id="u1", application_id=app.id, expected_status="submitted", new_status="interviewing", submitted_at=NOW, note=None)
        store.update(user_id="u1", application_id=app.id, expected_status="interviewing", new_status="interview_completed", submitted_at=NOW, note=note, reason=reason)
        applications.append((app, expected))
    with sqlite3.connect(path) as connection:
        connection.execute("ALTER TABLE application_events DROP COLUMN source_interview_round_id")
        connection.execute("UPDATE schema_versions SET version = 3 WHERE component = 'applications'")
    for _ in range(2):  # Opening the upgraded database again must be idempotent.
        migrated = SQLiteApplicationStore(path)
        for app, expected in applications:
            events = migrated.list_events(user_id="u1", application_id=app.id)
            assert len(events) == 3
            assert events[-1].source_interview_round_id == expected
        with sqlite3.connect(path) as connection:
            assert connection.execute("SELECT version FROM schema_versions WHERE component = 'applications'").fetchone() == (4,)


def test_v2_completion_undo_history_becomes_an_explicit_status_transition(tmp_path):
    path = tmp_path / "interviews.sqlite3"
    store = SQLiteInterviewStore(path)
    interview = store.create(user_id="u1", application_id="app-1", details=InterviewDetails(scheduled_start=NOW),
                             source="user_reported", email_event_id=None, source_thread_id=None, occurred_at=NOW)
    updated = store.update(round_=interview, details=InterviewDetails(change_type="details_updated", location="会议室"),
                           source="user_reported", email_event_id=None, source_thread_id=None, occurred_at=NOW)
    completed = store.complete(round_=updated, occurred_at=NOW)
    store.restore_completion(round_=completed)
    with sqlite3.connect(path) as connection:
        event_id, raw = connection.execute("SELECT id, details_json FROM interview_round_events WHERE event_type = 'completion_reverted'").fetchone()
        details = json.loads(raw)
        details["change_type"] = "details_updated"
        details.pop("previous_status")
        details.pop("new_status")
        connection.execute("UPDATE interview_round_events SET event_type = 'corrected', details_json = ? WHERE id = ?", (json.dumps(details), event_id))
        connection.execute("UPDATE schema_versions SET version = 2 WHERE component = 'interviews'")
    for _ in range(2):
        migrated = SQLiteInterviewStore(path)
        events = migrated.list_events(user_id="u1", interview_round_id=interview.id)
        assert [event.event_type for event in events] == ["created", "details_updated", "completed", "completion_reverted"]
        assert events[1].details.change_type == "details_updated"
        assert events[-1].details.change_type == "completion_reverted"
        assert events[-1].details.previous_status == "completed"
        assert events[-1].details.new_status == "scheduled"
        assert migrated.get(user_id="u1", interview_round_id=interview.id).status == "scheduled"
