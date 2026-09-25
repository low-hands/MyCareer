"""A resume version something still points at stays readable after deletion.

Applications and mock interviews name the exact resume version they used. The
library hides a deleted resume, but those versions keep their original file so
"which resume did I send" and "which resume was this practice run on" can still
be opened. Unreferenced versions are deleted as before.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from career_agent.agent.main_agent_contracts import ConversationTaskState
from career_agent.api.reads import WorkspaceReader
from career_agent.services.applications import ApplicationInputNotFoundError
from career_agent.storage.applications import SQLiteApplicationStore
from career_agent.storage.context import CareerContextStore
from career_agent.storage.mock_interviews import SQLiteMockInterviewStore
from career_agent.storage.resumes import ResumeStore
from tests.api.test_clear_applications import _paths, _seed_application


def _resume_of(paths, version_id: str):
    located = ResumeStore(Path(paths.resume_store)).get_version(
        user_id="u1", resume_version_id=version_id
    )
    assert located is not None
    return located


def test_a_deleted_resume_stays_viewable_from_what_references_it(tmp_path) -> None:
    paths = _paths(tmp_path)
    application, _ = _seed_application(paths, user_id="u1", source_job_id="sent", mock=False)
    store = ResumeStore(Path(paths.resume_store))
    referenced, version = _resume_of(paths, application.resume_version_id)
    role = store.create_target_role(user_id="u1", title="spare", priority=2)
    unreferenced, spare = store.import_document(
        user_id="u1", content=b"never sent", document_format="text",
        name="spare", target_role_id=role.id,
    )
    reader = WorkspaceReader(paths)

    assert reader.delete_resume(user_id="u1", resume_id=referenced.id)
    assert reader.delete_resume(user_id="u1", resume_id=unreferenced.id)

    listed = {item.id for item in store.list_resumes(user_id="u1")}
    assert referenced.id not in listed and unreferenced.id not in listed
    [view] = reader.applications(user_id="u1")
    assert (view.resume_name, view.resume_version_number, view.resume_deleted) == (
        referenced.name, version.version_number, True,
    )
    assert reader.resume_version_document(
        user_id="u1", resume_id=referenced.id, resume_version_id=version.id
    ) is not None
    assert store.get_version(user_id="u1", resume_version_id=spare.id) is None


def test_the_resume_version_of_an_application_can_be_backfilled(tmp_path) -> None:
    paths = _paths(tmp_path)
    application, _ = _seed_application(paths, user_id="u1", source_job_id="sent", mock=False)
    applications = SQLiteApplicationStore(Path(paths.application_store))
    applications.update(
        user_id="u1", application_id=application.id, expected_status="submitted",
        new_status="interviewing", submitted_at=application.submitted_at, note=None,
    )
    reader = WorkspaceReader(paths)

    cleared = reader.update_application_resume_version(
        user_id="u1", application_id=application.id, resume_version_id=None
    )
    assert cleared.resume_version_id is None
    restored = reader.update_application_resume_version(
        user_id="u1", application_id=application.id,
        resume_version_id=application.resume_version_id,
    )
    assert restored.resume_version_id == application.resume_version_id
    assert restored.status == "interviewing"

    events = applications.list_events(user_id="u1", application_id=application.id)
    changes = [event for event in events if event.event_type == "resume_version_changed"]
    # Two edits, and neither pretends the application moved back to submitted.
    assert [(event.previous_status, event.new_status) for event in changes] == [
        ("interviewing", "interviewing"), ("interviewing", "interviewing"),
    ]
    with pytest.raises(ApplicationInputNotFoundError):
        reader.update_application_resume_version(
            user_id="u1", application_id=application.id, resume_version_id="missing"
        )


def test_the_practice_list_names_both_kinds_and_links_back_to_a_live_run(
    tmp_path,
) -> None:
    paths = _paths(tmp_path)
    _seed_application(paths, user_id="u1", source_job_id="bound")
    mock = SQLiteMockInterviewStore(Path(paths.mock_interview_store))
    [bound] = mock.list_sessions(user_id="u1")
    mock.cancel(session=bound)
    free = mock.create_session(user_id="u1", interview_type="behavioral", target_role="PM")
    CareerContextStore(Path(paths.context_store)).upsert_task(
        user_id="u1", conversation_id="conversation-free",
        task=ConversationTaskState(active_workflow="mock_interview", run_id=free.id),
    )
    # A bound run whose application is already gone must not break the page.
    SQLiteApplicationStore(Path(paths.application_store)).clear_user(user_id="u1")

    sessions = {
        item.session_id: item
        for item in WorkspaceReader(paths).free_mock_interviews(user_id="u1").sessions
    }

    assert sessions[free.id].title == "自由练习"
    assert sessions[free.id].conversation_id == "conversation-free"
    assert sessions[bound.id].application_id == bound.application_id
    assert sessions[bound.id].company_name is None


def test_a_finished_practice_carries_what_repeating_it_needs(tmp_path) -> None:
    paths = _paths(tmp_path)
    application, _ = _seed_application(paths, user_id="u1", source_job_id="job", mock=False)
    from career_agent.storage.jobs import SQLiteJobPostingRepository

    job = SQLiteJobPostingRepository(Path(paths.job_store)).get_job(
        user_id="u1", job_posting_id=application.job_posting_id
    )
    mock = SQLiteMockInterviewStore(Path(paths.mock_interview_store))
    free = mock.create_session(
        user_id="u1", interview_type="technical", max_primary_questions=10,
        max_follow_ups_per_question=2, target_role="Java后端",
        job_posting_id=job.posting.id, jd_snapshot_id=job.snapshot.id,
        resume_version_id=application.resume_version_id,
    )
    mock.cancel(session=free)

    [view] = WorkspaceReader(paths).free_mock_interviews(user_id="u1").sessions

    assert (view.max_primary_questions, view.max_follow_ups_per_question) == (10, 2)
    assert view.target_role == "Java后端"
    assert (view.job_posting_id, view.jd_snapshot_id, view.jd_version) == (
        job.posting.id, job.snapshot.id, job.snapshot.version,
    )
    assert view.job_company_name == job.posting.company_name
    assert view.resume_document_format is not None and view.resume_byte_size is not None


def test_a_resume_moves_to_another_role_with_every_version_and_reference(tmp_path) -> None:
    paths = _paths(tmp_path)
    application, _ = _seed_application(paths, user_id="u1", source_job_id="sent", mock=False)
    store = ResumeStore(Path(paths.resume_store))
    resume, version = _resume_of(paths, application.resume_version_id)
    other = store.create_target_role(user_id="u1", title="Agent开发", priority=2)
    stranger_role = store.create_target_role(user_id="u2", title="别人的岗位", priority=1)
    reader = WorkspaceReader(paths)

    assert reader.move_resume(user_id="u1", resume_id=resume.id, target_role_id=other.id)

    [view] = reader.resumes(user_id="u1")
    assert (view.target_role_id, view.target_role, view.version_count) == (other.id, "Agent开发", 1)
    # The application still cites the same file.
    [application_view] = reader.applications(user_id="u1")
    assert application_view.resume_version_id == version.id
    with pytest.raises(ValueError, match="Target role not found"):
        reader.move_resume(user_id="u1", resume_id=resume.id, target_role_id=stranger_role.id)
    assert not reader.move_resume(user_id="u2", resume_id=resume.id, target_role_id=stranger_role.id)
    reader.delete_resume(user_id="u1", resume_id=resume.id)
    assert not reader.move_resume(user_id="u1", resume_id=resume.id, target_role_id=other.id)
