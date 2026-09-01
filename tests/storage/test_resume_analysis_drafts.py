from __future__ import annotations

from datetime import datetime, timedelta, timezone
import sqlite3

import pytest

from career_agent.agent.resume_analysis_contracts import (
    ExtractedCareerEvidence,
    ExtractedCareerRecord,
    ResumeAnalysisResult,
)
from career_agent.storage.resume_analysis import SQLiteResumeAnalysisDraftStore


def _result() -> ResumeAnalysisResult:
    return ResumeAnalysisResult(
        records=(
            ExtractedCareerRecord(
                record_type="project",
                organization=None,
                title="Career Agent",
                start_year=2026,
                source_locator="page 1, Projects",
                source_quote="Career Agent",
                evidence=(
                    ExtractedCareerEvidence(
                        claim="Built a career assistant",
                        source_locator="page 1, bullet 1",
                        source_quote="Built a career assistant",
                    ),
                ),
            ),
        ),
        warnings=("One icon was unreadable",),
    )


def test_draft_store_round_trips_structured_analysis_and_scopes_user(tmp_path) -> None:
    path = tmp_path / "resumes.sqlite3"
    store = SQLiteResumeAnalysisDraftStore(path)
    now = datetime(2026, 8, 25, tzinfo=timezone.utc)

    created = store.create(
        user_id="u1",
        resume_version_id="version-1",
        result=_result(),
        now=now,
    )

    assert created.status == "pending"
    assert created.expires_at == now + timedelta(days=30)
    assert store.get(user_id="u1", analysis_id=created.id, now=now) == created
    assert store.get(user_id="other", analysis_id=created.id, now=now) is None
    assert path.stat().st_mode & 0o777 == 0o600


def test_reanalysis_creates_a_new_id_without_mutating_the_old_result(tmp_path) -> None:
    store = SQLiteResumeAnalysisDraftStore(tmp_path / "resumes.sqlite3")
    first_result = _result()
    second_result = first_result.model_copy(
        update={"warnings": ("A later extraction produced different content",)}
    )

    first = store.create(
        user_id="u1",
        resume_version_id="version-1",
        result=first_result,
    )
    second = store.create(
        user_id="u1",
        resume_version_id="version-1",
        result=second_result,
    )

    assert second.id != first.id
    assert store.get(user_id="u1", analysis_id=first.id).result == first_result
    assert store.get(user_id="u1", analysis_id=second.id).result == second_result


def test_draft_store_does_not_store_original_resume_document(tmp_path) -> None:
    path = tmp_path / "resumes.sqlite3"
    store = SQLiteResumeAnalysisDraftStore(path)
    store.create(
        user_id="u1",
        resume_version_id="version-1",
        result=_result(),
    )

    with sqlite3.connect(path) as connection:
        row = connection.execute(
            "SELECT result_json FROM resume_analysis_drafts"
        ).fetchone()

    assert row is not None
    assert "raw_bytes" not in row[0]
    assert "file_data" not in row[0]
    assert "%PDF" not in row[0]


def test_expired_draft_is_hidden_and_can_be_deleted(tmp_path) -> None:
    store = SQLiteResumeAnalysisDraftStore(
        tmp_path / "resumes.sqlite3",
        ttl=timedelta(hours=1),
    )
    now = datetime(2026, 8, 25, tzinfo=timezone.utc)
    draft = store.create(
        user_id="u1",
        resume_version_id="version-1",
        result=_result(),
        now=now,
    )

    later = now + timedelta(hours=1)
    assert store.get(user_id="u1", analysis_id=draft.id, now=later) is None
    assert store.delete_expired(now=later) == 1
    assert store.delete_expired(now=later) == 0


def test_mark_confirmed_is_user_scoped_and_one_time(tmp_path) -> None:
    store = SQLiteResumeAnalysisDraftStore(tmp_path / "resumes.sqlite3")
    now = datetime(2026, 8, 25, tzinfo=timezone.utc)
    draft = store.create(
        user_id="u1",
        resume_version_id="version-1",
        result=_result(),
        now=now,
    )

    confirmed_at = now + timedelta(minutes=5)
    confirmed = store.mark_confirmed(
        user_id="u1",
        analysis_id=draft.id,
        now=confirmed_at,
    )
    with pytest.raises(ValueError, match="Cannot confirm confirmed"):
        store.mark_confirmed(
            user_id="u1",
            analysis_id=draft.id,
            now=confirmed_at + timedelta(minutes=1),
        )

    assert confirmed.status == "confirmed"
    assert confirmed.updated_at == confirmed_at
    assert store.get(user_id="other", analysis_id=draft.id, now=confirmed_at) is None


def test_reject_is_one_time_and_prevents_later_confirmation(tmp_path) -> None:
    store = SQLiteResumeAnalysisDraftStore(tmp_path / "resumes.sqlite3")
    now = datetime(2026, 8, 25, tzinfo=timezone.utc)
    draft = store.create(
        user_id="u1",
        resume_version_id="version-1",
        result=_result(),
        now=now,
    )

    rejected = store.mark_rejected(
        user_id="u1",
        analysis_id=draft.id,
        now=now + timedelta(minutes=1),
    )

    assert rejected.status == "rejected"
    with pytest.raises(ValueError, match="Cannot reject rejected"):
        store.mark_rejected(
            user_id="u1",
            analysis_id=draft.id,
            now=now + timedelta(minutes=2),
        )
    with pytest.raises(ValueError, match="Cannot confirm rejected"):
        store.mark_confirmed(
            user_id="u1",
            analysis_id=draft.id,
            now=now + timedelta(minutes=2),
        )
