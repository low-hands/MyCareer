from datetime import datetime, timedelta, timezone

from career_agent.agent.resume_job_match_contracts import ResumeJobMatchResult
from career_agent.agent.resume_tailoring_contracts import ResumeTailoringResult
from career_agent.api.reads import WorkspaceReader
from career_agent.domain.interviews import InterviewRetroReport
from career_agent.storage.resume_job_matches import StoredResumeJobMatch
from career_agent.storage.resume_tailoring import StoredResumeTailoringDraft


NOW = datetime(2026, 8, 31, tzinfo=timezone.utc)


class _One:
    def __init__(self, value) -> None:
        self.value = value

    def get(self, **kwargs):
        return self.value

    def get_retro(self, **kwargs):
        return self.value

    def get_for_display(self, **kwargs):
        return self.value

    def get_job(self, **kwargs):
        return self.value


class _RetroOnly(_One):
    def get(self, **kwargs):
        return None


def _reader() -> WorkspaceReader:
    reader = WorkspaceReader.__new__(WorkspaceReader)
    reader._jobs = _One(None)
    return reader


def test_real_interview_retro_is_reconstructed_from_its_entity() -> None:
    report = InterviewRetroReport(
        id="retro-1",
        user_id="u1",
        application_id="app-1",
        interview_round_id="round-1",
        source_notes="问了 RAG 评测。",
        summary="离线评测设计需要补强。",
        difficulties=("数据集构造没有讲清楚",),
        next_focus=("准备离线评测设计",),
        content_sha256="a" * 64,
        created_at=NOW,
    )
    reader = _reader()
    reader._interviews = _RetroOnly(report)

    view = reader._interview_retro_report("u1", "retro-1")

    assert view is not None
    assert view.kind == "interview_retro_report"
    assert "离线评测设计需要补强" in view.body
    assert "只依据你的复述" in view.body


def test_resume_match_is_reconstructed_from_the_immutable_match() -> None:
    stored = StoredResumeJobMatch(
        id="match-1",
        user_id="u1",
        resume_version_id="resume-v1",
        job_posting_id="job-1",
        jd_snapshot_id="jd-1",
        matcher_version="v1",
        evidence_fingerprint="none",
        result=ResumeJobMatchResult(
            overall_fit="moderate",
            summary="核心经验相关，但仍有缺口。",
        ),
        created_at=NOW,
    )
    reader = _reader()
    reader._resume_matches = _One(stored)

    view = reader._resume_job_match("u1", "match-1")

    assert view is not None
    assert view.kind == "resume_job_match"
    assert "整体判断：中等匹配" in view.body
    assert "核心经验相关" in view.body


def test_expired_tailoring_draft_remains_readable_but_is_marked_read_only() -> None:
    stored = StoredResumeTailoringDraft(
        id="draft-1",
        user_id="u1",
        match_id="match-1",
        worker_version="v1",
        result=ResumeTailoringResult(
            strategy_summary="突出已有的生产 RAG 经历。",
        ),
        created_at=NOW - timedelta(days=31),
        expires_at=NOW - timedelta(days=1),
    )
    reader = _reader()
    reader._tailoring_drafts = _One(stored)

    view = reader._resume_tailoring_draft("u1", "draft-1")

    assert view is not None
    assert view.kind == "resume_tailoring_draft"
    assert "expired" in view.subtitle
    assert "已经过期" in view.body
    assert "不能再审核或定稿" in view.body


def test_superseded_status_is_not_hidden_when_the_old_draft_also_expired() -> None:
    stored = StoredResumeTailoringDraft(
        id="draft-old",
        user_id="u1",
        match_id="match-1",
        status="superseded",
        worker_version="v1",
        result=ResumeTailoringResult(strategy_summary="旧修订。"),
        created_at=NOW - timedelta(days=40),
        expires_at=NOW - timedelta(days=10),
    )
    reader = _reader()
    reader._tailoring_drafts = _One(stored)

    view = reader._resume_tailoring_draft("u1", "draft-old")

    assert view is not None
    assert "superseded" in view.subtitle
    assert "已被更新的修订替代" in view.body
