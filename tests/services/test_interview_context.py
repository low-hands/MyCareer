from datetime import datetime, timezone
from types import SimpleNamespace

from career_agent.domain.interviews import (
    InterviewRetroQuestion,
    InterviewRetroReport,
    InterviewRound,
)
from career_agent.services.interview_context import InterviewPreparationContextFactory
from career_agent.services.interviews import InterviewDetail
from career_agent.storage.resumes import StoredResumeDocument


NOW = datetime(2026, 8, 29, tzinfo=timezone.utc)


def _round(round_id: str, sequence: int, status: str) -> InterviewRound:
    return InterviewRound(
        id=round_id,
        user_id="u1",
        application_id="app-1",
        sequence_number=sequence,
        status=status,
        interview_format="video",
        scheduled_start=NOW if status == "scheduled" else None,
        completed_at=NOW if status == "completed" else None,
        created_at=NOW,
        updated_at=NOW,
    )


def _retro(report_id: str, summary: str, source_notes: str) -> InterviewRetroReport:
    return InterviewRetroReport(
        id=report_id,
        user_id="u1",
        application_id="app-1",
        interview_round_id="round-1",
        source_notes=source_notes,
        summary=summary,
        questions=(InterviewRetroQuestion(
            question="How do you recover retrieval failures?",
            answer_summary="I described retries but not an objective.",
            self_assessment="mixed",
        ),),
        difficulties=("Recovery targets were vague.",),
        next_focus=("Define RTO and RPO.",),
        self_assessment="mixed",
        content_sha256="a" * 64,
        created_at=NOW,
    )


class Interviews:
    def __init__(self) -> None:
        self.current = _round("round-2", 2, "scheduled")
        self.prior = _round("round-1", 1, "completed")
        self.later = _round("round-3", 3, "completed")

    def get_interview(self, *, interview_round_id: str, **kwargs):
        if interview_round_id == self.current.id:
            return InterviewDetail(interview=self.current, events=())
        if interview_round_id == self.prior.id:
            return InterviewDetail(
                interview=self.prior,
                events=(),
                retros=(
                    _retro("retro-old", "Old version", "old raw notes"),
                    _retro("retro-latest", "Latest version", "secret raw recollection"),
                ),
            )
        return InterviewDetail(
            interview=self.later,
            events=(),
            retros=(_retro("retro-later", "Later round", "later raw notes"),),
        )

    def list_interviews(self, **kwargs):
        return (self.prior, self.later)


class Applications:
    def get_application(self, **kwargs):
        return SimpleNamespace(
            application=SimpleNamespace(
                id="app-1",
                job_posting_id="job-1",
                jd_snapshot_id="jd-1",
                resume_version_id="resume-v1",
            ),
            job=SimpleNamespace(
                posting=SimpleNamespace(company_name="Example Corp", title="RAG Engineer"),
                snapshot=SimpleNamespace(content="Build reliable retrieval systems."),
            ),
        )


class Resumes:
    def read_version_document(self, **kwargs):
        return StoredResumeDocument(
            resume_version_id="resume-v1",
            document_format="markdown",
            raw_bytes=b"Built retrieval evaluations.",
        )


class CareerHistory:
    def list_evidence(self, **kwargs):
        return (
            SimpleNamespace(
                claim="Built retrieval evaluations.",
                source_locator="experience.1",
                source_quote="Built retrieval evaluations.",
            ),
        )


def test_context_uses_latest_prior_retro_without_raw_source_notes() -> None:
    sources = InterviewPreparationContextFactory(
        interviews=Interviews(),
        applications=Applications(),
        resumes=Resumes(),
        career_history=CareerHistory(),
    ).build(
        user_id="u1",
        application_id="app-1",
        interview_round_id="round-2",
    )

    assert sources.document.resume_version_id == "resume-v1"
    assert sources.jd_snapshot_id == "jd-1"
    assert sources.context.company_name == "Example Corp"
    assert len(sources.context.confirmed_facts) == 1
    assert len(sources.context.prior_retros) == 1
    assert sources.context.prior_retros[0].summary == "Latest version"
    assert "secret raw recollection" not in sources.context.model_dump_json()
    assert "Later round" not in sources.context.model_dump_json()
