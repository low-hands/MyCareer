from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from career_agent.domain.interview_preparation import (
    EvidenceStory,
    InterviewFocusArea,
    InterviewPreparationResult,
    LikelyQuestion,
)
from career_agent.domain.interviews import InterviewRound
from career_agent.services.interview_preparation import (
    InterviewPreparationNotAvailableError,
    InterviewPreparationService,
)
from career_agent.services.interviews import InterviewDetail
from career_agent.storage.interview_preparations import SQLiteInterviewPreparationStore
from career_agent.storage.resumes import StoredResumeDocument


NOW = datetime(2026, 8, 26, tzinfo=timezone.utc)


class Interviews:
    def __init__(self):
        self.interview = InterviewRound(
            id="interview-1", user_id="u1", application_id="application-1",
            sequence_number=1, status="scheduled",
            scheduled_start=NOW + timedelta(days=2),
            scheduled_end=NOW + timedelta(days=2, hours=1),
            timezone="Asia/Shanghai", interview_format="video",
            created_at=NOW, updated_at=NOW,
        )

    def get_interview(self, **kwargs):
        return InterviewDetail(interview=self.interview, events=())

    def list_interviews(self, **kwargs):
        return ()


class Applications:
    def get_application(self, **kwargs):
        application = SimpleNamespace(
            id="application-1", job_posting_id="job-1", jd_snapshot_id="jd-1",
            resume_version_id="resume-version-1",
        )
        snapshot = SimpleNamespace(id="jd-1", content="Build reliable RAG systems.")
        posting = SimpleNamespace(company_name="Example Corp", title="RAG Engineer")
        return SimpleNamespace(
            application=application,
            job=SimpleNamespace(snapshot=snapshot, posting=posting),
        )


class Resumes:
    def read_version_document(self, **kwargs):
        return StoredResumeDocument(
            resume_version_id="resume-version-1", document_format="markdown",
            raw_bytes=b"Built a retrieval evaluation suite.",
        )


class CareerHistory:
    def list_evidence(self, **kwargs):
        return ()


class Worker:
    def __init__(self):
        self.calls = []

    def prepare(self, **kwargs):
        self.calls.append(kwargs)
        return InterviewPreparationResult(
            summary="重点准备 RAG 可靠性和评估。",
            focus_areas=(InterviewFocusArea(
                topic="RAG reliability", priority="high",
                rationale="The JD explicitly asks for reliable systems.",
                jd_quote="Build reliable RAG systems.",
            ),),
            evidence_stories=(EvidenceStory(
                theme="Evaluation", resume_locator="Project bullet 1",
                resume_quote="Built a retrieval evaluation suite.",
                preparation_prompt="补充你实际使用的指标和取舍。",
            ),),
            likely_questions=(LikelyQuestion(
                question="如何评估 RAG？", rationale="岗位强调可靠性。",
                answer_outline=("说明简历中已有的评估套件",),
            ),),
            checklist=("确认会议链接",),
        )


def build_service(tmp_path):
    interviews = Interviews()
    worker = Worker()
    service = InterviewPreparationService(
        interviews, Applications(), Resumes(), CareerHistory(), worker,
        SQLiteInterviewPreparationStore(tmp_path / "preparations.sqlite3"),
    )
    return service, interviews, worker


def test_preparation_uses_exact_documents_and_is_idempotent(tmp_path) -> None:
    service, _, worker = build_service(tmp_path)

    first = service.prepare(user_id="u1", interview_round_id="interview-1")
    repeated = service.prepare(user_id="u1", interview_round_id="interview-1")

    assert repeated.id == first.id
    assert len(worker.calls) == 1
    assert worker.calls[0]["context"].jd_text == "Build reliable RAG systems."
    assert worker.calls[0]["context"].company_name == "Example Corp"
    assert worker.calls[0]["document"].resume_version_id == "resume-version-1"
    assert first.jd_snapshot_id == "jd-1"
    assert first.resume_version_id == "resume-version-1"


def test_interview_change_creates_new_preparation_snapshot(tmp_path) -> None:
    service, interviews, worker = build_service(tmp_path)
    first = service.prepare(user_id="u1", interview_round_id="interview-1")
    interviews.interview = interviews.interview.model_copy(
        update={
            "interview_format": "onsite", "location": "Shanghai",
            "updated_at": NOW + timedelta(hours=1),
        }
    )

    changed = service.prepare(user_id="u1", interview_round_id="interview-1")

    assert changed.id != first.id
    assert len(worker.calls) == 2


def test_cancelled_interview_cannot_generate_preparation(tmp_path) -> None:
    service, interviews, worker = build_service(tmp_path)
    interviews.interview = interviews.interview.model_copy(update={"status": "cancelled"})

    with pytest.raises(InterviewPreparationNotAvailableError, match="cancelled"):
        service.prepare(user_id="u1", interview_round_id="interview-1")

    assert worker.calls == []
