"""Reading back a report the conversation only kept a summary of.

A report-producing turn stores the short prose the user read plus a reference to
the entity. That makes this endpoint the only path from a reloaded transcript to
the full report, so the tests here go through the real stores rather than a stub
reader: a rendered body that the UI cannot reach is the exact failure mode.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from fastapi.testclient import TestClient

from career_agent.api.app import create_app
from career_agent.api.reads import build_workspace_reader
from career_agent.agent.main_agent_contracts import ConversationTaskState
from career_agent.agent.resume_analysis_contracts import (
    ExtractedCareerEvidence,
    ExtractedCareerRecord,
    ResumeAnalysisResult,
)
from career_agent.agent.session_contracts import AgentSession
from career_agent.domain.interview_preparation import (
    InterviewFocusArea,
    InterviewPreparationResult,
    LikelyQuestion,
)
from career_agent.domain.job_discovery import JobDetail, Provenance
from career_agent.domain.job_research import (
    JobResearchDraft,
    JobResearchFindingDraft,
    JobResearchSourceDraft,
)
from career_agent.domain.mock_interviews import (
    MockInterviewAnswerEvaluation,
    MockInterviewPlan,
    MockInterviewPlanItem,
    MockInterviewQuestionResult,
    MockInterviewReport,
    MockInterviewScoreDimension,
)
from career_agent.services.job_research import JobResearchService
from career_agent.storage.interview_preparations import (
    SQLiteInterviewPreparationStore,
)
from career_agent.storage.job_research import SQLiteJobResearchStore
from career_agent.storage.jobs import SQLiteJobPostingRepository
from career_agent.storage.mock_interviews import SQLiteMockInterviewStore
from career_agent.storage.context import CareerContextStore
from career_agent.storage.resume_analysis import SQLiteResumeAnalysisDraftStore

NOW = datetime(2026, 8, 31, tzinfo=timezone.utc)


class _Runtime:
    def close(self) -> None:
        pass


def _args(tmp_path: Path) -> argparse.Namespace:
    """Every store the reader opens, pointed at one throwaway directory.

    Preparations share the resume store file, matching how the CLI wires them,
    so a wrong path here would read an empty table rather than fail loudly.
    """
    return argparse.Namespace(
        job_store=str(tmp_path / "jobs.sqlite3"),
        resume_store=str(tmp_path / "resumes.sqlite3"),
        application_store=str(tmp_path / "applications.sqlite3"),
        calendar_store=str(tmp_path / "calendar.sqlite3"),
        job_research_store=str(tmp_path / "research.sqlite3"),
        context_store=str(tmp_path / "context.sqlite3"),
        mock_interview_store=str(tmp_path / "mock.sqlite3"),
    )


def _client(tmp_path: Path) -> TestClient:
    args = _args(tmp_path)
    return TestClient(
        create_app(
            runtime_factory=_Runtime,
            action_center_factory=lambda: None,
            workspace_reader_factory=lambda: build_workspace_reader(args),
        )
    )


def test_pending_resume_analysis_rebuilds_body_and_bound_interaction(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path)
    draft = SQLiteResumeAnalysisDraftStore(
        Path(args.resume_store)
    ).create(
        user_id="u1",
        resume_version_id="resume-version-1",
        result=ResumeAnalysisResult(
            records=(
                ExtractedCareerRecord(
                    record_type="work",
                    organization="示例科技",
                    title="产品经理",
                    source_locator="第 1 页",
                    source_quote="示例科技 产品经理",
                    evidence=(
                        ExtractedCareerEvidence(
                            claim="负责知识库产品规划",
                            source_locator="第 1 页，第 1 条",
                            source_quote="负责知识库产品规划",
                        ),
                    ),
                ),
            ),
        ),
    )
    context = CareerContextStore(Path(args.context_store))
    context.upsert_session(
        AgentSession(
            session_id="c1",
            user_id="u1",
            created_at=NOW,
            last_active_at=NOW,
        )
    )
    context.upsert_task(
        user_id="u1",
        conversation_id="c1",
        task=ConversationTaskState(
            active_resume_analysis_id=draft.id,
            resume_analysis_status="pending",
        ),
    )

    transcript = build_workspace_reader(args).conversation_messages(
        user_id="u1",
        conversation_id="c1",
    )

    assert "负责知识库产品规划" in transcript.pending_interaction_body
    assert transcript.pending_interaction is not None
    assert transcript.pending_interaction.scope == "resume_analysis_confirmation"
    assert draft.id not in transcript.pending_interaction.interaction_id


def _saved_job(tmp_path: Path):
    jobs = SQLiteJobPostingRepository(Path(_args(tmp_path).job_store))
    detail = JobDetail(
        source_name="test",
        source_job_id="job-1",
        source_url="https://jobs.example.test/1",
        title="RAG 工程师",
        company_name="示例科技",
        description="负责企业检索系统的可靠性。",
        captured_at=NOW,
        provenance=Provenance(
            source_name="test",
            source_job_id="job-1",
            source_url="https://jobs.example.test/1",
            captured_at=NOW,
            operation="detail",
            adapter_version="test-v1",
        ),
    )
    return jobs.save_detail(
        user_id="u1",
        run_id="discovery-1",
        result_ref="result-1",
        selection_index=1,
        detail=detail,
    )


class _ResearchWorker:
    def research(self, *, run_id, request, resume=False):
        return JobResearchDraft(
            summary="公司主营企业级检索产品。",
            sources=(
                JobResearchSourceDraft(
                    source_key="S1",
                    url="https://example.com/product",
                    title="企业检索产品页",
                    publisher="示例科技",
                    relevant_excerpt="产品面向企业客户提供检索能力。",
                ),
            ),
            findings=(
                JobResearchFindingDraft(
                    topic="产品定位",
                    statement="公开资料显示产品面向企业检索场景。",
                    evidence_type="fact",
                    source_keys=("S1",),
                    confidence="high",
                ),
            ),
            open_questions=("检索评估由哪个团队负责？",),
        )

    def forget(self, run_id):
        pass


def _job_research_report(tmp_path: Path) -> str:
    saved = _saved_job(tmp_path)
    service = JobResearchService(
        jobs=SQLiteJobPostingRepository(Path(_args(tmp_path).job_store)),
        store=SQLiteJobResearchStore(Path(_args(tmp_path).job_research_store)),
        worker=_ResearchWorker(),
    )
    result = service.research(user_id="u1", job_posting_id=saved.posting.id)
    return result.report.id


def _mock_interview_report(tmp_path: Path) -> str:
    store = SQLiteMockInterviewStore(Path(_args(tmp_path).mock_interview_store))
    session = store.create_session(
        user_id="u1",
        application_id="app-1",
        job_posting_id="job-1",
        jd_snapshot_id="jd-1",
        resume_version_id="rv-1",
        interview_type="technical",
    )
    store.save_plan(
        session=session,
        plan=MockInterviewPlan(
            session_id=session.id,
            summary="一题",
            items=(
                MockInterviewPlanItem(
                    sequence_number=1,
                    question_type="project_deep_dive",
                    difficulty="intermediate",
                    focus="检索可靠性",
                    rationale="JD 要求",
                    jd_quotes=("RAG",),
                    resume_locators=("简历",),
                    resume_quotes=("负责检索系统",),
                ),
            ),
            created_at=NOW,
        ),
    )
    session = store.start(session=session)
    session, turn = store.ask(
        session=session,
        plan_item_number=1,
        question_type="project_deep_dive",
        question="介绍一个检索可靠性改进。",
    )
    session, turn = store.record_answer(
        session=session, turn=turn, answer="我设计了离线评估集。"
    )
    session, _ = store.record_evaluation(
        session=session,
        turn=turn,
        evaluation=MockInterviewAnswerEvaluation(
            rating="strong",
            summary="回答具体。",
            dimensions=(
                MockInterviewScoreDimension(
                    dimension="specificity", score=4, feedback="有数据"
                ),
            ),
            next_action="next_question",
            next_action_reason="够了",
        ),
    )
    report = MockInterviewReport(
        id="rep-1",
        session_id=session.id,
        completion_reason="plan_completed",
        summary="项目深度可以。",
        question_results=(
            MockInterviewQuestionResult(
                plan_item_number=1,
                question="介绍一个检索可靠性改进。",
                final_rating="strong",
                summary="好",
                follow_up_count=0,
            ),
        ),
        strengths=("能讲清个人职责",),
        development_areas=("补降级细节",),
        practice_actions=("重写系统设计回答",),
        created_at=NOW,
    )
    store.complete(session=session, report=report)
    return report.id


def _interview_preparation(tmp_path: Path) -> str:
    store = SQLiteInterviewPreparationStore(Path(_args(tmp_path).resume_store))
    stored = store.save(
        user_id="u1",
        interview_round_id="round-1",
        application_id="app-1",
        job_posting_id=_saved_job(tmp_path).posting.id,
        jd_snapshot_id="jd-1",
        resume_version_id="rv-1",
        input_fingerprint="f" * 64,
        worker_version="test-v1",
        result=InterviewPreparationResult(
            summary="重点准备检索可靠性。",
            focus_areas=(
                InterviewFocusArea(
                    topic="检索评估",
                    priority="high",
                    rationale="JD 明确要求",
                    jd_quote="负责企业检索系统的可靠性",
                ),
            ),
            likely_questions=(
                LikelyQuestion(
                    question="你怎么衡量召回质量？",
                    rationale="JD 提到可靠性",
                ),
            ),
        ),
    )
    return stored.id


def test_each_report_kind_reads_back_its_full_rendered_body(tmp_path) -> None:
    """All three kinds, because a reference the UI cannot follow is worse than none."""
    research_id = _job_research_report(tmp_path)
    mock_id = _mock_interview_report(tmp_path)
    preparation_id = _interview_preparation(tmp_path)

    with _client(tmp_path) as client:
        research = client.get(
            f"/v1/reports/job_research_report/{research_id}",
            params={"user_id": "u1"},
        ).json()
        mock = client.get(
            f"/v1/reports/mock_interview_report/{mock_id}",
            params={"user_id": "u1"},
        ).json()
        preparation = client.get(
            f"/v1/reports/interview_preparation/{preparation_id}",
            params={"user_id": "u1"},
        ).json()

    # The body is the report itself, not the one-line summary the row kept.
    assert "公司主营企业级检索产品。" in research["body"]
    assert "[S1]" in research["body"]
    assert research["title"] == "示例科技 公司调研"
    assert research["subtitle"] == "1 条结论 · 1 个来源"

    assert "项目深度可以。" in mock["body"]
    assert "能讲清个人职责" in mock["body"]
    assert mock["subtitle"] == "技术面 · 1 题"

    assert "重点准备检索可靠性。" in preparation["body"]
    assert "你怎么衡量召回质量？" in preparation["body"]
    assert preparation["subtitle"] == "示例科技 RAG 工程师 · 1 个可能问题"


def test_job_research_replays_delivery_time_warnings(tmp_path) -> None:
    """A card renders the historical turn, not today's mutable report state."""
    research_id = _job_research_report(tmp_path)

    with _client(tmp_path) as client:
        response = client.get(
            f"/v1/reports/job_research_report/{research_id}",
            params={
                "user_id": "u1",
                "status_at_delivery": "outdated",
                "anchored_by_other_job": "true",
            },
        )

    assert response.status_code == 200
    body = response.json()["body"]
    assert "另一个岗位所触发" in body
    assert "已超过当前时效窗口" in body


def test_another_users_report_is_not_readable(tmp_path) -> None:
    """The id in a transcript is not authorization; the owner filter is."""
    ids = {
        "job_research_report": _job_research_report(tmp_path),
        "mock_interview_report": _mock_interview_report(tmp_path),
        "interview_preparation": _interview_preparation(tmp_path),
    }

    with _client(tmp_path) as client:
        for kind, resource_id in ids.items():
            response = client.get(
                f"/v1/reports/{kind}/{resource_id}", params={"user_id": "u2"}
            )
            assert response.status_code == 404, kind


def test_a_kind_the_transcript_never_produces_is_not_dispatched(tmp_path) -> None:
    """An unknown kind must not fall through to a lookup in another store.

    Ids are unique per store, not globally, so a kind that resolved by trying
    each store in turn could serve an unrelated entity for a colliding id.
    """
    mock_id = _mock_interview_report(tmp_path)

    with _client(tmp_path) as client:
        assert (
            client.get(
                f"/v1/reports/resume_analysis/{mock_id}", params={"user_id": "u1"}
            ).status_code
            == 404
        )
        # The same id under its own kind still resolves, so this is the kind
        # check failing the lookup rather than the seed being absent.
        assert (
            client.get(
                f"/v1/reports/mock_interview_report/{mock_id}",
                params={"user_id": "u1"},
            ).status_code
            == 200
        )


def test_an_unknown_id_is_a_missing_report_not_an_error(tmp_path) -> None:
    with _client(tmp_path) as client:
        response = client.get(
            "/v1/reports/mock_interview_report/rep-missing",
            params={"user_id": "u1"},
        )

    assert response.status_code == 404
