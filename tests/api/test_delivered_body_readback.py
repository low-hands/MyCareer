from argparse import Namespace
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sqlite3
from unittest.mock import patch

from fastapi.testclient import TestClient
import pytest

from career_agent.agent.context_manager import ContextManager
from career_agent.agent.delivered_body_contracts import (
    BodyDependency,
    MockInterviewBodySource,
    ResumeAnalysisBodySource,
    SavedJobBodySource,
)
from career_agent.agent.main_agent_contracts import (
    ConversationMessageContext,
    ConversationResourceReference,
    ConversationTaskState,
    ToolResult,
)
from career_agent.agent.main_agent_runtime import MainAgentRuntime
from career_agent.agent.resume_analysis_contracts import (
    ExtractedCareerRecord,
    ResumeAnalysisResult,
)
from career_agent.api.app import create_app
from career_agent.api.reads import WorkspaceReader
from career_agent.domain.job_discovery import JobDetail, Provenance
from career_agent.domain.mock_interviews.models import (
    MockInterviewAnswerEvaluation,
    MockInterviewPlan,
    MockInterviewPlanItem,
    MockInterviewScoreDimension,
)
from career_agent.storage.context import CareerContextStore, DeliveredBodyDraft
from career_agent.storage.jobs import SQLiteJobPostingRepository
from career_agent.storage.mock_interviews import SQLiteMockInterviewStore
from career_agent.storage.resume_analysis import SQLiteResumeAnalysisDraftStore
from career_agent.storage.turn_receipts import SQLiteTurnReceiptStore
from career_agent.harness.streaming import ContentDeltaEvent


def _args(path: Path) -> Namespace:
    return Namespace(
        **{
            name: str(path / f"{name}.sqlite3")
            for name in (
                "context_store",
                "resume_store",
                "application_store",
                "job_store",
                "interview_store",
                "email_store",
                "action_center_store",
                "calendar_store",
                "mock_interview_store",
                "job_research_store",
            )
        }
    )


class _Runtime:
    def close(self) -> None:
        pass


def _store_body(
    path: Path, body: DeliveredBodyDraft, *, turn_id: str | None = None
) -> str:
    store = CareerContextStore(Path(_args(path).context_store))
    ContextManager(store).load_for_turn(
        user_id="u1", conversation_id="c1", user_message="查看结果"
    )
    now = datetime.now(timezone.utc)
    store.commit_turn(
        user_id="u1",
        conversation_id="c1",
        task=ConversationTaskState(),
        user_message=ConversationMessageContext(
            role="user", content="查看结果", created_at=now
        ),
        assistant_message=ConversationMessageContext(
            role="assistant",
            content="结果已展示",
            created_at=now,
            resource_refs=(
                ConversationResourceReference(
                    kind="job_research_report",
                    resource_id="original",
                    status_at_delivery="current",
                    anchored_by_other_job=False,
                ),
            ),
        ),
        assistant_bodies=(body,),
        memory_scope_keys=("career_evidence/private/claim",),
        turn_id=turn_id,
    )
    return store.list_delivered_body_references("u1", "c1", from_sequence=1)[-1].body_id


def _job(path: Path, description: str = "原始岗位描述"):
    now = datetime.now(timezone.utc)
    return SQLiteJobPostingRepository(Path(_args(path).job_store)).save_detail(
        user_id="u1",
        run_id="discovery-1",
        result_ref="result-1",
        selection_index=1,
        detail=JobDetail(
            source_name="test",
            source_job_id="job-1",
            source_url="https://example.test/jobs/1",
            title="工程师",
            company_name="示例公司",
            description=description,
            captured_at=now,
            provenance=Provenance(
                source_name="test",
                source_job_id="job-1",
                source_url="https://example.test/jobs/1",
                captured_at=now,
                operation="detail",
                adapter_version="test-v1",
            ),
        ),
    )


def test_source_card_reads_current_job_without_copying_body_or_model_refs(
    tmp_path, api_keys, auth, issue_key
):
    saved = _job(tmp_path)
    result = ToolResult(
        tool_name="get_saved_job",
        state="saved_job_ready",
        message="已读取岗位",
        body_source=SavedJobBodySource(job_posting_id=saved.posting.id),
    )
    body_id = _store_body(
        tmp_path, MainAgentRuntime._delivered_bodies((result,))[0], turn_id="t1"
    )
    changed = _job(tmp_path, "更新后的岗位描述")
    assert changed.posting.id == saved.posting.id
    reader = WorkspaceReader(_args(tmp_path))
    app = create_app(
        runtime_factory=_Runtime,
        workspace_reader_factory=lambda: reader,
        api_key_store_factory=lambda: api_keys,
        action_center_factory=lambda: None,
    )
    with TestClient(app) as client:
        report = client.get(f"/v1/reports/delivered_body/{body_id}", headers=auth)
        assert report.status_code == 200
        assert report.json()["body"] == "更新后的岗位描述"
        assert (
            client.get(
                f"/v1/reports/delivered_body/{body_id}", headers=issue_key("u2")
            ).status_code
            == 404
        )
    transcript = reader.conversation_messages(user_id="u1", conversation_id="c1")
    assert [ref.kind for ref in transcript.messages[-1].resources] == [
        "job_research_report",
        "delivered_body",
    ]
    store = CareerContextStore(Path(_args(tmp_path).context_store))
    message = store.list_messages("u1", "c1", limit=10)[-1]
    assert [ref.kind for ref in message.resource_refs] == ["job_research_report"]
    assert body_id not in message.model_dump_json()
    with sqlite3.connect(store.path) as connection:
        body, source = connection.execute(
            "SELECT body, source_json FROM conversation_delivered_bodies"
        ).fetchone()
    assert body == "" and saved.posting.id in source
    assert "岗位描述" not in source
    receipts = SQLiteTurnReceiptStore(store.path)
    key = dict(user_id="u1", conversation_id="c1", request_id="r1")
    receipts.begin(**key, turn_id="t1")
    receipts.commit(**key, turn_id="t1", events=(ContentDeltaEvent(delta="岗位描述"),))
    other = dict(key, request_id="r2")
    receipts.begin(**other, turn_id="t2")
    receipts.commit(**other, turn_id="t2", events=(ContentDeltaEvent(delta="无关"),))
    assert reader.delete_job(user_id="u1", job_posting_id="job-missing") == "not_found"
    assert store.get_delivered_body("u1", body_id) is not None
    assert receipts.get(**key).content_status == "available"
    assert reader.delete_job(user_id="u1", job_posting_id=saved.posting.id) == "deleted"
    assert store.get_delivered_body("u1", body_id) is None
    receipt = receipts.get(**key)
    assert receipt.status == "COMMITTED" and receipt.content_status == "deleted"
    assert receipt.events == ()
    assert receipts.get(**other).content_status == "available"
    with sqlite3.connect(store.path) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM conversation_delivered_bodies"
        ).fetchone() == (0,)


def test_snapshot_survives_job_update_and_is_physically_deleted_with_job(tmp_path):
    saved = _job(tmp_path)
    body_id = _store_body(
        tmp_path,
        DeliveredBodyDraft(
            kind="saved_jobs_compared",
            title="岗位对比",
            retention="snapshot",
            body="生成时的岗位对比",
            dependencies=(BodyDependency(kind="job", resource_id=saved.posting.id),),
        ),
    )
    _job(tmp_path, "新的岗位要求")
    reader = WorkspaceReader(_args(tmp_path))
    report = reader.report(user_id="u1", kind="delivered_body", resource_id=body_id)
    assert report.body == "生成时的岗位对比"
    assert report.subtitle == "历史快照（截至生成时间）"
    assert reader.delete_job(user_id="u1", job_posting_id=saved.posting.id) == "deleted"
    assert (
        reader.report(user_id="u1", kind="delivered_body", resource_id=body_id) is None
    )
    with sqlite3.connect(_args(tmp_path).context_store) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM conversation_delivered_bodies"
        ).fetchone() == (0,)


@pytest.mark.parametrize(
    "expired,removed", [(False, False), (True, False), (True, True)]
)
def test_resume_handle_uses_original_expiry_even_after_cleanup(
    tmp_path, api_keys, auth, issue_key, expired, removed
):
    drafts = SQLiteResumeAnalysisDraftStore(
        Path(_args(tmp_path).resume_store), ttl=timedelta(hours=1)
    )
    now = datetime.now(timezone.utc)
    draft = drafts.create(
        user_id="u1",
        resume_version_id="rv-1",
        result=ResumeAnalysisResult(
            records=(
                ExtractedCareerRecord(
                    record_type="work",
                    title="PRIVATE CANDIDATE",
                    source_locator="第 1 页",
                    source_quote="PRIVATE CANDIDATE",
                ),
            )
        ),
        now=now - timedelta(hours=2) if expired else now,
    )
    body_id = _store_body(
        tmp_path,
        DeliveredBodyDraft(
            kind="resume_analysis_ready",
            title="简历分析",
            retention="source",
            source=ResumeAnalysisBodySource(
                analysis_id=draft.id, expires_at=draft.expires_at
            ),
        ),
    )
    if removed:
        assert drafts.delete_expired(now=now) == 1
    reader = WorkspaceReader(_args(tmp_path))
    app = create_app(
        runtime_factory=_Runtime,
        workspace_reader_factory=lambda: reader,
        api_key_store_factory=lambda: api_keys,
        action_center_factory=lambda: None,
    )
    with TestClient(app) as client:
        response = client.get(f"/v1/reports/delivered_body/{body_id}", headers=auth)
        assert response.status_code == 200
        report = response.json()
        assert report["availability"] == ("expired" if expired else "available")
        assert ("PRIVATE CANDIDATE" in report["body"]) is not expired
        assert (
            client.get(
                f"/v1/reports/delivered_body/{body_id}", headers=issue_key("u2")
            ).status_code
            == 404
        )
    assert (
        reader.conversation_messages(user_id="u1", conversation_id="c1")
        .messages[-1]
        .resources[-1]
        .resource_id
        == body_id
    )
    with sqlite3.connect(_args(tmp_path).context_store) as connection:
        row = connection.execute(
            "SELECT body, source_json FROM conversation_delivered_bodies"
        ).fetchone()
    assert row[0] == "" and "PRIVATE CANDIDATE" not in row[1]
    assert draft.expires_at.isoformat().replace("+00:00", "Z") in row[1]


def test_mock_interview_handle_includes_follow_up_and_checks_message_visibility_first(
    tmp_path,
):
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
            summary="项目题",
            items=(
                MockInterviewPlanItem(
                    sequence_number=1,
                    question_type="project_deep_dive",
                    difficulty="intermediate",
                    focus="项目证据",
                    rationale="岗位要求",
                ),
            ),
            created_at=datetime.now(timezone.utc),
        ),
    )
    session = store.start(session=session)
    session, primary = store.ask(
        session=session,
        plan_item_number=1,
        question_type="project_deep_dive",
        question="主问题",
    )
    session, primary = store.record_answer(
        session=session, turn=primary, answer="主回答"
    )
    session, _ = store.record_evaluation(
        session=session,
        turn=primary,
        evaluation=MockInterviewAnswerEvaluation(
            rating="adequate",
            summary="需要澄清",
            dimensions=(
                MockInterviewScoreDimension(
                    dimension="reasoning",
                    score=3,
                    feedback="需要澄清",
                ),
            ),
            next_action="follow_up",
            next_action_reason="追问",
            follow_up_question="追问问题",
        ),
    )
    session, follow_up = store.ask(
        session=session,
        plan_item_number=1,
        question_type="project_deep_dive",
        question="追问问题",
        turn_type="follow_up",
        parent_turn_id=primary.id,
    )
    store.record_answer(session=session, turn=follow_up, answer="追问回答")
    body_id = _store_body(
        tmp_path,
        DeliveredBodyDraft(
            kind="mock_interview_question_found",
            title="模拟面试问答",
            retention="source",
            source=MockInterviewBodySource(session_id=session.id, question_number=1),
        ),
    )
    reader = WorkspaceReader(_args(tmp_path))
    report = reader.report(user_id="u1", kind="delivered_body", resource_id=body_id)
    assert all(
        text in report.body for text in ("主问题", "主回答", "追问问题", "追问回答")
    )
    CareerContextStore(Path(_args(tmp_path).context_store)).purge_derived_memory(
        user_id="u1", scope_key="career_evidence/private/claim"
    )
    with patch.object(
        reader._mock_interviews,
        "get_session",
        side_effect=AssertionError("source read before visibility"),
    ):
        assert (
            reader.report(user_id="u1", kind="delivered_body", resource_id=body_id)
            is None
        )
    assert (
        reader.conversation_messages(user_id="u1", conversation_id="c1").messages == ()
    )
