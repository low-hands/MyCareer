from datetime import datetime, timezone

import pytest

from career_agent.agent.job_research_contracts import JobResearchWorkerRequest
from career_agent.agent.openai_compatible_client import AgentWorkerError
from career_agent.domain.job_discovery import JobDetail, Provenance
from career_agent.domain.job_research import (
    JobResearchDraft,
    JobResearchFindingDraft,
    JobResearchSourceDraft,
)
from career_agent.services.job_research import (
    JobResearchExecutionError,
    JobResearchService,
)
from career_agent.storage.job_research import SQLiteJobResearchStore
from career_agent.storage.jobs import SQLiteJobPostingRepository


NOW = datetime(2026, 8, 30, tzinfo=timezone.utc)


def _seed(jobs: SQLiteJobPostingRepository):
    detail = JobDetail(
        source_name="test",
        source_job_id="job-1",
        source_url="https://jobs.example.test/1",
        title="RAG Engineer",
        company_name="Example Corp",
        description="Build reliable retrieval systems for enterprise users.",
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


def _draft(*, source_key: str = "S1", cited_key: str = "S1") -> JobResearchDraft:
    return JobResearchDraft(
        summary="The role is tied to enterprise retrieval reliability.",
        sources=(JobResearchSourceDraft(
            source_key=source_key,
            url="https://example.com/product?utm_source=test&token=secret",
            title="Enterprise retrieval product",
            publisher="Example Corp",
            relevant_excerpt="The product provides retrieval for enterprise users.",
        ),),
        findings=(JobResearchFindingDraft(
            topic="Product context",
            statement="The public product targets enterprise retrieval use cases.",
            evidence_type="fact",
            source_keys=(cited_key,),
            confidence="high",
        ),),
        open_questions=("Which team owns retrieval evaluation?",),
    )


class Worker:
    def __init__(self, draft: JobResearchDraft | None = None) -> None:
        self.draft = draft or _draft()
        self.calls: list[tuple[str, JobResearchWorkerRequest, bool]] = []
        self.forgotten: list[str] = []
        self.fail_once = False

    def research(self, *, run_id, request, resume=False):
        self.calls.append((run_id, request, resume))
        if self.fail_once:
            self.fail_once = False
            raise AgentWorkerError(
                "JOB_RESEARCH_TRANSPORT_ERROR",
                "temporary failure",
                retryable=True,
            )
        return self.draft

    def forget(self, run_id):
        self.forgotten.append(run_id)


def _service(tmp_path, worker: Worker):
    jobs = SQLiteJobPostingRepository(tmp_path / "jobs.sqlite3")
    saved = _seed(jobs)
    store = SQLiteJobResearchStore(tmp_path / "research.sqlite3")
    return (
        JobResearchService(jobs=jobs, store=store, worker=worker),
        store,
        saved,
    )


def test_research_persists_sources_report_and_reuses_fresh_result(tmp_path) -> None:
    worker = Worker()
    service, store, saved = _service(tmp_path, worker)

    first = service.research(
        user_id="u1",
        job_posting_id=saved.posting.id,
        focus="technical constraints",
        user_provided_context="一面提到企业知识库产品。",
    )
    repeated = service.research(
        user_id="u1",
        job_posting_id=saved.posting.id,
        focus="technical constraints",
        user_provided_context="一面提到企业知识库产品。",
    )

    assert first.run.status == "completed"
    assert first.report.jd_snapshot_id == saved.snapshot.id
    assert first.cached is False
    assert repeated.report.id == first.report.id
    assert repeated.cached is True
    assert len(worker.calls) == 1
    assert (
        worker.calls[0][1].scope.user_provided_context
        == "一面提到企业知识库产品。"
    )
    assert worker.forgotten == [first.run.id]
    assert len(first.sources) == 1
    assert "secret" not in first.sources[0].url
    assert "utm_source" not in first.sources[0].normalized_url
    assert store.get_report(user_id="other", report_id=first.report.id) is None


def test_user_context_is_part_of_research_identity_not_public_evidence(tmp_path) -> None:
    worker = Worker()
    service, _, saved = _service(tmp_path, worker)

    first = service.research(
        user_id="u1",
        job_posting_id=saved.posting.id,
        user_provided_context="一面提到企业知识库。",
    )
    second = service.research(
        user_id="u1",
        job_posting_id=saved.posting.id,
        user_provided_context="二面提到智能客服。",
    )

    assert first.run.input_fingerprint != second.run.input_fingerprint
    assert first.report.id != second.report.id
    assert len(worker.calls) == 2
    assert second.report.scope.user_provided_context == "二面提到智能客服。"


def test_failed_run_keeps_checkpoint_identity_and_can_retry(tmp_path) -> None:
    worker = Worker()
    worker.fail_once = True
    service, store, saved = _service(tmp_path, worker)

    with pytest.raises(JobResearchExecutionError):
        service.research(user_id="u1", job_posting_id=saved.posting.id)

    failed_run_id = worker.calls[0][0]
    failed = store.get_run(user_id="u1", run_id=failed_run_id)
    assert failed is not None and failed.status == "failed"
    assert worker.forgotten == []

    result = service.retry(user_id="u1", run_id=failed_run_id)

    assert result.run.id == failed_run_id
    assert worker.calls[-1][2] is True
    assert worker.forgotten == [failed_run_id]


def test_unknown_or_unused_source_keys_fail_before_persistence(tmp_path) -> None:
    worker = Worker(_draft(cited_key="S2"))
    service, store, saved = _service(tmp_path, worker)

    with pytest.raises(JobResearchExecutionError, match="unknown source keys"):
        service.research(user_id="u1", job_posting_id=saved.posting.id)

    run = store.get_run(user_id="u1", run_id=worker.calls[0][0])
    assert run is not None and run.status == "failed"
    assert run.report_id is None
