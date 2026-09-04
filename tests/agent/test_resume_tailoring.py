from __future__ import annotations

import base64
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from career_agent.agent.context_manager import ContextManager
from career_agent.agent.main_agent_contracts import (
    AgentDecision,
    CareerProfileContext,
    ToolCall,
)
from career_agent.agent.main_agent_runtime import MainAgentRuntime
from career_agent.agent.main_agent_tools import MainAgentToolRegistry
from career_agent.agent.openai_compatible_client import (
    AgentWorkerError,
    OpenAICompatibleAgentConfig,
)
from career_agent.agent.openai_resume_tailoring_reviewer import (
    OpenAIResumeTailoringReviewer,
)
from career_agent.agent.deepagent_resume_tailoring_worker import (
    DeepAgentResumeFinalizationWorker,
    DeepAgentResumeTailoringWorker,
)
from career_agent.agent.resume_job_match_contracts import ResumeJobMatchResult
from career_agent.agent.resume_tailoring_contracts import (
    AcceptedTailoringChange,
    FinalizedResumeDocument,
    ResumeReviewIssue,
    ResumeReviewResult,
    ResumeTailoringResult,
)
from career_agent.agent.resume_tailoring_review_graph import ResumeTailoringReviewGraph
from career_agent.domain.job_discovery import JobDetail, Provenance
from career_agent.services.resume_export import ResumeExportService
from career_agent.services.resume_tailoring import (
    ResumeFinalReviewBlockedError,
    ResumeTailoringAlreadyFinalizedError,
    ResumeTailoringDraftNotFoundError,
    ResumeTailoringNotReadyError,
    ResumeTailoringService,
    ResumeTailoringSupersededError,
)
from career_agent.storage.career_history import CareerHistoryStore
from career_agent.storage.context import CareerContextStore
from career_agent.storage.jobs import SQLiteJobPostingRepository
from career_agent.storage.resume_artifacts import SQLiteResumeArtifactStore
from career_agent.storage.resumes import ResumeStore, StoredResumeDocument
from career_agent.storage.resume_job_matches import SQLiteResumeJobMatchStore
from career_agent.storage.resume_tailoring import SQLiteResumeTailoringDraftStore


VALID_MATCH = ResumeJobMatchResult(
    overall_fit="moderate",
    summary="Relevant RAG experience with some gaps.",
    requirements=(),
)

VALID_DRAFT = {
    "strategy_summary": "Lead with directly supported production RAG experience.",
    "changes": [
        {
            "target_locator": "Experience, bullet 1",
            "original_quote": "Built RAG systems",
            "proposed_text": "Built production RAG systems for knowledge retrieval.",
            "rationale": "Makes the directly supported production scope clearer.",
            "addresses_requirements": ["Production RAG experience"],
            "support_evidence": [
                {
                    "source_locator": "Experience, bullet 1",
                    "source_quote": "Built RAG systems",
                }
            ],
        },
        {
            "target_locator": "Skills",
            "original_quote": "Python",
            "proposed_text": "Python · Retrieval-Augmented Generation",
            "rationale": "Surfaces an explicitly demonstrated specialization.",
            "addresses_requirements": ["RAG development"],
            "support_evidence": [
                {
                    "source_locator": "Experience, bullet 1",
                    "source_quote": "Built RAG systems",
                }
            ],
        },
    ],
    "preserved_strengths": ["RAG experience"],
    "unresolved_gaps": ["Go is not stated"],
    "clarification_questions": [],
    "warnings": [],
}


class FakeDeepAgent:
    def __init__(self, output=VALID_DRAFT) -> None:
        self.output = output
        self.state = None

    def invoke(self, state):
        self.state = state
        return {"structured_response": self.output}


class FakeResponsesClient:
    def __init__(self, output: dict) -> None:
        self.output = output
        self.calls = []
        self.responses = self

    def create(self, **kwargs):
        self.calls.append(kwargs)

        class Response:
            output_text = ""

        response = Response()
        response.output_text = json.dumps(self.output)
        return response


def skill_root(tmp_path: Path) -> Path:
    root = tmp_path / "skills"
    skill = root / "resume-tailoring"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\nname: resume-tailoring\ndescription: Tailor a grounded resume.\n---\n\n"
        "Use only grounded resume evidence.\n",
        encoding="utf-8",
    )
    return root


def deep_worker(tmp_path: Path, agent: FakeDeepAgent) -> DeepAgentResumeTailoringWorker:
    return DeepAgentResumeTailoringWorker(
        OpenAICompatibleAgentConfig(
            endpoint="https://example.test/v1/chat/completions",
            api_key="secret",
            model="multimodal-model",
        ),
        skills_root=skill_root(tmp_path),
        agent=agent,
    )


def test_tailoring_worker_sends_grounded_text_inputs_and_user_goal(tmp_path) -> None:
    agent = FakeDeepAgent()

    result = deep_worker(tmp_path, agent).tailor(
        document=StoredResumeDocument(
            resume_version_id="v1",
            document_format="markdown",
            raw_bytes=b"Built RAG systems",
        ),
        jd_text="Build production RAG systems. Strong Go experience.",
        match_result=VALID_MATCH,
        tailoring_goal="Keep it concise",
    )

    assert result.changes[0].proposed_text.startswith("Built production")
    text = agent.state["messages"][0]["content"][0]["text"]
    assert "<resume_document>" in text
    assert "<job_description>" in text
    assert "<grounded_match_result>" in text
    assert "Keep it concise" in text


def test_tailoring_worker_sends_pdf_as_input_file(tmp_path) -> None:
    raw_pdf = b"%PDF-1.7\x00\xffbinary"
    agent = FakeDeepAgent()

    deep_worker(tmp_path, agent).tailor(
        document=StoredResumeDocument(
            resume_version_id="pdf-v1",
            document_format="pdf",
            raw_bytes=raw_pdf,
        ),
        jd_text="Build RAG systems",
        match_result=VALID_MATCH,
    )

    content = agent.state["messages"][0]["content"]
    assert content[0]["type"] == "file"
    assert content[0]["base64"] == base64.b64encode(raw_pdf).decode("ascii")
    assert content[0]["mime_type"] == "application/pdf"
    assert content[0]["filename"] == "pdf-v1.pdf"
    assert "detail" not in content[0]


def test_tailoring_contract_rejects_change_without_resume_evidence(tmp_path) -> None:
    invalid = dict(VALID_DRAFT)
    invalid["changes"] = [
        {
            "target_locator": "Summary",
            "original_quote": None,
            "proposed_text": "Expert Go engineer",
            "rationale": "Matches the JD",
            "addresses_requirements": ["Go"],
            "support_evidence": [],
        }
    ]

    with pytest.raises(AgentWorkerError) as error:
        deep_worker(tmp_path, FakeDeepAgent(invalid)).tailor(
            document=StoredResumeDocument(
                resume_version_id="v1",
                document_format="text",
                raw_bytes=b"resume",
            ),
            jd_text="Go",
            match_result=VALID_MATCH,
        )

    assert error.value.code == "RESUME_TAILORING_INVALID_RESPONSE"


def test_tailoring_worker_requires_local_skill_source(tmp_path) -> None:
    with pytest.raises(ValueError, match="Resume tailoring skill is missing"):
        DeepAgentResumeTailoringWorker(
            OpenAICompatibleAgentConfig(
                endpoint="https://example.test/v1/chat/completions",
                api_key="secret",
                model="multimodal-model",
            ),
            skills_root=tmp_path / "missing",
            agent=FakeDeepAgent(),
        )


def test_tailoring_worker_configures_isolated_deep_agent_with_skill(tmp_path) -> None:
    captured = {}

    def factory(**kwargs):
        captured.update(kwargs)
        return FakeDeepAgent()

    root = skill_root(tmp_path)
    worker = DeepAgentResumeTailoringWorker(
        OpenAICompatibleAgentConfig(
            endpoint="https://example.test/v1/chat/completions",
            api_key="secret",
            model="multimodal-model",
        ),
        skills_root=root,
        agent_factory=factory,
    )

    assert isinstance(worker._agent, FakeDeepAgent)
    assert captured["skills"] == ["/"]
    assert captured["response_format"] is ResumeTailoringResult
    assert captured["subagents"] == []
    assert captured["backend"].cwd == root.resolve()
    assert captured["backend"].virtual_mode is True
    assert captured["permissions"][0].operations == ["write"]
    assert captured["permissions"][0].mode == "deny"


def test_finalization_worker_applies_only_supplied_accepted_changes(tmp_path) -> None:
    agent = FakeDeepAgent(
        {
            "markdown": "# Candidate\n\n- Built production RAG systems.",
            "applied_change_indices": [1],
            "warnings": [],
        }
    )
    worker = DeepAgentResumeFinalizationWorker(
        OpenAICompatibleAgentConfig(
            endpoint="https://example.test/v1/chat/completions",
            api_key="secret",
            model="multimodal-model",
        ),
        skills_root=skill_root(tmp_path),
        agent=agent,
    )
    accepted = AcceptedTailoringChange(
        change_index=1,
        change=ResumeTailoringResult.model_validate(VALID_DRAFT).changes[0],
    )

    result = worker.finalize(
        document=StoredResumeDocument(
            resume_version_id="v1",
            document_format="text",
            raw_bytes=b"Built RAG systems",
        ),
        accepted_changes=(accepted,),
    )

    assert result.applied_change_indices == (1,)
    text = agent.state["messages"][0]["content"][0]["text"]
    assert "<resume_document>" in text
    assert "<accepted_changes>" in text
    assert '"change_index": 1' in text


def test_finalization_worker_sends_pdf_as_input_file(tmp_path) -> None:
    agent = FakeDeepAgent(
        {
            "markdown": "# Candidate\n\n- Built production RAG systems.",
            "applied_change_indices": [1],
            "warnings": [],
        }
    )
    worker = DeepAgentResumeFinalizationWorker(
        OpenAICompatibleAgentConfig(
            endpoint="https://example.test/v1/chat/completions",
            api_key="secret",
            model="multimodal-model",
        ),
        skills_root=skill_root(tmp_path),
        agent=agent,
    )
    accepted = AcceptedTailoringChange(
        change_index=1,
        change=ResumeTailoringResult.model_validate(VALID_DRAFT).changes[0],
    )
    raw_pdf = b"%PDF-1.7\x00\xffbinary"

    worker.finalize(
        document=StoredResumeDocument(
            resume_version_id="pdf-v1",
            document_format="pdf",
            raw_bytes=raw_pdf,
        ),
        accepted_changes=(accepted,),
    )

    content = agent.state["messages"][0]["content"]
    assert content[0]["type"] == "file"
    assert content[0]["base64"] == base64.b64encode(raw_pdf).decode("ascii")
    assert content[0]["mime_type"] == "application/pdf"
    assert '"change_index": 1' in content[1]["text"]


def test_independent_reviewer_receives_pdf_and_structured_candidate() -> None:
    client = FakeResponsesClient(
        {"verdict": "pass", "summary": "Grounded and relevant.", "issues": []}
    )
    reviewer = OpenAIResumeTailoringReviewer(
        OpenAICompatibleAgentConfig(
            endpoint="https://example.test/v1/chat/completions",
            api_key="secret",
            model="multimodal-model",
        ),
        client=client,
    )
    raw_pdf = b"%PDF-1.7 reviewer"

    result = reviewer.review_draft(
        document=StoredResumeDocument(
            resume_version_id="pdf-v1",
            document_format="pdf",
            raw_bytes=raw_pdf,
        ),
        jd_text="Build RAG systems",
        match_result=VALID_MATCH,
        draft=ResumeTailoringResult.model_validate(VALID_DRAFT),
    )

    assert result.verdict == "pass"
    call = client.calls[0]
    content = call["input"][0]["content"]
    assert content[0]["type"] == "input_file"
    assert content[0]["file_data"].startswith("data:application/pdf;base64,")
    assert "<candidate_change_set>" in content[1]["text"]
    assert call["text"]["format"]["schema"] == ResumeReviewResult.model_json_schema()


class RecordingTailoringWorker:
    def __init__(self) -> None:
        self.calls = []

    def tailor(self, **kwargs) -> ResumeTailoringResult:
        self.calls.append(kwargs)
        return ResumeTailoringResult.model_validate(VALID_DRAFT)


class RecordingFinalizationWorker:
    def __init__(self) -> None:
        self.calls = []

    def finalize(self, **kwargs) -> FinalizedResumeDocument:
        self.calls.append(kwargs)
        return FinalizedResumeDocument(
            markdown=(
                "# Candidate\n\n## Experience\n\n"
                "- Built production RAG systems for knowledge retrieval."
            ),
            applied_change_indices=tuple(
                item.change_index for item in kwargs["accepted_changes"]
            ),
        )


class RecordingReviewer:
    def __init__(self, *, final_verdict="pass") -> None:
        self.draft_calls = []
        self.final_calls = []
        self.final_verdict = final_verdict

    def review_draft(self, **kwargs) -> ResumeReviewResult:
        self.draft_calls.append(kwargs)
        return ResumeReviewResult(verdict="pass", summary="Draft is grounded.")

    def review_final(self, **kwargs) -> ResumeReviewResult:
        self.final_calls.append(kwargs)
        if self.final_verdict == "pass":
            return ResumeReviewResult(verdict="pass", summary="Final resume is exact.")
        return ResumeReviewResult(
            verdict="block",
            summary="An unapproved substantive edit was introduced.",
            issues=(
                ResumeReviewIssue(
                    category="change_set_mismatch",
                    severity="blocking",
                    explanation="Final content differs beyond accepted changes.",
                ),
            ),
        )


def seed_service(tmp_path, *, reviewer=None):
    resume_path = tmp_path / "resumes.sqlite3"
    resumes = ResumeStore(resume_path)
    role = resumes.create_target_role(user_id="u1", title="AI Engineer", priority=1)
    _, version = resumes.import_document(
        user_id="u1",
        target_role_id=role.id,
        name="AI Resume",
        content=b"PRIVATE RESUME: Built RAG systems",
        document_format="text",
    )
    jobs = SQLiteJobPostingRepository(tmp_path / "jobs.sqlite3")
    captured_at = datetime(2026, 8, 25, tzinfo=timezone.utc)
    job = jobs.save_detail(
        user_id="u1",
        run_id="run-1",
        result_ref="ref-1",
        selection_index=1,
        detail=JobDetail(
            source_name="test",
            source_job_id="job-1",
            title="RAG Engineer",
            company_name="Acme",
            description="PRIVATE JD: Build production RAG systems.",
            captured_at=captured_at,
            provenance=Provenance(
                source_name="test",
                source_job_id="job-1",
                captured_at=captured_at,
                operation="detail",
                adapter_version="test-v1",
            ),
        ),
    )
    matches = SQLiteResumeJobMatchStore(resume_path)
    stored_match = matches.save(
        user_id="u1",
        resume_version_id=version.id,
        job_posting_id=job.posting.id,
        jd_snapshot_id=job.snapshot.id,
        matcher_version="test-v1",
        evidence_fingerprint="none",
        result=VALID_MATCH,
    )
    history = CareerHistoryStore(resume_path)
    drafts = SQLiteResumeTailoringDraftStore(resume_path)
    tailoring_worker = RecordingTailoringWorker()
    finalization_worker = RecordingFinalizationWorker()
    service = ResumeTailoringService(
        resumes,
        jobs,
        history,
        matches,
        drafts,
        tailoring_worker,
        finalization_worker,
        reviewer=reviewer,
    )
    return service, tailoring_worker, finalization_worker, stored_match


def _scanned_pdf() -> bytes:
    """A structurally valid PDF whose only page draws graphics, never text."""

    return _single_page_pdf(None)


def _single_page_pdf(text: str | None) -> bytes:
    objects = [
        b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj",
        b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj",
        b"3 0 obj<</Type/Page/Parent 2 0 R/Resources<</Font<</F1 4 0 R>>>>"
        b"/MediaBox[0 0 612 792]/Contents 5 0 R>>endobj",
        b"4 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj",
    ]
    stream = (
        b"0 0 0 RG 10 10 m 20 20 l S"
        if text is None
        else b"BT /F1 12 Tf 72 720 Td (" + text.encode("ascii") + b") Tj ET"
    )
    objects.append(
        b"5 0 obj<</Length "
        + str(len(stream)).encode("ascii")
        + b">>stream\n"
        + stream
        + b"\nendstream endobj"
    )
    out = b"%PDF-1.4\n"
    offsets = []
    for obj in objects:
        offsets.append(len(out))
        out += obj + b"\n"
    xref_at = len(out)
    size = str(len(objects) + 1).encode("ascii")
    out += b"xref\n0 " + size + b"\n0000000000 65535 f \n"
    for offset in offsets:
        out += ("%010d 00000 n \n" % offset).encode("ascii")
    out += (
        b"trailer<</Size " + size + b"/Root 1 0 R>>\nstartxref\n"
        + str(xref_at).encode("ascii")
        + b"\n%%EOF\n"
    )
    return out


def test_pdf_grounding_blocks_a_quote_absent_from_extracted_text() -> None:
    invalid = ResumeTailoringResult.model_validate(VALID_DRAFT).model_dump(mode="json")
    invalid["changes"][0]["support_evidence"][0]["source_quote"] = "Led a team of 50"

    issues = ResumeTailoringReviewGraph._validate_grounding(
        document=StoredResumeDocument(
            resume_version_id="v1",
            document_format="pdf",
            raw_bytes=_single_page_pdf("Built RAG systems and Python"),
        ),
        match_result=VALID_MATCH,
        confirmed_facts=(),
        draft=ResumeTailoringResult.model_validate(invalid),
    )

    assert [issue.severity for issue in issues] == ["blocking"]
    assert issues[0].category == "unsupported_fact"
    assert issues[0].source_quote == "Led a team of 50"


def test_scanned_pdf_warns_instead_of_skipping_grounding() -> None:
    invalid = ResumeTailoringResult.model_validate(VALID_DRAFT).model_dump(mode="json")
    invalid["changes"][0]["support_evidence"][0]["source_quote"] = "Led a team of 50"

    issues = ResumeTailoringReviewGraph._validate_grounding(
        document=StoredResumeDocument(
            resume_version_id="v2",
            document_format="pdf",
            raw_bytes=_scanned_pdf(),
        ),
        match_result=VALID_MATCH,
        confirmed_facts=(),
        draft=ResumeTailoringResult.model_validate(invalid),
    )

    # The page is readable but carries no text, so every quote outside the
    # confirmed set is flagged without blocking: unverifiable is not disproven.
    assert issues
    assert {issue.severity for issue in issues} == {"warning"}
    assert {issue.category for issue in issues} == {"unsupported_fact"}
    assert "Led a team of 50" in {issue.source_quote for issue in issues}


def test_unparseable_pdf_blocks_instead_of_warning() -> None:
    issues = ResumeTailoringReviewGraph._validate_grounding(
        document=StoredResumeDocument(
            resume_version_id="v4",
            document_format="pdf",
            raw_bytes=b"%PDF-1.7 truncated garbage",
        ),
        match_result=VALID_MATCH,
        confirmed_facts=(),
        draft=ResumeTailoringResult.model_validate(VALID_DRAFT),
    )

    # A corrupt upload proves nothing, so it must not buy the warning-only path.
    assert issues
    assert {issue.severity for issue in issues} == {"blocking"}


def test_undecodable_text_resume_still_blocks_grounding() -> None:
    issues = ResumeTailoringReviewGraph._validate_grounding(
        document=StoredResumeDocument(
            resume_version_id="v3",
            document_format="text",
            raw_bytes=b"\xff\xfe not utf-8",
        ),
        match_result=VALID_MATCH,
        confirmed_facts=(),
        draft=ResumeTailoringResult.model_validate(VALID_DRAFT),
    )

    assert issues
    assert {issue.severity for issue in issues} == {"blocking"}


def test_review_graph_revises_grounding_failure_then_persists_passed_trace(tmp_path) -> None:
    valid = ResumeTailoringResult.model_validate(VALID_DRAFT)
    invalid_data = valid.model_dump(mode="json")
    invalid_data["changes"][0]["support_evidence"][0]["source_quote"] = "Invented fact"

    class SequenceWorker:
        def __init__(self) -> None:
            self.outputs = [ResumeTailoringResult.model_validate(invalid_data), valid]
            self.calls = []

        def tailor(self, **kwargs):
            self.calls.append(kwargs)
            return self.outputs.pop(0)

    worker = SequenceWorker()
    reviewer = RecordingReviewer()
    outcome = ResumeTailoringReviewGraph(worker, reviewer).run(
        document=StoredResumeDocument(
            resume_version_id="v1",
            document_format="text",
            raw_bytes=b"Built RAG systems and Python",
        ),
        jd_text="Build production RAG systems",
        match_result=VALID_MATCH,
        user_feedback="Keep every change concise.",
    )

    assert outcome.trace.status == "passed"
    assert outcome.trace.stop_reason == "passed"
    assert len(outcome.trace.attempts) == 2
    assert outcome.trace.attempts[0].result.verdict == "revise"
    assert worker.calls[1]["review_feedback"]
    assert worker.calls[0]["user_feedback"] == "Keep every change concise."
    assert worker.calls[1]["user_feedback"] == "Keep every change concise."
    assert len(reviewer.draft_calls) == 1


def test_review_graph_stops_when_writer_returns_unchanged_draft(tmp_path) -> None:
    draft = ResumeTailoringResult.model_validate(VALID_DRAFT)

    class UnchangedWorker:
        def tailor(self, **kwargs):
            return draft

    class ReviseReviewer(RecordingReviewer):
        def review_draft(self, **kwargs) -> ResumeReviewResult:
            self.draft_calls.append(kwargs)
            return ResumeReviewResult(
                verdict="revise",
                summary="Clarify one sentence.",
                issues=(
                    ResumeReviewIssue(
                        category="unclear_expression",
                        severity="blocking",
                        change_index=1,
                        explanation="The sentence is ambiguous.",
                        revision_instruction="Clarify the sentence.",
                    ),
                ),
            )

    reviewer = ReviseReviewer()
    outcome = ResumeTailoringReviewGraph(UnchangedWorker(), reviewer).run(
        document=StoredResumeDocument(
            resume_version_id="v1",
            document_format="text",
            raw_bytes=b"Built RAG systems and Python",
        ),
        jd_text="Build production RAG systems",
        match_result=VALID_MATCH,
    )

    assert outcome.trace.status == "blocked"
    assert outcome.trace.stop_reason == "no_progress"
    assert len(outcome.trace.attempts) == 2
    assert len(reviewer.draft_calls) == 1


def test_tailoring_service_reads_private_inputs_and_persists_reviewable_draft(tmp_path) -> None:
    service, worker, finalization_worker, stored_match = seed_service(tmp_path)

    draft = service.create_draft(
        user_id="u1",
        match_id=stored_match.id,
        tailoring_goal="Keep it concise",
    )

    assert draft.id.startswith("resume_tailoring_")
    assert draft.status == "pending"
    assert worker.calls[0]["document"].raw_bytes.startswith(b"PRIVATE RESUME")
    assert worker.calls[0]["jd_text"].startswith("PRIVATE JD")
    assert worker.calls[0]["match_result"] == VALID_MATCH
    assert service.get_draft(user_id="u1", draft_id=draft.id) == draft

    partial = service.review_draft(
        user_id="u1",
        draft_id=draft.id,
        accepted_change_indices=(1,),
    )
    assert partial.status == "in_review"
    assert partial.pending_change_indices == (2,)
    assert partial.change_reviews[0].decision == "accepted"

    rebuilt = SQLiteResumeTailoringDraftStore(tmp_path / "resumes.sqlite3")
    assert rebuilt.get(user_id="u1", draft_id=draft.id) == partial

    reviewed = service.review_draft(
        user_id="u1",
        draft_id=draft.id,
        rejected_change_indices=(2,),
        feedback="Keep the skills section unchanged.",
    )
    assert reviewed.status == "reviewed"
    assert reviewed.pending_change_indices == ()
    assert [item.decision for item in reviewed.change_reviews] == [
        "accepted",
        "rejected",
    ]
    assert reviewed.change_reviews[1].feedback == "Keep the skills section unchanged."

    finalized = service.finalize_draft(user_id="u1", draft_id=draft.id)
    assert finalized.created is True
    assert finalized.applied_change_indices == (1,)
    assert finalized.resume_version.version_number == 2
    assert finalized.resume_version.source_type == "agent_tailoring"
    assert finalized.resume_version.document_format == "markdown"
    stored_document = service._resume_store.read_version_document(
        user_id="u1",
        resume_version_id=finalized.resume_version.id,
    )
    assert stored_document is not None
    assert stored_document.raw_bytes.startswith(b"# Candidate")
    assert service.get_draft(user_id="u1", draft_id=draft.id).status == "finalized"
    repeated = service.finalize_draft(user_id="u1", draft_id=draft.id)
    assert repeated.created is False
    assert repeated.resume_version.id == finalized.resume_version.id
    assert len(finalization_worker.calls) == 1

    with pytest.raises(ResumeTailoringAlreadyFinalizedError):
        service.review_draft(
            user_id="u1",
            draft_id=draft.id,
            accepted_change_indices=(3,),
        )
    with pytest.raises(ResumeTailoringDraftNotFoundError):
        service.review_draft(
            user_id="other",
            draft_id=draft.id,
            accepted_change_indices=(1,),
        )


def test_tailoring_finalization_requires_complete_review_and_an_accepted_change(
    tmp_path,
) -> None:
    service, _, finalization_worker, stored_match = seed_service(tmp_path)
    draft = service.create_draft(user_id="u1", match_id=stored_match.id)

    with pytest.raises(ResumeTailoringNotReadyError, match="Every tailoring change"):
        service.finalize_draft(user_id="u1", draft_id=draft.id)

    service.review_draft(
        user_id="u1",
        draft_id=draft.id,
        rejected_change_indices=(1, 2),
    )
    with pytest.raises(ResumeTailoringNotReadyError, match="At least one"):
        service.finalize_draft(user_id="u1", draft_id=draft.id)

    assert finalization_worker.calls == []


def test_tailoring_finalization_rejects_worker_change_index_mismatch(tmp_path) -> None:
    service, _, finalization_worker, stored_match = seed_service(tmp_path)
    draft = service.create_draft(user_id="u1", match_id=stored_match.id)
    service.review_draft(
        user_id="u1",
        draft_id=draft.id,
        accepted_change_indices=(1,),
        rejected_change_indices=(2,),
    )

    def mismatched_finalize(**kwargs) -> FinalizedResumeDocument:
        finalization_worker.calls.append(kwargs)
        return FinalizedResumeDocument(
            markdown="# Candidate\n\nChanged resume",
            applied_change_indices=(2,),
        )

    finalization_worker.finalize = mismatched_finalize
    with pytest.raises(ValueError, match="exactly the accepted changes"):
        service.finalize_draft(user_id="u1", draft_id=draft.id)

    assert service.get_draft(user_id="u1", draft_id=draft.id).status == "reviewed"
    assert service._resume_store.get_tailored_version(
        user_id="u1",
        tailoring_draft_id=draft.id,
    ) is None


def test_service_persists_automated_review_and_runs_final_qa(tmp_path) -> None:
    reviewer = RecordingReviewer()
    service, _, _, stored_match = seed_service(tmp_path, reviewer=reviewer)

    draft = service.create_draft(user_id="u1", match_id=stored_match.id)

    assert draft.automated_review is not None
    assert draft.automated_review.status == "passed"
    rebuilt = SQLiteResumeTailoringDraftStore(tmp_path / "resumes.sqlite3").get(
        user_id="u1", draft_id=draft.id
    )
    assert rebuilt is not None
    assert rebuilt.automated_review == draft.automated_review

    service.review_draft(
        user_id="u1",
        draft_id=draft.id,
        accepted_change_indices=(1,),
        rejected_change_indices=(2,),
    )
    finalized = service.finalize_draft(user_id="u1", draft_id=draft.id)

    assert finalized.created is True
    assert len(reviewer.draft_calls) == 1
    assert len(reviewer.final_calls) == 1


def test_user_feedback_creates_reviewed_child_draft_without_old_decisions(tmp_path) -> None:
    reviewer = RecordingReviewer()
    service, worker, _, stored_match = seed_service(tmp_path, reviewer=reviewer)
    original = service.create_draft(user_id="u1", match_id=stored_match.id)
    service.review_draft(
        user_id="u1",
        draft_id=original.id,
        accepted_change_indices=(1,),
    )

    revised = service.revise_draft(
        user_id="u1",
        draft_id=original.id,
        feedback="第一条更简洁一些，但不要增加数字。",
    )

    assert revised.id != original.id
    assert revised.parent_draft_id == original.id
    assert revised.revision_number == 2
    assert revised.revision_feedback == "第一条更简洁一些，但不要增加数字。"
    assert revised.status == "pending"
    assert revised.change_reviews == ()
    assert revised.pending_change_indices == (1, 2)
    assert revised.automated_review is not None
    assert revised.automated_review.status == "passed"
    assert worker.calls[1]["previous_draft"] == original.result
    assert worker.calls[1]["user_feedback"] == "第一条更简洁一些，但不要增加数字。"
    assert worker.calls[1]["review_feedback"] == ()
    stored_original = service.get_draft(user_id="u1", draft_id=original.id)
    assert stored_original.status == "superseded"
    assert stored_original.change_reviews[0].decision == "accepted"

    with pytest.raises(ResumeTailoringSupersededError):
        service.review_draft(
            user_id="u1",
            draft_id=original.id,
            rejected_change_indices=(2,),
        )
    with pytest.raises(ResumeTailoringNotReadyError, match="newer"):
        service.finalize_draft(user_id="u1", draft_id=original.id)

    rebuilt = SQLiteResumeTailoringDraftStore(tmp_path / "resumes.sqlite3").get(
        user_id="u1", draft_id=revised.id
    )
    assert rebuilt == revised


def test_final_reviewer_blocks_version_creation_after_user_approval(tmp_path) -> None:
    reviewer = RecordingReviewer(final_verdict="block")
    service, _, _, stored_match = seed_service(tmp_path, reviewer=reviewer)
    draft = service.create_draft(user_id="u1", match_id=stored_match.id)
    service.review_draft(
        user_id="u1",
        draft_id=draft.id,
        accepted_change_indices=(1,),
        rejected_change_indices=(2,),
    )

    with pytest.raises(ResumeFinalReviewBlockedError, match="new user review"):
        service.finalize_draft(user_id="u1", draft_id=draft.id)

    assert service.get_draft(user_id="u1", draft_id=draft.id).status == "reviewed"
    assert service._resume_store.get_tailored_version(
        user_id="u1", tailoring_draft_id=draft.id
    ) is None


class SequenceDecisionMaker:
    def __init__(self, *decisions: AgentDecision) -> None:
        self.decisions = list(decisions)
        self.contexts = []

    def decide(self, context, tool_specs):
        self.contexts.append(context)
        return self.decisions.pop(0)


def test_main_agent_regenerates_active_draft_from_user_feedback(tmp_path) -> None:
    reviewer = RecordingReviewer()
    service, _, _, stored_match = seed_service(tmp_path, reviewer=reviewer)
    manager = ContextManager(CareerContextStore(tmp_path / "context.sqlite3"))
    manager.upsert_profile(CareerProfileContext(user_id="u1"))
    seeded = manager.load_for_turn(
        user_id="u1", conversation_id="c1", user_message="seed active match"
    )
    manager.commit_turn(
        context=seeded,
        task=seeded.task.model_copy(
            update={"active_resume_job_match_id": stored_match.id}
        ),
        assistant_message="seeded",
    )
    tools = MainAgentToolRegistry(
        resume_tailoring_service=service,
    )
    revise_schema = next(
        spec
        for spec in tools.schemas()
        if spec["function"]["name"] == "revise_resume_tailoring"
    )
    assert "user_id" not in revise_schema["function"]["parameters"].get(
        "properties", {}
    )

    create_decisions = SequenceDecisionMaker(
        AgentDecision(
            action="tool_call",
            tool_call=ToolCall(
                name="draft_resume_tailoring",
                arguments={},
            ),
        ),
        AgentDecision(action="final", message="请先看这份修改建议。"),
    )
    created = MainAgentRuntime(
        context_manager=manager,
        decision_maker=create_decisions,
        tools=tools,
    ).run_turn(
        user_id="u1",
        conversation_id="c1",
        user_message="先给我一版定制建议",
    )
    parent_id = created.context.task.active_resume_tailoring_draft_id
    assert parent_id is not None

    revise_decisions = SequenceDecisionMaker(
        AgentDecision(
            action="tool_call",
            tool_call=ToolCall(
                name="revise_resume_tailoring",
                arguments={"feedback": "第一条再简洁一点，不要增加数字。"},
            ),
        ),
        AgentDecision(action="final", message="已按意见生成新草稿，请重新审阅。"),
    )
    revised = MainAgentRuntime(
        context_manager=manager,
        decision_maker=revise_decisions,
        tools=tools,
    ).run_turn(
        user_id="u1",
        conversation_id="c1",
        user_message="第一条再简洁一点，不要增加数字",
    )

    observation = revised.tool_result
    assert observation.tool_name == "revise_resume_tailoring"
    assert observation.state == "resume_tailoring_draft_ready"
    assert observation.payload["parent_draft_id"] == parent_id
    assert observation.payload["revision_number"] == 2
    assert observation.payload["change_reviews"] == []
    assert revised.context.task.active_resume_tailoring_draft_id == observation.payload[
        "draft_id"
    ]
    assert revised.context.task.resume_tailoring_status == "pending"


def test_main_agent_creates_and_recalls_active_tailoring_draft(tmp_path) -> None:
    service, _, _, stored_match = seed_service(tmp_path)
    manager = ContextManager(CareerContextStore(tmp_path / "context.sqlite3"))
    manager.upsert_profile(CareerProfileContext(user_id="u1"))
    seeded = manager.load_for_turn(
        user_id="u1", conversation_id="c1", user_message="seed active match"
    )
    manager.commit_turn(
        context=seeded,
        task=seeded.task.model_copy(
            update={"active_resume_job_match_id": stored_match.id}
        ),
        assistant_message="seeded",
    )
    decisions = SequenceDecisionMaker(
        AgentDecision(
            action="tool_call",
            tool_call=ToolCall(
                name="draft_resume_tailoring",
                arguments={
                    "tailoring_goal": "Keep it concise",
                },
            ),
        ),
        AgentDecision(action="final", message="我生成了一份待审阅的修改草稿。"),
    )
    tools = MainAgentToolRegistry(
        resume_tailoring_service=service,
        resume_export_service=ResumeExportService(
            service._resume_store,
            SQLiteResumeArtifactStore(tmp_path / "resumes.sqlite3"),
        ),
    )
    review_schema = next(
        spec
        for spec in tools.schemas()
        if spec["function"]["name"] == "review_resume_tailoring"
    )
    assert "user_id" not in review_schema["function"]["parameters"].get(
        "properties", {}
    )
    runtime = MainAgentRuntime(
        context_manager=manager,
        decision_maker=decisions,
        tools=tools,
    )

    result = runtime.run_turn(
        user_id="u1",
        conversation_id="c1",
        user_message="按这个岗位优化一下简历",
    )

    observation = result.tool_result
    assert observation.state == "resume_tailoring_draft_ready"
    assert [change["change_index"] for change in observation.payload["changes"]] == [1, 2]
    assert result.context.task.active_resume_tailoring_draft_id == observation.payload["draft_id"]
    assert result.context.task.resume_tailoring_status == "pending"
    serialized = observation.model_dump_json()
    assert "PRIVATE RESUME" not in serialized
    assert "PRIVATE JD" not in serialized
    rendered = MainAgentRuntime._assistant_message(result.tool_result)
    assert rendered.startswith("# 简历定制草稿 · 修订 1")
    assert "Built production RAG systems for knowledge retrieval." in rendered
    assert observation.resource_ref is not None
    assert observation.resource_ref.kind == "resume_tailoring_draft"
    assert observation.resource_ref.title == "简历改写稿 v1"
    assert observation.resource_ref.description == (
        "Lead with directly supported production RAG experience."
    )

    stored = service._draft_store.get_for_display(
        user_id="u1", draft_id=observation.payload["draft_id"]
    )
    assert stored is not None
    assert service._draft_store.get(
        user_id="u1",
        draft_id=stored.id,
        now=stored.expires_at,
    ) is None
    assert service._draft_store.get_for_display(
        user_id="u1", draft_id=stored.id
    ) is not None

    review_decisions = SequenceDecisionMaker(
        AgentDecision(
            action="tool_call",
            tool_call=ToolCall(name="get_resume_tailoring_draft", arguments={}),
        ),
        AgentDecision(action="final", message="这是刚才的定制草稿。"),
    )
    review_runtime = MainAgentRuntime(
        context_manager=manager,
        decision_maker=review_decisions,
        tools=tools,
    )
    recalled_result = review_runtime.run_turn(
        user_id="u1",
        conversation_id="c1",
        user_message="再看一下草稿",
    )
    reviewed = recalled_result.tool_result
    assert reviewed.payload["draft_id"] == observation.payload["draft_id"]

    decision_maker = SequenceDecisionMaker(
        AgentDecision(
            action="tool_call",
            tool_call=ToolCall(
                name="review_resume_tailoring",
                arguments={
                    "accepted_change_indices": [1],
                    "rejected_change_indices": [2],
                    "feedback": "Keep the skills section unchanged.",
                },
            ),
        ),
        AgentDecision(action="final", message="已记录你的逐条决定。"),
    )
    review_action_runtime = MainAgentRuntime(
        context_manager=manager,
        decision_maker=decision_maker,
        tools=tools,
    )
    reviewed_result = review_action_runtime.run_turn(
        user_id="u1",
        conversation_id="c1",
        user_message="接受第一条，拒绝第二条，技能区保持原样",
    )
    review_observation = reviewed_result.tool_result
    assert review_observation.tool_name == "review_resume_tailoring"
    assert review_observation.payload["status"] == "reviewed"
    assert review_observation.payload["pending_change_indices"] == ()
    assert reviewed_result.context.task.resume_tailoring_status == "reviewed"

    finalize_decisions = SequenceDecisionMaker(
        AgentDecision(
            action="tool_call",
            tool_call=ToolCall(name="finalize_resume_tailoring", arguments={}),
        ),
        AgentDecision(action="final", message="已生成新的 Markdown 简历版本。"),
    )
    finalize_runtime = MainAgentRuntime(
        context_manager=manager,
        decision_maker=finalize_decisions,
        tools=tools,
    )
    finalized_result = finalize_runtime.run_turn(
        user_id="u1",
        conversation_id="c1",
        user_message="确认，生成新的简历版本",
    )
    finalization_observation = finalized_result.tool_result
    assert finalization_observation.state == "resume_tailoring_finalized"
    assert finalization_observation.payload["source_type"] == "agent_tailoring"
    assert "markdown" not in finalization_observation.payload
    assert finalized_result.context.task.resume_tailoring_status == "finalized"
    assert finalized_result.context.task.active_resume_version_id == (
        finalization_observation.payload["resume_version_id"]
    )

    export_decisions = SequenceDecisionMaker(
        AgentDecision(
            action="tool_call",
            tool_call=ToolCall(name="export_resume_artifact", arguments={}),
        ),
        AgentDecision(action="final", message="简历文件已经准备好。"),
    )
    export_runtime = MainAgentRuntime(
        context_manager=manager,
        decision_maker=export_decisions,
        tools=tools,
    )
    exported_result = export_runtime.run_turn(
        user_id="u1",
        conversation_id="c1",
        user_message="把刚生成的简历给我下载",
    )
    export_observation = exported_result.tool_result
    assert export_observation.state == "resume_artifact_ready"
    assert export_observation.payload["resume_version_id"] == (
        finalization_observation.payload["resume_version_id"]
    )
    assert export_observation.payload["filename"] == "AI Resume-v2.md"
    assert "PRIVATE RESUME" not in export_observation.model_dump_json()
    assert "content" not in export_observation.payload
    assert "path" not in export_observation.payload
    assert exported_result.context.task.active_resume_artifact_id == (
        export_observation.payload["artifact_id"]
    )
    assert len(exported_result.artifacts) == 1
    assert exported_result.artifacts[0].reference.id == (
        export_observation.payload["artifact_id"]
    )
    assert exported_result.artifacts[0].content.startswith(b"# Candidate")
