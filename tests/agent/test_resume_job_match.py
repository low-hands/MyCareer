from __future__ import annotations

import base64
from datetime import datetime, timedelta, timezone
import json
import sqlite3

import pytest

from career_agent.agent.context_manager import ContextManager
from career_agent.agent.main_agent_contracts import (
    AgentDecision,
    CareerProfileContext,
    ConversationTaskState,
    ResumeVersionCandidateContextItem,
    SavedJobCandidateContextItem,
    ToolCall,
)
from career_agent.agent.main_agent_runtime import MainAgentRuntime
from career_agent.agent.main_agent_tools import MainAgentToolRegistry
from career_agent.agent.openai_compatible_client import (
    AgentWorkerError,
    OpenAICompatibleAgentConfig,
)
from career_agent.agent.openai_resume_job_match_worker import (
    OpenAIResumeJobMatchWorker,
)
from career_agent.agent.resume_job_match_contracts import ResumeJobMatchResult
from career_agent.agent.resume_job_match_contracts import (
    ResumeJobMatchAuditProposal,
    ResumeJobMatchStateFinding,
)
from career_agent.domain.job_discovery import JobDetail, Provenance
from career_agent.services.resume_job_match import (
    ResumeJobMatchInputNotFoundError,
    ResumeJobMatchService,
)
from career_agent.storage.career_history import CareerHistoryStore
from career_agent.storage.context import CareerContextStore
from career_agent.storage.jobs import SQLiteJobPostingRepository
from career_agent.storage.resumes import ResumeStore, StoredResumeDocument
from career_agent.storage.resume_job_matches import SQLiteResumeJobMatchStore
from conftest import enter_tool_profile


VALID_MATCH = {
    "overall_fit": "moderate",
    "summary": "The resume demonstrates relevant RAG experience.",
    "requirements": [
        {
            "requirement": "Production RAG experience",
            "jd_quote": "Build production RAG systems",
            "status": "matched",
            "rationale": "The resume explicitly describes production RAG work.",
            "resume_evidence": [
                {
                    "source_locator": "Experience, bullet 1",
                    "source_quote": "Built production RAG systems",
                }
            ],
        },
        {
            "requirement": "Go experience",
            "jd_quote": "Strong Go experience",
            "status": "missing",
            "rationale": "Go is not stated in the resume.",
            "resume_evidence": [],
        },
    ],
    "recommendations": ["Make reliability outcomes more explicit."],
    "clarification_questions": [],
    "limitations": [],
}


class FakeResponses:
    def __init__(self, output: dict[str, object] | str = VALID_MATCH) -> None:
        self.output = output
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        output_text = self.output if isinstance(self.output, str) else json.dumps(self.output)
        return type("Response", (), {"output_text": output_text})()


class FakeClient:
    def __init__(self, output: dict[str, object] | str = VALID_MATCH) -> None:
        self.responses = FakeResponses(output)


def worker(client: FakeClient) -> OpenAIResumeJobMatchWorker:
    return OpenAIResumeJobMatchWorker(
        OpenAICompatibleAgentConfig(
            endpoint="https://example.test/v1/chat/completions",
            api_key="secret",
            model="multimodal-model",
        ),
        client=client,
    )


def test_text_match_sends_complete_resume_and_jd_as_untrusted_data() -> None:
    client = FakeClient()

    result = worker(client).match(
        document=StoredResumeDocument(
            resume_version_id="resume_version_1",
            document_format="markdown",
            raw_bytes=b"Built production RAG systems",
        ),
        jd_text="Build production RAG systems. Strong Go experience.",
    )

    assert result.overall_fit == "moderate"
    kwargs = client.responses.kwargs
    assert kwargs is not None
    text = kwargs["input"][0]["content"][0]["text"]
    assert "<resume_document>" in text
    assert "Built production RAG systems" in text
    assert "<job_description>" in text
    assert "Strong Go experience" in text
    assert "numeric fit scores" in kwargs["instructions"]


def test_pdf_match_sends_original_pdf_as_input_file_alongside_jd() -> None:
    raw_pdf = b"%PDF-1.7\x00\xffbinary"
    client = FakeClient()

    worker(client).match(
        document=StoredResumeDocument(
            resume_version_id="resume_version_pdf",
            document_format="pdf",
            raw_bytes=raw_pdf,
        ),
        jd_text="Build production RAG systems.",
    )

    content = client.responses.kwargs["input"][0]["content"]
    assert content[0] == {
        "type": "input_file",
        "filename": "resume_version_pdf.pdf",
        "file_data": "data:application/pdf;base64," + base64.b64encode(raw_pdf).decode("ascii"),
    }
    assert "Build production RAG systems" in content[1]["text"]
    assert "detail" not in content[0]


def test_match_contract_rejects_positive_assessment_without_resume_quote() -> None:
    invalid = dict(VALID_MATCH)
    invalid["requirements"] = [
        {
            "requirement": "RAG",
            "jd_quote": "RAG",
            "status": "matched",
            "rationale": "Relevant",
            "resume_evidence": [],
        }
    ]

    with pytest.raises(AgentWorkerError) as error:
        worker(FakeClient(invalid)).match(
            document=StoredResumeDocument(
                resume_version_id="v1",
                document_format="text",
                raw_bytes=b"resume",
            ),
            jd_text="RAG",
        )

    assert error.value.code == "RESUME_JOB_MATCH_INVALID_RESPONSE"


class RecordingMatchWorker:
    def __init__(self) -> None:
        self.calls = []

    def match(self, **kwargs) -> ResumeJobMatchResult:
        self.calls.append(kwargs)
        return ResumeJobMatchResult.model_validate(VALID_MATCH)


class PreferenceSensitiveMatchWorker(RecordingMatchWorker):
    def match(self, **kwargs) -> ResumeJobMatchResult:
        self.calls.append(kwargs)
        free_text = tuple(
            item
            for item in kwargs["intent_states"]
            if item.pref_scope.startswith("freeform")
        )
        return ResumeJobMatchResult.model_validate(
            {
                **VALID_MATCH,
                "overall_fit": "weak" if free_text else "moderate",
                "summary": (
                    "The confirmed employer preference downranks this role."
                    if free_text
                    else "The quarantined employer preference has no authority."
                ),
            }
        )


class CascadingPreferenceMatchWorker(RecordingMatchWorker):
    def match(self, **kwargs) -> ResumeJobMatchResult:
        self.calls.append(kwargs)
        values = tuple(
            item.value
            for item in kwargs["intent_states"]
            if item.scope_key == "person_intent/self/company_scale"
        )
        exception_applies = any("例外" in value for value in values)
        default_applies = any("不去大厂" in value for value in values)
        return ResumeJobMatchResult.model_validate(
            {
                **VALID_MATCH,
                "overall_fit": (
                    "moderate"
                    if exception_applies
                    else "weak"
                    if default_applies
                    else "moderate"
                ),
                "summary": "Resolved from the effective preference view.",
            }
        )


class StateAuditingMatchWorker(RecordingMatchWorker):
    def __init__(self) -> None:
        super().__init__()
        self.audits = []

    def match(self, **kwargs) -> ResumeJobMatchResult:
        self.calls.append(kwargs)
        stale = {
            **VALID_MATCH,
            "summary": "The Hangzhou preference makes this role a weak location fit.",
        }
        return ResumeJobMatchResult.model_validate(stale)

    def audit_state(self, **kwargs) -> ResumeJobMatchAuditProposal:
        self.audits.append(kwargs)
        transition = kwargs["transitions"][0]
        repaired = {
            **VALID_MATCH,
            "summary": (
                f"The current {transition.new_value} preference makes "
                "the location acceptable."
            ),
        }
        return ResumeJobMatchAuditProposal(
            findings=(
                ResumeJobMatchStateFinding(
                    scope_key=transition.scope_key,
                    pref_scope=transition.pref_scope,
                    old_value=transition.old_value,
                    new_value=transition.new_value,
                    status="stale",
                    material=True,
                    rationale="The draft planned around the superseded city.",
                ),
            ),
            repaired_result=ResumeJobMatchResult.model_validate(repaired),
        )


class ConflictFreeAuditingWorker(RecordingMatchWorker):
    def audit_state(self, **kwargs) -> ResumeJobMatchAuditProposal:
        transition = kwargs["transitions"][0]
        changed = {**VALID_MATCH, "summary": "An unnecessary rewrite."}
        return ResumeJobMatchAuditProposal(
            findings=(
                ResumeJobMatchStateFinding(
                    scope_key=transition.scope_key,
                    pref_scope=transition.pref_scope,
                    old_value=transition.old_value,
                    new_value=transition.new_value,
                    status="current",
                    material=False,
                    rationale="The draft does not rely on the old state.",
                ),
            ),
            repaired_result=ResumeJobMatchResult.model_validate(changed),
        )


def seed_inputs(tmp_path):
    resume_path = tmp_path / "resumes.sqlite3"
    resume_store = ResumeStore(resume_path)
    role = resume_store.create_target_role(user_id="u1", title="AI Engineer", priority=1)
    _, version = resume_store.import_document(
        user_id="u1",
        target_role_id=role.id,
        name="AI Resume",
        content=b"PRIVATE RESUME: Built production RAG systems",
        document_format="text",
    )
    jobs = SQLiteJobPostingRepository(tmp_path / "jobs.sqlite3")
    captured_at = datetime(2026, 8, 25, tzinfo=timezone.utc)
    saved = jobs.save_detail(
        user_id="u1",
        run_id="run-1",
        result_ref="ref-1",
        selection_index=1,
        detail=JobDetail(
            source_name="test",
            source_job_id="job-1",
            title="RAG Engineer",
            company_name="Acme",
            description="PRIVATE JD: Build production RAG systems. Strong Go experience.",
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
    history = CareerHistoryStore(resume_path)
    record = history.create_record(user_id="u1", record_type="work", title="Engineer")
    evidence = history.create_evidence(
        user_id="u1",
        career_record_id=record.id,
        claim="Built production RAG systems",
        origin="resume_extraction",
        source_resume_version_id=version.id,
        source_locator="Experience, bullet 1",
        source_quote="Built production RAG systems",
    )
    history.confirm_evidence(user_id="u1", career_evidence_id=evidence.id)
    return resume_store, jobs, history, version, saved


def test_service_loads_owned_complete_inputs_and_only_exact_version_facts(tmp_path) -> None:
    resumes, jobs, history, version, saved = seed_inputs(tmp_path)
    worker_stub = RecordingMatchWorker()
    match_store = SQLiteResumeJobMatchStore(tmp_path / "resumes.sqlite3")
    service = ResumeJobMatchService(resumes, jobs, history, worker_stub, match_store)

    result = service.match(
        user_id="u1",
        resume_version_id=version.id,
        job_posting_id=saved.posting.id,
    )

    call = worker_stub.calls[0]
    assert call["document"].raw_bytes.startswith(b"PRIVATE RESUME")
    assert call["jd_text"].startswith("PRIVATE JD")
    assert [fact.claim for fact in call["confirmed_facts"]] == ["Built production RAG systems"]
    assert result.id.startswith("resume_job_match_")
    assert result.result.overall_fit == "moderate"

    cached = service.match(
        user_id="u1",
        resume_version_id=version.id,
        job_posting_id=saved.posting.id,
    )
    assert cached.id == result.id
    assert len(worker_stub.calls) == 1
    rebuilt = SQLiteResumeJobMatchStore(tmp_path / "resumes.sqlite3")
    assert rebuilt.get(user_id="u1", match_id=result.id) == result

    record = history.list_records(user_id="u1")[0]
    added = history.create_evidence(
        user_id="u1",
        career_record_id=record.id,
        claim="Improved retrieval reliability",
        origin="resume_extraction",
        source_resume_version_id=version.id,
        source_locator="Experience, bullet 2",
        source_quote="Improved retrieval reliability",
    )
    history.confirm_evidence(user_id="u1", career_evidence_id=added.id)
    refreshed = service.match(
        user_id="u1",
        resume_version_id=version.id,
        job_posting_id=saved.posting.id,
    )
    assert refreshed.id != result.id
    assert len(worker_stub.calls) == 2


def test_only_active_free_text_preferences_can_change_a_match_result(tmp_path) -> None:
    resumes, jobs, history, version, saved = seed_inputs(tmp_path)
    context = CareerContextStore(tmp_path / "context.sqlite3")
    candidate = context.capture_free_text_preference_from_message(
        user_id="u1",
        conversation_id="c1",
        message="我想清楚了，不去大厂。",
    )
    assert candidate is not None
    worker_stub = PreferenceSensitiveMatchWorker()
    service = ResumeJobMatchService(
        resumes,
        jobs,
        history,
        worker_stub,
        SQLiteResumeJobMatchStore(tmp_path / "resumes.sqlite3"),
        career_profile_store=context,
    )

    before_confirmation = service.match(
        user_id="u1",
        resume_version_id=version.id,
        job_posting_id=saved.posting.id,
    )
    assert before_confirmation.result.overall_fit == "moderate"
    assert worker_stub.calls[0]["intent_states"] == ()

    confirmed = context.confirm_free_text_preference(
        user_id="u1",
        update_id=candidate.update_id,
    )
    assert confirmed is not None
    after_confirmation = service.match(
        user_id="u1",
        resume_version_id=version.id,
        job_posting_id=saved.posting.id,
    )
    assert after_confirmation.result.overall_fit == "weak"
    assert [
        item.value
        for item in worker_stub.calls[1]["intent_states"]
        if item.pref_scope.startswith("freeform")
    ] == ["我想清楚了，不去大厂。"]

    context.capture_free_text_preference_from_message(
        user_id="u1",
        conversation_id="c2",
        message="删除这条大厂偏好",
    )
    after_deletion = service.match(
        user_id="u1",
        resume_version_id=version.id,
        job_posting_id=saved.posting.id,
    )
    assert after_deletion.id == before_confirmation.id
    assert after_deletion.result.overall_fit == "moderate"


def test_situational_exception_overrides_default_for_only_one_job(
    tmp_path,
) -> None:
    resumes, jobs, history, version, saved = seed_inputs(tmp_path)
    context = CareerContextStore(tmp_path / "context.sqlite3")
    default = context.capture_free_text_preference_from_message(
        user_id="u1",
        conversation_id="c1",
        message="我不去大厂。",
    )
    assert default is not None
    assert context.confirm_free_text_preference(
        user_id="u1", update_id=default.update_id
    ) is not None

    context.upsert_task(
        user_id="u1",
        conversation_id="c1",
        task=ConversationTaskState(
            active_job_posting_id=saved.posting.id
        ),
    )
    exception = context.capture_free_text_preference_from_message(
        user_id="u1",
        conversation_id="c1",
        message="这个岗位的话，这家例外。",
    )
    assert exception is not None
    assert context.confirm_free_text_preference(
        user_id="u1", update_id=exception.update_id
    ) is not None
    active_preferences = context.list_free_text_preferences(
        user_id="u1", statuses=("active",)
    )
    situational = next(
        item for item in active_preferences if "例外" in item.value
    )
    assert situational.layer == "transient"
    assert situational.timescale == "situational"
    assert situational.valid_until is not None

    captured_at = datetime(2026, 8, 25, tzinfo=timezone.utc)
    other = jobs.save_detail(
        user_id="u1",
        run_id="run-2",
        result_ref="ref-2",
        selection_index=1,
        detail=JobDetail(
            source_name="test",
            source_job_id="job-2",
            title="Platform Engineer",
            company_name="Other Corp",
            description="Build a large-scale platform.",
            captured_at=captured_at,
            provenance=Provenance(
                source_name="test",
                source_job_id="job-2",
                captured_at=captured_at,
                operation="detail",
                adapter_version="test-v1",
            ),
        ),
    )
    worker_stub = CascadingPreferenceMatchWorker()
    service = ResumeJobMatchService(
        resumes,
        jobs,
        history,
        worker_stub,
        SQLiteResumeJobMatchStore(tmp_path / "resumes.sqlite3"),
        career_profile_store=context,
    )

    matching = service.match(
        user_id="u1",
        resume_version_id=version.id,
        job_posting_id=saved.posting.id,
    )
    unrelated = service.match(
        user_id="u1",
        resume_version_id=version.id,
        job_posting_id=other.posting.id,
    )

    assert matching.result.overall_fit == "moderate"
    assert unrelated.result.overall_fit == "weak"
    assert [
        item.value
        for item in worker_stub.calls[0]["intent_states"]
        if item.scope_key == "person_intent/self/company_scale"
    ] == ["这个岗位的话，这家例外。"]
    assert [
        item.value
        for item in worker_stub.calls[1]["intent_states"]
        if item.scope_key == "person_intent/self/company_scale"
    ] == ["我不去大厂。"]

    with sqlite3.connect(context.path) as connection:
        connection.execute(
            """
            UPDATE career_intent_versions
            SET valid_until = ?
            WHERE update_id = ?
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                situational.update_id,
            ),
        )
    after_expiry = service.match(
        user_id="u1",
        resume_version_id=version.id,
        job_posting_id=saved.posting.id,
    )
    assert after_expiry.result.overall_fit == "weak"
    assert [
        item.value
        for item in worker_stub.calls[2]["intent_states"]
        if item.scope_key == "person_intent/self/company_scale"
    ] == ["我不去大厂。"]


def test_sr_pr_and_ipa_path_repairs_a_visible_revised_state(tmp_path) -> None:
    resumes, jobs, history, version, saved = seed_inputs(tmp_path)
    context = CareerContextStore(tmp_path / "context.sqlite3")
    context.upsert_profile(
        CareerProfileContext(user_id="u1", default_city="Shanghai"),
        source="test",
    )
    context.upsert_profile(
        CareerProfileContext(user_id="u1", default_city="Hangzhou"),
        source="test",
    )
    context.upsert_profile(
        CareerProfileContext(user_id="u1", default_city="Shanghai"),
        source="test",
    )
    worker_stub = StateAuditingMatchWorker()
    service = ResumeJobMatchService(
        resumes,
        jobs,
        history,
        worker_stub,
        SQLiteResumeJobMatchStore(tmp_path / "resumes.sqlite3"),
        career_profile_store=context,
    )

    result = service.match(
        user_id="u1",
        resume_version_id=version.id,
        job_posting_id=saved.posting.id,
    )

    transition = worker_stub.audits[0]["transitions"][0]
    # SR: the state-anchored pass receives and recognizes the old→new change.
    assert (transition.old_value, transition.new_value) == (
        "Hangzhou",
        "Shanghai",
    )
    # PR: a draft presupposing the old city is not allowed through unchanged.
    assert "Hangzhou" not in result.result.summary
    # IPA: the open-ended result materially follows the new state.
    assert "Shanghai" in result.result.summary
    assert result.result.limitations[-1] == (
        "已按当前求职状态修正：Hangzhou → Shanghai。"
    )
    assert worker_stub.calls[0]["intent_states"][0].value == "Shanghai"


def test_an_intent_confirmed_long_ago_is_reported_not_withheld(tmp_path) -> None:
    resumes, jobs, history, version, saved = seed_inputs(tmp_path)
    context = CareerContextStore(tmp_path / "context.sqlite3")
    context.upsert_profile(
        CareerProfileContext(user_id="u1", default_city="Hangzhou"),
        source="test",
    )
    context.upsert_profile(
        CareerProfileContext(user_id="u1", default_city="Shanghai"),
        source="test",
    )
    confirmed_at = datetime.now(timezone.utc) - timedelta(days=400)
    database = sqlite3.connect(tmp_path / "context.sqlite3")
    with database:
        database.execute(
            "UPDATE career_intent_versions SET last_corroborated_at = ?",
            (confirmed_at.isoformat(),),
        )
    database.close()
    worker_stub = StateAuditingMatchWorker()
    service = ResumeJobMatchService(
        resumes,
        jobs,
        history,
        worker_stub,
        SQLiteResumeJobMatchStore(tmp_path / "resumes.sqlite3"),
        career_profile_store=context,
    )

    result = service.match(
        user_id="u1",
        resume_version_id=version.id,
        job_posting_id=saved.posting.id,
    )

    # The preference is still delivered, carrying the date it was last stated
    # rather than a score that would have excluded it for being old.
    anchor = worker_stub.calls[0]["intent_states"][0]
    assert anchor.value == "Shanghai"
    assert anchor.last_confirmed_at == confirmed_at
    # Its transition survives too, so stale-state repair keeps the input it
    # needs most for intent nobody has restated in a long time.
    assert worker_stub.audits[0]["transitions"][0].new_value == "Shanghai"
    assert "Shanghai" in result.result.summary


def test_repair_only_keeps_conflict_free_draft_byte_stable(tmp_path) -> None:
    resumes, jobs, history, version, saved = seed_inputs(tmp_path)
    context = CareerContextStore(tmp_path / "context.sqlite3")
    context.upsert_profile(CareerProfileContext(user_id="u1", default_city="A"))
    context.upsert_profile(CareerProfileContext(user_id="u1", default_city="B"))
    service = ResumeJobMatchService(
        resumes,
        jobs,
        history,
        ConflictFreeAuditingWorker(),
        SQLiteResumeJobMatchStore(tmp_path / "resumes.sqlite3"),
        career_profile_store=context,
    )

    result = service.match(
        user_id="u1",
        resume_version_id=version.id,
        job_posting_id=saved.posting.id,
    )

    assert result.result == ResumeJobMatchResult.model_validate(VALID_MATCH)


def test_service_hides_foreign_inputs_before_calling_worker(tmp_path) -> None:
    resumes, jobs, history, version, saved = seed_inputs(tmp_path)
    worker_stub = RecordingMatchWorker()
    service = ResumeJobMatchService(
        resumes,
        jobs,
        history,
        worker_stub,
        SQLiteResumeJobMatchStore(tmp_path / "resumes.sqlite3"),
    )

    with pytest.raises(ResumeJobMatchInputNotFoundError) as error:
        service.match(
            user_id="other",
            resume_version_id=version.id,
            job_posting_id=saved.posting.id,
        )

    assert error.value.input_kind == "resume_version"
    assert worker_stub.calls == []


class SequenceDecisionMaker:
    def __init__(self, *decisions: AgentDecision) -> None:
        self.decisions = list(decisions)
        self.contexts = []

    def decide(self, context, tool_specs):
        self.contexts.append(context)
        return self.decisions.pop(0)


def test_main_agent_match_tool_returns_analysis_without_original_documents(tmp_path) -> None:
    resumes, jobs, history, version, saved = seed_inputs(tmp_path)
    service = ResumeJobMatchService(
        resumes,
        jobs,
        history,
        RecordingMatchWorker(),
        SQLiteResumeJobMatchStore(tmp_path / "resumes.sqlite3"),
    )
    manager = ContextManager(CareerContextStore(tmp_path / "context.sqlite3"))
    manager.upsert_profile(CareerProfileContext(user_id="u1"))
    seeded = manager.load_for_turn(
        user_id="u1", conversation_id="c1", user_message="seed active inputs"
    )
    manager.commit_turn(
        context=seeded,
        task=seeded.task.model_copy(
            update={
                "tool_profile": "resume",
                "resume_version_candidates": (
                    ResumeVersionCandidateContextItem(
                        resume_version_id=version.id,
                        version_number=version.version_number,
                        source_type=version.source_type,
                        document_format=version.document_format,
                        byte_size=version.byte_size,
                    ),
                ),
                "saved_job_candidates": (
                    SavedJobCandidateContextItem(
                        job_posting_id=saved.posting.id,
                        title=saved.posting.title,
                        company_name=saved.posting.company_name,
                        city=saved.city,
                        salary=saved.salary,
                    ),
                ),
            }
        ),
        assistant_message="seeded",
    )
    decisions = SequenceDecisionMaker(
        AgentDecision(
            action="tool_call",
            tool_call=ToolCall(
                name="match_resume_to_job",
                arguments={
                    "resume_version_selection_index": 1,
                    "job_selection_index": 1,
                },
            ),
        ),
        AgentDecision(action="final", message="这份简历与岗位整体中等匹配。"),
    )
    tools = MainAgentToolRegistry(
        job_repository=jobs,
        resume_store=resumes,
        resume_job_match_service=service,
    )
    runtime = MainAgentRuntime(
        context_manager=manager,
        decision_maker=decisions,
        tools=tools,
    )

    result = runtime.run_turn(
        user_id="u1",
        conversation_id="c1",
        user_message="把这份简历和这个岗位匹配一下",
    )

    schema = next(
        spec for spec in tools.schemas()
        if spec["function"]["name"] == "match_resume_to_job"
    )
    assert "user_id" not in schema["function"]["parameters"].get("properties", {})
    assert set(schema["function"]["parameters"]["properties"]) == {
        "resume_version_selection_index",
        "job_selection_index",
    }
    observation = result.tool_result
    assert observation.state == "resume_job_match_ready"
    assert observation.payload["requirements"][0]["status"] == "matched"
    serialized = observation.model_dump_json()
    assert "PRIVATE RESUME" not in serialized
    assert "PRIVATE JD" not in serialized
    assert result.context.task.active_resume_job_match_id == observation.payload["match_id"]
    assert result.context.task.resume_job_match_status == "ready"
    rendered = MainAgentRuntime._assistant_message(result.tool_result)
    assert rendered.startswith("# 简历与岗位匹配")
    assert "整体判断：中等匹配" in rendered
    assert "The resume demonstrates relevant RAG experience." in rendered
    assert observation.resource_ref is not None
    assert observation.resource_ref.kind == "resume_job_match"
    assert observation.resource_ref.title == (
        "AI Resume v1 × Acme · RAG Engineer · 简历岗位匹配"
    )
    assert observation.resource_ref.description == "简历与岗位匹配；整体匹配度为 moderate。"

    review_decisions = SequenceDecisionMaker(
        AgentDecision(
            action="tool_call",
            tool_call=ToolCall(name="get_resume_job_match", arguments={}),
        ),
        AgentDecision(action="final", message="这是刚才的匹配结果。"),
    )
    review_runtime = MainAgentRuntime(
        context_manager=manager,
        decision_maker=review_decisions,
        tools=tools,
    )
    review_result = review_runtime.run_turn(
        user_id="u1",
        conversation_id="c1",
        user_message="再看一下刚才的匹配",
    )
    reviewed = review_result.tool_result
    assert reviewed.tool_name == "get_resume_job_match"
    assert reviewed.payload["match_id"] == observation.payload["match_id"]


def test_main_agent_match_tool_rejects_model_supplied_user_id(tmp_path) -> None:
    resumes, jobs, history, version, saved = seed_inputs(tmp_path)
    manager = ContextManager(CareerContextStore(tmp_path / "context.sqlite3"))
    manager.upsert_profile(CareerProfileContext(user_id="u1"))
    enter_tool_profile(manager, "resume")
    decisions = SequenceDecisionMaker(
        AgentDecision(
            action="tool_call",
            tool_call=ToolCall(
                name="match_resume_to_job",
                arguments={
                    "resume_version_id": version.id,
                    "job_posting_id": saved.posting.id,
                    "user_id": "other",
                },
            ),
        )
    )
    runtime = MainAgentRuntime(
        context_manager=manager,
        decision_maker=decisions,
        tools=MainAgentToolRegistry(
        resume_job_match_service=ResumeJobMatchService(
                resumes,
                jobs,
                history,
                RecordingMatchWorker(),
                SQLiteResumeJobMatchStore(tmp_path / "resumes.sqlite3"),
            ),
    ),
    )

    with pytest.raises(ValueError, match="cannot accept internal identifier"):
        runtime.run_turn(
            user_id="u1",
            conversation_id="c1",
            user_message="越权匹配",
        )
