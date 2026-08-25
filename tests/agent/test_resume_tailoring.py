from __future__ import annotations

import base64
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
from career_agent.agent.deepagent_resume_tailoring_worker import (
    DeepAgentResumeTailoringWorker,
)
from career_agent.agent.resume_job_match_contracts import ResumeJobMatchResult
from career_agent.agent.resume_tailoring_contracts import ResumeTailoringResult
from career_agent.domain.job_discovery import JobDetail, Provenance
from career_agent.services.resume_tailoring import (
    ResumeTailoringDraftNotFoundError,
    ResumeTailoringService,
)
from career_agent.storage.career_history import CareerHistoryStore
from career_agent.storage.context import CareerContextStore
from career_agent.storage.jobs import SQLiteJobPostingRepository
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


class RecordingTailoringWorker:
    def __init__(self) -> None:
        self.calls = []

    def tailor(self, **kwargs) -> ResumeTailoringResult:
        self.calls.append(kwargs)
        return ResumeTailoringResult.model_validate(VALID_DRAFT)


def seed_service(tmp_path):
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
    service = ResumeTailoringService(
        resumes,
        jobs,
        history,
        matches,
        drafts,
        tailoring_worker,
    )
    return service, tailoring_worker, stored_match


def test_tailoring_service_reads_private_inputs_and_persists_reviewable_draft(tmp_path) -> None:
    service, worker, stored_match = seed_service(tmp_path)

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

    with pytest.raises(ValueError, match="Unknown tailoring change index"):
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


class UnusedGateway:
    def advance(self, **kwargs):
        raise AssertionError("Tailoring must not enter Job Discovery")


class SequenceDecisionMaker:
    def __init__(self, *decisions: AgentDecision) -> None:
        self.decisions = list(decisions)
        self.contexts = []

    def decide(self, context, tool_specs):
        self.contexts.append(context)
        return self.decisions.pop(0)


def test_main_agent_creates_and_recalls_active_tailoring_draft(tmp_path) -> None:
    service, _, stored_match = seed_service(tmp_path)
    manager = ContextManager(CareerContextStore(tmp_path / "context.sqlite3"))
    manager.upsert_profile(CareerProfileContext(user_id="u1"))
    decisions = SequenceDecisionMaker(
        AgentDecision(
            action="tool_call",
            tool_call=ToolCall(
                name="draft_resume_tailoring",
                arguments={
                    "match_id": stored_match.id,
                    "tailoring_goal": "Keep it concise",
                },
            ),
        ),
        AgentDecision(action="final", message="我生成了一份待审阅的修改草稿。"),
    )
    tools = MainAgentToolRegistry(
        UnusedGateway(),
        resume_tailoring_service=service,
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

    observation = decisions.contexts[1].tool_observations[-1]
    assert observation.state == "resume_tailoring_draft_ready"
    assert [change["change_index"] for change in observation.payload["changes"]] == [1, 2]
    assert result.context.task.active_resume_tailoring_draft_id == observation.payload["draft_id"]
    assert result.context.task.resume_tailoring_status == "pending"
    serialized = observation.model_dump_json()
    assert "PRIVATE RESUME" not in serialized
    assert "PRIVATE JD" not in serialized

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
    review_runtime.run_turn(
        user_id="u1",
        conversation_id="c1",
        user_message="再看一下草稿",
    )
    reviewed = review_decisions.contexts[1].tool_observations[-1]
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
    review_observation = decision_maker.contexts[1].tool_observations[-1]
    assert review_observation.tool_name == "review_resume_tailoring"
    assert review_observation.payload["status"] == "reviewed"
    assert review_observation.payload["pending_change_indices"] == ()
    assert reviewed_result.context.task.resume_tailoring_status == "reviewed"
