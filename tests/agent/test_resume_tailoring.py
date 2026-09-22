from __future__ import annotations

import base64
import copy
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import TypeAdapter, ValidationError

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
from career_agent.agent.resume_job_match_contracts import (
    ConfirmedResumeFact,
    RequirementAssessment,
    ResumeJobMatchResult,
)
from career_agent.agent.resume_tailoring_contracts import (
    AcceptedTailoringChange,
    GapMitigationPayload,
    GapLearningPlan,
    FinalizedResumeDocument,
    ResumeReviewIssue,
    ResumeReviewResult,
    ResumeTailoringResult,
    ResumeTailoringGenerationResult,
    canonicalize_gap_mitigations,
    gap_mitigation_errors,
)
from career_agent.agent.resume_tailoring_presenter import render_resume_tailoring
from career_agent.agent.resume_tailoring_review_graph import ResumeTailoringReviewGraph
from career_agent.agent.resume_tailoring_review_graph import EvidenceCheck
from career_agent.domain.job_discovery import JobDetail, Provenance
from career_agent.services.resume_export import ResumeExportService
from career_agent.services.resume_tailoring import (
    ResumeFinalReviewBlockedError,
    ResumeTailoringAlreadyFinalizedError,
    ResumeTailoringDraftNotFoundError,
    ResumeTailoringNotReadyError,
    ResumeTailoringReviewBlockedError,
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


REQUIREMENT_ID = "job_requirement_00000000000000000001"


VALID_MATCH = ResumeJobMatchResult(
    overall_fit="moderate",
    summary="Relevant RAG experience with some gaps.",
    requirements=(
        RequirementAssessment(
            requirement_id=REQUIREMENT_ID,
            requirement="Production Go experience",
            jd_quote="Strong production Go experience is required.",
            tier="A",
            kind="fact",
            status="missing",
            rationale="The resume does not establish this requirement.",
        ),
    ),
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
    "gap_mitigations": [
        {
            "gap": "Go is not stated",
            "requirement_id": REQUIREMENT_ID,
            "gap_type": "strengthenable",
            "priority": "P1",
            "resolution_mode": "learn",
            "rationale": "The resume does not establish Go proficiency.",
            "adjacent_experience": [
                {
                    "source_locator": "Experience, bullet 1",
                    "source_quote": "Built RAG systems",
                    "relevance": "Shows production software delivery, not Go proficiency.",
                }
            ],
            "alternative_evidence": [
                {
                    "description": (
                        "A small reviewed Go service with tests and a deployment README"
                    ),
                    "status": "planned",
                    "acceptance_criteria": (
                        "The repository builds, tests pass, and the README explains one "
                        "concurrency trade-off."
                    ),
                }
            ],
            "next_action": "Build one small Go API and document the trade-offs.",
            "learning_plan": {
                "objective": "Implement and explain a small idiomatic Go service.",
                "resource_directions": [
                    "Official Go tutorial and effective Go guidance",
                    "A testing and HTTP-service lab",
                ],
                "minimum_acceptable_level": (
                    "Can implement, test, and explain one concurrent HTTP service."
                ),
                "estimated_effort": "1-2 weeks",
            },
            "interview_talking_point": {
                "acknowledge_gap": "My current resume does not show production Go work.",
                "bridge_to_evidence": (
                    "I have delivered production RAG systems and can discuss the "
                    "engineering practices that transfer."
                ),
                "close_with_action": (
                    "I am building a tested Go service so I can demonstrate the "
                    "language-specific fundamentals directly."
                ),
            },
        }
    ],
    "clarification_questions": [],
    "warnings": [],
}

COMPACT_DRAFT = copy.deepcopy(VALID_DRAFT)
COMPACT_DRAFT["gap_mitigations"][0].pop("adjacent_experience")
COMPACT_DRAFT["gap_mitigations"][0].pop("alternative_evidence")


def _mitigation(**updates):
    mitigation = copy.deepcopy(VALID_DRAFT["gap_mitigations"][0])
    mitigation.update(updates)
    return mitigation


def _match_requirement(
    *,
    requirement_id: str = REQUIREMENT_ID,
    tier: str = "S",
    kind: str = "fact",
    status: str = "missing",
) -> RequirementAssessment:
    return RequirementAssessment(
        requirement_id=requirement_id,
        requirement="Production Go experience",
        jd_quote="Strong production Go experience is required.",
        tier=tier,
        kind=kind,
        status=status,
        rationale="The resume does not establish this requirement.",
    )


class FakeDeepAgent:
    def __init__(self, output=COMPACT_DRAFT) -> None:
        self.output = output
        self.state = None

    def invoke(self, state, *, config=None):
        self.state = state
        self.config = config
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


def test_legacy_tailoring_result_without_gap_mitigations_remains_readable() -> None:
    legacy = copy.deepcopy(VALID_DRAFT)
    legacy.pop("gap_mitigations")

    result = ResumeTailoringResult.model_validate(legacy)

    assert result.unresolved_gaps == ("Go is not stated",)
    assert result.gap_mitigations == ()


def test_gap_coverage_uses_requirement_ids_not_model_gap_text() -> None:
    draft = copy.deepcopy(VALID_DRAFT)
    draft["unresolved_gaps"] = ["unrelated model wording"]
    draft["gap_mitigations"][0].pop("gap")

    canonical = canonicalize_gap_mitigations(
        ResumeTailoringResult.model_validate(draft),
        VALID_MATCH,
    )

    assert gap_mitigation_errors(canonical, VALID_MATCH) == ()
    assert canonical.gap_mitigations[0].gap == "Production Go experience"
    assert canonical.unresolved_gaps == ("Production Go experience",)


def test_s_fact_missing_requirement_is_a_p0_hard_blocker() -> None:
    draft = copy.deepcopy(VALID_DRAFT)
    draft["gap_mitigations"] = [
        _mitigation(
            requirement_id=REQUIREMENT_ID,
            gap_type="hard_blocker",
            priority="P0",
        )
    ]
    result = ResumeTailoringResult.model_validate(draft)
    match = ResumeJobMatchResult(
        overall_fit="weak",
        summary="A stated hard requirement is missing.",
        requirements=(_match_requirement(),),
    )

    assert gap_mitigation_errors(result, match) == ()


@pytest.mark.parametrize(
    ("tier", "kind", "status"),
    (("S", "inference", "missing"), ("S", "fact", "unclear"), ("B", "fact", "missing")),
)
def test_only_s_fact_missing_may_be_a_hard_blocker(tier, kind, status) -> None:
    draft = copy.deepcopy(VALID_DRAFT)
    draft["gap_mitigations"] = [
        _mitigation(
            requirement_id=REQUIREMENT_ID,
            gap_type="hard_blocker",
            priority="P0",
        )
    ]
    result = ResumeTailoringResult.model_validate(draft)
    match = ResumeJobMatchResult(
        overall_fit="weak",
        summary="One requirement remains unresolved.",
        requirements=(
            _match_requirement(tier=tier, kind=kind, status=status),
        ),
    )

    assert any(
        "only S/fact/missing can be a hard blocker" in error
        for error in gap_mitigation_errors(result, match)
    )


def test_s_fact_missing_cannot_be_downgraded_or_deprioritized() -> None:
    match = ResumeJobMatchResult(
        overall_fit="weak",
        summary="A stated hard requirement is missing.",
        requirements=(_match_requirement(),),
    )
    strengthenable = copy.deepcopy(VALID_DRAFT)
    strengthenable["gap_mitigations"] = [
        _mitigation(requirement_id=REQUIREMENT_ID, gap_type="strengthenable")
    ]
    non_p0 = copy.deepcopy(VALID_DRAFT)
    non_p0["gap_mitigations"] = [
        _mitigation(
            requirement_id=REQUIREMENT_ID,
            gap_type="hard_blocker",
            priority="P1",
        )
    ]

    assert any(
        "must be a hard blocker" in error
        for error in gap_mitigation_errors(
            ResumeTailoringResult.model_validate(strengthenable), match
        )
    )
    assert any(
        "must have P0 priority" in error
        for error in gap_mitigation_errors(
            ResumeTailoringResult.model_validate(non_p0), match
        )
    )


def test_unclear_requirement_must_clarify_without_a_learning_plan() -> None:
    match = ResumeJobMatchResult(
        overall_fit="insufficient_evidence",
        summary="One stated requirement needs clarification.",
        requirements=(_match_requirement(status="unclear"),),
    )
    valid = copy.deepcopy(VALID_DRAFT)
    valid["gap_mitigations"] = [
        _mitigation(
            resolution_mode="clarify",
            priority="P0",
            learning_plan=None,
            clarification_question="What production scale is expected?",
            adjacent_experience=[],
            alternative_evidence=[],
            next_action="Ask the recruiter what production scale is expected.",
        )
    ]
    invalid = copy.deepcopy(VALID_DRAFT)

    assert gap_mitigation_errors(
        ResumeTailoringResult.model_validate(valid), match
    ) == ()
    errors = gap_mitigation_errors(
        ResumeTailoringResult.model_validate(invalid), match
    )
    assert any("must use clarify mode" in error for error in errors)
    assert any("cannot prescribe learning" in error for error in errors)


def test_p0_is_rejected_for_non_blocking_non_clarification_gap() -> None:
    draft = copy.deepcopy(VALID_DRAFT)
    draft["gap_mitigations"] = [_mitigation(priority="P0")]

    errors = gap_mitigation_errors(
        ResumeTailoringResult.model_validate(draft), VALID_MATCH
    )

    assert any("P0 is reserved" in error for error in errors)


def test_alternative_evidence_distinguishes_existing_from_planned() -> None:
    existing = _mitigation(
        resolution_mode="provide_evidence",
        learning_plan=None,
        alternative_evidence=[
            {
                "description": "Existing architecture sample",
                "status": "existing",
                "source_locator": "Experience, bullet 1",
                "source_quote": "Built RAG systems",
            }
        ],
    )
    planned = _mitigation(
        resolution_mode="build_artifact",
        learning_plan=None,
        alternative_evidence=[
            {
                "description": "Small Go service",
                "status": "planned",
                "acceptance_criteria": "Builds cleanly and passes its tests.",
            }
        ],
    )

    for mitigation in (existing, planned):
        draft = copy.deepcopy(VALID_DRAFT)
        draft["gap_mitigations"] = [mitigation]
        assert gap_mitigation_errors(
            ResumeTailoringResult.model_validate(draft), VALID_MATCH
        ) == ()


@pytest.mark.parametrize(
    "alternative",
    (
        {"description": "Unproven existing item", "status": "existing"},
        {"description": "Unspecified future item", "status": "planned"},
    ),
)
def test_alternative_evidence_requires_proof_or_acceptance_criteria(
    alternative,
) -> None:
    draft = copy.deepcopy(VALID_DRAFT)
    draft["gap_mitigations"][0]["alternative_evidence"] = [alternative]

    with pytest.raises(ValidationError):
        ResumeTailoringResult.model_validate(draft)


def test_new_mitigation_cannot_use_unbound_gap_text() -> None:
    draft = copy.deepcopy(VALID_DRAFT)
    draft["gap_mitigations"][0]["requirement_id"] = None

    errors = gap_mitigation_errors(
        ResumeTailoringResult.model_validate(draft), VALID_MATCH
    )

    assert "new gap mitigations require an authoritative requirement ID" in errors


def test_legacy_free_text_alternative_evidence_is_readable_but_not_new_valid() -> None:
    draft = copy.deepcopy(VALID_DRAFT)
    draft["gap_mitigations"][0]["alternative_evidence"] = ["Legacy portfolio text"]
    result = ResumeTailoringResult.model_validate(draft)

    assert result.gap_mitigations[0].alternative_evidence == (
        "Legacy portfolio text",
    )
    assert any(
        "must declare existing or planned status" in error
        for error in gap_mitigation_errors(result, VALID_MATCH)
    )


def test_unknown_and_duplicate_requirement_ids_are_rejected() -> None:
    unknown = copy.deepcopy(VALID_DRAFT)
    unknown["gap_mitigations"] = [
        _mitigation(requirement_id=REQUIREMENT_ID)
    ]
    unknown_errors = gap_mitigation_errors(
        ResumeTailoringResult.model_validate(unknown),
        ResumeJobMatchResult(overall_fit="moderate", summary="No requirements."),
    )
    assert unknown_errors == (f"unknown requirement ID: {REQUIREMENT_ID}",)

    duplicate = copy.deepcopy(VALID_DRAFT)
    duplicate["unresolved_gaps"] = ["Go is not stated", "Go delivery is not stated"]
    duplicate["gap_mitigations"] = [
        _mitigation(requirement_id=REQUIREMENT_ID),
        _mitigation(
            gap="Go delivery is not stated",
            requirement_id=REQUIREMENT_ID,
        ),
    ]
    duplicate_errors = gap_mitigation_errors(
        ResumeTailoringResult.model_validate(duplicate),
        ResumeJobMatchResult(
            overall_fit="weak",
            summary="Go remains unresolved.",
            requirements=(_match_requirement(tier="A"),),
        ),
    )
    assert "a requirement can have only one gap mitigation" in duplicate_errors


def test_presenter_renders_actionable_gap_plan() -> None:
    rendered = render_resume_tailoring(
        ResumeTailoringResult.model_validate(VALID_DRAFT),
        status="pending",
        revision_number=1,
    )

    assert "## 缺口缓解与学习路线" in rendered
    assert "[P1] Go is not stated" in rendered
    assert "分类：可补强项" in rendered
    assert "处理方式：学习并达到可演示水平" in rendered
    assert "Built RAG systems" in rendered
    assert "计划产物" in rendered
    assert "最低可接受水平" in rendered
    assert "面试诚实表达" in rendered


def test_presenter_marks_ocr_evidence_as_waiting_for_confirmation() -> None:
    draft = copy.deepcopy(VALID_DRAFT)
    draft["gap_mitigations"][0]["adjacent_experience"][0][
        "evidence_quality"
    ] = "ocr_unverified"

    rendered = render_resume_tailoring(
        ResumeTailoringResult.model_validate(draft),
        status="pending",
        revision_number=1,
    )

    assert "待用户确认（OCR 未验证）" in rendered


@pytest.mark.parametrize(
    ("mode", "updates"),
    (
        (
            "clarify",
            {
                "clarification_question": "What production scale is expected?",
                "learning_plan": None,
                "adjacent_experience": [],
                "alternative_evidence": [],
            },
        ),
        (
            "provide_evidence",
            {
                "learning_plan": None,
                "alternative_evidence": [
                    {
                        "description": "Existing RAG delivery evidence",
                        "status": "existing",
                        "source_locator": "Experience, bullet 1",
                        "source_quote": "Built RAG systems",
                    }
                ],
            },
        ),
        (
            "build_artifact",
            {
                "learning_plan": None,
                "adjacent_experience": [],
                "alternative_evidence": [
                    {
                        "description": "A tested Go service",
                        "status": "planned",
                        "acceptance_criteria": "Builds and tests pass.",
                    }
                ],
            },
        ),
        (
            "learn",
            {
                "alternative_evidence": [],
            },
        ),
    ),
)
def test_each_resolution_mode_has_a_minimal_legal_shape(mode, updates) -> None:
    draft = copy.deepcopy(VALID_DRAFT)
    draft["gap_mitigations"] = [_mitigation(resolution_mode=mode, **updates)]

    result = ResumeTailoringResult.model_validate(draft)
    assert result.gap_mitigations[0].resolution_mode == mode


def test_conditional_fields_are_rejected_when_mode_does_not_need_them() -> None:
    draft = copy.deepcopy(VALID_DRAFT)
    draft["gap_mitigations"] = [
        _mitigation(
            resolution_mode="clarify",
            clarification_question="What scale is expected?",
            learning_plan=VALID_DRAFT["gap_mitigations"][0]["learning_plan"],
            adjacent_experience=[],
            alternative_evidence=[],
        )
    ]

    with pytest.raises(ValidationError, match="cannot include a learning plan"):
        ResumeTailoringResult.model_validate(draft)


def test_evidence_check_reports_normalized_quality_and_page() -> None:
    check = ResumeTailoringReviewGraph._check_evidence(
        document=StoredResumeDocument(
            resume_version_id="pdf-layout",
            document_format="pdf",
            raw_bytes=_single_page_pdf("micro- service"),
        ),
        quote="micro-service",
        declared_quality="exact",
        page=1,
    )

    assert check == EvidenceCheck(
        matched=True,
        quality="normalized",
        page=1,
        reason="hyphenation_normalized",
    )


def test_declared_page_must_contain_the_quote() -> None:
    check = ResumeTailoringReviewGraph._check_evidence(
        document=StoredResumeDocument(
            resume_version_id="pdf-layout",
            document_format="pdf",
            raw_bytes=_single_page_pdf("machine-learning platform"),
        ),
        quote="machine-learning platform",
        declared_quality="exact",
        page=2,
    )

    assert check.matched is False
    assert check.reason == "page_out_of_range"


def test_known_quote_cannot_override_an_invalid_pdf_page() -> None:
    draft = copy.deepcopy(VALID_DRAFT)
    draft["changes"][0]["support_evidence"][0]["page"] = 99
    draft["gap_mitigations"][0]["adjacent_experience"][0]["page"] = 99
    issues = ResumeTailoringReviewGraph._validate_grounding(
        document=StoredResumeDocument(
            resume_version_id="one-page-pdf",
            document_format="pdf",
            raw_bytes=_single_page_pdf("Built RAG systems"),
        ),
        match_result=VALID_MATCH,
        confirmed_facts=(
            ConfirmedResumeFact(
                claim="Built RAG systems",
                source_locator="Experience",
                source_quote="Built RAG systems",
            ),
        ),
        draft=ResumeTailoringResult.model_validate(draft),
    )

    assert len([issue for issue in issues if "page_out_of_range" in issue.explanation]) == 2
    assert all(
        issue.severity == "blocking"
        for issue in issues if "page_out_of_range" in issue.explanation
    )


def test_historical_known_quote_cannot_override_the_exact_resume_document() -> None:
    draft = copy.deepcopy(VALID_DRAFT)
    invented = "Invented leadership credential"
    draft["changes"][0]["support_evidence"][0]["source_quote"] = invented
    issues = ResumeTailoringReviewGraph._validate_grounding(
        document=StoredResumeDocument(
            resume_version_id="v1", document_format="text",
            raw_bytes=b"Built RAG systems",
        ),
        match_result=VALID_MATCH,
        confirmed_facts=(ConfirmedResumeFact(
            claim=invented, source_locator="Experience", source_quote=invented,
        ),),
        draft=ResumeTailoringResult.model_validate(draft),
    )

    assert any(
        issue.severity == "blocking" and issue.source_quote == invented
        for issue in issues
    )


def test_server_canonicalizes_declared_exact_to_normalized() -> None:
    draft = ResumeTailoringResult(
        strategy_summary="Surface the exact supported phrase.",
        changes=(
            {
                "target_locator": "Experience",
                "original_quote": "machine-learning platform",
                "proposed_text": "Machine-learning platform",
                "rationale": "Normalize capitalization.",
                "support_evidence": (
                    {
                        "source_locator": "Experience",
                        "source_quote": "machine-learning platform",
                        "evidence_quality": "exact",
                    },
                ),
            },
        ),
    )

    canonical = ResumeTailoringReviewGraph._canonicalize_evidence(
        draft,
        StoredResumeDocument(
            resume_version_id="pdf-layout",
            document_format="pdf",
            raw_bytes=_single_page_pdf("machine- learning platform"),
        ),
    )

    evidence = canonical.changes[0].support_evidence[0]
    assert evidence.evidence_quality == "normalized"
    assert evidence.page == 1


def test_server_promotes_ocr_declaration_when_text_layer_verifies_quote() -> None:
    check = ResumeTailoringReviewGraph._check_evidence(
        document=StoredResumeDocument(
            resume_version_id="text-layer",
            document_format="pdf",
            raw_bytes=_single_page_pdf("Built reliable RAG systems"),
        ),
        quote="built reliable rag systems",
        declared_quality="ocr_unverified",
        page=1,
    )

    assert check == EvidenceCheck(
        matched=True,
        quality="exact",
        page=1,
        reason="exact_text_match",
    )


def test_scanned_pdf_downgrades_declared_exact_to_ocr_unverified() -> None:
    draft = copy.deepcopy(VALID_DRAFT)
    canonical = ResumeTailoringReviewGraph._canonicalize_evidence(
        ResumeTailoringResult.model_validate(draft),
        StoredResumeDocument(
            resume_version_id="scan",
            document_format="pdf",
            raw_bytes=_scanned_pdf(),
        ),
    )

    assert canonical.changes[0].support_evidence[0].evidence_quality == "ocr_unverified"


def test_ocr_evidence_still_validates_declared_page_number() -> None:
    check = ResumeTailoringReviewGraph._check_evidence(
        document=StoredResumeDocument(
            resume_version_id="scan",
            document_format="pdf",
            raw_bytes=_scanned_pdf(),
        ),
        quote="anything",
        declared_quality="ocr_unverified",
        page=2,
    )

    assert check.matched is False
    assert check.reason == "page_out_of_range"


def test_ocr_unverified_evidence_is_not_accepted_for_a_resume_change() -> None:
    invalid = copy.deepcopy(VALID_DRAFT)
    invalid["changes"][0]["support_evidence"][0]["evidence_quality"] = "ocr_unverified"

    issues = ResumeTailoringReviewGraph._validate_grounding(
        document=StoredResumeDocument(
            resume_version_id="scan",
            document_format="pdf",
            raw_bytes=_scanned_pdf(),
        ),
        match_result=VALID_MATCH,
        confirmed_facts=(),
        draft=ResumeTailoringResult.model_validate(invalid),
    )

    assert any(
        issue.severity == "blocking"
        and "cannot be used to write" in issue.explanation
        for issue in issues
    )


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
    assert captured["skills"] == []
    assert "Use only grounded resume evidence." in captured["system_prompt"]
    assert captured["response_format"] is ResumeTailoringGenerationResult
    generation_schema = captured["response_format"].model_json_schema()
    assert "GapMitigation" not in generation_schema["$defs"]
    assert "LearnMitigation" in generation_schema["$defs"]
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


def seed_service(tmp_path, *, reviewer=None, document_format="text", content=b"PRIVATE RESUME: Built RAG systems"):
    resume_path = tmp_path / "resumes.sqlite3"
    resumes = ResumeStore(resume_path)
    role = resumes.create_target_role(user_id="u1", title="AI Engineer", priority=1)
    _, version = resumes.import_document(
        user_id="u1",
        target_role_id=role.id,
        name="AI Resume",
        content=content,
        document_format=document_format,
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


def test_service_without_reviewer_blocks_scanned_pdf_support_evidence(tmp_path) -> None:
    service, _, _, stored_match = seed_service(
        tmp_path,
        document_format="pdf",
        content=_scanned_pdf(),
    )

    with pytest.raises(ResumeTailoringReviewBlockedError, match="grounding"):
        service.create_draft(user_id="u1", match_id=stored_match.id)

    with sqlite3.connect(tmp_path / "resumes.sqlite3") as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM resume_tailoring_drafts WHERE match_id = ?",
            (stored_match.id,),
        ).fetchone()[0] == 0


def test_finalization_blocks_a_legacy_scanned_pdf_draft_without_reviewer(tmp_path) -> None:
    service, _, finalization_worker, stored_match = seed_service(
        tmp_path,
        document_format="pdf",
        content=_scanned_pdf(),
    )
    legacy_draft = service._draft_store.create(
        user_id="u1",
        match_id=stored_match.id,
        parent_draft_id=None,
        revision_number=1,
        revision_feedback=None,
        tailoring_goal=None,
        worker_version="legacy",
        result=ResumeTailoringResult.model_validate(VALID_DRAFT),
        automated_review=None,
    )
    service.review_draft(
        user_id="u1", draft_id=legacy_draft.id,
        accepted_change_indices=(1,), rejected_change_indices=(2,),
    )

    with pytest.raises(ResumeFinalReviewBlockedError, match="grounding"):
        service.finalize_draft(user_id="u1", draft_id=legacy_draft.id)

    assert finalization_worker.calls == []


def test_finalization_ignores_rejected_ungrounded_change(tmp_path) -> None:
    service, _, finalization_worker, stored_match = seed_service(tmp_path)
    legacy_result = copy.deepcopy(VALID_DRAFT)
    legacy_result["changes"][1]["support_evidence"][0]["source_quote"] = (
        "Invented credential absent from the resume"
    )
    legacy_draft = service._draft_store.create(
        user_id="u1",
        match_id=stored_match.id,
        tailoring_goal=None,
        worker_version="legacy",
        result=ResumeTailoringResult.model_validate(legacy_result),
    )
    service.review_draft(
        user_id="u1", draft_id=legacy_draft.id,
        accepted_change_indices=(1,), rejected_change_indices=(2,),
    )

    finalized = service.finalize_draft(user_id="u1", draft_id=legacy_draft.id)

    assert finalized.created is True
    assert finalized.applied_change_indices == (1,)
    assert tuple(item.change_index for item in finalization_worker.calls[0]["accepted_changes"]) == (1,)


def test_compact_gap_mitigation_union_rejects_unrelated_fields() -> None:
    payload = {
        "requirement_id": REQUIREMENT_ID,
        "gap_type": "strengthenable",
        "priority": "P1",
        "next_action": "Ask the hiring team.",
        "rationale": "The threshold is not stated.",
        "resolution_mode": "clarify",
        "clarification_question": "What level is expected?",
        "learning_plan": VALID_DRAFT["gap_mitigations"][0]["learning_plan"],
    }
    with pytest.raises(ValidationError):
        TypeAdapter(GapMitigationPayload).validate_python(payload)


def test_generation_schema_rejects_legacy_superset_while_read_schema_accepts_it() -> None:
    assert ResumeTailoringResult.model_validate(VALID_DRAFT).gap_mitigations
    with pytest.raises(ValidationError):
        ResumeTailoringGenerationResult.model_validate(VALID_DRAFT)
    assert ResumeTailoringGenerationResult.model_validate(COMPACT_DRAFT).gap_mitigations


def test_compact_gap_mitigation_union_accepts_each_mode_minimum() -> None:
    common = {
        "requirement_id": REQUIREMENT_ID,
        "gap_type": "strengthenable",
        "priority": "P1",
        "next_action": "Take the next concrete step.",
        "rationale": "The current evidence is incomplete.",
    }
    values = (
        {**common, "resolution_mode": "clarify", "clarification_question": "What is required?"},
        {
            **common,
            "resolution_mode": "provide_evidence",
            "adjacent_experience": [
                {"source_locator": "Experience", "source_quote": "Built RAG systems", "relevance": "adjacent"}
            ],
        },
        {
            **common,
            "resolution_mode": "build_artifact",
            "alternative_evidence": [
                {"description": "Tested service", "status": "planned", "acceptance_criteria": "Passes tests."}
            ],
        },
        {**common, "resolution_mode": "learn", "learning_plan": VALID_DRAFT["gap_mitigations"][0]["learning_plan"]},
    )
    parsed = tuple(TypeAdapter(GapMitigationPayload).validate_python(value) for value in values)
    assert [item.resolution_mode for item in parsed] == ["clarify", "provide_evidence", "build_artifact", "learn"]


def test_legacy_learning_plan_quality_is_not_revalidated_on_read(tmp_path) -> None:
    legacy = copy.deepcopy(VALID_DRAFT)
    legacy["gap_mitigations"][0]["learning_plan"] = {
        "objective": "理解相关概念",
        "resource_directions": ["学习相关知识。"],
        "minimum_acceptable_level": "理解相关概念",
        "estimated_effort": "时间待定",
    }
    result = ResumeTailoringResult.model_validate(legacy)
    store = SQLiteResumeTailoringDraftStore(tmp_path / "drafts.sqlite3")
    stored = store.create(
        user_id="u1", match_id="match-1", tailoring_goal=None,
        worker_version="historical", result=result,
    )

    reloaded = store.get(user_id="u1", draft_id=stored.id)
    assert reloaded is not None
    assert reloaded.result.gap_mitigations[0].learning_plan.estimated_effort == "时间待定"


@pytest.mark.parametrize("effort", ["0 hours", "2-1 weeks", "0-2 weeks", "2-0 weeks", "as needed", "2 weeks or 0 hours"])
def test_new_learning_plan_rejects_invalid_effort_at_generation_boundary(effort) -> None:
    draft = copy.deepcopy(VALID_DRAFT)
    draft["gap_mitigations"][0]["learning_plan"]["estimated_effort"] = effort
    result = ResumeTailoringResult.model_validate(draft)

    assert any("estimated effort" in error for error in gap_mitigation_errors(result, VALID_MATCH))


def test_learning_plan_accepts_positive_ascending_effort() -> None:
    valid = GapLearningPlan(
        objective="Implement a Go HTTP service",
        resource_directions=("Official Go HTTP and testing tutorials",),
        minimum_acceptable_level="A tested, deployable project that can be demoed",
        estimated_effort="1-2 weeks",
    )
    assert valid.estimated_effort == "1-2 weeks"
    draft = copy.deepcopy(VALID_DRAFT)
    draft["gap_mitigations"][0]["learning_plan"]["estimated_effort"] = "1-2 weeks"
    assert gap_mitigation_errors(ResumeTailoringResult.model_validate(draft), VALID_MATCH) == ()


def test_semantically_related_chinese_plan_is_not_rejected_by_token_overlap() -> None:
    match = VALID_MATCH.model_copy(update={
        "requirements": (
            VALID_MATCH.requirements[0].model_copy(
                update={"requirement": "掌握数据库优化能力"}
            ),
        ),
    })
    draft = copy.deepcopy(VALID_DRAFT)
    draft["gap_mitigations"][0]["learning_plan"]["objective"] = "学习索引与事务"
    draft["gap_mitigations"][0]["learning_plan"]["resource_directions"] = [
        "数据库索引和事务的官方文档"
    ]
    draft["gap_mitigations"][0]["learning_plan"]["minimum_acceptable_level"] = (
        "完成索引优化项目并用查询计划验证改进"
    )
    result = ResumeTailoringResult.model_validate(draft)
    assert gap_mitigation_errors(result, match) == ()


def test_reviewer_is_instructed_to_check_semantic_relevance_and_plan_quality() -> None:
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
    reviewer.review_draft(
        document=StoredResumeDocument(
            resume_version_id="v1", document_format="text", raw_bytes=b"Built RAG systems"
        ),
        jd_text="Production Go experience",
        match_result=VALID_MATCH,
        draft=ResumeTailoringResult.model_validate(VALID_DRAFT),
    )

    instructions = client.calls[0]["instructions"]
    assert "semantically" in instructions
    assert "negated outcomes" in instructions
    assert "plausible" in instructions


def test_keyword_only_or_negated_learning_outcome_is_revised_by_reviewer() -> None:
    invalid_data = copy.deepcopy(VALID_DRAFT)
    invalid_data["gap_mitigations"][0]["learning_plan"].update({
        "resource_directions": ["学习相关知识。"],
        "minimum_acceptable_level": "No project or test required",
        "estimated_effort": "2 weeks",
    })
    invalid = ResumeTailoringResult.model_validate(invalid_data)
    valid = ResumeTailoringResult.model_validate(VALID_DRAFT)
    assert gap_mitigation_errors(invalid, VALID_MATCH) == ()

    class SequenceWorker:
        def __init__(self) -> None:
            self.outputs = [invalid, valid]

        def tailor(self, **kwargs):
            return self.outputs.pop(0)

    class PlanReviewer(RecordingReviewer):
        def review_draft(self, **kwargs) -> ResumeReviewResult:
            self.draft_calls.append(kwargs)
            plan = kwargs["draft"].gap_mitigations[0].learning_plan
            if plan.minimum_acceptable_level == "No project or test required":
                return ResumeReviewResult(
                    verdict="revise",
                    summary="The learning plan has no verifiable outcome.",
                    issues=(ResumeReviewIssue(
                        category="jd_misalignment",
                        severity="blocking",
                        explanation="Generic resources and a negated outcome cannot demonstrate progress.",
                        revision_instruction="Name specific material and an outcome that can be checked.",
                    ),),
                )
            return ResumeReviewResult(verdict="pass", summary="Plan is actionable.")

    reviewer = PlanReviewer()
    outcome = ResumeTailoringReviewGraph(SequenceWorker(), reviewer).run(
        document=StoredResumeDocument(
            resume_version_id="v1",
            document_format="text",
            raw_bytes=b"Built RAG systems and Python",
        ),
        jd_text="Production Go experience",
        match_result=VALID_MATCH,
    )

    assert outcome.trace.status == "passed"
    assert [attempt.result.verdict for attempt in outcome.trace.attempts] == [
        "revise", "pass",
    ]
    assert len(reviewer.draft_calls) == 2


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


def test_grounding_blocks_unverified_adjacent_gap_evidence() -> None:
    invalid = copy.deepcopy(VALID_DRAFT)
    invalid["gap_mitigations"][0]["adjacent_experience"][0][
        "source_quote"
    ] = "Led a production Go platform"

    issues = ResumeTailoringReviewGraph._validate_grounding(
        document=StoredResumeDocument(
            resume_version_id="v1",
            document_format="text",
            raw_bytes=b"Built RAG systems and Python",
        ),
        match_result=VALID_MATCH,
        confirmed_facts=(),
        draft=ResumeTailoringResult.model_validate(invalid),
    )

    adjacent_issue = next(
        issue for issue in issues if issue.source_quote == "Led a production Go platform"
    )
    assert adjacent_issue.severity == "blocking"
    assert adjacent_issue.category == "unsupported_fact"


def test_grounding_blocks_unverified_existing_alternative_evidence() -> None:
    invalid = copy.deepcopy(VALID_DRAFT)
    invalid["gap_mitigations"][0]["resolution_mode"] = "provide_evidence"
    invalid["gap_mitigations"][0]["learning_plan"] = None
    invalid["gap_mitigations"][0]["alternative_evidence"] = [
        {
            "description": "Claimed existing Go service",
            "status": "existing",
            "source_locator": "Projects",
            "source_quote": "Built a production Go platform",
        }
    ]

    issues = ResumeTailoringReviewGraph._validate_grounding(
        document=StoredResumeDocument(
            resume_version_id="v1",
            document_format="text",
            raw_bytes=b"Built RAG systems and Python",
        ),
        match_result=VALID_MATCH,
        confirmed_facts=(),
        draft=ResumeTailoringResult.model_validate(invalid),
    )

    issue = next(
        item for item in issues if item.source_quote == "Built a production Go platform"
    )
    assert issue.severity == "blocking"
    assert "mark the material as planned" in (issue.revision_instruction or "")


def test_pdf_grounding_tolerates_layout_hyphen_and_whitespace_artifacts() -> None:
    draft = ResumeTailoringResult(
        strategy_summary="Surface the exact supported phrase.",
        changes=(
            {
                "target_locator": "Experience",
                "original_quote": "machine-learning platform",
                "proposed_text": "Machine-learning platform",
                "rationale": "Normalize capitalization.",
                "support_evidence": (
                    {
                        "source_locator": "Experience",
                        "source_quote": "machine-learning platform",
                    },
                ),
            },
        ),
    )
    empty_match = ResumeJobMatchResult(
        overall_fit="moderate",
        summary="No unresolved requirements.",
    )

    issues = ResumeTailoringReviewGraph._validate_grounding(
        document=StoredResumeDocument(
            resume_version_id="pdf-layout",
            document_format="pdf",
            raw_bytes=_single_page_pdf("machine- learning platform"),
        ),
        match_result=empty_match,
        confirmed_facts=(),
        draft=draft,
    )

    assert issues == ()


def test_scanned_pdf_keeps_adjacent_candidates_visible_but_blocks_resume_writes() -> None:
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

    # The page is readable but carries no text. Adjacent evidence remains a
    # warning candidate, while support evidence cannot authorize a write.
    assert issues
    assert {issue.category for issue in issues} == {"unsupported_fact"}
    assert "blocking" in {issue.severity for issue in issues}
    assert "Led a team of 50" in {issue.source_quote for issue in issues}


@pytest.mark.parametrize("evidence_kind", ["adjacent", "alternative"])
@pytest.mark.parametrize("page, expected_severity", [(1, "warning"), (99, "blocking")])
def test_scanned_pdf_cited_page_preserves_candidate_downgrade(
    evidence_kind: str, page: int, expected_severity: str
) -> None:
    draft = copy.deepcopy(VALID_DRAFT)
    quote = "Built RAG systems"
    if evidence_kind == "adjacent":
        draft["gap_mitigations"][0]["adjacent_experience"][0]["page"] = page
    else:
        mitigation = draft["gap_mitigations"][0]
        mitigation["resolution_mode"] = "provide_evidence"
        mitigation["learning_plan"] = None
        mitigation["adjacent_experience"] = []
        mitigation["alternative_evidence"] = [{
            "description": "Existing RAG project",
            "status": "existing",
            "source_locator": "Experience",
            "source_quote": quote,
            "page": page,
        }]

    issues = ResumeTailoringReviewGraph._validate_grounding(
        document=StoredResumeDocument(
            resume_version_id="scan",
            document_format="pdf",
            raw_bytes=_scanned_pdf(),
        ),
        match_result=VALID_MATCH,
        confirmed_facts=(),
        draft=ResumeTailoringResult.model_validate(draft),
    )

    candidate_issues = [issue for issue in issues if issue.source_quote == quote and issue.change_index is None]
    assert len(candidate_issues) == 1
    assert candidate_issues[0].severity == expected_severity
    assert any(issue.change_index is not None and issue.severity == "blocking" for issue in issues)


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


def test_service_rejects_new_gap_without_mitigation_before_persisting(tmp_path) -> None:
    service, worker, _, stored_match = seed_service(tmp_path)
    legacy_shape = copy.deepcopy(VALID_DRAFT)
    legacy_shape.pop("gap_mitigations")
    worker.tailor = lambda **kwargs: ResumeTailoringResult.model_validate(legacy_shape)

    with pytest.raises(
        ResumeTailoringReviewBlockedError,
        match="missing mitigations for requirement IDs",
    ):
        service.create_draft(user_id="u1", match_id=stored_match.id)

    assert service._draft_store.list_with_unresolved_gaps(user_id="u1") == ()


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


def test_staged_draft_can_be_seen_before_background_review_but_not_finalized(tmp_path) -> None:
    reviewer = RecordingReviewer()
    service, _, _, stored_match = seed_service(tmp_path, reviewer=reviewer)

    draft = service.create_draft(
        user_id="u1", match_id=stored_match.id, run_automated_review=False
    )

    assert draft.automated_review is None
    with pytest.raises(ResumeTailoringNotReadyError, match="后台"):
        service.finalize_draft(user_id="u1", draft_id=draft.id)

    reviewed = service.review_draft_automatically(user_id="u1", draft_id=draft.id)
    assert reviewed.automated_review is not None
    assert reviewed.automated_review.status == "passed"
    assert len(reviewer.draft_calls) == 1

    service.review_draft(
        user_id="u1",
        draft_id=draft.id,
        accepted_change_indices=(1,),
        rejected_change_indices=(2,),
    )
    finalized = service.finalize_draft(user_id="u1", draft_id=draft.id)
    assert finalized.created is True


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
            update={
                "active_resume_job_match_id": stored_match.id,
                "tool_profile": "resume",
            }
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
            update={
                "active_resume_job_match_id": stored_match.id,
                "tool_profile": "resume",
            }
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
    assert observation.payload["gap_mitigations"][0]["priority"] == "P1"
    rebuilt_result = MainAgentRuntime._resume_tailoring_result(observation)
    assert rebuilt_result is not None
    assert rebuilt_result.gap_mitigations[0].learning_plan is not None
    assert (
        rebuilt_result.gap_mitigations[0].learning_plan.minimum_acceptable_level
        == "Can implement, test, and explain one concurrent HTTP service."
    )
    assert result.context.task.active_resume_tailoring_draft_id == observation.payload["draft_id"]
    assert result.context.task.resume_tailoring_status == "pending"
    serialized = observation.model_dump_json()
    assert "PRIVATE RESUME" not in serialized
    assert "PRIVATE JD" not in serialized
    rendered = MainAgentRuntime._assistant_message(result.tool_result)
    assert rendered.startswith("# 简历定制草稿 · 修订 1")
    assert "Built production RAG systems for knowledge retrieval." in rendered
    assert "面试诚实表达" in rendered
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


def test_a_weak_match_may_list_more_unresolved_gaps_than_short_list_fields() -> None:
    """Gaps scale with how weak the match is, so they are bounded like changes.

    The skill requires every missing or unclear requirement to stay an
    unresolved gap. A weak match with coarse questionnaire answers therefore
    produces more gaps than a strong one, and a real run was rejected for
    listing eleven — discarding a completed tailoring run because the model
    was thorough about what it could not support. The cap still exists; it is
    the cap on the work, not a cap on admitting ignorance.
    """

    draft = dict(
        VALID_DRAFT,
        unresolved_gaps=[f"gap {n}" for n in range(1, 12)],
        gap_mitigations=[],
    )

    result = ResumeTailoringResult.model_validate(draft)

    assert len(result.unresolved_gaps) == 11
    with pytest.raises(ValidationError):
        ResumeTailoringResult.model_validate(
            dict(
                VALID_DRAFT,
                unresolved_gaps=[f"gap {n}" for n in range(31)],
                gap_mitigations=[],
            )
        )


def test_a_revision_revises_the_draft_the_feedback_is_about() -> None:
    """The writer must see the draft its feedback describes.

    It is told to change only the stated issues and keep every unchallenged
    change. Handed the caller's starting point instead — None on a fresh
    tailoring request — it is asked to preserve something it was never shown
    and to locate changes by an index that refers to a draft it does not
    have, so it rewrites blind.
    """

    valid = ResumeTailoringResult.model_validate(VALID_DRAFT)
    ungrounded = valid.model_dump(mode="json")
    ungrounded["changes"][0]["support_evidence"][0]["source_quote"] = "Invented fact"
    first = ResumeTailoringResult.model_validate(ungrounded)

    class SequenceWorker:
        def __init__(self) -> None:
            self.outputs = [first, valid]
            self.calls = []

        def tailor(self, **kwargs):
            self.calls.append(kwargs)
            return self.outputs.pop(0)

    worker = SequenceWorker()
    ResumeTailoringReviewGraph(worker, RecordingReviewer()).run(
        document=StoredResumeDocument(
            resume_version_id="v1",
            document_format="text",
            raw_bytes=b"Built RAG systems and Python",
        ),
        jd_text="Build production RAG systems",
        match_result=VALID_MATCH,
    )

    assert worker.calls[0]["previous_draft"] is None
    assert worker.calls[1]["previous_draft"] == ResumeTailoringReviewGraph._canonicalize_evidence(
        canonicalize_gap_mitigations(first, VALID_MATCH),
        StoredResumeDocument(
            resume_version_id="v1",
            document_format="text",
            raw_bytes=b"Built RAG systems and Python",
        ),
    )


@pytest.mark.parametrize('effort', [
    '30 focused hours over 4 weeks, approximately 7–8 hours per week.',
    '20-30 hours (5 hours per week)',
    '共30小时，分4周完成，每周投入7-8小时',
])
def test_compound_learning_effort_keeps_all_quantities_bounded(effort):
    draft = copy.deepcopy(VALID_DRAFT)
    draft['gap_mitigations'][0]['learning_plan']['estimated_effort'] = effort
    assert gap_mitigation_errors(ResumeTailoringResult.model_validate(draft), VALID_MATCH) == ()


@pytest.mark.parametrize('effort', [
    '-2 hours', '30 hours over 0 weeks', '30 hours over 4-2 weeks',
    '30 hours over unlimited weeks', '30 hours and then as long as needed',
])
def test_compound_effort_cannot_hide_invalid_or_unbounded_duration(effort):
    draft = copy.deepcopy(VALID_DRAFT)
    draft['gap_mitigations'][0]['learning_plan']['estimated_effort'] = effort
    assert any('estimated effort' in e for e in gap_mitigation_errors(
        ResumeTailoringResult.model_validate(draft), VALID_MATCH,
    ))


def test_partial_coverage_uses_optional_evidence_without_becoming_a_missing_skill():
    from career_agent.agent.resume_job_match_contracts import ResumeMatchEvidence
    match = VALID_MATCH.model_copy(update={'requirements': (
        VALID_MATCH.requirements[0].model_copy(update={
            'status': 'partial',
            'resume_evidence': (ResumeMatchEvidence(source_locator='Experience', source_quote='Built RAG systems'),),
        }),
    )})
    draft = copy.deepcopy(VALID_DRAFT)
    draft['gap_mitigations'] = []
    canonical = canonicalize_gap_mitigations(ResumeTailoringResult.model_validate(draft), match)
    assert canonical.unresolved_gaps == ()
    assert gap_mitigation_errors(canonical, match) == ()
    draft['gap_mitigations'] = [_mitigation(
        resolution_mode='provide_evidence', learning_plan=None, alternative_evidence=[],
    )]
    assert gap_mitigation_errors(ResumeTailoringResult.model_validate(draft), match) == ()
    errors = gap_mitigation_errors(ResumeTailoringResult.model_validate(VALID_DRAFT), match)
    assert any('partial requirement' in e for e in errors)


def test_effort_repair_feedback_includes_actual_failure_not_only_generic_instruction():
    class RepairWorker:
        def __init__(self):
            self.calls = []

        def tailor(self, **kwargs):
            self.calls.append(kwargs)
            data = copy.deepcopy(VALID_DRAFT)
            if len(self.calls) == 1:
                data['gap_mitigations'][0]['learning_plan']['estimated_effort'] = '0 hours'
            else:
                feedback = ' '.join(kwargs['review_feedback'])
                assert 'estimated effort must use positive, ascending bounds' in feedback
                assert REQUIREMENT_ID in feedback
            return ResumeTailoringResult.model_validate(data)

    worker = RepairWorker()
    outcome = ResumeTailoringReviewGraph(worker, RecordingReviewer()).run(
        document=StoredResumeDocument(resume_version_id='v1', document_format='text', raw_bytes=b'Built RAG systems and Python'),
        jd_text='Production Go experience', match_result=VALID_MATCH,
    )
    assert outcome.trace.status == 'passed'
    assert len(worker.calls) == 2


@pytest.mark.parametrize('tier,status,expected_type,expected_priorities', [
    ('A', 'missing', 'strengthenable', ('P1', 'P2')),
    ('B', 'missing', 'strengthenable', ('P1', 'P2')),
    ('C', 'missing', 'strengthenable', ('P1', 'P2')),
    ('S', 'missing', 'hard_blocker', ('P0',)),
    ('S', 'unclear', 'strengthenable', ('P0', 'P1', 'P2')),
])
def test_writer_and_reviewer_receive_the_same_authoritative_mitigation_rules(
    tier, status, expected_type, expected_priorities,
):
    from career_agent.agent.resume_tailoring_contracts import mitigation_policy
    match = VALID_MATCH.model_copy(update={'requirements': (
        VALID_MATCH.requirements[0].model_copy(update={'tier': tier, 'status': status}),
    )})
    policy = mitigation_policy(match)
    assert policy[0]['gap_type'] == expected_type
    assert policy[0]['priorities'] == expected_priorities
    encoded = json.dumps(policy, ensure_ascii=False)
    context = DeepAgentResumeTailoringWorker._context_text(
        jd_text='Go', match_result=match, confirmed_facts=(), tailoring_goal=None,
        user_feedback=None, review_feedback=(), previous_draft=None,
    )
    assert encoded in context
    client = FakeResponsesClient({'verdict': 'pass', 'summary': 'Grounded.', 'issues': []})
    reviewer = OpenAIResumeTailoringReviewer(OpenAICompatibleAgentConfig(
        endpoint='https://example.test/v1/chat/completions', api_key='secret', model='test',
    ), client=client)
    reviewer.review_draft(
        document=StoredResumeDocument(resume_version_id='v1', document_format='text', raw_bytes=b'Built RAG systems'),
        jd_text='Go', match_result=match, draft=ResumeTailoringResult.model_validate(VALID_DRAFT),
    )
    assert any(encoded in item.get('text', '') for item in client.calls[0]['input'][0]['content'])


def test_reviewer_repairs_zero_change_index_with_structural_feedback():
    from types import SimpleNamespace
    invalid = {
        'verdict': 'revise', 'summary': 'Artifact needs acceptance criteria.',
        'issues': [{'category': 'jd_misalignment', 'severity': 'blocking',
                    'change_index': 0, 'explanation': 'The artifact has no checkable outcome.'}],
    }
    valid = copy.deepcopy(invalid)
    valid['issues'][0]['change_index'] = None

    class Client:
        def __init__(self):
            self.responses = self
            self.calls = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            if len(self.calls) == 2:
                assert 'change_index' in kwargs['instructions']
                assert 'greater_than_equal' in kwargs['instructions']
                assert 'The artifact has no checkable outcome.' not in kwargs['instructions']
            return SimpleNamespace(output_text=json.dumps(invalid if len(self.calls) == 1 else valid))

    client = Client()
    reviewer = OpenAIResumeTailoringReviewer(OpenAICompatibleAgentConfig(
        endpoint='https://example.test/v1/chat/completions', api_key='secret', model='test',
    ), client=client)
    result = reviewer.review_draft(
        document=StoredResumeDocument(resume_version_id='v1', document_format='text', raw_bytes=b'Built RAG systems'),
        jd_text='Go', match_result=VALID_MATCH, draft=ResumeTailoringResult.model_validate(VALID_DRAFT),
    )
    assert result.verdict == 'revise'
    assert result.issues[0].change_index is None
    assert len(client.calls) == 2


def test_empty_projection_does_not_erase_unbound_historical_gaps():
    match = VALID_MATCH.model_copy(update={'requirements': (
        VALID_MATCH.requirements[0].model_copy(update={'requirement_id': None}),
    )})
    draft = copy.deepcopy(VALID_DRAFT)
    draft['gap_mitigations'] = []
    canonical = canonicalize_gap_mitigations(ResumeTailoringResult.model_validate(draft), match)
    assert canonical.unresolved_gaps == ('Go is not stated',)
    assert gap_mitigation_errors(canonical, match)
