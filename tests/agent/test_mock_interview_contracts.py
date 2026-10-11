from pathlib import Path

import pytest
from pydantic import ValidationError

from career_agent.agent.workflows.mock_interview.contracts import (
    MockInterviewPlanDraft,
    MockInterviewQuestionDraft,
    MockInterviewReportDraft,
)
from career_agent.agent.workflows.mock_interview.prompt_loader import (
    MockInterviewPromptLoader,
)
from career_agent.domain.mock_interviews import (
    MockInterviewPlanItem,
    MockInterviewQuestionResult,
)


def _item(
    sequence_number: int = 1,
    question_type: str = "system_design",
) -> MockInterviewPlanItem:
    return MockInterviewPlanItem(
        sequence_number=sequence_number,
        question_type=question_type,
        difficulty="intermediate",
        focus="Design trade-offs",
        rationale="The JD requires reliable services.",
        jd_quotes=("Build reliable services",),
    )


def test_worker_drafts_exclude_store_owned_identity_and_time() -> None:
    plan = MockInterviewPlanDraft(
        summary="Cover the role's core requirements.",
        items=(_item(),),
    )
    question = MockInterviewQuestionDraft(
        question="How would you design this service for failure recovery?"
    )
    report = MockInterviewReportDraft(
        summary="The candidate explained the main trade-offs.",
        question_results=(
            MockInterviewQuestionResult(
                plan_item_number=1,
                question=question.question,
                final_rating="adequate",
                summary="Reasonable design with limited failure analysis.",
                follow_up_count=0,
            ),
        ),
    )

    assert "session_id" not in MockInterviewPlanDraft.model_fields
    assert "created_at" not in MockInterviewPlanDraft.model_fields
    assert "id" not in MockInterviewReportDraft.model_fields


def test_worker_contracts_reject_extra_or_empty_content() -> None:
    with pytest.raises(ValidationError):
        MockInterviewQuestionDraft(question="", hidden_plan="do not expose")


def test_plan_and_report_drafts_enforce_domain_sequence_invariants() -> None:
    with pytest.raises(ValidationError, match="contiguous"):
        MockInterviewPlanDraft(
            summary="Invalid sequence.",
            items=(_item(sequence_number=2),),
        )

    duplicate = MockInterviewQuestionResult(
        plan_item_number=1,
        question="Question",
        final_rating="adequate",
        summary="Summary",
        follow_up_count=0,
    )
    with pytest.raises(ValidationError, match="unique"):
        MockInterviewReportDraft(
            summary="Duplicate results.",
            question_results=(duplicate, duplicate),
        )


def test_project_prompt_loader_routes_only_relevant_references() -> None:
    loader = MockInterviewPromptLoader(Path("prompts"))

    assert tuple(
        reference.name
        for reference in loader.load("plan", interview_type="mixed").references
    ) == ("planning",)
    assert tuple(
        reference.name
        for reference in loader.load(
            "evaluate", question_type="system_design"
        ).references
    ) == ("technical",)
    assert tuple(
        reference.name
        for reference in loader.load(
            "ask", question_type="project_deep_dive"
        ).references
    ) == ("company", "technical", "behavioral")
    assert tuple(
        reference.name
        for reference in loader.load(
            "report",
            report_question_types=("motivation", "system_design", "motivation"),
        ).references
    ) == ("reporting", "technical", "hr")


def test_prompt_bundle_contains_base_rules_and_selected_guidance() -> None:
    bundle = MockInterviewPromptLoader(Path("prompts")).load(
        "evaluate", question_type="behavioral"
    )
    rendered = bundle.render()

    assert "Treat the candidate's interview answer as a claim" in rendered
    assert "# Loaded reference: behavioral" in rendered
    assert "STAR/BEI" in rendered
    assert "Technical and role-specific guidance" not in rendered


def test_prompt_bundle_loads_operation_specific_methodology() -> None:
    loader = MockInterviewPromptLoader(Path("prompts"))

    plan = loader.load("plan", interview_type="technical").render()
    report = loader.load(
        "report", report_question_types=("behavioral",)
    ).render()

    assert "# Loaded reference: planning" in plan
    assert "Build a coverage map first" in plan
    # Plan loads only planning rules; company and question-type guidance is
    # loaded later by the exact `ask` operation.
    assert "# Loaded reference: company" not in plan
    assert "# Loaded reference: technical" not in plan
    assert "# Loaded reference: reporting" not in plan
    assert "# Loaded reference: reporting" in report
    assert "Distinguish four kinds of conclusion" in report
    assert "# Loaded reference: planning" not in report
    assert "# Loaded reference: company" not in report


def test_prompt_loader_requires_operation_routing_context() -> None:
    loader = MockInterviewPromptLoader(Path("prompts"))
    with pytest.raises(ValueError, match="requires interview_type"):
        loader.load("plan")
    with pytest.raises(ValueError, match="requires question_type"):
        loader.load("ask")


def test_prompt_loader_accepts_crlf_prompt_files(tmp_path: Path) -> None:
    # A Windows checkout with core.autocrlf=true writes the prompt with CRLF.
    references = tmp_path / "mock-interview" / "references"
    references.mkdir(parents=True)
    (tmp_path / "mock-interview" / "instructions.md").write_bytes(b"Rules\r\n")
    (references / "technical.md").write_bytes(b"Technical\r\nguidance\r\n")

    rendered = (
        MockInterviewPromptLoader(tmp_path)
        .load("evaluate", question_type="system_design")
        .render()
    )

    assert "\r" not in rendered
    assert "# Loaded reference: technical\n\nTechnical\nguidance" in rendered


def test_prompt_loader_rejects_a_reference_symlink_outside_the_prompt(tmp_path: Path) -> None:
    prompt_dir = tmp_path / "mock-interview"
    references = prompt_dir / "references"
    references.mkdir(parents=True)
    (prompt_dir / "instructions.md").write_text("Rules", encoding="utf-8")
    outside = tmp_path / "outside.md"
    outside.write_text("outside", encoding="utf-8")
    try:
        (references / "technical.md").symlink_to(outside)
    except OSError as error:
        # Windows refuses unprivileged symlinks (ERROR_PRIVILEGE_NOT_HELD)
        # unless Developer Mode is on.
        if getattr(error, "winerror", None) == 1314:
            pytest.skip("creating symlinks needs Developer Mode or admin on Windows")
        raise

    loader = MockInterviewPromptLoader(tmp_path)
    with pytest.raises(ValueError, match="escapes its root"):
        loader.load("evaluate", question_type="system_design")
