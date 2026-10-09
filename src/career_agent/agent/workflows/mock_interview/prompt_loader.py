from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from career_agent.domain.mock_interviews import (
    MockInterviewQuestionType,
    MockInterviewType,
)


MockInterviewOperation = Literal["plan", "ask", "follow_up", "evaluate", "report"]
MockInterviewReferenceName = Literal[
    "planning",
    "company",
    "technical",
    "behavioral",
    "hr",
    "reporting",
]


@dataclass(frozen=True)
class MockInterviewPromptReference:
    name: MockInterviewReferenceName
    content: str


@dataclass(frozen=True)
class MockInterviewPromptBundle:
    operation: MockInterviewOperation
    instructions: str
    references: tuple[MockInterviewPromptReference, ...]

    def render(self) -> str:
        sections = [self.instructions]
        sections.extend(
            f"# Loaded reference: {reference.name}\n\n{reference.content}"
            for reference in self.references
        )
        return "\n\n".join(sections)


class MockInterviewPromptLoader:
    """Loads the project-owned mock-interview prompt with bounded references.

    This is a prompt, not a Skill: code chooses which references each operation
    receives, so untrusted JD, resume, or answer text can never choose a
    filesystem path. The returned bundle is intended for a worker's system
    instructions, not Main Agent state.
    """

    _CONTENT_REFERENCE_ORDER: tuple[MockInterviewReferenceName, ...] = (
        "technical",
        "behavioral",
        "hr",
    )
    _QUESTION_REFERENCES: dict[
        MockInterviewQuestionType, tuple[MockInterviewReferenceName, ...]
    ] = {
        "introduction": ("behavioral",),
        "knowledge": ("technical",),
        "problem_solving": ("technical",),
        "system_design": ("technical",),
        "project_deep_dive": ("technical", "behavioral"),
        "role_scenario": ("technical", "behavioral"),
        "behavioral": ("behavioral",),
        "motivation": ("hr",),
        "career_planning": ("hr",),
    }
    _MAX_FILE_BYTES = 128_000

    def __init__(self, prompts_root: Path) -> None:
        self._prompt_dir = (prompts_root.expanduser().resolve() / "mock-interview")
        self._instructions_file = self._prompt_dir / "instructions.md"
        if not self._prompt_dir.is_dir() or not self._instructions_file.is_file():
            raise ValueError(
                "Mock interview prompt is missing; expected "
                f"{self._instructions_file}"
            )
        self._instructions = self._read_confined(self._instructions_file)

    def load(
        self,
        operation: MockInterviewOperation,
        *,
        interview_type: MockInterviewType | None = None,
        question_type: MockInterviewQuestionType | None = None,
        report_question_types: tuple[MockInterviewQuestionType, ...] = (),
    ) -> MockInterviewPromptBundle:
        names = self._route(
            operation,
            interview_type=interview_type,
            question_type=question_type,
            report_question_types=report_question_types,
        )
        references = tuple(
            MockInterviewPromptReference(
                name=name,
                content=self._read_confined(
                    self._prompt_dir / "references" / f"{name}.md"
                ),
            )
            for name in names
        )
        return MockInterviewPromptBundle(
            operation=operation,
            instructions=self._instructions,
            references=references,
        )

    def _route(
        self,
        operation: MockInterviewOperation,
        *,
        interview_type: MockInterviewType | None,
        question_type: MockInterviewQuestionType | None,
        report_question_types: tuple[MockInterviewQuestionType, ...],
    ) -> tuple[MockInterviewReferenceName, ...]:
        if operation == "plan":
            if interview_type is None:
                raise ValueError("plan prompt loading requires interview_type")
            # Planning only needs coverage and sequencing rules. Company and
            # question-type references are loaded later by the exact `ask`
            # operation; including them here makes lightweight compatible
            # models spend their whole timeout before returning a bounded plan.
            return ("planning",)
        if operation in {"ask", "follow_up", "evaluate"}:
            if question_type is None:
                raise ValueError(
                    f"{operation} prompt loading requires question_type"
                )
            references = self._QUESTION_REFERENCES[question_type]
            return ("company", *references) if operation == "ask" else references
        if operation == "report":
            selected = {
                reference
                for item_type in report_question_types
                for reference in self._QUESTION_REFERENCES[item_type]
            }
            return (
                "reporting",
                *(
                    name
                    for name in self._CONTENT_REFERENCE_ORDER
                    if name in selected
                ),
            )
        raise ValueError(f"Unsupported mock interview operation: {operation}")

    def _read_confined(self, path: Path) -> str:
        resolved = path.resolve()
        try:
            resolved.relative_to(self._prompt_dir)
        except ValueError as error:
            raise ValueError("Mock interview prompt reference escapes its root") from error
        if not resolved.is_file():
            raise ValueError(f"Mock interview prompt reference is missing: {resolved.name}")
        if resolved.stat().st_size > self._MAX_FILE_BYTES:
            raise ValueError(
                f"Mock interview prompt file exceeds {self._MAX_FILE_BYTES} bytes: "
                f"{resolved.name}"
            )
        try:
            content = resolved.read_text(encoding="utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(
                f"Mock interview prompt file must be UTF-8: {resolved.name}"
            ) from error
        if not content.strip():
            raise ValueError(f"Mock interview prompt file is empty: {resolved.name}")
        return content.strip()
