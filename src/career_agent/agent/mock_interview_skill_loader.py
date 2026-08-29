from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Literal

from career_agent.domain.mock_interviews import (
    MockInterviewQuestionType,
    MockInterviewType,
)


MockInterviewOperation = Literal["plan", "ask", "evaluate", "report"]
MockInterviewReferenceName = Literal[
    "planning",
    "company",
    "technical",
    "behavioral",
    "hr",
    "reporting",
]


@dataclass(frozen=True)
class MockInterviewSkillReference:
    name: MockInterviewReferenceName
    content: str


@dataclass(frozen=True)
class MockInterviewSkillBundle:
    operation: MockInterviewOperation
    instructions: str
    references: tuple[MockInterviewSkillReference, ...]

    def render(self) -> str:
        sections = [self.instructions]
        sections.extend(
            f"# Loaded reference: {reference.name}\n\n{reference.content}"
            for reference in self.references
        )
        return "\n\n".join(sections)


class MockInterviewSkillLoader:
    """Loads the project-owned mock-interview Skill with bounded references.

    Reference routing is code-owned rather than model-owned: untrusted JD,
    resume, or answer text can never choose a filesystem path. The returned
    bundle is intended for a worker's system instructions, not Main Agent state.
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
    _INTERVIEW_REFERENCES: dict[
        MockInterviewType, tuple[MockInterviewReferenceName, ...]
    ] = {
        "technical": ("technical",),
        "role_specific": ("technical", "behavioral"),
        "behavioral": ("behavioral",),
        "hr": ("hr",),
        "mixed": _CONTENT_REFERENCE_ORDER,
    }
    _MAX_FILE_BYTES = 128_000

    def __init__(self, skills_root: Path) -> None:
        self._skill_dir = (skills_root.expanduser().resolve() / "mock-interview")
        self._skill_file = self._skill_dir / "SKILL.md"
        if not self._skill_dir.is_dir() or not self._skill_file.is_file():
            raise ValueError(
                "Mock interview skill is missing; expected "
                f"{self._skill_file}"
            )
        instructions = self._read_confined(self._skill_file)
        if not re.search(r"(?m)^name:\s*mock-interview\s*$", instructions):
            raise ValueError("Mock interview SKILL.md has the wrong or missing name")
        self._instructions = instructions

    def load(
        self,
        operation: MockInterviewOperation,
        *,
        interview_type: MockInterviewType | None = None,
        question_type: MockInterviewQuestionType | None = None,
        report_question_types: tuple[MockInterviewQuestionType, ...] = (),
    ) -> MockInterviewSkillBundle:
        names = self._route(
            operation,
            interview_type=interview_type,
            question_type=question_type,
            report_question_types=report_question_types,
        )
        references = tuple(
            MockInterviewSkillReference(
                name=name,
                content=self._read_confined(
                    self._skill_dir / "references" / f"{name}.md"
                ),
            )
            for name in names
        )
        return MockInterviewSkillBundle(
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
                raise ValueError("plan skill loading requires interview_type")
            return (
                "planning",
                "company",
                *self._INTERVIEW_REFERENCES[interview_type],
            )
        if operation in {"ask", "evaluate"}:
            if question_type is None:
                raise ValueError(
                    f"{operation} skill loading requires question_type"
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
            resolved.relative_to(self._skill_dir)
        except ValueError as error:
            raise ValueError("Mock interview skill reference escapes its root") from error
        if not resolved.is_file():
            raise ValueError(f"Mock interview skill reference is missing: {resolved.name}")
        if resolved.stat().st_size > self._MAX_FILE_BYTES:
            raise ValueError(
                f"Mock interview skill file exceeds {self._MAX_FILE_BYTES} bytes: "
                f"{resolved.name}"
            )
        try:
            content = resolved.read_text(encoding="utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(
                f"Mock interview skill file must be UTF-8: {resolved.name}"
            ) from error
        if not content.strip():
            raise ValueError(f"Mock interview skill file is empty: {resolved.name}")
        return content.strip()
