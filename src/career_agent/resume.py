"""Resume parsing and structured data extraction.

This module handles reading resume files (PDF/text) and converting them
into structured ResumeData. The parsing logic is decoupled from CLI/API
so it can be reused by any interface.
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

from career_agent.prompts import RESUME_STRUCTURE_PROMPT


class WorkExperience(BaseModel):
    """Work or internship experience."""

    company: str = ""
    title: str = ""
    duration: str = ""
    highlights: list[str] = Field(default_factory=list)


class ProjectExperience(BaseModel):
    """A project entry (academic, personal, or open-source)."""

    name: str = ""
    role: str = ""
    duration: str = ""
    highlights: list[str] = Field(default_factory=list)


class Publication(BaseModel):
    """A paper, article, or conference publication."""

    title: str = ""
    venue: str = ""
    year: str = ""


class Award(BaseModel):
    """A competition, award, or certification."""

    name: str = ""
    organization: str = ""


class Education(BaseModel):
    """A single education entry."""

    school: str = ""
    degree: str = ""
    major: str = ""
    duration: str = ""


class ResumeData(BaseModel):
    """Structured resume data extracted from a file.

    Every value in this model must be grounded in the source document.
    Inferred career direction belongs in a separate analysis step.
    """

    name: str = ""
    email: str = ""
    phone: str = ""
    summary: str = ""
    skills: list[str] = Field(default_factory=list)
    tech_stack: list[str] = Field(default_factory=list)
    work_experience: list[WorkExperience] = Field(default_factory=list)
    project_experience: list[ProjectExperience] = Field(default_factory=list)
    publications: list[Publication] = Field(default_factory=list)
    awards: list[Award] = Field(default_factory=list)
    education: list[Education] = Field(default_factory=list)
    stated_target_roles: list[str] = Field(default_factory=list)
    stated_target_cities: list[str] = Field(default_factory=list)
    stated_expected_salary: str = ""
    raw_text: str = Field(default="", exclude=True)

    def to_facts(self) -> list[str]:
        """Convert resume data into concise facts for prompt injection."""
        facts: list[str] = []
        if self.name:
            facts.append(f"用户姓名: {self.name}")
        if self.skills:
            facts.append(f"核心技能: {', '.join(self.skills)}")
        if self.tech_stack:
            facts.append(f"技术栈: {', '.join(self.tech_stack)}")
        if self.stated_target_roles:
            facts.append(f"简历明确目标岗位: {', '.join(self.stated_target_roles)}")
        if self.stated_target_cities:
            facts.append(f"简历明确目标城市: {', '.join(self.stated_target_cities)}")
        if self.stated_expected_salary:
            facts.append(f"简历明确期望薪资: {self.stated_expected_salary}")
        if self.education:
            entries = [
                " ".join(
                    part
                    for part in (edu.school, edu.degree, edu.major, edu.duration)
                    if part
                )
                for edu in self.education
            ]
            facts.append(f"教育经历: {'; '.join(entries)}")
        if self.work_experience:
            entries = [
                _format_experience(w.company, w.title, w.duration, w.highlights)
                for w in self.work_experience
            ]
            facts.append(f"工作/实习经历: {'; '.join(entries)}")
        if self.project_experience:
            entries = [
                _format_experience(p.name, p.role, p.duration, p.highlights)
                for p in self.project_experience
            ]
            facts.append(f"项目经历: {'; '.join(entries)}")
        if self.publications:
            entries = [
                " | ".join(part for part in (pub.title, pub.venue, pub.year) if part)
                for pub in self.publications
            ]
            facts.append(f"论文/发表: {'; '.join(entries)}")
        if self.awards:
            entries = [
                " | ".join(part for part in (award.name, award.organization) if part)
                for award in self.awards
            ]
            facts.append(f"竞赛/奖项: {'; '.join(entries)}")
        return facts


def _format_experience(
    organization: str,
    role: str,
    duration: str,
    highlights: list[str],
) -> str:
    """Render one structured experience without dropping extracted details."""

    heading = " | ".join(part for part in (organization, role, duration) if part)
    detail = "；".join(highlight for highlight in highlights if highlight)
    if heading and detail:
        return f"{heading}: {detail}"
    return heading or detail


def extract_text_from_pdf(file_path: Path) -> str:
    """Extract raw text from a PDF file using PyMuPDF for better CJK support."""
    import fitz  # PyMuPDF

    doc = fitz.open(str(file_path))
    pages = [page.get_text() for page in doc]
    doc.close()
    return "\n".join(pages).strip()


def extract_text_from_file(file_path: Path) -> str:
    """Extract text from a file, supporting PDF and plain text."""
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return extract_text_from_pdf(file_path)
    # Treat everything else as plain text
    return file_path.read_text(encoding="utf-8").strip()


async def parse_resume_with_llm(
    raw_text: str,
    model: BaseChatModel,
) -> ResumeData:
    """Use LLM to structure raw resume text into ResumeData.

    Tries structured output first (json_schema / function_calling).
    Falls back to manual JSON parsing if the API doesn't support them.
    """
    from langchain_core.messages import HumanMessage

    prompt = RESUME_STRUCTURE_PROMPT.format(text=raw_text[:8000])

    # Try structured output (works with OpenAI and compatible APIs)
    for method in (None, "function_calling"):
        try:
            kwargs = {"method": method} if method else {}
            response = await model.with_structured_output(
                ResumeData, **kwargs
            ).ainvoke([HumanMessage(content=prompt)])
            response.raw_text = raw_text
            return response
        except Exception:
            continue

    # Fallback: call model normally, parse JSON from text
    from langchain_core.messages import AIMessage

    ai: AIMessage = await model.ainvoke([HumanMessage(content=prompt)])
    data = _parse_json_from_text(ai.content or "")
    data.raw_text = raw_text
    return data


def _parse_json_from_text(text: str) -> ResumeData:
    """Extract and parse JSON from model text output."""
    import json
    import re

    # Try direct parse
    text = text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Failed to parse resume JSON from model output: {exc}"
        ) from exc

    # Coerce common type mismatches (model may return str instead of list)
    _list_fields = [
        "skills", "tech_stack", "stated_target_roles", "stated_target_cities",
        "work_experience", "project_experience", "publications",
        "awards", "education",
    ]
    for field in _list_fields:
        val = obj.get(field)
        if isinstance(val, str):
            obj[field] = [val] if val else []

    # Coerce int → str for year/duration fields in nested objects
    _nested_list_fields = [
        "work_experience", "project_experience", "publications",
        "awards", "education",
    ]
    for field in _nested_list_fields:
        for item in obj.get(field, []):
            if isinstance(item, dict):
                for key, val in item.items():
                    if isinstance(val, int):
                        item[key] = str(val)
                highlights = item.get("highlights")
                if isinstance(highlights, str):
                    item["highlights"] = [highlights] if highlights else []

    try:
        return ResumeData.model_validate(obj)
    except Exception as exc:
        raise ValueError(
            f"Failed to validate resume data: {exc}"
        ) from exc


async def load_resume(file_path: str | Path, model: BaseChatModel) -> ResumeData:
    """Load a resume file and return structured ResumeData.

    This is the main entry point for resume handling.
    CLI and API both call this function.
    """
    path = Path(file_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Resume file not found: {path}")

    raw_text = extract_text_from_file(path)
    if not raw_text:
        raise ValueError(f"Resume file is empty: {path}")

    return await parse_resume_with_llm(raw_text, model)
