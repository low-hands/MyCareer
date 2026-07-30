"""PDF/text extraction and LLM structuring for resumes.

Domain models live in :mod:`career_agent.resumes.models`; this module owns only
the conversion from a source file into those models.
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.language_models import BaseChatModel

from career_agent.prompts import RESUME_STRUCTURE_PROMPT
from career_agent.resumes.models import ResumeData


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
            response = await model.with_structured_output(ResumeData, **kwargs).ainvoke([HumanMessage(content=prompt)])
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
        raise ValueError(f"Failed to parse resume JSON from model output: {exc}") from exc

    # Coerce common type mismatches (model may return str instead of list)
    _list_fields = [
        "skills",
        "technologies",
        "stated_target_roles",
        "stated_target_cities",
        "work_experience",
        "project_experience",
        "publications",
        "awards",
        "education",
    ]
    for field in _list_fields:
        val = obj.get(field)
        if isinstance(val, str):
            obj[field] = [val] if val else []

    # Coerce int → str for year/duration fields in nested objects
    _nested_list_fields = [
        "work_experience",
        "project_experience",
        "publications",
        "awards",
        "education",
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
        raise ValueError(f"Failed to validate resume data: {exc}") from exc


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
