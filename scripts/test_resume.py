"""Quick integration test: parse a real resume file end-to-end.

Usage:
    uv run python scripts/test_resume.py <path_to_resume>

Requires RESUME_MODEL, RESUME_API_KEY, RESUME_BASE_URL in .env.
"""

import asyncio
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

from career_agent.models import model_from_env
from career_agent.resumes.models import ResumeData
from career_agent.resumes.parser import extract_text_from_file, parse_resume_with_llm


async def main() -> None:
    load_dotenv()

    if len(sys.argv) < 2:
        print("Usage: uv run python scripts/test_resume.py <path_to_resume>")
        sys.exit(1)

    file_path = Path(sys.argv[1])
    print(f"Loading resume: {file_path}")

    # Step 1: Extract raw text
    raw_text = extract_text_from_file(file_path)
    print(f"\n===== Step 1: PDF Text Extraction ({len(raw_text)} chars) =====")
    print(raw_text[:500])
    print("... (truncated)")

    # Step 2: LLM structured parse
    model = model_from_env("resume", temperature=0)
    print(f"\n===== Step 2: LLM Parse (model={model.__class__.__name__}) =====")

    resume = await parse_resume_with_llm(raw_text, model)

    # Step 3: Show Pydantic result
    print("\n===== Step 3: Pydantic ResumeData =====")
    print(f"type: {type(resume).__name__}")
    print(f"is ResumeData: {isinstance(resume, ResumeData)}")
    print(json.dumps(resume.model_dump(exclude={"raw_text"}), ensure_ascii=False, indent=2))

    print("\n===== Step 4: to_facts() =====")
    for fact in resume.to_facts():
        print(f"  - {fact}")


if __name__ == "__main__":
    asyncio.run(main())
