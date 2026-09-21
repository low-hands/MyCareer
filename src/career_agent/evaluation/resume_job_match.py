"""Live repeated-sample probe for the 096 resume/JD judgment rubric.

This deliberately does not use cassettes: the point is to measure whether a
real configured model keeps the same requirement mapping and fit band across
independent calls.
"""

from __future__ import annotations

from dataclasses import dataclass

from career_agent.agent.job_analysis_contracts import TieredRequirement
from career_agent.agent.openai_compatible_client import OpenAICompatibleAgentConfig
from career_agent.agent.openai_resume_job_match_worker import OpenAIResumeJobMatchWorker
from career_agent.agent.resume_job_match_contracts import ResumeJobMatchResult
from career_agent.services.resume_job_match import ResumeJobMatchService
from career_agent.storage.resumes import StoredResumeDocument


@dataclass(frozen=True)
class Sample:
    index: int
    result: ResumeJobMatchResult | None
    error: str | None


_JD = (
    "We need a backend engineer with 3+ years of Python, production PostgreSQL, "
    "and experience operating services. Kubernetes is a plus."
)
_RESUME = (
    "Experience\n"
    "- Built and operated Python services for four years.\n"
    "- Designed PostgreSQL schemas and production migrations.\n"
    "- On-call ownership for Linux services."
)
_REQUIREMENTS = (
    TieredRequirement(
        requirement_id="job_requirement_00000000000000000001",
        text="3+ years of Python",
        tier="S",
        kind="fact",
        jd_quote="3+ years of Python",
    ),
    TieredRequirement(
        requirement_id="job_requirement_00000000000000000002",
        text="production PostgreSQL",
        tier="A",
        kind="fact",
        jd_quote="production PostgreSQL",
    ),
    TieredRequirement(
        requirement_id="job_requirement_00000000000000000003",
        text="Kubernetes",
        tier="B",
        kind="fact",
        jd_quote="Kubernetes is a plus",
    ),
)


def run_samples(
    config: OpenAICompatibleAgentConfig,
    *,
    sample_count: int = 6,
) -> tuple[Sample, ...]:
    if not 1 <= sample_count <= 20:
        raise ValueError("sample_count must be between 1 and 20")
    worker = OpenAIResumeJobMatchWorker(config)
    document = StoredResumeDocument(
        resume_version_id="eval_resume_096",
        document_format="text",
        raw_bytes=_RESUME.encode("utf-8"),
    )
    samples: list[Sample] = []
    for index in range(1, sample_count + 1):
        try:
            raw = worker.match(
                document=document,
                jd_text=_JD,
                tiered_requirements=_REQUIREMENTS,
            )
            # Reuse the production binding and deterministic rubric, without
            # constructing stores or writing a match row during an evaluation.
            checked = ResumeJobMatchService._bind_and_grade_requirements(
                result=raw,
                requirements=_REQUIREMENTS,
            )
            samples.append(Sample(index=index, result=checked, error=None))
        except Exception as error:  # report every sample; one bad call must not hide the rest
            samples.append(Sample(index=index, result=None, error=type(error).__name__))
    return tuple(samples)
