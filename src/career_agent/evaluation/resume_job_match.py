"""Live repeated-sample probe for the 096 resume/JD judgment rubric.

This deliberately does not use cassettes: the point is to measure whether a
real configured model keeps the same requirement mapping and fit band across
independent calls.
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
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


@dataclass(frozen=True)
class BoundaryCase:
    name: str
    resume: str
    requirements: tuple[TieredRequirement, ...]
    expected_fit: str
    expected_statuses: tuple[str | tuple[str, ...], ...]


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


def _req(number: int, text: str, tier: str, kind: str, quote: str) -> TieredRequirement:
    return TieredRequirement(
        requirement_id=f"job_requirement_{number:020x}",
        text=text,
        tier=tier,  # type: ignore[arg-type]
        kind=kind,  # type: ignore[arg-type]
        jd_quote=quote,
    )


BOUNDARY_CASES = (
    BoundaryCase(
        "strong_all_core_supported",
        "Four years Python services. PostgreSQL migrations in production. Kubernetes deployments.",
        (
            _req(101, "Python", "S", "fact", "Python"),
            _req(102, "PostgreSQL", "A", "fact", "PostgreSQL"),
        ),
        "strong",
        ("matched", "matched"),
    ),
    BoundaryCase(
        "weak_s_hard_gate_missing",
        "Four years Python services.",
        (
            _req(103, "Go", "S", "fact", "Go"),
            _req(104, "Python", "A", "fact", "Python"),
        ),
        "weak",
        ("missing", "matched"),
    ),
    BoundaryCase(
        "moderate_a_mixed",
        "Built production Python services.",
        (
            _req(105, "Python", "A", "fact", "Python"),
            _req(106, "PostgreSQL", "A", "fact", "PostgreSQL"),
        ),
        "moderate",
        ("matched", "missing"),
    ),
    BoundaryCase(
        "weak_a_gaps_outnumber_support",
        "Built production Python services.",
        (
            _req(107, "Python", "A", "fact", "Python"),
            _req(108, "Go", "A", "fact", "Go"),
            _req(109, "Kubernetes", "A", "fact", "Kubernetes"),
        ),
        "weak",
        ("matched", "missing", "missing"),
    ),
    BoundaryCase(
        "insufficient_unclear_majority",
        "Built production Python services.",
        (
            _req(110, "Python", "A", "fact", "Python"),
            _req(
                111,
                "appropriate domain background",
                "A",
                "fact",
                "appropriate domain background",
            ),
            _req(
                112,
                "sufficient scale experience",
                "A",
                "fact",
                "sufficient scale experience",
            ),
            _req(
                113,
                "strong cultural fit",
                "A",
                "fact",
                "strong cultural fit",
            ),
        ),
        "insufficient_evidence",
        ("matched", "unclear", "unclear", "unclear"),
    ),
    BoundaryCase(
        "inference_support_is_moderate",
        (
            "Owned Python services in production. Participated in on-call "
            "rotations and incident response."
        ),
        (
            _req(
                114,
                "reliable service ownership",
                "A",
                "inference",
                "operating reliable services",
            ),
            _req(115, "on-call maturity", "A", "inference", "on-call maturity"),
        ),
        "moderate",
        (("matched", "partial"), ("matched", "partial")),
    ),
    BoundaryCase(
        "inference_only_without_support_is_insufficient",
        "Worked on documentation and code review.",
        (
            _req(
                116,
                "reliable service ownership",
                "A",
                "inference",
                "operating reliable services",
            ),
            _req(117, "on-call maturity", "A", "inference", "on-call maturity"),
        ),
        "insufficient_evidence",
        ("missing", "missing"),
    ),
    BoundaryCase(
        "optional_gap_does_not_lower_core_fit",
        "Four years Python services and production PostgreSQL.",
        (
            _req(118, "Python", "S", "fact", "Python"),
            _req(119, "PostgreSQL", "A", "fact", "PostgreSQL"),
            _req(120, "Kubernetes", "B", "fact", "Kubernetes"),
        ),
        "strong",
        ("matched", "matched", "missing"),
    ),
)


_SUMMARY_FIT_BAND = re.compile(
    r"\boverall[ -]fit\s+(?:is\s+)?(?:therefore\s+)?"
    r"(strong|moderate|weak|insufficient(?:_evidence| evidence)?)\b",
    re.IGNORECASE,
)


def _summary_fit_band(summary: str) -> str | None:
    match = _SUMMARY_FIT_BAND.search(summary)
    if match is None:
        return None
    return match.group(1).lower().replace(" ", "_")


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


def run_boundary_suite(
    config: OpenAICompatibleAgentConfig,
    *,
    sample_count: int = 6,
    case_names: tuple[str, ...] | None = None,
) -> dict[str, tuple[Sample, ...]]:
    """Run every 096 rubric boundary independently against the live model."""
    if not 1 <= sample_count <= 20:
        raise ValueError("sample_count must be between 1 and 20")
    selected_cases = BOUNDARY_CASES
    if case_names is not None:
        requested = set(case_names)
        if not requested:
            raise ValueError("at least one boundary case is required")
        known = {case.name for case in BOUNDARY_CASES}
        unknown = requested - known
        if unknown:
            raise ValueError(f"unknown boundary case: {', '.join(sorted(unknown))}")
        selected_cases = tuple(
            case for case in BOUNDARY_CASES if case.name in requested
        )

    def one(case: BoundaryCase, index: int) -> Sample:
        try:
            raw = OpenAIResumeJobMatchWorker(config).match(
                document=StoredResumeDocument(
                    resume_version_id=f"eval_{case.name}_{index}",
                    document_format="text",
                    raw_bytes=case.resume.encode("utf-8"),
                ),
                jd_text="\n".join(item.jd_quote for item in case.requirements),
                tiered_requirements=case.requirements,
            )
            checked = ResumeJobMatchService._bind_and_grade_requirements(
                result=raw, requirements=case.requirements
            )
            statuses = tuple(item.status for item in checked.requirements)
            statuses_match = all(
                actual in expected if isinstance(expected, tuple) else actual == expected
                for actual, expected in zip(statuses, case.expected_statuses, strict=True)
            )
            core_text = tuple(
                token
                for item in case.requirements
                if item.tier in {"S", "A"}
                for token in (item.text, item.jd_quote)
            )
            summary_mentions_core = any(text in checked.summary for text in core_text)
            summary_fit_band = _summary_fit_band(checked.summary)
            mismatches: list[str] = []
            if checked.overall_fit != case.expected_fit:
                mismatches.append(
                    f"fit expected={case.expected_fit} actual={checked.overall_fit}"
                )
            if not statuses_match:
                mismatches.append(
                    f"statuses expected={case.expected_statuses!r} actual={statuses!r}"
                )
            if not summary_mentions_core:
                mismatches.append("summary omitted every S/A requirement anchor")
            if summary_fit_band not in {None, case.expected_fit}:
                mismatches.append(
                    "summary fit band conflicts with deterministic result "
                    f"expected={case.expected_fit} actual={summary_fit_band}"
                )
            error = "; ".join(mismatches) or None
            return Sample(index, checked, error)
        except Exception as error:
            return Sample(index, None, type(error).__name__)

    output: dict[str, list[Sample]] = {case.name: [] for case in selected_cases}
    with ThreadPoolExecutor(max_workers=min(8, len(selected_cases) * sample_count)) as pool:
        futures = {
            pool.submit(one, case, index): (case.name, index)
            for case in selected_cases
            for index in range(1, sample_count + 1)
        }
        for future in as_completed(futures):
            name, _ = futures[future]
            output[name].append(future.result())
    return {name: tuple(sorted(samples, key=lambda item: item.index)) for name, samples in output.items()}
