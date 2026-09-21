from __future__ import annotations

import pytest

from career_agent.agent.job_analysis_contracts import TieredRequirement
from career_agent.agent.resume_job_fit import derive_overall_fit
from career_agent.agent.resume_job_match_contracts import (
    RequirementAssessment,
    ResumeMatchEvidence,
)


def _requirement(
    index: int,
    *,
    tier: str = "A",
    kind: str = "fact",
) -> TieredRequirement:
    return TieredRequirement(
        requirement_id=f"job_requirement_{index:020x}",
        text=f"requirement {index}",
        tier=tier,
        kind=kind,
        jd_quote=f"JD quote {index}",
    )


def _assessment(
    requirement: TieredRequirement,
    status: str,
) -> RequirementAssessment:
    evidence = (
        ResumeMatchEvidence(
            source_locator="Experience, bullet 1",
            source_quote="verbatim resume evidence",
        ),
    ) if status in {"matched", "partial"} else ()
    return RequirementAssessment(
        requirement_id=requirement.requirement_id,
        requirement=requirement.text,
        jd_quote=requirement.jd_quote,
        status=status,
        rationale="deterministic rubric fixture",
        resume_evidence=evidence,
    )


@pytest.mark.parametrize(
    ("shape", "expected"),
    [
        ((("S", "fact", "matched"), ("A", "fact", "matched"), ("A", "fact", "matched"), ("A", "fact", "partial")), "strong"),
        ((("S", "fact", "missing"), ("A", "fact", "matched")), "weak"),
        ((("A", "fact", "matched"), ("A", "fact", "missing")), "moderate"),
        ((("A", "fact", "matched"), ("A", "fact", "missing"), ("A", "fact", "missing")), "weak"),
        ((("S", "fact", "matched"), ("A", "fact", "unclear"), ("A", "fact", "unclear"), ("A", "fact", "unclear")), "insufficient_evidence"),
        ((("S", "inference", "matched"), ("A", "inference", "partial")), "moderate"),
        ((("S", "inference", "missing"), ("A", "inference", "unclear")), "insufficient_evidence"),
        # A partially supported S requirement is the strong/moderate border.
        # `no_factual_gap` only rejects missing and unclear, so without this
        # case nothing holds the anchor that every S fact must be matched.
        ((("S", "fact", "partial"),), "moderate"),
        ((("S", "fact", "partial"), ("A", "fact", "matched")), "moderate"),
        ((("S", "fact", "matched"), ("S", "fact", "partial"), ("A", "fact", "matched")), "moderate"),
    ],
)
def test_overall_fit_boundary_anchors(shape, expected) -> None:
    items = shape
    requirements = tuple(
        _requirement(index, tier=tier, kind=kind)
        for index, (tier, kind, _) in enumerate(items, start=1)
    )
    assessments = tuple(
        _assessment(requirement, item[2])
        for requirement, item in zip(requirements, items)
    )

    assert derive_overall_fit(requirements, assessments) == expected


def test_b_and_c_gaps_do_not_lower_an_otherwise_strong_fit() -> None:
    requirements = (
        _requirement(1, tier="S"),
        _requirement(2, tier="A"),
        _requirement(3, tier="B"),
        _requirement(4, tier="C"),
    )
    assessments = (
        _assessment(requirements[0], "matched"),
        _assessment(requirements[1], "matched"),
        _assessment(requirements[2], "missing"),
        _assessment(requirements[3], "missing"),
    )

    assert derive_overall_fit(requirements, assessments) == "strong"
