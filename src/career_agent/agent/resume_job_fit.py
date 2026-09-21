from __future__ import annotations

from collections import Counter
from typing import Literal

from career_agent.agent.job_analysis_contracts import TieredRequirement
from career_agent.agent.resume_job_match_contracts import RequirementAssessment


OverallFit = Literal["strong", "moderate", "weak", "insufficient_evidence"]


def derive_overall_fit(
    requirements: tuple[TieredRequirement, ...],
    assessments: tuple[RequirementAssessment, ...],
) -> OverallFit:
    """Apply the 096 anchors to assessments already bound to requirements."""
    by_id = {
        item.requirement_id: item
        for item in requirements
        if item.requirement_id is not None
    }
    material = [
        (by_id[item.requirement_id], item)
        for item in assessments
        if item.requirement_id in by_id
        and by_id[item.requirement_id].tier in {"S", "A"}
    ]
    if not material:
        return "insufficient_evidence"

    if any(
        requirement.tier == "S"
        and requirement.kind == "fact"
        and assessment.status == "missing"
        for requirement, assessment in material
    ):
        return "weak"

    statuses = Counter(assessment.status for _, assessment in material)
    if statuses["unclear"] == len(material) or statuses["unclear"] * 2 > len(material):
        return "insufficient_evidence"
    if all(requirement.kind == "inference" for requirement, _ in material) and not any(
        assessment.status in {"matched", "partial"}
        for _, assessment in material
    ):
        return "insufficient_evidence"

    factual_a = [
        assessment
        for requirement, assessment in material
        if requirement.tier == "A" and requirement.kind == "fact"
    ]
    a_statuses = Counter(item.status for item in factual_a)
    if a_statuses["missing"] > a_statuses["matched"] + a_statuses["partial"]:
        return "weak"
    factual_material = [
        assessment for requirement, assessment in material if requirement.kind == "fact"
    ]
    if factual_material and not any(
        item.status in {"matched", "partial"} for item in factual_material
    ) and any(item.status == "missing" for item in factual_material):
        return "weak"

    factual_s = [
        assessment
        for requirement, assessment in material
        if requirement.tier == "S" and requirement.kind == "fact"
    ]
    has_factual_material = bool(factual_material)
    no_factual_gap = all(
        item.status not in {"missing", "unclear"} for item in factual_material
    )
    all_s_matched = all(item.status == "matched" for item in factual_s)
    a_has_match = not factual_a or a_statuses["matched"] > 0
    a_match_dominates_partial = a_statuses["matched"] >= a_statuses["partial"]
    if (
        has_factual_material
        and no_factual_gap
        and all_s_matched
        and a_has_match
        and a_match_dominates_partial
    ):
        return "strong"
    return "moderate"
