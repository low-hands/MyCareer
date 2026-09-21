from __future__ import annotations

from career_agent.evaluation.resume_job_match import (
    BOUNDARY_CASES,
    _summary_fit_band,
)
from career_agent.cli import build_parser


def test_096_boundary_suite_covers_every_documented_shape() -> None:
    assert [case.name for case in BOUNDARY_CASES] == [
        "strong_all_core_supported",
        "weak_s_hard_gate_missing",
        "moderate_a_mixed",
        "weak_a_gaps_outnumber_support",
        "insufficient_unclear_majority",
        "inference_support_is_moderate",
        "inference_only_without_support_is_insufficient",
        "optional_gap_does_not_lower_core_fit",
    ]


def test_summary_fit_band_finds_a_conflicting_model_claim() -> None:
    assert _summary_fit_band("Overall fit is therefore moderate.") == "moderate"
    assert _summary_fit_band("Overall-fit insufficient evidence.") == (
        "insufficient_evidence"
    )
    assert _summary_fit_band("Python is supported; Kubernetes is optional.") is None


def test_boundary_cli_can_select_one_case_for_targeted_resampling() -> None:
    args = build_parser().parse_args(
        [
            "eval",
            "resume-job-match",
            "--case",
            "optional_gap_does_not_lower_core_fit",
        ]
    )

    assert args.case == ["optional_gap_does_not_lower_core_fit"]
