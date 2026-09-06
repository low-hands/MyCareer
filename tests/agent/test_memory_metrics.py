from career_agent.evaluation.memory_metrics import (
    compare_paired_runs,
    probe_state_drift,
    summarize_memory_metrics,
)
from career_agent.harness.memory_telemetry import content_digest


def test_p2_uptake_and_context_churn_are_best_effort() -> None:
    city = content_digest("杭州")
    events = (
        {
            "event_type": "memory_write_observed",
            "entries": [
                {
                    "entry_id": "person_intent/self/default_city",
                    "content_digest": city,
                }
            ],
        },
        {
            "event_type": "memory_context_observed",
            "conversation_key": "c",
            "slot_fingerprints": {"career_profile": "a", "task": "x"},
        },
        {
            "event_type": "memory_context_observed",
            "conversation_key": "c",
            "slot_fingerprints": {"career_profile": "b", "task": "x"},
        },
        {
            "event_type": "memory_use_observed",
            "entries": [
                {
                    "entry_id": "person_intent/self/default_city",
                    "content_digest": city,
                }
            ],
        },
    )

    summary = summarize_memory_metrics(events)

    assert summary.uptake_rate.value == 1.0
    assert summary.uptake_rate.comparability == "BEST_EFFORT"
    assert summary.context_churn_rate.value == 0.5
    assert summary.context_churn_rate.comparability == "BEST_EFFORT"


def test_version_sensitive_metrics_fail_closed_in_m6a() -> None:
    summary = summarize_memory_metrics(())

    assert summary.staleness_exposure.measurable is False
    assert summary.staleness_exposure.comparability == "NONCOMPARABLE"
    assert summary.zombie_exposure.comparability == "NONCOMPARABLE"
    assert summary.supersedence_exposure.comparability == "NONCOMPARABLE"


def test_state_drift_is_action_on_a_superseded_value_not_just_exposure() -> None:
    probe = probe_state_drift(
        current_value="杭州",
        superseded_values=("上海",),
        surfaces={
            "profile": "当前城市：杭州",
            "conversation_summary": "此前目标城市：上海",
        },
        action="我继续按上海岗位帮你筛选。",
    )

    assert probe.stale_surfaces == ("conversation_summary",)
    assert probe.current_surfaces == ("profile",)
    assert probe.classification == "superseded"
    assert probe.action_used_stale is True


def test_current_action_can_coexist_with_stale_context_without_being_drift() -> None:
    probe = probe_state_drift(
        current_value="杭州",
        superseded_values=("上海",),
        surfaces={"summary": "上海", "profile": "杭州"},
        action="接下来只看杭州。",
    )

    assert probe.stale_surfaces == ("summary",)
    assert probe.classification == "current"
    assert probe.action_used_stale is False


def test_memory_on_off_results_are_compared_as_pairs() -> None:
    result = compare_paired_runs(
        memory_on_successes=(True, True, False),
        memory_off_successes=(False, True, False),
    )

    assert result.pair_count == 3
    assert result.success_rate_delta == 1 / 3
    assert result.comparability == "COMPARABLE"
