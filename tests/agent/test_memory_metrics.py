from datetime import datetime, timezone

from career_agent.agent.main_agent_contracts import (
    CareerMemoryClaim,
    CareerMemoryContext,
    CareerMemoryRecord,
    CareerProfileContext,
    MainAgentContext,
)
from career_agent.evaluation.memory_metrics import (
    compare_paired_runs,
    probe_state_drift,
    summarize_memory_metrics,
)
from career_agent.harness.memory_telemetry import (
    content_digest,
    memory_context_observation,
)


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
    assert summary.context_churn_rate.comparability == "NONCOMPARABLE"
    assert summary.context_churn_by_slot["career_profile"].value == 1.0
    assert summary.context_churn_by_slot["task"].value == 0.0


def test_query_reranking_churn_is_reported_per_slot_without_memory_writes() -> None:
    recorded_at = datetime(2026, 9, 1, tzinfo=timezone.utc)

    def context(title: str) -> MainAgentContext:
        return MainAgentContext(
            conversation_id="conversation-1",
            profile=CareerProfileContext(user_id="u1"),
            career_memory=CareerMemoryContext(
                records=(
                    CareerMemoryRecord(
                        record_type="project",
                        title=title,
                        confirmed_highlights=(
                            CareerMemoryClaim(
                                claim=f"Worked on {title}",
                                origin="user_input",
                                recorded_at=recorded_at,
                            ),
                        ),
                    ),
                )
            ),
            user_message=f"Tell me about {title}",
        )

    observations = tuple(
        memory_context_observation(item, career_memory_enabled=True)
        for item in (context("Retrieval"), context("Planning"), context("Agents"))
    )
    summary = summarize_memory_metrics(
        tuple(
            {"event_type": "memory_context_observed", **observation}
            for observation in observations
        )
    )

    assert summary.uptake_rate.measurable is False
    assert summary.context_churn_by_slot["career_profile"].value == 1.0
    assert summary.context_churn_by_slot["task"].value == 0.0
    assert summary.context_churn_by_slot["conversation_summary"].value == 0.0
    assert summary.context_churn_by_slot["recent_messages"].value == 0.0
    assert summary.context_churn_rate.value == 0.25
    assert summary.context_churn_rate.comparability == "NONCOMPARABLE"
    assert "not a memory-change signal" in summary.context_churn_rate.reason
    latest = observations[-1]
    assert set(latest["slot_chars"]) == {
        "career_profile",
        "task",
        "conversation_summary",
        "recent_messages",
    }
    assert sum(latest["career_profile_chars"].values()) == (
        latest["slot_chars"]["career_profile"]
    )


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
