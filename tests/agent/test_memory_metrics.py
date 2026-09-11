from datetime import datetime, timezone

from career_agent.agent.main_agent_contracts import (
    CareerMemoryClaim,
    CareerMemoryContext,
    CareerMemoryRecord,
    CareerProfileBudgets,
    CareerProfileContext,
    HardConstraintContext,
    MainAgentContext,
)
from career_agent.evaluation.memory_metrics import summarize_memory_metrics
from career_agent.harness.memory_telemetry import memory_context_observation


def _context(*, records_input_units: int = 2_800) -> MainAgentContext:
    return MainAgentContext(
        conversation_id="conversation-1",
        profile=CareerProfileContext(
            user_id="u1",
            hard_constraints=(
                HardConstraintContext(
                    relation="work_arrangement",
                    value="必须远程",
                ),
            ),
        ),
        career_memory=CareerMemoryContext(
            records=(
                CareerMemoryRecord(
                    record_type="project",
                    title="Retrieval",
                    confirmed_highlights=(
                        CareerMemoryClaim(
                            claim="Built a retriever",
                            origin="user_input",
                            recorded_at=datetime(
                                2026, 9, 1, tzinfo=timezone.utc
                            ),
                            revision=1,
                            detail_ref="detail_" + "a" * 24,
                        ),
                    ),
                ),
            ),
        ),
        career_profile_budgets=CareerProfileBudgets(
            records_input_units=records_input_units
        ),
        user_message="帮我找岗位",
    )


def test_context_churn_is_reported_per_slot() -> None:
    summary = summarize_memory_metrics(
        (
            {
                "event_type": "memory_context_observed",
                "conversation_key": "c",
                "slot_fingerprints": {"career_memory": "a", "task": "x"},
            },
            {
                "event_type": "memory_context_observed",
                "conversation_key": "c",
                "slot_fingerprints": {"career_memory": "b", "task": "x"},
            },
        )
    )

    assert summary.context_churn_rate.value == 0.5
    assert summary.context_churn_rate.comparability == "NONCOMPARABLE"
    assert summary.context_churn_by_slot["career_memory"].value == 1.0
    assert summary.context_churn_by_slot["task"].value == 0.0


def test_context_observation_reports_named_projection_delivery() -> None:
    observation = memory_context_observation(
        _context(),
        career_memory_enabled=True,
    )

    assert observation["career_profile_delivery"]["records_returned"] == 1
    assert observation["career_profile_delivery"]["claims_returned"] == 1
    assert observation["career_profile_delivery"]["records_dropped"] == 0
    assert observation["career_profile_delivery"]["claims_dropped"] == 0
    assert observation["career_profile_truncation"] == {
        "any_truncated": False,
        "all_truncation_model_visible": True,
    }
    assert "career_profile_chars" not in observation
    assert "career_profile_source_chars" not in observation
    assert "career_profile_schema_chars" not in observation


def test_zero_record_budget_keeps_truncation_visible() -> None:
    observation = memory_context_observation(
        _context(records_input_units=0),
        career_memory_enabled=True,
    )
    delivery = observation["career_profile_delivery"]

    assert delivery["records_returned"] == 0
    assert delivery["records_total"] == 1
    assert delivery["claims_returned"] == 0
    assert delivery["claims_total"] == 1
    assert observation["career_profile_truncation"] == {
        "any_truncated": True,
        "all_truncation_model_visible": True,
    }


def test_true_empty_memory_is_not_reported_as_truncated() -> None:
    context = MainAgentContext(
        conversation_id="conversation-1",
        profile=CareerProfileContext(user_id="u1"),
        user_message="帮我找岗位",
    )

    observation = memory_context_observation(
        context,
        career_memory_enabled=True,
    )

    assert observation["career_profile_delivery"]["records_total"] == 0
    assert observation["career_profile_truncation"] == {
        "any_truncated": False,
        "all_truncation_model_visible": True,
    }


def test_version_metrics_use_context_observations_only() -> None:
    current = {
        "entry_id": "career_evidence/record-1/claim",
        "update_id": "career_evidence_update_" + "a" * 32,
        "content_digest": "sha256:" + "a" * 64,
        "revision": 2,
        "lifecycle_status": "current",
    }
    superseded = {
        **current,
        "update_id": "career_evidence_update_" + "b" * 32,
        "content_digest": "sha256:" + "b" * 64,
        "revision": 1,
        "lifecycle_status": "superseded",
    }
    summary = summarize_memory_metrics(
        (
            {
                "event_type": "memory_context_observed",
                "conversation_key": "c",
                "binding_profile": "p1",
                "version_inventory_complete": True,
                "entries": [current, superseded],
                "slot_fingerprints": {},
            },
        )
    )

    assert summary.version_observation_count == 1
    assert summary.p1_complete_observation_count == 1
    assert summary.version_context_entry_count == 2
    assert summary.supersedence_exposure.value == 0.5


def test_zombie_exposure_uses_post_tombstone_context() -> None:
    update_id = "career_evidence_update_" + "a" * 32
    entry = {
        "entry_id": "career_evidence/record-1/claim",
        "update_id": update_id,
        "content_digest": "sha256:" + "a" * 64,
        "revision": 1,
        "lifecycle_status": "current",
    }
    summary = summarize_memory_metrics(
        (
            {
                "event_type": "memory_tombstone_observed",
                "entries": [
                    {
                        **entry,
                        "lifecycle_status": "tombstoned",
                    }
                ],
            },
            {
                "event_type": "memory_context_observed",
                "conversation_key": "c",
                "binding_profile": "p1",
                "version_inventory_complete": True,
                "entries": [entry],
                "slot_fingerprints": {},
            },
        )
    )

    assert summary.zombie_exposure.value == 1.0


def test_working_notes_influence_is_the_guarded_decision_rate() -> None:
    summary = summarize_memory_metrics(
        (
            {
                "event_type": "memory_context_observed",
                "working_notes_chars": 20,
                "working_notes_only_tokens": 2,
                "working_notes_only_argument": 1,
                "slot_fingerprints": {},
            },
            {
                "event_type": "memory_context_observed",
                "working_notes_chars": 20,
                "working_notes_only_tokens": 0,
                "working_notes_only_argument": 0,
                "slot_fingerprints": {},
            },
            {
                "event_type": "memory_context_observed",
                "working_notes_chars": 0,
                "slot_fingerprints": {},
            },
        )
    )

    assert summary.working_notes_influence.value == 0.5
    assert summary.working_notes_influence.comparability == "BEST_EFFORT"
