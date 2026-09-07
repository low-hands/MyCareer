from datetime import datetime, timezone

import pytest

from career_agent.agent.main_agent_contracts import (
    AgentDecision,
    CareerMemoryClaim,
    CareerMemoryContext,
    CareerMemoryRecord,
    CareerProfileBudgets,
    CareerProfileContext,
    CurrentTargetContext,
    ConversationMessageContext,
    HardConstraintContext,
    MainAgentContext,
    MemoryTelemetryBinding,
)
from career_agent.agent.context_manager import ContextManager
from career_agent.evaluation.memory_metrics import (
    probe_state_drift,
    summarize_memory_budget_metrics,
    summarize_memory_metrics,
)
from career_agent.harness.memory_telemetry import (
    content_digest,
    memory_context_observation,
    memory_use_observation,
)
from career_agent.storage.context import CareerContextStore
from career_agent.storage.resumes import ResumeStore


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
            "slot_fingerprints": {"career_memory": "a", "task": "x"},
        },
        {
            "event_type": "memory_context_observed",
            "conversation_key": "c",
            "slot_fingerprints": {"career_memory": "b", "task": "x"},
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
    assert summary.context_churn_by_slot["career_memory"].value == 1.0
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
                                    revision=1,
                                    detail_ref="detail_" + "a" * 24,
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
    assert summary.context_churn_by_slot["career_identity"].value == 0.0
    assert summary.context_churn_by_slot["career_memory"].value == 1.0
    assert summary.context_churn_by_slot["task"].value == 0.0
    assert summary.context_churn_by_slot["conversation_summary"].value == 0.0
    assert summary.context_churn_by_slot["recent_messages"].value == 0.0
    assert summary.context_churn_rate.value == 0.2
    assert summary.context_churn_rate.comparability == "NONCOMPARABLE"
    assert "not a memory-change signal" in summary.context_churn_rate.reason
    latest = observations[-1]
    assert set(latest["slot_chars"]) == {
        "career_identity",
        "career_memory",
        "career_profile",
        "task",
        "conversation_summary",
        "recent_messages",
    }
    composition = latest["career_profile_chars"]
    assert (
        composition["keys"]
        + composition["values"]
        + composition["punctuation"]
        + composition["schema_chars"]
    ) == latest["slot_chars"]["career_profile"]
    assert composition["schema_chars"] == latest["career_profile_schema_chars"][
        "label_chars"
    ]
    assert latest["career_profile_schema_chars"]["schema_chars"] > 0
    assert latest["career_profile_schema_chars"]["label_chars"] > 0
    assert sum(latest["career_profile_source_chars"].values()) == (
        latest["slot_chars"]["career_profile"]
    )
    assert set(latest["career_profile_source_chars"]) == {
        "records",
        "current_targets",
        "hard_constraints",
        "shared",
    }
    assert latest["career_profile_source_chars"]["current_targets"] == 0
    assert latest["career_profile_dynamic_ratio"] == (
        latest["slot_chars"]["career_profile"]
        / latest["dynamic_context_chars"]
    )
    assert latest["career_profile_delivery"]["records_returned"] == 1
    assert latest["career_profile_delivery"]["records_dropped"] == 0
    assert latest["career_profile_delivery"]["claims_returned"] == 1
    assert latest["career_profile_delivery"]["claims_dropped"] == 0


def test_profile_source_buckets_include_business_blocks_and_shared_framing() -> None:
    context = MainAgentContext(
        conversation_id="conversation-1",
        profile=CareerProfileContext(
            user_id="u1",
            default_city="杭州",
            hard_constraints=(
                HardConstraintContext(
                    relation="work_schedule",
                    value="不接受 996",
                ),
            ),
            current_targets=(
                CurrentTargetContext(
                    title="ML Engineer",
                    priority=1,
                    salary_expectation="40-60k",
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
        user_message="帮我找岗位",
    )

    observation = memory_context_observation(
        context,
        career_memory_enabled=True,
    )
    buckets = observation["career_profile_source_chars"]

    assert all(buckets[name] > 0 for name in buckets)
    assert sum(buckets.values()) == observation["slot_chars"]["career_profile"]
    # Person-level default_city and the outer JSON framing are deliberately
    # shared rather than misattributed to the role-scoped target block.
    assert buckets["shared"] >= len('"default_city": "杭州"')


def test_zero_record_budget_keeps_dropped_counts_visible_in_telemetry() -> None:
    context = MainAgentContext(
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
        career_profile_budgets=CareerProfileBudgets(records_input_units=0),
        user_message="帮我找岗位",
    )

    observation = memory_context_observation(
        context,
        career_memory_enabled=True,
    )
    delivery = observation["career_profile_delivery"]

    assert delivery["records_returned"] == 0
    assert delivery["records_total"] == 1
    assert delivery["records_dropped"] == 1
    assert delivery["claims_returned"] == 0
    assert delivery["claims_total"] == 1
    assert delivery["claims_dropped"] == 1
    assert delivery["hard_constraints_returned"] == 1
    assert delivery["hard_constraints_dropped"] == 0
    assert observation["career_profile_budgets"]["records_input_units"] == 0
    truncation = observation["career_profile_truncation"]
    assert truncation["any_truncated"] is True
    assert truncation["all_truncation_model_visible"] is True
    assert truncation["records"] == {
        "truncated": True,
        "model_visible": True,
        "indistinguishable_from_empty": False,
        "fetch_required": True,
        "fetch_tool": "search_career_memory",
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
    assert observation["career_profile_truncation"]["any_truncated"] is False
    assert observation["career_profile_truncation"]["records"] == {
        "truncated": False,
        "model_visible": True,
        "indistinguishable_from_empty": False,
        "fetch_required": False,
    }


def test_hard_constraint_dynamic_expansion_is_observable() -> None:
    context = MainAgentContext(
        conversation_id="conversation-1",
        profile=CareerProfileContext(
            user_id="u1",
            hard_constraints=(
                HardConstraintContext(
                    relation="work_schedule",
                    value="不接受 996",
                ),
            ),
        ),
        career_profile_budgets=CareerProfileBudgets(
            hard_constraints_input_units=0,
        ),
        user_message="推荐岗位",
    )

    observation = memory_context_observation(
        context,
        career_memory_enabled=True,
    )
    truncation = observation["career_profile_truncation"]

    assert truncation["hard_constraints"]["truncated"] is False
    assert truncation["hard_constraints"]["budget_expanded"] is True
    assert truncation["any_budget_expanded"] is True


def test_production_budget_summary_counts_turns_not_model_attempts() -> None:
    def events(
        run_id: str,
        *,
        budget: int,
        fetched: bool = False,
    ) -> tuple[dict, ...]:
        context = {
            "run_id": run_id,
            "event_type": "memory_context_observed",
            "details": {
                "career_profile_budgets": {
                    "budget_unit": "estimated_input_tokens",
                    "records_input_units": budget,
                    "current_targets_input_units": 800,
                    "hard_constraints_input_units": 600,
                },
                "career_memory_enabled": True,
                "career_profile_delivery": {
                    "records_dropped": 1,
                    "claims_dropped": 2,
                },
                "career_profile_truncation": {
                    "records": {
                        "model_visible": True,
                        "fetch_required": True,
                    },
                    "all_truncation_model_visible": True,
                    "any_fetch_required": True,
                },
            },
        }
        model_events = (
            (
                {
                    "run_id": run_id,
                    "event_type": "model_succeeded",
                    "details": {
                        "decision_action": "tool_call",
                        "tool_name": "search_career_memory",
                    },
                },
            )
            if fetched
            else ()
        )
        return (
            context,
            context,
            *model_events,
            {
                "run_id": run_id,
                "event_type": "model_succeeded",
                "details": {"decision_action": "final"},
            },
            {
                "run_id": run_id,
                "event_type": "turn_completed",
                "details": {},
            },
        )

    summary = summarize_memory_budget_metrics(
        (
            *events("low-1", budget=256),
            *events("low-2", budget=256),
            *events("high-1", budget=512, fetched=True),
            *events("high-2", budget=512),
        ),
    )

    assert summary.observed_run_count == 4
    assert summary.eligible_run_count == 4
    assert summary.missing_budget_run_count == 0
    low, high = summary.cohorts
    assert low.records_input_unit_budget == 256
    assert low.turn_count == 2
    assert low.fetch_hit_count == 0
    assert low.fetch_miss_count == 2
    assert low.fetch_miss_rate.value == 1.0
    assert high.records_input_unit_budget == 512
    assert high.turn_count == 2
    assert high.fetch_turn_count == 1
    assert high.fetch_hit_count == 1
    assert high.fetch_miss_count == 1
    assert high.fetch_hit_rate.value == 0.5
    assert high.fetch_miss_rate.value == 0.5
    assert high.technical_failure_rate.value == 0.0
    assert not hasattr(summary, "recommended_records_char_floor")
    assert not hasattr(high, "unfinished_wilson_95")


def test_fetch_hit_requires_a_visible_overflow_before_the_fetch() -> None:
    budgets = {
        "budget_unit": "estimated_input_tokens",
        "records_input_units": 512,
        "current_targets_input_units": 800,
        "hard_constraints_input_units": 600,
    }

    def context(run_id: str, *, visible: bool) -> dict:
        return {
            "run_id": run_id,
            "event_type": "memory_context_observed",
            "details": {
                "career_profile_budgets": budgets,
                "career_memory_enabled": True,
                "career_profile_delivery": {
                    "records_dropped": 1,
                    "claims_dropped": 1,
                },
                "career_profile_truncation": {
                    "records": {
                        "model_visible": visible,
                        "fetch_required": visible,
                    },
                    "all_truncation_model_visible": visible,
                    "any_fetch_required": visible,
                },
            },
        }

    summary = summarize_memory_budget_metrics(
        (
            {
                "run_id": "prefetch",
                "event_type": "model_succeeded",
                "details": {
                    "decision_action": "tool_call",
                    "tool_name": "search_career_memory",
                },
            },
            context("prefetch", visible=True),
            context("invisible", visible=False),
        )
    )
    cohort = summary.cohorts[0]

    assert cohort.fetch_turn_count == 1
    assert cohort.fetch_required_turn_count == 1
    assert cohort.fetch_hit_count == 0
    assert cohort.fetch_miss_count == 1
    assert cohort.model_invisible_truncation_count == 1


def test_fetch_hit_requires_each_overflow_sections_named_tool() -> None:
    budgets = {
        "budget_unit": "estimated_input_tokens",
        "records_input_units": 512,
        "current_targets_input_units": 800,
        "hard_constraints_input_units": 600,
    }

    def context(run_id: str, required_tools: list[str]) -> dict:
        return {
            "run_id": run_id,
            "event_type": "memory_context_observed",
            "details": {
                "career_profile_budgets": budgets,
                "career_memory_enabled": True,
                "career_profile_delivery": {"records_dropped": 1},
                "career_profile_truncation": {
                    "any_fetch_required": True,
                    "all_truncation_model_visible": True,
                    "required_fetch_tools": required_tools,
                },
            },
        }

    summary = summarize_memory_budget_metrics(
        (
            context(
                "partial",
                ["search_career_memory", "list_target_roles"],
            ),
            {
                "run_id": "partial",
                "event_type": "model_succeeded",
                "details": {
                    "decision_action": "tool_call",
                    "tool_name": "search_career_memory",
                },
            },
            context("target", ["list_target_roles"]),
            {
                "run_id": "target",
                "event_type": "model_succeeded",
                "details": {
                    "decision_action": "tool_call",
                    "tool_name": "list_target_roles",
                },
            },
        )
    )
    cohort = summary.cohorts[0]

    assert cohort.fetch_required_turn_count == 2
    assert cohort.fetch_hit_count == 1
    assert cohort.fetch_miss_count == 1


def test_version_sensitive_metrics_fail_closed_in_m6a() -> None:
    summary = summarize_memory_metrics(())

    assert summary.staleness_exposure.measurable is False
    assert summary.staleness_exposure.comparability == "NONCOMPARABLE"
    assert summary.zombie_exposure.comparability == "NONCOMPARABLE"
    assert summary.supersedence_exposure.comparability == "NONCOMPARABLE"


def test_use_detection_ignores_versioned_values_hidden_by_the_budget() -> None:
    digest = content_digest("Built a private retriever")
    binding = MemoryTelemetryBinding(
        entry_id="career_evidence/root/claim",
        update_id="career_evidence_update_" + "a" * 32,
        content_digest=digest,
        value="Built a private retriever",
        revision=1,
        lifecycle_status="current",
    )
    context = MainAgentContext(
        conversation_id="c1",
        profile=CareerProfileContext(user_id="u1"),
        career_memory=CareerMemoryContext(
            records=(
                CareerMemoryRecord(
                    record_type="project",
                    title="Private",
                    confirmed_highlights=(
                        CareerMemoryClaim(
                            claim=binding.value,
                            origin="user_input",
                            recorded_at=datetime(
                                2026, 9, 1, tzinfo=timezone.utc
                            ),
                            revision=1,
                            detail_ref="detail_" + "a" * 24,
                            telemetry_binding=binding,
                        ),
                    ),
                ),
            ),
            telemetry_bindings=(binding,),
            telemetry_inventory_complete=True,
        ),
        career_profile_budgets=CareerProfileBudgets(records_input_units=0),
        user_message="continue",
    )

    observed = memory_use_observation(
        context,
        AgentDecision(action="final", message=binding.value),
    )

    assert observed is None


def test_repeated_lineage_surface_downgrades_p1_instead_of_false_zero() -> None:
    digest = content_digest("上海")
    bindings = (
        MemoryTelemetryBinding(
            entry_id="person_intent/self/default_city",
            update_id="intent_update_" + "a" * 32,
            content_digest=digest,
            value="上海",
            revision=1,
            lifecycle_status="superseded",
        ),
        MemoryTelemetryBinding(
            entry_id="person_intent/self/default_city",
            update_id="intent_update_" + "b" * 32,
            content_digest=digest,
            value="上海",
            revision=3,
            lifecycle_status="current",
        ),
    )
    context = MainAgentContext(
        conversation_id="c1",
        profile=CareerProfileContext(
            user_id="u1",
            default_city="上海",
            telemetry_bindings=bindings,
            telemetry_inventory_complete=True,
        ),
        user_message="continue",
    )

    observation = memory_context_observation(
        context,
        career_memory_enabled=False,
    )

    assert observation["binding_profile"] == "p2"
    assert observation["version_inventory_complete"] is False


@pytest.mark.parametrize(
    ("old_entry", "current_entry"),
    (
        (
            "target_role_intent/role/salary_expectation",
            "target_role_intent/role/salary_expectation",
        ),
        (
            "target_role_intent/role-a/salary_expectation",
            "target_role_intent/role-b/salary_expectation",
        ),
    ),
)
def test_overlapping_values_are_not_claimed_as_p1(
    old_entry,
    current_entry,
) -> None:
    bindings = (
        MemoryTelemetryBinding(
            entry_id=old_entry,
            update_id="intent_update_" + "a" * 32,
            content_digest=content_digest("40k"),
            value="40k",
            revision=1,
            lifecycle_status="superseded",
        ),
        MemoryTelemetryBinding(
            entry_id=current_entry,
            update_id="intent_update_" + "b" * 32,
            content_digest=content_digest("40k-60k"),
            value="40k-60k",
            revision=2,
            lifecycle_status="current",
        ),
    )
    context = MainAgentContext(
        conversation_id="c1",
        profile=CareerProfileContext(
            user_id="u1",
            current_targets=(
                CurrentTargetContext(
                    target_role_id="role",
                    title="ML Engineer",
                        priority=1,
                    salary_expectation="40k-60k",
                ),
            ),
            telemetry_bindings=bindings,
            telemetry_inventory_complete=True,
        ),
        user_message="continue",
    )

    observation = memory_context_observation(
        context,
        career_memory_enabled=False,
    )

    assert observation["binding_profile"] == "p2"
    assert observation["version_inventory_complete"] is False


def test_ascii_surface_detection_does_not_match_inside_another_word() -> None:
    binding = MemoryTelemetryBinding(
        entry_id="person_intent/self/default_city",
        update_id="intent_update_" + "a" * 32,
        content_digest=content_digest("AI"),
        value="AI",
        revision=1,
        lifecycle_status="current",
    )
    context = MainAgentContext(
        conversation_id="c1",
        profile=CareerProfileContext(
            user_id="u1",
            default_city="AI",
            telemetry_bindings=(binding,),
            telemetry_inventory_complete=True,
        ),
        user_message="continue",
    )

    observed = memory_use_observation(
        context,
        AgentDecision(action="final", message="Send an email."),
    )

    assert observed is None


def test_malformed_p1_entry_fails_version_metrics_closed() -> None:
    summary = summarize_memory_metrics(
        (
            {
                "event_type": "memory_context_observed",
                "binding_profile": "p1",
                "version_inventory_complete": True,
                "slot_fingerprints": {},
                "entries": [
                    {
                        "entry_id": "person_intent/self/default_city",
                        "content_digest": "sha256:" + "a" * 64,
                        "lifecycle_status": "superseded",
                    }
                ],
            },
        )
    )

    assert summary.staleness_exposure.measurable is False
    assert summary.supersedence_exposure.measurable is False
    assert summary.version_observation_count == 1
    assert summary.p1_complete_observation_count == 0
    assert summary.version_context_observation_count == 1
    assert summary.p1_complete_context_observation_count == 0
    assert summary.p1_complete_ratio.value == 0.0


def test_one_degraded_observation_does_not_invalidate_complete_p1_history() -> None:
    complete = {
        "event_type": "memory_context_observed",
        "binding_profile": "p1",
        "version_inventory_complete": True,
        "slot_fingerprints": {},
        "entries": [
            {
                "entry_id": "person_intent/self/default_city",
                "update_id": "intent_update_" + "a" * 32,
                "content_digest": "sha256:" + "b" * 64,
                "revision": 1,
                "lifecycle_status": "superseded",
            }
        ],
    }
    city_binding = MemoryTelemetryBinding(
        entry_id="person_intent/self/default_city",
        update_id="intent_update_" + "c" * 32,
        content_digest=content_digest("上海"),
        value="上海",
        revision=1,
        lifecycle_status="current",
    )
    claim_binding = MemoryTelemetryBinding(
        entry_id="career_evidence/root/claim",
        update_id="career_evidence_update_" + "d" * 32,
        content_digest=content_digest("在上海主导推荐系统改造"),
        value="在上海主导推荐系统改造",
        revision=1,
        lifecycle_status="current",
    )
    degraded = {
        "event_type": "memory_context_observed",
        **memory_context_observation(
            MainAgentContext(
                conversation_id="c1",
                profile=CareerProfileContext(
                    user_id="u1",
                    default_city="上海",
                    telemetry_bindings=(city_binding,),
                    telemetry_inventory_complete=True,
                ),
                career_memory=CareerMemoryContext(
                    records=(
                        CareerMemoryRecord(
                            record_type="project",
                            title="推荐系统",
                            confirmed_highlights=(
                                CareerMemoryClaim(
                                    claim=claim_binding.value,
                                    origin="user_input",
                                    recorded_at=datetime(
                                        2026, 9, 1, tzinfo=timezone.utc
                                    ),
                                    revision=1,
                                    detail_ref="detail_" + "a" * 24,
                                    telemetry_binding=claim_binding,
                                ),
                            ),
                        ),
                    ),
                    telemetry_bindings=(claim_binding,),
                    telemetry_inventory_complete=True,
                ),
                user_message="继续",
            ),
            career_memory_enabled=True,
        ),
    }

    summary = summarize_memory_metrics((*((complete,) * 100), degraded))

    assert degraded["binding_profile"] == "p2"
    assert degraded["version_inventory_complete"] is False
    assert summary.version_observation_count == 101
    assert summary.p1_complete_observation_count == 100
    assert summary.version_context_observation_count == 101
    assert summary.p1_complete_context_observation_count == 100
    assert summary.p1_complete_ratio.value == 100 / 101
    assert summary.supersedence_exposure.value == 1.0
    assert summary.supersedence_exposure.measurable is True


def test_p1_cross_store_bindings_measure_superseded_context_and_use(
    tmp_path,
) -> None:
    context_store = CareerContextStore(tmp_path / "context.sqlite3")
    context_store.upsert_profile(
        CareerProfileContext(user_id="u1", default_city="上海"),
        source="test",
    )
    context_store.upsert_profile(
        CareerProfileContext(user_id="u1", default_city="杭州"),
        source="test",
    )
    resumes = ResumeStore(tmp_path / "resumes.sqlite3")
    role = resumes.create_target_role(
        user_id="u1",
        title="ML Engineer",
        priority=1,
    )
    resumes.update_target_role_intent(
        user_id="u1",
        target_role_id=role.id,
        salary_expectation="30-40k",
    )
    resumes.update_target_role_intent(
        user_id="u1",
        target_role_id=role.id,
        salary_expectation="40-50k",
    )
    manager = ContextManager(
        context_store,
        target_role_source=resumes,
    )
    context = manager.load_for_turn(
        user_id="u1",
        conversation_id="c1",
        user_message="继续",
    ).model_copy(
        update={
            "recent_messages": (
                ConversationMessageContext(
                    role="assistant",
                    content="之前按：上海；薪资 30-40k 筛选。",
                    created_at=datetime(2026, 9, 1, tzinfo=timezone.utc),
                ),
            )
        }
    )
    context_observation = memory_context_observation(
        context,
        career_memory_enabled=False,
    )
    use_observation = memory_use_observation(
        context,
        AgentDecision(
            action="final",
            message="继续按：上海；薪资 30-40k 推荐。",
        ),
    )

    assert context_observation["binding_profile"] == "p1"
    assert context_observation["version_inventory_complete"] is True
    assert use_observation is not None
    assert use_observation["binding_profile"] == "p1"
    statuses = {
        (entry["entry_id"], entry["lifecycle_status"])
        for entry in context_observation["entries"]
    }
    assert ("person_intent/self/default_city", "current") in statuses
    assert ("person_intent/self/default_city", "superseded") in statuses
    assert (
        f"target_role_intent/{role.id}/salary_expectation",
        "current",
    ) in statuses
    assert (
        f"target_role_intent/{role.id}/salary_expectation",
        "superseded",
    ) in statuses

    summary = summarize_memory_metrics(
        (
            {
                "event_type": "memory_context_observed",
                **context_observation,
            },
            {
                "event_type": "memory_use_observed",
                **use_observation,
            },
        )
    )

    assert summary.supersedence_exposure.value == 0.5
    assert summary.supersedence_exposure.comparability == "BEST_EFFORT"
    assert summary.staleness_exposure.value == 1.0
    assert summary.staleness_exposure.comparability == "BEST_EFFORT"
    assert summary.version_observation_count == 2
    assert summary.p1_complete_observation_count == 2
    assert summary.p1_complete_ratio.value == 1.0
    assert summary.version_use_observation_count == 1
    assert summary.p1_complete_use_observation_count == 1
    assert summary.zombie_exposure.value is None
    assert summary.zombie_exposure.comparability == "NONCOMPARABLE"
    assert "M3 tombstones" in summary.zombie_exposure.reason


@pytest.mark.parametrize(
    ("value", "message"),
    (
        ("远程", "这个岗位支持远程办公，可以聊聊。"),
        ("上海", "上海外国语大学的校招也在开。"),
        ("北京", "推荐几个北京银行的岗位。"),
    ),
)
def test_short_cjk_values_do_not_match_inside_compound_words(
    value,
    message,
) -> None:
    binding = MemoryTelemetryBinding(
        entry_id="person_intent/self/default_city",
        update_id="intent_update_" + "a" * 32,
        content_digest=content_digest(value),
        value=value,
        revision=1,
        lifecycle_status="current",
    )
    context = MainAgentContext(
        conversation_id="c1",
        profile=CareerProfileContext(
            user_id="u1",
            default_city=value,
            telemetry_bindings=(binding,),
            telemetry_inventory_complete=True,
        ),
        user_message="continue",
    )

    assert memory_use_observation(
        context,
        AgentDecision(action="final", message=message),
    ) is None


def test_short_cjk_value_matches_when_delimited_by_other_scripts() -> None:
    binding = MemoryTelemetryBinding(
        entry_id="person_intent/self/default_city",
        update_id="intent_update_" + "a" * 32,
        content_digest=content_digest("上海"),
        value="上海",
        revision=1,
        lifecycle_status="current",
    )
    context = MainAgentContext(
        conversation_id="c1",
        profile=CareerProfileContext(
            user_id="u1",
            default_city="上海",
            telemetry_bindings=(binding,),
            telemetry_inventory_complete=True,
        ),
        user_message="continue",
    )

    observed = memory_use_observation(
        context,
        AgentDecision(action="final", message="目标：上海。"),
    )

    assert observed is not None
    assert observed["entries"][0]["update_id"] == binding.update_id


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
