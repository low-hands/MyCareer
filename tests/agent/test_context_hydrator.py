from __future__ import annotations

from datetime import datetime, timezone

from career_agent.agent.contracts.context import MainAgentContext
from career_agent.agent.contracts.profile import (
    CareerProfileContext,
    FreeTextPreferenceContext,
    MemoryTelemetryBinding,
)
from career_agent.agent.contracts.resources import CareerMemoryContext
from career_agent.harness.context_hydrator import ContextHydrator
from career_agent.storage.intent_versions import intent_entry_id


NOW = datetime(2026, 10, 2, tzinfo=timezone.utc)


def _context() -> MainAgentContext:
    return MainAgentContext(
        conversation_id="c1",
        profile=CareerProfileContext(user_id="u1"),
        free_text_preferences=(
            FreeTextPreferenceContext(
                scope_key="freeform.person_default/rust",
                topic_key="language",
                statement="偏好 Rust",
                status="active",
                observed_at=NOW,
                confirmed_at=NOW,
                update_id="intent_update_" + "a" * 32,
            ),
        ),
        user_message="帮我规划下一步",
    )


def test_without_projector_hydration_only_resets_pending_and_tracks_preferences() -> None:
    context = _context()
    hydrator = ContextHydrator(career_context_projector=None)

    update = hydrator.hydrate(
        {
            "context": context,
            "pending": {"name": "read_conversation_span", "policy_owned": True},
        }
    )

    assert update == {
        "pending": {},
        "career_memory_scope_keys": (
            intent_entry_id(
                "freeform.person_default/rust",
                "freeform.person_default",
            ),
        ),
    }
    assert hydrator.project_context(context) is context


def test_projected_memory_and_preferences_share_one_deduplicated_scope_inventory() -> None:
    context = _context()
    preference_entry_id = intent_entry_id(
        "freeform.person_default/rust",
        "freeform.person_default",
    )
    memory = CareerMemoryContext(
        telemetry_bindings=(
            MemoryTelemetryBinding(
                entry_id="career-entry-1",
                update_id="career_evidence_update_" + "b" * 32,
                content_digest="sha256:" + "c" * 64,
                value="负责检索平台",
                revision=1,
                lifecycle_status="current",
            ),
            MemoryTelemetryBinding(
                entry_id=preference_entry_id,
                update_id="intent_update_" + "a" * 32,
                content_digest="sha256:" + "d" * 64,
                value="偏好 Rust",
                revision=1,
                lifecycle_status="current",
            ),
        )
    )

    class Projector:
        def project(self, *, user_id: str, query: str) -> CareerMemoryContext:
            assert user_id == "u1"
            assert query == "帮我规划下一步"
            return memory

    hydrator = ContextHydrator(
        career_context_projector=Projector(),  # type: ignore[arg-type]
    )

    update = hydrator.hydrate({"context": context})

    assert update["pending"] == {}
    assert update["context"].career_memory == memory
    assert update["career_memory_scope_keys"] == (
        "career-entry-1",
        preference_entry_id,
    )
