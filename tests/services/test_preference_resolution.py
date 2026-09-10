from datetime import datetime, timedelta, timezone

from career_agent.domain.intent_memory import IntentMemoryVersion
from career_agent.services.preference_resolution import (
    PreferenceResolutionContext,
    resolve_effective_preferences,
)


NOW = datetime(2026, 9, 10, tzinfo=timezone.utc)


def _version(
    *,
    update: int,
    value: str,
    pref_scope: str,
    layer: str,
    timescale: str = "permanent",
    valid_until: datetime | None = None,
) -> IntentMemoryVersion:
    return IntentMemoryVersion(
        update_id=f"intent_update_{update:032x}",
        user_id="u1",
        scope_key="person_intent/self/company_scale",
        value=value,
        content_digest=f"sha256:{update:064x}",
        revision=1,
        valid_from=NOW - timedelta(days=1),
        valid_until=valid_until,
        pref_scope=pref_scope,
        timescale=timescale,
        layer=layer,
        last_corroborated_at=NOW - timedelta(days=1),
        base_confidence=1.0,
        admission_status="active",
        capture_action="add",
        semantic_stance="positive",
        source="test",
    )


def test_narrow_layer_overrides_only_for_its_matching_context() -> None:
    stable = _version(
        update=1,
        value="始终优先考虑诚信透明的雇主",
        pref_scope="freeform.person_stable",
        layer="stable",
    )
    default = _version(
        update=2,
        value="我默认不去大厂",
        pref_scope="freeform.person_default",
        layer="contextual",
    )
    structured_default = _version(
        update=3,
        value="exclude_large_companies",
        pref_scope="person_default",
        layer="contextual",
    )
    exception = _version(
        update=4,
        value="这家例外",
        pref_scope="freeform.job.job-1",
        layer="transient",
        timescale="situational",
        valid_until=NOW + timedelta(days=1),
    )

    matching = resolve_effective_preferences(
        (default, exception, stable, structured_default),
        context=PreferenceResolutionContext(job_posting_id="job-1"),
        now=NOW,
    )
    assert [item.value for item in matching] == [
        "始终优先考虑诚信透明的雇主",
        "这家例外",
    ]

    other = resolve_effective_preferences(
        (default, exception, stable, structured_default),
        context=PreferenceResolutionContext(job_posting_id="job-2"),
        now=NOW,
    )
    assert [item.value for item in other] == [
        "始终优先考虑诚信透明的雇主",
        "exclude_large_companies",
        "我默认不去大厂",
    ]


def test_expired_situational_value_leaves_the_view_not_history() -> None:
    default = _version(
        update=1,
        value="我默认不去大厂",
        pref_scope="freeform.person_default",
        layer="contextual",
    )
    expired = _version(
        update=2,
        value="这家例外",
        pref_scope="freeform.job.job-1",
        layer="transient",
        timescale="situational",
        valid_until=NOW,
    )

    resolved = resolve_effective_preferences(
        (default, expired),
        context=PreferenceResolutionContext(job_posting_id="job-1"),
        now=NOW,
    )

    assert [item.value for item in resolved] == ["我默认不去大厂"]
    assert expired.value == "这家例外"


def test_person_level_situational_value_needs_no_named_job_context() -> None:
    default = _version(
        update=1,
        value="我默认不去大厂",
        pref_scope="freeform.person_default",
        layer="contextual",
    )
    until_year_end = _version(
        update=2,
        value="我年底前不考虑大厂",
        pref_scope="freeform.person_situational",
        layer="transient",
        timescale="situational",
        valid_until=datetime(2027, 1, 1, tzinfo=timezone.utc),
    )

    active = resolve_effective_preferences(
        (default, until_year_end),
        context=PreferenceResolutionContext(),
        now=NOW,
    )
    expired = resolve_effective_preferences(
        (default, until_year_end),
        context=PreferenceResolutionContext(),
        now=datetime(2027, 1, 1, tzinfo=timezone.utc),
    )

    assert [item.value for item in active] == ["我年底前不考虑大厂"]
    assert [item.value for item in expired] == ["我默认不去大厂"]
