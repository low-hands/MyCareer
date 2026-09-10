import pytest
from pydantic import ValidationError

from career_agent.services.free_text_preferences import (
    FreeTextPreferenceMutation,
    extract_free_text_preference,
    normalize_preference_stance,
    preference_topic_is_relevant,
)


@pytest.mark.parametrize(
    ("message", "stance", "ownership"),
    (
        ("我想清楚了，不去大厂。", "negative", "person_default"),
        ("我还是不考虑大公司", "negative", "person_default"),
        ("我改主意了，想去大厂。", "positive", "person_default"),
        ("我现在可以去大公司", "positive", "person_default"),
        ("我以后都不去大厂", "negative", "person_stable"),
        (
            "我年底前不考虑大厂",
            "negative",
            "person_situational",
        ),
        ("这类岗位不考虑大厂", "negative", "role"),
        ("这个岗位的话，这家例外", "positive", "situational"),
    ),
)
def test_supported_employer_scale_statements_expose_their_stance(
    message: str,
    stance: str,
    ownership: str,
) -> None:
    mutation = extract_free_text_preference(message)

    assert mutation is not None
    assert mutation.action == "quarantine"
    assert mutation.topic_key == "employer_scale"
    assert mutation.statement == message
    assert mutation.stance == stance
    assert mutation.ownership == ownership


def test_non_preference_mentions_abstain() -> None:
    assert extract_free_text_preference("这家公司算大厂吗？") is None
    assert extract_free_text_preference("朋友说他不去大厂") is None


def test_delete_mutations_cannot_smuggle_a_preference_value() -> None:
    with pytest.raises(ValidationError):
        FreeTextPreferenceMutation(
            action="delete",
            topic_key="employer_scale",
            statement="不去大厂",
            stance="negative",
        )


def test_quarantine_is_only_relevant_to_job_topics_or_confirmation() -> None:
    assert preference_topic_is_relevant("employer_scale", "推荐几个岗位")
    assert preference_topic_is_relevant("employer_scale", "确认")
    assert not preference_topic_is_relevant("employer_scale", "帮我准备 Python 面试")
    assert not preference_topic_is_relevant(
        "team_open_source_culture",
        "开源社区",
        "偏好开源社区活跃的团队",
    )


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("avoid", "negative"),
        ("avoid_large_companies", "negative"),
        ("reject", "negative"),
        ("prefer", "positive"),
        ("favor_remote_work", "positive"),
        ("allow", "positive"),
    ],
)
def test_distilled_stance_aliases_have_controlled_polarity(
    raw: str,
    expected: str,
) -> None:
    assert normalize_preference_stance(raw) == expected
