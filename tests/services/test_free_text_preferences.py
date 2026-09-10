import pytest
from pydantic import ValidationError

from career_agent.services.free_text_preferences import (
    FreeTextPreferenceMutation,
    extract_free_text_preference,
    preference_topic_is_relevant,
)


@pytest.mark.parametrize(
    ("message", "stance"),
    (
        ("我想清楚了，不去大厂。", "negative"),
        ("我还是不考虑大公司", "negative"),
        ("我改主意了，想去大厂。", "positive"),
        ("我现在可以去大公司", "positive"),
    ),
)
def test_supported_employer_scale_statements_expose_their_stance(
    message: str,
    stance: str,
) -> None:
    mutation = extract_free_text_preference(message)

    assert mutation is not None
    assert mutation.action == "quarantine"
    assert mutation.topic_key == "employer_scale"
    assert mutation.statement == message
    assert mutation.stance == stance


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
