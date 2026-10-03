from career_agent.agent.context_builder import keyword_tool_profile
from career_agent.agent.main_agent_runtime import (
    keyword_tool_profile as runtime_keyword_tool_profile,
)


def test_runtime_reexports_the_canonical_keyword_router() -> None:
    assert runtime_keyword_tool_profile is keyword_tool_profile


def test_named_capability_wins_over_incidental_domain_words() -> None:
    assert keyword_tool_profile("用我的简历做一次模拟面试") == "interview"


def test_ambiguous_topical_message_does_not_pre_route() -> None:
    assert keyword_tool_profile("看看岗位，再优化简历") is None
