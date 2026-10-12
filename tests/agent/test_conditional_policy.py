from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from career_agent.agent.capabilities.catalog import CAPABILITIES
from career_agent.agent.contracts.context import MainAgentContext
from career_agent.agent.contracts.observations import (
    WorkingNotesContext,
)
from career_agent.agent.contracts.profile import CareerProfileContext
from career_agent.agent.contracts.resources import ConversationMessageContext
from career_agent.agent.providers.main_agent import OpenAICompatibleMainAgentDecisionMaker
from career_agent.agent.runtime.conditional_policy import conditional_policy
from career_agent.agent.runtime.decision_messages import project_decision_messages


def _context(message: str = "hello") -> MainAgentContext:
    return MainAgentContext(
        conversation_id="c1",
        profile=CareerProfileContext(user_id="u1"),
        user_message=message,
    )


def _policy(context: MainAgentContext) -> str:
    return project_decision_messages(context).control.get("conditional_policy", "")


def test_cold_context_keeps_conditional_rules_out_of_cached_system_prompt() -> None:
    context = _context()
    assert _policy(context) == ""
    prompt = OpenAICompatibleMainAgentDecisionMaker._system_prompt()
    assert "working_notes is an unconfirmed agent scratchpad" not in prompt
    assert "A user may ask for interview preparation" not in prompt
    assert "Example shape:" in prompt


def test_compression_uses_the_same_switch_as_history_readback() -> None:
    cold = _context()
    compressed = cold.model_copy(update={"through_sequence": 2, "recent_from_sequence": 3})
    assert not cold.has_compressed_history
    assert compressed.has_compressed_history
    assert "archived_fact_count" not in _policy(cold)
    assert "archived_fact_count" in _policy(compressed)


def test_empty_notes_do_not_trigger_but_existing_or_new_notes_do() -> None:
    empty = _context().model_copy(update={"working_notes": WorkingNotesContext(markdown="", revision="a" * 12)})
    assert "Working notes:" not in _policy(empty)
    existing = empty.model_copy(update={"working_notes": WorkingNotesContext(markdown="unconfirmed preference", revision="a" * 12)})
    assert "Working notes:" in _policy(existing)
    assert "Working notes:" in _policy(_context("更新工作笔记"))
    offered_only = empty.model_copy(update={
        "capability_selection": SimpleNamespace(offered_names=("update_working_notes",)),
    })
    assert "Working notes:" not in conditional_policy(offered_only)
    assert "projected revision" in CAPABILITIES["update_working_notes"].tool_schema()[
        "function"
    ]["description"]


def test_preference_and_episode_rules_require_projected_entries() -> None:
    cold = _context()
    assert "Free-text preferences:" not in _policy(cold)
    assert "Career episodes" not in _policy(cold)
    preference = cold.model_copy(update={"free_text_preferences_quarantined_total": 1})
    assert "Free-text preferences:" in _policy(preference)
    episodes = cold.model_copy(update={"career_episodes": (object(),)})
    assert "Career episodes" in conditional_policy(episodes)


def test_interview_request_and_short_followup_receive_rule() -> None:
    interview = _context("明天有面试")
    assert "Interview preparation and tracking:" in _policy(interview)
    followup = _context("好的，记录吧").model_copy(
        update={
            "recent_messages": (
                ConversationMessageContext(
                    role="user",
                    content="明天有面试",
                    created_at=datetime.now(timezone.utc),
                ),
            ),
        }
    )
    assert "Interview preparation and tracking:" in _policy(followup)
    assert "Interview preparation and tracking:" not in _policy(_context("帮我改简历"))


@pytest.mark.parametrize(
    "message",
    (
        "下周三二面", "明天终面", "HR面怎么准备", "笔试准备", "拿到offer了",
        "onsite prep", "I have two interviews next week", "my applications",
        "interviewing tomorrow", "offers", "interviewer asked", "My application",
        "Offer letter",
    ),
)
def test_interview_variants_receive_rule(message: str) -> None:
    assert "Interview preparation and tracking:" in _policy(_context(message))


@pytest.mark.parametrize(
    "message",
    ("The tool was offered", "reapplication is unrelated", "interviewed", "preonsite"),
)
def test_interview_english_terms_do_not_match_word_fragments(message: str) -> None:
    assert "Interview preparation and tracking:" not in _policy(_context(message))


def test_interview_followup_window_boundaries() -> None:
    recent = ConversationMessageContext(
        role="user", content="明天有面试", created_at=datetime.now(timezone.utc)
    )
    at_limit = _context("好" * 60).model_copy(update={"recent_messages": (recent,)})
    beyond_limit = _context("好" * 61).model_copy(update={"recent_messages": (recent,)})
    assert "Interview preparation and tracking:" in _policy(at_limit)
    assert "Interview preparation and tracking:" not in _policy(beyond_limit)
    older = _context("好的").model_copy(update={
        "recent_messages": (
            recent,
            ConversationMessageContext(
                role="assistant", content="别的话题", created_at=datetime.now(timezone.utc)
            ),
            ConversationMessageContext(
                role="user", content="其他问题", created_at=datetime.now(timezone.utc)
            ),
        ),
    })
    assert "Interview preparation and tracking:" not in _policy(older)


def test_memory_md_rule_stays_in_system_without_exact_file_name_in_request() -> None:
    prompt = OpenAICompatibleMainAgentDecisionMaker._system_prompt()
    assert "MEMORY.md review is an owner-operated CLI workflow" in prompt
    assert "MEMORY.md review" not in _policy(_context("导出我的记忆文件"))
