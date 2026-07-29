from career_agent.prompts import CORE_AGENT_SYSTEM_PROMPT, render_system_prompt


def test_render_system_prompt_without_facts_returns_base_prompt() -> None:
    assert render_system_prompt() == CORE_AGENT_SYSTEM_PROMPT


def test_render_system_prompt_adds_only_non_empty_facts() -> None:
    rendered = render_system_prompt(
        base_prompt="Base instruction.",
        facts=["熟悉 Python", " ", "目标岗位是 AI Engineer"],
    )

    assert rendered == ("Base instruction.\n\nKnown facts:\n- 熟悉 Python\n- 目标岗位是 AI Engineer")
