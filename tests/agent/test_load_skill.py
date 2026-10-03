"""The main agent loads a skill's instructions for itself; the reader never sees them."""

from __future__ import annotations

from pathlib import Path

from career_agent.agent.contracts.main_agent import AgentDecision
from career_agent.agent.presentation.factory import build_turn_presenter, present_turn
from career_agent.agent.runtime.observation_reducer import tool_observation
from career_agent.agent.capabilities.registry import MainAgentToolRegistry

SKILLS = Path(__file__).resolve().parents[2] / "skills"


def _loaded():
    tools = MainAgentToolRegistry(skills_root=SKILLS)
    return tools.invoke_atomic_tool("load_skill", {"user_id": "u1", "skill": "resume-critique"})


def test_the_skill_is_offered_only_when_skills_are_configured() -> None:
    names = lambda tools: {spec["function"]["name"] for spec in tools.schemas()}
    assert "load_skill" in names(MainAgentToolRegistry(skills_root=SKILLS))
    assert "load_skill" not in names(MainAgentToolRegistry())


def test_the_model_reads_the_instructions_without_their_frontmatter() -> None:
    result = _loaded()

    assert result.state == "skill_loaded"
    body = result.payload["body"]
    assert body.startswith("# Resume Critique")
    assert "name: resume-critique" not in body
    observation = tool_observation("load_skill", result)
    assert observation.body is not None and "Read it three times" in observation.body


def test_the_reply_carries_the_critique_not_the_skill() -> None:
    result = _loaded()

    answered = present_turn(
        {
            "decision": AgentDecision(action="final", message="## 总体评价\n缺少成果。"),
            "tool_results": (result,),
        }
    )

    assert answered["assistant_message"] == "## 总体评价\n缺少成果。"
    assert "Read it three times" not in build_turn_presenter()._screen_message(result)
