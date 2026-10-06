"""Migration guard: legacy prompt, schema and task projection stay unchanged."""

from __future__ import annotations

import json

from career_agent.agent.capabilities.legacy_profile import LegacyProfileStrategy
from career_agent.agent.capabilities.catalog import CAPABILITIES
from career_agent.agent.capabilities.profiles import profile_schemas
from career_agent.agent.contracts.context import MainAgentContext
from career_agent.agent.contracts.profile import CareerProfileContext
from career_agent.agent.contracts.task_state import ConversationTaskState
from career_agent.agent.providers.main_agent import OpenAICompatibleMainAgentDecisionMaker
from career_agent.agent.runtime.decision_messages import project_decision_messages
from career_agent.evaluation.trajectory import prompt_fingerprint
from career_agent.evaluation.trajectory import advance_trajectory_context
from career_agent.evaluation.main_agent_scenarios import SCENARIOS
from career_agent.evaluation.tool_selection_scenarios import TOOL_SELECTION_SCENARIOS


SCHEMAS = tuple(d.tool_schema() for d in CAPABILITIES.values() if d.model_callable)


def test_legacy_strategy_preserves_schemas_prompt_and_model_context() -> None:
    strategy = LegacyProfileStrategy()
    context = MainAgentContext(
        conversation_id="c1", profile=CareerProfileContext(user_id="u1"),
        task=ConversationTaskState(tool_profile="job"), user_message="看看岗位",
    )
    selection = strategy.select(context, SCHEMAS)
    selected_context = context.model_copy(update={"capability_selection": selection})
    assert selection.schemas == profile_schemas("job", SCHEMAS, context.task)
    assert json.dumps(selected_context.model_context(), ensure_ascii=False) == json.dumps(
        context.model_context(), ensure_ascii=False,
    )
    assert project_decision_messages(selected_context).control["task"] == project_decision_messages(context).control["task"]
    assert OpenAICompatibleMainAgentDecisionMaker._system_prompt(strategy.tool_policy()) == OpenAICompatibleMainAgentDecisionMaker._system_prompt()
    assert prompt_fingerprint(profile_schemas("job", SCHEMAS)) == (
        "5f6094b1a3b30ada53ada96df5113e454b4753dcee5f1b93e1a6fb3fc1fcda4d"
    )


def test_legacy_strategy_matches_every_recorded_decision_snapshot() -> None:
    strategy = LegacyProfileStrategy()
    compared = 0
    for scenario in (*SCENARIOS, *TOOL_SELECTION_SCENARIOS):
        context = scenario.context
        for step in scenario.steps:
            context = advance_trajectory_context(context, step)
            selection = strategy.select(context, SCHEMAS)
            selected = context.model_copy(update={"capability_selection": selection})
            assert selection.schemas == profile_schemas(
                context.task.tool_profile, SCHEMAS, context.task,
            ), (scenario.name, step)
            assert selected.model_context() == context.model_context(), (scenario.name, step)
            clock = {"now": "2026-10-06T00:00:00Z", "timezone": "UTC"}
            assert project_decision_messages(selected, clock=clock) == project_decision_messages(context, clock=clock), (scenario.name, step)
            compared += 1
    assert compared > len(SCENARIOS) + len(TOOL_SELECTION_SCENARIOS)
