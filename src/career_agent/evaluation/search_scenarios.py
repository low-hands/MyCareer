"""Trajectory fixtures with runtime-loaded capabilities and turn continuations."""

from __future__ import annotations

from dataclasses import replace

from career_agent.agent.capabilities.proactive import succeeded
from career_agent.evaluation.main_agent_scenarios import SCENARIOS
from career_agent.evaluation.trajectory import TrajectoryScenario


_SEARCH_KNOWN_GAPS = {
    "intent_is_not_inferred_from_a_job_the_user_liked": (
        "2026-10-07 qwen3.7-plus search-mode A replay passed 2/3: one sample "
        "called analyze_job after the user only said the posting looked good. "
        "The forbidden-tool assertion still fails that sample; DeepSeek and "
        "the Qwen legacy-mode samples passed 3/3."
    ),
    "saved_jd_body_without_language_requirement_finishes": (
        "2026-10-07 qwen3.7-plus search-mode A replay passed 2/3: one sample "
        "called analyze_job although the saved JD body did not contain the "
        "language condition and the correct next decision was final. The "
        "action assertion still fails that sample; Qwen legacy passed 3/3."
    ),
}


def search_mode_scenarios() -> tuple[TrajectoryScenario, ...]:
    adapted = []
    for scenario in SCENARIOS:
        context = scenario.context
        if scenario.name == "submitted_resume_questionnaire_continues_tailoring":
            # This fixture starts after submission. A real turn gets the bound
            # target from PendingQuestionnaire in InteractionCoordinator.
            context = context.model_copy(update={
                "turn_continuation_capability": "draft_resume_tailoring",
            })
        previously_used = tuple(
            observation.tool_name for observation in context.tool_observations
            if succeeded(observation)
        )
        if previously_used:
            # The snapshot starts after these calls. Live search-mode turns
            # would already have retained their tools in task state.
            context = context.model_copy(update={
                "task": context.task.add_loaded_capabilities(previously_used),
            })
        adapted.append(replace(
            scenario,
            context=context,
            recording_samples=3,
            known_gap=(
                _SEARCH_KNOWN_GAPS.get(scenario.name, scenario.known_gap)
            ),
        ))
    return tuple(adapted)


SEARCH_SCENARIOS = search_mode_scenarios()
