"""Search-mode views of the existing trajectory scenarios.

Only the mechanics of obtaining a tool change here. The business assertions
remain in the original catalogue and are reused by these views.
"""

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
        facts = tuple(
            fact for fact in scenario.decisive_facts if fact != "task.tool_profile"
        )
        policy = scenario.policy
        if scenario.name in {
            "a_core_request_routes_before_job_analysis",
            "tool_selection_switches_from_resume_to_job_research",
        }:
            # Search is an allowed intermediate call before the business step.
            # The former route step and its synthetic profile switch disappear.
            steps = (replace(scenario.steps[1], observation=None, task_update={}),)
            policy = (
                "When the business tool is not offered, discover it with "
                "search_capabilities and then continue the user's task."
            )
        elif scenario.name == "tool_selection_combines_job_analysis_and_resume_match":
            # The analysis result advances the fixture directly to matching.
            job = scenario.context.task.domain_context.job.model_copy(update={
                "active_analysis_id": "analysis-1",
                "active_analysis_jd_snapshot_id": "jd-1",
                "analysis_status": "ready",
            })
            domain = scenario.context.task.domain_context.model_copy(update={"job": job})
            steps = (
                scenario.steps[0],
                replace(scenario.steps[2],
                    observation=scenario.steps[1].observation,
                    task_update={"domain_context": domain}),
            )
            policy = (
                "Analyze the selected job, then match the bound resume using "
                "the resulting analysis without restarting the user's task."
            )
        else:
            steps = scenario.steps
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
            steps=steps,
            policy=policy,
            decisive_facts=facts,
            recording_samples=3,
            known_gap=(
                None if scenario.name.startswith("tool_selection_")
                else _SEARCH_KNOWN_GAPS.get(scenario.name, scenario.known_gap)
            ),
        ))
    return tuple(adapted)


SEARCH_SCENARIOS = search_mode_scenarios()
