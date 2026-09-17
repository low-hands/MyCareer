"""Which fixed group of tools each decision is made against.

A profile is a stable set of tool names. ``core`` is always present: general
question answering, owner settings, the entry point of every domain, and
``route_to_capability``. A domain profile adds that domain's tools. The model
switches profile by calling ``route_to_capability`` explicitly; the runtime
persists the choice on ``ConversationTaskState.tool_profile`` and re-enters
``decide``. No keyword rule guesses the domain, because a compound request
("tailor my resume for this job and prepare the interview") is a sequence of
routes, not one classification.

Stability is the point. Within a profile the schema array and the prompt are
byte-stable, so the cached request prefix survives every decision in the
profile and is lost exactly once, at the switch.

This module is the single place that says where a tool lives. ``tool_effects``
says what a tool does to the world and ``tool_reachability`` says what it needs
from task state; the three are checked against each other at import so a tool
cannot be routable without an effect, or have a profile that no route reaches.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping, TypeVar

from career_agent.agent.main_agent_contracts import (
    DOMAIN_TOOL_PROFILES,
    TOOL_PROFILE_NAMES,
    ConversationTaskState,
    ToolProfile,
)
from career_agent.agent.tool_effects import TOOL_EFFECTS, effect_for
from career_agent.agent.tool_reachability import (
    PRECONDITIONS,
    REQUIREMENTS,
    reachable,
)

ROUTE_TOOL = "route_to_capability"

CORE_TOOLS = frozenset(
    {
        ROUTE_TOOL,
        # General conversation support.
        "read_conversation_span",
        "update_working_notes",
        "fetch_archived_constraints",
        "search_career_memory",
        # Owner settings.
        "update_owner_settings",
        # Today's agenda.
        "get_daily_brief",
        "list_action_items",
        "complete_action_item",
        "dismiss_action_item",
        "snooze_action_item",
        # One entry per domain: enough to answer "what do I have" without
        # leaving core, and to select the object a domain will then act on.
        "open_job_search",
        "find_saved_jobs",
        "list_resumes",
        "list_applications",
        "list_interviews",
    }
)

_DOMAIN_TOOLS: Mapping[ToolProfile, frozenset[str]] = MappingProxyType(
    {
        "job": frozenset(
            {
                "open_job_search",
                "find_saved_jobs",
                "get_saved_job",
                "analyze_job",
                "compare_saved_jobs",
                "research_job",
                "retry_job_research",
                "get_job_research",
                "list_target_roles",
                "propose_job_intent",
                "confirm_job_intent",
            }
        ),
        "resume": frozenset(
            {
                "list_resumes",
                "get_resume_metadata",
                "list_target_roles",
                "find_saved_jobs",
                "get_saved_job",
                "analyze_resume",
                "get_resume_analysis",
                "match_resume_to_job",
                "get_resume_job_match",
                "draft_resume_tailoring",
                "get_resume_tailoring_draft",
                "review_resume_tailoring",
                "revise_resume_tailoring",
                "finalize_resume_tailoring",
                "export_resume_artifact",
            }
        ),
        "application": frozenset(
            {
                "list_applications",
                "get_application",
                "create_application",
                "update_application_status",
                "sync_application_emails",
                "list_email_events",
                "resolve_email_event",
                "find_saved_jobs",
                "list_resumes",
            }
        ),
        "interview": frozenset(
            {
                "list_interviews",
                "get_interview",
                "create_interview",
                "update_interview",
                "complete_interview",
                "record_interview_retro",
                "prepare_interview",
                "get_interview_preparation",
                "list_calendar_accounts",
                "list_calendar_links",
                "prepare_interview_calendar_sync",
                "get_calendar_proposal",
                "execute_calendar_proposal",
                "start_mock_interview",
                "restart_mock_interview",
                "get_mock_interview_result",
                "list_applications",
            }
        ),
        "memory": frozenset(
            {
                "search_career_memory",
                "search_career_history",
                "search_career_episodes",
                "get_career_memory_detail",
                "resolve_claim_source",
                "fetch_archived_constraints",
                "propose_job_intent",
                "confirm_job_intent",
                "propose_free_text_preference_confirmation",
                "confirm_free_text_preference",
                "propose_memory_amendment",
                "confirm_memory_amendment",
                "propose_memory_tombstone",
                "confirm_memory_tombstone",
                "propose_career_fact",
                "confirm_career_fact",
                "propose_constraint_retirement",
                "confirm_constraint_retirement",
            }
        ),
    }
)

TOOL_PROFILES: Mapping[ToolProfile, frozenset[str]] = MappingProxyType(
    {
        "core": CORE_TOOLS,
        **{name: CORE_TOOLS | _DOMAIN_TOOLS[name] for name in DOMAIN_TOOL_PROFILES},
    }
)
"""The complete tool set offered under each profile; every profile contains core."""

ROUTABLE_TOOLS: frozenset[str] = frozenset().union(*TOOL_PROFILES.values())
"""Every model-callable tool. Runtime-owned continuations are deliberately absent."""

_RUNTIME_ONLY = frozenset({"handle_mock_interview_input", "retry_mock_interview"})

if set(TOOL_PROFILES) != set(TOOL_PROFILE_NAMES):
    raise RuntimeError("every ToolProfile needs a tool set")
if ROUTABLE_TOOLS - set(TOOL_EFFECTS):
    raise RuntimeError(
        f"a profiled tool has no declared effect: {sorted(ROUTABLE_TOOLS - set(TOOL_EFFECTS))!r}"
    )
if (set(TOOL_EFFECTS) - _RUNTIME_ONLY) - ROUTABLE_TOOLS:
    raise RuntimeError(
        "a model-callable tool belongs to no profile: "
        f"{sorted((set(TOOL_EFFECTS) - _RUNTIME_ONLY) - ROUTABLE_TOOLS)!r}"
    )
if set(PRECONDITIONS) - ROUTABLE_TOOLS:
    raise RuntimeError("a precondition names a tool outside every profile")
if effect_for(ROUTE_TOOL) != "CONTROL":
    raise RuntimeError("the route tool must be a CONTROL capability")

MAX_NEXT_REQUIREMENTS = 3
Schema = TypeVar("Schema", bound=Mapping[str, object])


def profile_tools(profile: ToolProfile) -> frozenset[str]:
    return TOOL_PROFILES[profile]


def profile_schemas(
    profile: ToolProfile, schemas: tuple[Schema, ...]
) -> tuple[Schema, ...]:
    offered = profile_tools(profile)
    return tuple(
        schema
        for schema in schemas
        if isinstance(function := schema.get("function"), Mapping)
        and function.get("name") in offered
    )


def project_tool_availability(task: ConversationTaskState) -> dict[str, object]:
    """What the model can do now under its profile, and the nearest gaps.

    ``available_now`` lists the profile's tools whose preconditions hold.
    ``next_requirements`` groups blocked tools by their unmet precondition,
    with the domain's tools first and the number of groups capped.
    """

    tools = profile_tools(task.tool_profile)
    available = sorted(name for name in tools if reachable(name, task))
    requirements: dict[str, list[str]] = {}
    domain_first = sorted(tools, key=lambda name: (name in CORE_TOOLS, name))
    for name in domain_first:
        if name in PRECONDITIONS and not reachable(name, task):
            requirement = REQUIREMENTS[name]
            requirements.setdefault(requirement, []).append(name)
    return {
        "tool_profile": task.tool_profile,
        "available_now": available,
        "next_requirements": [
            f"{', '.join(names)}: {requirement}"
            for requirement, names in list(requirements.items())[:MAX_NEXT_REQUIREMENTS]
        ],
    }
