from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol

from career_agent.agent.main_agent_contracts import (
    GetCareerMemoryDetailToolArguments,
    LoadSkillToolArguments,
    MainAgentContext,
    ReadConversationSpanToolArguments,
    ResolveClaimSourceToolArguments,
    RouteToCapabilityToolArguments,
    SearchCareerEpisodesToolArguments,
    SearchCareerHistoryToolArguments,
    SearchCareerMemoryToolArguments,
    ToolObservation,
    UpdateOwnerSettingsToolArguments,
    project_action_center_arguments,
    project_calendar_arguments,
    project_career_fact_arguments,
    project_constraint_retirement_arguments,
    project_email_arguments,
    project_free_text_preference_arguments,
    project_interview_arguments,
    project_interview_preparation_arguments,
    project_job_intent_arguments,
    project_job_research_arguments,
    project_memory_amendment_arguments,
    project_memory_tombstone_arguments,
    project_mock_interview_arguments,
    project_mock_interview_result_arguments,
    project_open_job_search_arguments,
    project_restart_mock_interview_arguments,
    project_resume_arguments,
    project_saved_job_arguments,
    project_working_notes_arguments,
)
from career_agent.agent.main_state import MainAgentState


class ArgumentProjectionHost(Protocol):
    """Projection hooks kept injectable for runtime variants and focused tests."""

    def _project_runtime_workflow_arguments(
        self, state: MainAgentState, name: str
    ) -> dict[str, Any]: ...

    def _project_atomic_tool_arguments(
        self, context: MainAgentContext, name: str, arguments: dict[str, Any]
    ) -> dict[str, Any]: ...

    def _project_workflow_arguments(
        self, context: MainAgentContext, name: str, arguments: dict[str, Any]
    ) -> dict[str, Any]: ...

    def _reraise_security_refusal(self, error: ValueError) -> None: ...

    def _rejection_observation(
        self, name: str, error: ValueError
    ) -> ToolObservation: ...


@dataclass(frozen=True)
class ProjectedArguments:
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ProjectionRefusal:
    state_update: MainAgentState


ProjectionOutcome = ProjectedArguments | ProjectionRefusal


class ArgumentProjectionMiddleware:
    """Bind model arguments to owner-controlled state before execution.

    Expected projection mistakes become graph observations so the model may
    repair its call. Attempts to cross the least-privilege boundary remain hard
    failures through ``_reraise_security_refusal``.
    """

    def __init__(
        self, *, host: ArgumentProjectionHost, max_projection_refusals: int
    ) -> None:
        self._host = host
        self._max_projection_refusals = max_projection_refusals

    def project(
        self,
        state: MainAgentState,
        *,
        name: str,
        kind: Literal["atomic_tool", "workflow"],
        runtime_owned: bool,
        owner_confirmed: bool,
        policy_owned: bool,
        policy_prelude: bool,
    ) -> ProjectionOutcome:
        decision = state["decision"]
        if decision.tool_call is None:
            raise ValueError("tool_call action requires tool_call arguments")
        try:
            arguments = (
                state["pending"]["arguments"]
                if owner_confirmed
                else self._host._project_runtime_workflow_arguments(state, name)
                if runtime_owned
                else self._host._project_atomic_tool_arguments(
                    state["context"], name, decision.tool_call.arguments
                )
                if kind == "atomic_tool"
                else self._host._project_workflow_arguments(
                    state["context"], name, decision.tool_call.arguments
                )
            )
            if owner_confirmed and name == "update_owner_settings":
                arguments = {
                    **arguments,
                    "confirmation_id": state["pending"]["confirmation_id"],
                }
            return ProjectedArguments(arguments=arguments)
        except ValueError as error:
            self._host._reraise_security_refusal(error)
            control = state.get("control", {})
            if (
                control.get("projection_refusals", 0)
                >= self._max_projection_refusals
            ):
                return ProjectionRefusal(
                    state_update={"authorization_route": "present"}
                )
            result = self._host._rejection_observation(name, error)
            return ProjectionRefusal(
                state_update={
                    "authorization_route": "observe",
                    "pending": {
                        "name": name,
                        "result": result,
                        "synthetic_kind": "projection",
                        "runtime_owned": runtime_owned,
                        "policy_owned": policy_owned,
                        "policy_prelude": policy_prelude,
                    },
                }
            )


def project_runtime_owned_arguments(
    state: MainAgentState, name: str
) -> dict[str, Any]:
    """Bind private workflow input to the durable owner selected at ingress."""

    context = state["context"]
    task = context.task
    if task.active_workflow != "mock_interview" or not task.run_id:
        raise ValueError("runtime-owned mock interview has no active session")
    supplied = state.get("pending", {}).get("arguments", {})
    if name == "retry_mock_interview":
        if task.phase != "failed" or supplied:
            raise ValueError("retry_mock_interview requires one failed active run")
        return {
            "user_id": context.profile.user_id,
            "session_id": task.run_id,
        }
    if name == "handle_mock_interview_input":
        if task.phase == "failed" or set(supplied) != {"message"}:
            raise ValueError(
                "handle_mock_interview_input requires one workflow-owned message"
            )
        message = supplied["message"]
        if not isinstance(message, str):
            raise ValueError("workflow-owned mock interview message must be text")
        return {
            "user_id": context.profile.user_id,
            "session_id": task.run_id,
            "message": message,
        }
    raise ValueError(f"Unknown runtime-owned workflow: {name}")


def project_workflow_arguments(
    context: MainAgentContext,
    name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    if name == "sync_application_emails":
        return project_email_arguments(context, name, arguments)
    if name in {"research_job", "retry_job_research"}:
        return project_job_research_arguments(context, name, arguments)
    if name == "start_mock_interview":
        return project_mock_interview_arguments(context, arguments)
    if name == "restart_mock_interview":
        return project_restart_mock_interview_arguments(context, arguments)
    raise ValueError(f"Unknown main-agent workflow: {name}")


def project_atomic_arguments(
    context: MainAgentContext,
    name: str,
    arguments: dict[str, object],
    *,
    source_turn_id: str | None,
) -> dict[str, object]:
    if name == "load_skill":
        return {
            "user_id": context.profile.user_id,
            **LoadSkillToolArguments.model_validate(arguments).model_dump(),
        }
    if name == "route_to_capability":
        model_arguments = RouteToCapabilityToolArguments.model_validate(arguments)
        return {
            "current_tool_profile": context.task.tool_profile,
            **model_arguments.model_dump(),
        }
    if name == "read_conversation_span":
        model_arguments = ReadConversationSpanToolArguments.model_validate(arguments)
        return {
            "user_id": context.profile.user_id,
            "conversation_id": context.conversation_id,
            **model_arguments.model_dump(exclude_none=True),
        }
    if name == "resolve_claim_source":
        model_arguments = ResolveClaimSourceToolArguments.model_validate(arguments)
        return {
            "user_id": context.profile.user_id,
            **model_arguments.model_dump(),
        }
    if name == "get_career_memory_detail":
        model_arguments = GetCareerMemoryDetailToolArguments.model_validate(arguments)
        return {
            "user_id": context.profile.user_id,
            **model_arguments.model_dump(),
        }
    if name == "search_career_episodes":
        model_arguments = SearchCareerEpisodesToolArguments.model_validate(arguments)
        if (
            model_arguments.detail_ref is not None
            and model_arguments.detail_ref
            not in {item.detail_ref for item in context.career_episodes}
        ):
            raise ValueError("episode detail_ref was not projected in this turn")
        return {
            "user_id": context.profile.user_id,
            **model_arguments.model_dump(),
        }
    if name == "search_career_memory":
        model_arguments = SearchCareerMemoryToolArguments.model_validate(arguments)
        return {
            "user_id": context.profile.user_id,
            **model_arguments.model_dump(),
        }
    if name == "search_career_history":
        model_arguments = SearchCareerHistoryToolArguments.model_validate(arguments)
        return {
            "user_id": context.profile.user_id,
            **model_arguments.model_dump(),
        }
    if name == "update_owner_settings":
        proposed = UpdateOwnerSettingsToolArguments.model_validate(arguments)
        changes = proposed.model_dump(exclude_none=True)
        if all(
            (
                value == context.preferences.boss_search
                if key == "boss_search"
                else value == context.preferences.application_confirmation
            )
            for key, value in changes.items()
        ):
            raise ValueError("owner-settings proposal does not change current settings")
        return {
            "user_id": context.profile.user_id,
            "expected_revision": context.preferences.revision,
            **changes,
        }
    if name == "open_job_search":
        return {
            **project_open_job_search_arguments(context, arguments),
            "source_turn_id": source_turn_id,
        }
    if name in {"propose_job_intent", "confirm_job_intent"}:
        return project_job_intent_arguments(context, name, arguments)
    if name in {
        "propose_free_text_preference_confirmation",
        "confirm_free_text_preference",
    }:
        return project_free_text_preference_arguments(context, name, arguments)
    if name in {"propose_memory_tombstone", "confirm_memory_tombstone"}:
        return project_memory_tombstone_arguments(context, name, arguments)
    if name in {"propose_memory_amendment", "confirm_memory_amendment"}:
        return project_memory_amendment_arguments(context, name, arguments)
    if name == "update_working_notes":
        return project_working_notes_arguments(context, name, arguments)
    if name in {"propose_career_fact", "confirm_career_fact"}:
        return project_career_fact_arguments(context, name, arguments)
    if name in {
        "fetch_archived_constraints",
        "propose_constraint_retirement",
        "confirm_constraint_retirement",
    }:
        return project_constraint_retirement_arguments(context, name, arguments)
    if name in {
        "find_saved_jobs",
        "get_saved_job",
        "compare_saved_jobs",
        "analyze_job",
    }:
        return project_saved_job_arguments(context, name, arguments)
    if name == "get_job_research":
        return project_job_research_arguments(context, name, arguments)
    if name in {"list_email_events", "resolve_email_event"}:
        return project_email_arguments(context, name, arguments)
    if name in {
        "list_interviews",
        "get_interview",
        "create_interview",
        "update_interview",
        "complete_interview",
        "record_interview_retro",
    }:
        return project_interview_arguments(context, name, arguments)
    if name in {"prepare_interview", "get_interview_preparation"}:
        return project_interview_preparation_arguments(context, name, arguments)
    if name == "get_mock_interview_result":
        return project_mock_interview_result_arguments(context, arguments)
    if name in {
        "get_daily_brief",
        "list_action_items",
        "complete_action_item",
        "dismiss_action_item",
        "snooze_action_item",
    }:
        return project_action_center_arguments(context, name, arguments)
    if name in {
        "list_calendar_accounts",
        "list_calendar_links",
        "prepare_interview_calendar_sync",
        "get_calendar_proposal",
        "execute_calendar_proposal",
    }:
        return project_calendar_arguments(context, name, arguments)
    if name in {
        "list_target_roles",
        "list_resumes",
        "get_resume_metadata",
        "match_resume_to_job",
        "get_resume_job_match",
        "draft_resume_tailoring",
        "get_resume_tailoring_draft",
        "review_resume_tailoring",
        "revise_resume_tailoring",
        "finalize_resume_tailoring",
        "export_resume_artifact",
        "create_application",
        "update_application_status",
        "list_applications",
        "get_application",
    }:
        return project_resume_arguments(context, name, arguments)
    return arguments
