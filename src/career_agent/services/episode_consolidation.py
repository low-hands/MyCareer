from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Iterable

from career_agent.agent.main_agent_contracts import (
    ConversationResourceReference,
    ToolObservation,
)
from career_agent.domain.episodes import (
    CareerEpisodeDraft,
    EPISODE_SUMMARY_MAX_CHARS,
    EpisodeResourceRef,
)


_APPLICATION_TOOLS = {"create_application", "update_application_status"}
_INTERVIEW_TOOLS = {
    "create_interview",
    "update_interview",
    "complete_interview",
    "record_interview_retro",
}
_JOB_RESEARCH_TOOLS = {"research_job", "retry_job_research"}
_RESUME_ANALYSIS_TOOLS = {"confirm_resume_analysis"}
_INTENT_CONFIRMATION_TOOLS = {"confirm_job_intent"}
_RESUME_TAILORING_TOOLS = {"finalize_resume_tailoring"}
_EPISODIC_SUCCESS_STATES = {
    **{tool: frozenset({"application_ready"}) for tool in _APPLICATION_TOOLS},
    **{
        tool: frozenset(
            {"interview_ready", "interview_retro_recorded"}
        )
        for tool in _INTERVIEW_TOOLS
    },
    **{tool: frozenset({"job_research_ready"}) for tool in _JOB_RESEARCH_TOOLS},
    **{
        tool: frozenset({"resume_analysis_confirmed"})
        for tool in _RESUME_ANALYSIS_TOOLS
    },
    **{
        tool: frozenset({"job_intent_recorded"})
        for tool in _INTENT_CONFIRMATION_TOOLS
    },
    **{
        tool: frozenset({"resume_tailoring_finalized"})
        for tool in _RESUME_TAILORING_TOOLS
    },
}


class EpisodeDraftCoverageError(RuntimeError):
    """A committed episodic tool omitted the stable source pointer L1 needs."""


def drafts_from_tool_results(
    *,
    user_id: str,
    conversation_id: str,
    tool_results: Iterable[object],
    occurred_at: datetime | None = None,
) -> tuple[CareerEpisodeDraft, ...]:
    """Build L1 rows only from writes whose durable outcome is known."""

    default_time = occurred_at or datetime.now(timezone.utc)
    drafts: list[CareerEpisodeDraft] = []
    for result in tool_results:
        if not isinstance(result, ToolObservation):
            continue
        if result.execution_outcome != "committed":
            continue
        success_states = _EPISODIC_SUCCESS_STATES.get(result.tool_name)
        if success_states is not None and result.state not in success_states:
            # Replayed receipts and failed capabilities did not produce a new
            # domain entity in this turn. Startup/failure reconciliation owns
            # any pre-existing row they refer to.
            continue
        payload = result.payload
        if result.tool_name in _APPLICATION_TOOLS:
            source_id = _required_source_id(
                result.tool_name,
                payload,
                "application_id",
            )
            drafts.append(
                CareerEpisodeDraft(
                    user_id=user_id,
                    kind="application",
                    source_run_id=source_id,
                    occurred_at=_payload_time(payload, default_time),
                    title=_bounded(
                        " · ".join(
                            part
                            for part in (
                                _text(payload.get("company_name")),
                                _text(payload.get("title")),
                            )
                            if part
                        )
                        or "投递记录",
                        80,
                    ),
                    summary=_bounded(result.message, EPISODE_SUMMARY_MAX_CHARS),
                    conversation_id=conversation_id,
                    resource_refs=_resource_refs(result.resource_ref),
                )
            )
        elif result.tool_name in _INTERVIEW_TOOLS:
            source_id = _required_source_id(
                result.tool_name,
                payload,
                "interview_round_id",
            )
            rich_summary = (
                _text(payload.get("summary"))
                if result.tool_name == "record_interview_retro"
                else None
            )
            sequence = payload.get("sequence_number")
            label = _text(payload.get("employer_label"))
            suffix = f"第 {sequence} 轮面试" if sequence else "面试"
            resource_title = (
                result.resource_ref.title
                if result.resource_ref is not None
                else None
            )
            drafts.append(
                CareerEpisodeDraft(
                    user_id=user_id,
                    kind="interview_round",
                    source_run_id=source_id,
                    occurred_at=_payload_time(payload, default_time),
                    title=_bounded(
                        resource_title
                        or (f"{label} · {suffix}" if label else suffix),
                        80,
                    ),
                    summary=_bounded(
                        rich_summary or result.message,
                        EPISODE_SUMMARY_MAX_CHARS,
                    ),
                    conversation_id=conversation_id,
                    resource_refs=_resource_refs(result.resource_ref),
                )
            )
        elif result.tool_name in _JOB_RESEARCH_TOOLS:
            source_id = _required_source_id(
                result.tool_name,
                payload,
                "run_id",
            )
            research = payload.get("research")
            rich_summary = (
                _text(research.get("summary"))
                if isinstance(research, dict)
                else None
            )
            drafts.append(
                CareerEpisodeDraft(
                    user_id=user_id,
                    kind="job_research",
                    source_run_id=source_id,
                    occurred_at=_payload_time(payload, default_time),
                    title=_bounded(
                        (
                            result.resource_ref.title
                            if result.resource_ref is not None
                            else None
                        )
                        or "岗位研究",
                        80,
                    ),
                    summary=_bounded(
                        rich_summary or result.message,
                        EPISODE_SUMMARY_MAX_CHARS,
                    ),
                    conversation_id=conversation_id,
                    resource_refs=_resource_refs(result.resource_ref),
                )
            )
        elif result.tool_name in _RESUME_ANALYSIS_TOOLS:
            source_id = _required_source_id(
                result.tool_name,
                payload,
                "analysis_id",
            )
            drafts.append(
                CareerEpisodeDraft(
                    user_id=user_id,
                    kind="resume_analysis",
                    source_run_id=source_id,
                    occurred_at=_payload_time(payload, default_time),
                    title="简历分析已确认",
                    summary=_bounded(
                        result.message, EPISODE_SUMMARY_MAX_CHARS
                    ),
                    conversation_id=conversation_id,
                    resource_refs=_resource_refs(result.resource_ref),
                )
            )
        elif result.tool_name in _INTENT_CONFIRMATION_TOOLS:
            source_id = _required_source_id(
                result.tool_name,
                payload,
                "intent_episode_id",
            )
            drafts.append(
                CareerEpisodeDraft(
                    user_id=user_id,
                    kind="intent_confirmation",
                    source_run_id=source_id,
                    occurred_at=_payload_time(payload, default_time),
                    title="求职意图已确认",
                    summary=_bounded(
                        result.message, EPISODE_SUMMARY_MAX_CHARS
                    ),
                    conversation_id=conversation_id,
                    resource_refs=_resource_refs(result.resource_ref),
                )
            )
        elif result.tool_name in _RESUME_TAILORING_TOOLS:
            if payload.get("created") is False:
                continue
            source_id = _required_source_id(
                result.tool_name,
                payload,
                "draft_id",
            )
            drafts.append(
                CareerEpisodeDraft(
                    user_id=user_id,
                    kind="resume_tailoring",
                    source_run_id=source_id,
                    occurred_at=_payload_time(payload, default_time),
                    title="简历定制已完成",
                    summary=_bounded(
                        result.message, EPISODE_SUMMARY_MAX_CHARS
                    ),
                    conversation_id=conversation_id,
                    resource_refs=_resource_refs(result.resource_ref),
                )
            )
    return tuple(drafts)


def mock_interview_exit_draft(
    *,
    user_id: str,
    conversation_id: str,
    source_run_id: str,
    assistant_message: str,
    resource_refs: tuple[ConversationResourceReference, ...] = (),
    occurred_at: datetime | None = None,
) -> CareerEpisodeDraft:
    titled_reference = next(
        (reference for reference in resource_refs if reference.title), None
    )
    return CareerEpisodeDraft(
        user_id=user_id,
        kind="mock_interview",
        source_run_id=source_run_id,
        occurred_at=occurred_at or datetime.now(timezone.utc),
        title=_bounded(
            titled_reference.title if titled_reference else "模拟面试",
            80,
        ),
        summary=_bounded(assistant_message, EPISODE_SUMMARY_MAX_CHARS),
        conversation_id=conversation_id,
        resource_refs=tuple(_episode_ref(reference) for reference in resource_refs),
    )


def _payload_time(payload: dict[str, object], fallback: datetime) -> datetime:
    for field in ("updated_at", "created_at", "completed_at", "submitted_at"):
        value = payload.get(field)
        if isinstance(value, datetime):
            return value
        if isinstance(value, str) and value.strip():
            try:
                return datetime.fromisoformat(value)
            except ValueError:
                continue
    return fallback


def _resource_refs(
    reference: ConversationResourceReference | None,
) -> tuple[EpisodeResourceRef, ...]:
    return (_episode_ref(reference),) if reference is not None else ()


def _episode_ref(reference: ConversationResourceReference) -> EpisodeResourceRef:
    return EpisodeResourceRef(
        kind=reference.kind,
        resource_id=reference.resource_id,
        title=reference.title,
    )


def _text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = " ".join(value.split())
    return normalized or None


def _required_source_id(
    tool_name: str,
    payload: Mapping[str, object],
    field: str,
) -> str:
    source_id = _text(payload.get(field))
    if source_id is None:
        raise EpisodeDraftCoverageError(
            f"{tool_name} committed without the required {field} episode pointer"
        )
    return source_id


def _bounded(value: str, limit: int) -> str:
    normalized = " ".join(value.split())
    if not normalized:
        normalized = "已完成一项职业任务。"
    return normalized[:limit]
