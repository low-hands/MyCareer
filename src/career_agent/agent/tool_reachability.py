"""When a tool's prerequisites are met, stated as a table over task state.

This no longer selects what the model is offered. The schema array is kept
byte-stable across task-state changes so the cached request prefix survives,
and the capability boundary is each handler's own argument projection, which
answers an unmet precondition with a bounded soft refusal.

What remains here is the declarative statement of those preconditions: one
place to read what a tool needs, checked against the registry so it cannot name
a tool that does not exist. ``tool_profiles`` reads it to tell the model which
tools of the current profile are usable now and what the nearest unmet
requirement is; the schema array itself never consumes it, so a wrong entry
misleads the model's planning rather than stranding a legal path.

Reference-index readbacks are the one case the table cannot express. Their
anchor is the conversation window (``recent_messages`` / ``archived_resources``)
rather than ``ConversationTaskState``, so task state can neither prove nor
disprove that some report is still reachable.
"""

from __future__ import annotations

from collections.abc import Callable

from career_agent.agent.main_agent_contracts import ConversationTaskState

Precondition = Callable[[ConversationTaskState], bool]


# Readbacks reached through a resource handle: the handle is derived per conversation
# into the conversation's resource references, which live outside task state.
_REFERENCE_READBACKS = frozenset(
    {
        "get_job_research",
        "get_interview_preparation",
        "get_mock_interview_result",
    }
)


def _reachable_via_job(task: ConversationTaskState) -> bool:
    return bool(task.active_job_posting_id or task.saved_job_candidates)


def _reachable_via_resume_version(task: ConversationTaskState) -> bool:
    return bool(task.active_resume_version_id or task.resume_version_candidates)


def _reachable_via_application(task: ConversationTaskState) -> bool:
    return bool(task.active_application_id or task.application_candidates)


def _reachable_via_interview(task: ConversationTaskState) -> bool:
    return bool(task.active_interview_round_id or task.interview_candidates)


def _reachable_via_action_item(task: ConversationTaskState) -> bool:
    return bool(task.active_action_item_id or task.action_candidates)


def _has_current_job_analysis(task: ConversationTaskState) -> bool:
    return bool(
        task.active_job_analysis_id
        and task.job_analysis_status == "ready"
        and task.active_job_analysis_jd_snapshot_id
        and task.active_job_analysis_jd_snapshot_id == task.active_jd_snapshot_id
    )


PRECONDITIONS: dict[str, Precondition] = {
    # Saved jobs and their derived runs.
    "get_saved_job": _reachable_via_job,
    "analyze_job": _reachable_via_job,
    "research_job": _reachable_via_job,
    "compare_saved_jobs": lambda t: bool(t.saved_job_candidates),
    # Resumes and their immutable versions.
    "get_resume_metadata": lambda t: bool(t.resume_candidates),
    "analyze_resume": _reachable_via_resume_version,
    "export_resume_artifact": lambda t: bool(t.active_resume_version_id),
    # Active-object-only analysis / match / tailoring chain.
    "get_resume_analysis": lambda t: bool(t.active_resume_analysis_id),
    "get_resume_job_match": lambda t: bool(t.active_resume_job_match_id),
    "draft_resume_tailoring": lambda t: bool(t.active_resume_job_match_id),
    "get_resume_tailoring_draft": lambda t: bool(
        t.active_resume_tailoring_draft_id
    ),
    "review_resume_tailoring": lambda t: bool(
        t.active_resume_tailoring_draft_id
    ),
    "revise_resume_tailoring": lambda t: bool(
        t.active_resume_tailoring_draft_id
    ),
    "finalize_resume_tailoring": lambda t: bool(
        t.active_resume_tailoring_draft_id
    ),
    # Matching and applying need a job *and* a resume version.
    "match_resume_to_job": lambda t: (
        _reachable_via_job(t)
        and _reachable_via_resume_version(t)
        and _has_current_job_analysis(t)
    ),
    "create_application": _reachable_via_job,
    # Applications.
    "get_application": _reachable_via_application,
    "update_application_status": _reachable_via_application,
    "create_interview": lambda t: bool(
        _reachable_via_application(t) or _reachable_via_job(t)
    ),
    # Interviews.
    "get_interview": _reachable_via_interview,
    "update_interview": _reachable_via_interview,
    "complete_interview": _reachable_via_interview,
    "record_interview_retro": _reachable_via_interview,
    "prepare_interview": lambda t: bool(
        _reachable_via_interview(t) or t.action_candidates
    ),
    # Email events.
    "resolve_email_event": lambda t: bool(t.email_event_candidates),
    # Stated intent can only be confirmed once a proposal has been read back.
    "confirm_job_intent": lambda t: t.pending_job_intent_update is not None,
    # Action center items.
    "complete_action_item": _reachable_via_action_item,
    "dismiss_action_item": _reachable_via_action_item,
    "snooze_action_item": _reachable_via_action_item,
    # Calendar.
    "prepare_interview_calendar_sync": _reachable_via_interview,
    "get_calendar_proposal": lambda t: bool(t.active_calendar_proposal_id),
    "execute_calendar_proposal": lambda t: bool(t.active_calendar_proposal_id),
    # Job research retry.
    "retry_job_research": lambda t: bool(t.active_job_research_run_id),
    # Mock interview.
    # Free practice is always available; an application/interview only changes
    # where the source materials come from.
    "start_mock_interview": lambda t: True,
    "restart_mock_interview": lambda t: (
        t.active_workflow == "mock_interview"
        and t.phase
        in {
            "mock_interview_checkpoint_missing",
            "mock_interview_graph_incompatible",
        }
    ),
    # Memory writes can only be confirmed once their proposal has been shown.
    "confirm_free_text_preference": lambda t: (
        t.pending_free_text_preference is not None
    ),
    "confirm_memory_amendment": lambda t: t.pending_memory_amendment is not None,
    "confirm_memory_tombstone": lambda t: t.pending_memory_tombstone is not None,
    "confirm_career_fact": lambda t: t.pending_career_fact is not None,
    "confirm_constraint_retirement": lambda t: (
        t.pending_constraint_retirement is not None
    ),
}


_NEEDS_JOB = "先用 find_saved_jobs 列出或选定一个已收藏岗位"
_NEEDS_RESUME_VERSION = "先用 list_resumes 列出或选定一个简历版本"
_NEEDS_APPLICATION = "先用 list_applications 列出或选定一条投递记录"
_NEEDS_INTERVIEW = "先用 list_interviews 列出或选定一轮面试"
_NEEDS_ACTION_ITEM = "先用 list_action_items 列出待办事项"
_NEEDS_TAILORING_DRAFT = "先用 draft_resume_tailoring 生成定制草稿"
_NEEDS_PROPOSAL = "先调用对应的 propose_* 工具向用户展示提案"

REQUIREMENTS: dict[str, str] = {
    "get_saved_job": _NEEDS_JOB,
    "analyze_job": _NEEDS_JOB,
    "research_job": _NEEDS_JOB,
    "compare_saved_jobs": "先用 find_saved_jobs 列出可比较的岗位",
    "get_resume_metadata": "先用 list_resumes 列出简历",
    "analyze_resume": _NEEDS_RESUME_VERSION,
    "export_resume_artifact": "先选定一个简历版本（定制完成后自动选定）",
    "get_resume_analysis": "先用 analyze_resume 完成简历分析",
    "get_resume_job_match": "先用 match_resume_to_job 完成岗位匹配",
    "draft_resume_tailoring": "定制前需先用 match_resume_to_job 完成岗位匹配",
    "get_resume_tailoring_draft": _NEEDS_TAILORING_DRAFT,
    "review_resume_tailoring": _NEEDS_TAILORING_DRAFT,
    "revise_resume_tailoring": _NEEDS_TAILORING_DRAFT,
    "finalize_resume_tailoring": _NEEDS_TAILORING_DRAFT,
    "match_resume_to_job": "先用 analyze_job 分析当前 JD，并同时选定一个岗位和一个简历版本",
    "create_application": _NEEDS_JOB,
    "get_application": _NEEDS_APPLICATION,
    "update_application_status": _NEEDS_APPLICATION,
    "create_interview": (
        "需要上下文唯一指向一条投递记录或一个已保存岗位；若都没有，"
        "先询问是否纳入跟踪，并请用户提供或选择公司与岗位，不能关联无关 JD"
    ),
    "get_interview": _NEEDS_INTERVIEW,
    "update_interview": _NEEDS_INTERVIEW,
    "complete_interview": _NEEDS_INTERVIEW,
    "record_interview_retro": _NEEDS_INTERVIEW,
    "prepare_interview": "先选定一轮面试或一条待办事项",
    "resolve_email_event": "先用 list_email_events 列出邮件事件",
    "confirm_job_intent": "先用 propose_job_intent 展示意图变更",
    "complete_action_item": _NEEDS_ACTION_ITEM,
    "dismiss_action_item": _NEEDS_ACTION_ITEM,
    "snooze_action_item": _NEEDS_ACTION_ITEM,
    "prepare_interview_calendar_sync": _NEEDS_INTERVIEW,
    "get_calendar_proposal": "先用 prepare_interview_calendar_sync 生成日历预览",
    "execute_calendar_proposal": "先用 prepare_interview_calendar_sync 生成日历预览",
    "retry_job_research": "只能重试当前会话里已发起的公司调研",
    "start_mock_interview": "可直接自由练习，也可选择一条投递或面试",
    "restart_mock_interview": "只有模拟面试检查点丢失或不兼容时才能重启",
    "confirm_free_text_preference": _NEEDS_PROPOSAL,
    "confirm_memory_amendment": _NEEDS_PROPOSAL,
    "confirm_memory_tombstone": _NEEDS_PROPOSAL,
    "confirm_career_fact": _NEEDS_PROPOSAL,
    "confirm_constraint_retirement": _NEEDS_PROPOSAL,
}
"""One line per precondition, worded as the step that satisfies it."""

if set(REQUIREMENTS) != set(PRECONDITIONS):
    raise RuntimeError(
        "every precondition needs exactly one requirement line: "
        f"{sorted(set(REQUIREMENTS) ^ set(PRECONDITIONS))!r}"
    )


def reachable(name: str, task: ConversationTaskState) -> bool:
    """Whether ``name`` should be offered given ``task``.

    A tool with no declared precondition is always offered. Reference-index
    readbacks are always offered because their reachability lives in the
    conversation window rather than in task state.
    """
    if name in _REFERENCE_READBACKS:
        return True
    precondition = PRECONDITIONS.get(name)
    return precondition(task) if precondition is not None else True
