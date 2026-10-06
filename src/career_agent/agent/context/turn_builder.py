from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import re
from typing import Literal

from career_agent.agent.context.manager import ContextManager
from career_agent.agent.resources.input import (
    InputResourceRejectedError,
    resolve_application_input_resource,
    resolve_input_resources,
    resolve_job_input_resources,
)
from career_agent.agent.capabilities.catalog import ToolProfile
from career_agent.agent.contracts.candidates import (
    ActiveSavedJobContextItem,
    SavedJobCandidateContextItem,
)
from career_agent.agent.contracts.context import MainAgentContext
from career_agent.agent.contracts.observations import AttachedResumeContext
from career_agent.agent.contracts.task_state import ConversationTaskState
from career_agent.agent.capabilities.registry import MainAgentToolRegistry
from career_agent.domain.applications.models import ApplicationStatus
from career_agent.harness.streaming import TurnInputResource


BareConfirmationTarget = Literal[
    "career_fact",
    "job_intent",
    "free_text_preference",
]


@dataclass(frozen=True)
class PreparedTurnContext:
    routing_task: ConversationTaskState
    bare_confirmation_target: BareConfirmationTarget | None
    attached_resumes: tuple[AttachedResumeContext, ...]
    attached_jobs: tuple[SavedJobCandidateContextItem, ...]
    active_application: tuple[str | None, ApplicationStatus | None]


_FAST_PROFILE_PATTERNS: tuple[tuple[ToolProfile, re.Pattern[str]], ...] = (
    (
        "interview",
        re.compile(
            r"(?:模拟面试|面试(?:准备|安排|通知|记录|复盘|题|官)|"
            r"(?:准备|参加|安排|记录|模拟).{0,4}面试|"
            r"(?:有|收到|约了|参加).{0,8}面试|mock\s+interview|interview\s+prep)",
            re.IGNORECASE,
        ),
    ),
    (
        "application",
        re.compile(
            r"(?:投递(?:记录|进度|状态)?|申请进度|招聘邮件|offer(?:\s|$)|"
            r"跟进招聘|application\s+status)",
            re.IGNORECASE,
        ),
    ),
    (
        "resume",
        re.compile(
            r"(?:简历(?:分析|优化|修改|润色|匹配|导出)?|(?:^|\s)CV(?:\s|$)|resume)",
            re.IGNORECASE,
        ),
    ),
    (
        "job",
        re.compile(
            r"(?:找.{0,12}(?:工作|岗位|职位|实习)|搜(?:索)?.{0,8}(?:岗位|职位|实习)|"
            r"(?:分析|看看|对比).{0,8}(?:JD|岗位|职位)|岗位库|职位描述|job\s+search)",
            re.IGNORECASE,
        ),
    ),
)

_NAMED_CAPABILITY_PATTERNS: tuple[tuple[ToolProfile, re.Pattern[str]], ...] = (
    ("interview", re.compile(r"模拟面试|mock\s+interview", re.IGNORECASE)),
)


def keyword_tool_profile(message: str) -> ToolProfile | None:
    """Route only unmistakable domain language; actions still need a decision."""

    normalized = " ".join(message.split())
    if not normalized:
        return None
    named = {
        profile
        for profile, pattern in _NAMED_CAPABILITY_PATTERNS
        if pattern.search(normalized)
    }
    if len(named) == 1:
        return next(iter(named))
    matches = [
        profile
        for profile, pattern in _FAST_PROFILE_PATTERNS
        if pattern.search(normalized)
    ]
    return matches[0] if len(matches) == 1 else None


class TurnContextBuilder:
    """Resolve trusted turn inputs and construct the context given to a runner."""

    def __init__(
        self,
        *,
        context_manager: ContextManager,
        tools: MainAgentToolRegistry,
        owns_next_turn: Callable[[ConversationTaskState], bool],
        ingress_profile: bool = True,
    ) -> None:
        self._context_manager = context_manager
        self._tools = tools
        self._owns_next_turn = owns_next_turn
        self._ingress_profile = ingress_profile

    def prepare(
        self,
        *,
        user_id: str,
        conversation_id: str,
        input_resources: tuple[TurnInputResource, ...],
    ) -> PreparedTurnContext:
        routing_task = self._context_manager.get_task(
            user_id=user_id,
            conversation_id=conversation_id,
        )
        if input_resources and self._owns_next_turn(routing_task):
            raise InputResourceRejectedError(
                f"{routing_task.active_workflow} owns this conversation; "
                "attachments are not read until it completes"
            )
        attached_resumes = (
            resolve_input_resources(
                self._tools.resume_store,
                user_id=user_id,
                resources=input_resources,
                text_service=self._tools.resume_text_service,
            )
            if input_resources
            else ()
        )
        attached_jobs = (
            resolve_job_input_resources(
                self._tools.job_repository,
                user_id=user_id,
                resources=input_resources,
            )
            if input_resources
            else ()
        )
        active_application = (
            resolve_application_input_resource(
                self._tools.application_service,
                user_id=user_id,
                resources=input_resources,
            )
            if input_resources
            else (None, None)
        )
        bare_confirmation_target = routing_task.bare_confirmation_target
        if bare_confirmation_target is not None:
            routing_task = self._context_manager.disarm_bare_confirmation(
                user_id=user_id,
                conversation_id=conversation_id,
                task=routing_task,
            )
        return PreparedTurnContext(
            routing_task=routing_task,
            bare_confirmation_target=bare_confirmation_target,
            attached_resumes=attached_resumes,
            attached_jobs=attached_jobs,
            active_application=active_application,
        )

    def load_turn(
        self,
        prepared: PreparedTurnContext,
        *,
        user_id: str,
        conversation_id: str,
        user_message: str,
        route_profile: bool,
    ) -> MainAgentContext:
        context = self._context_manager.load_for_turn(
            user_id=user_id,
            conversation_id=conversation_id,
            user_message=user_message,
        )
        context = self.attach_input_resources(
            context,
            prepared.attached_resumes,
            prepared.attached_jobs,
            prepared.active_application,
        )
        context = self.refresh_saved_job_focus(context)
        if route_profile and self._ingress_profile:
            fast_profile = keyword_tool_profile(context.user_message)
            if fast_profile is not None and context.task.tool_profile == "core":
                context = context.model_copy(
                    update={
                        "task": context.task.model_copy(
                            update={"tool_profile": fast_profile}
                        )
                    }
                )
        return context

    def load_workflow_turn(
        self,
        prepared: PreparedTurnContext,
        *,
        user_id: str,
        conversation_id: str,
    ) -> MainAgentContext:
        return self._context_manager.load_for_workflow_turn(
            user_id=user_id,
            conversation_id=conversation_id,
            task=prepared.routing_task,
        )

    @staticmethod
    def attach_input_resources(
        context: MainAgentContext,
        attached_resumes: tuple[AttachedResumeContext, ...],
        attached_jobs: tuple[SavedJobCandidateContextItem, ...] = (),
        active_application: tuple[
            str | None, ApplicationStatus | None
        ] = (None, None),
    ) -> MainAgentContext:
        """Place verified attachments on the turn and focus their final item."""

        if (
            not attached_resumes
            and not attached_jobs
            and active_application[0] is None
        ):
            return context
        task = context.task
        if attached_resumes:
            task = task.update_resume_context(
                active_version_id=attached_resumes[-1].resume_version_id
            )
        if attached_jobs:
            attached_ids = {item.job_posting_id for item in attached_jobs}
            focused = attached_jobs[-1]
            pinned = (
                ActiveSavedJobContextItem(
                    job_posting_id=focused.job_posting_id,
                    jd_snapshot_id=focused.jd_snapshot_id,
                    title=focused.title,
                    company_name=focused.company_name,
                    jd_version=focused.jd_version,
                )
                if focused.jd_snapshot_id is not None
                and focused.jd_version is not None
                else None
            )
            task = task.update_job_context(
                active_posting_id=focused.job_posting_id,
                active_jd_snapshot_id=(
                    pinned.jd_snapshot_id if pinned is not None else None
                ),
                active_saved_job=pinned,
                saved_job_candidates=(
                    *attached_jobs,
                    *(
                        item
                        for item in context.task.saved_job_candidates
                        if item.job_posting_id not in attached_ids
                    ),
                ),
            )
        if active_application[0] is not None:
            task = task.update_application_context(
                active_id=active_application[0],
                active_status=active_application[1],
            )
        return context.model_copy(
            update={
                **(
                    {"attached_resumes": attached_resumes}
                    if attached_resumes
                    else {}
                ),
                **({"attached_jobs": attached_jobs} if attached_jobs else {}),
                "task": task,
            }
        )

    def refresh_saved_job_focus(self, context: MainAgentContext) -> MainAgentContext:
        """Refresh whether the pinned JD remains readable without exposing it."""

        focus = context.task.focused_saved_job()
        if focus is None or self._tools.job_repository is None:
            return context
        readable = (
            self._tools.job_repository.get_snapshot(
                user_id=context.profile.user_id,
                jd_snapshot_id=focus.jd_snapshot_id,
            )
            is not None
        )
        if readable == focus.readable:
            return context
        return context.model_copy(
            update={
                "task": context.task.focus_saved_job(
                    focus.model_copy(update={"readable": readable})
                )
            }
        )
