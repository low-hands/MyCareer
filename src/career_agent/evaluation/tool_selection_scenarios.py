"""Offline-only tool-selection probes; no cassette or model recording needed."""

from __future__ import annotations

from career_agent.agent.contracts.context import MainAgentContext
from career_agent.agent.contracts.observations import DecisionObservation
from career_agent.agent.contracts.profile import CareerProfileContext
from career_agent.agent.contracts.task_state import ConversationTaskState
from career_agent.evaluation.trajectory import TrajectoryScenario, TrajectoryStep


def _context(message: str, task: ConversationTaskState) -> MainAgentContext:
    return MainAgentContext(
        conversation_id="offline-tool-selection",
        profile=CareerProfileContext(user_id="offline-tool-selection"),
        user_message=message,
        task=task,
    )


_MATCH_TASK = ConversationTaskState(
    tool_profile="resume",
    active_job_posting_id="job-1",
    active_jd_snapshot_id="jd-1",
    active_job_analysis_id="analysis-1",
    active_job_analysis_jd_snapshot_id="jd-1",
    job_analysis_status="ready",
    active_resume_version_id="resume-1",
)
_MATCH_READY = _MATCH_TASK.update_resume_context(
    active_job_match_id="match-1", job_match_status="ready"
)


TOOL_SELECTION_SCENARIOS: tuple[TrajectoryScenario, ...] = (
    TrajectoryScenario(
        name="offline_resume_to_interview_preparation",
        policy="A resume task can be followed by preparation for the selected interview.",
        context=_context(
            "简历先看到这里，接着准备当前这轮面试。",
            ConversationTaskState(tool_profile="resume", active_interview_round_id="round-1"),
        ),
        steps=(
            TrajectoryStep(expect_tool="route_to_capability"),
            TrajectoryStep(
                task_update={"tool_profile": "interview"},
                observation=DecisionObservation(
                    tool_name="route_to_capability", state="tool_profile_switched",
                    message="已切换。", arguments={"domain": "interview"},
                ),
                expect_tool="prepare_interview",
            ),
        ),
    ),
    TrajectoryScenario(
        name="offline_job_to_application_creation",
        policy="A selected job can lead to an explicitly requested application record.",
        context=_context(
            "把当前选中的岗位建立投递记录。",
            ConversationTaskState(tool_profile="job", active_job_posting_id="job-1"),
        ),
        steps=(
            TrajectoryStep(expect_tool="route_to_capability"),
            TrajectoryStep(
                task_update={"tool_profile": "application"},
                observation=DecisionObservation(
                    tool_name="route_to_capability", state="tool_profile_switched",
                    message="已切换。", arguments={"domain": "application"},
                ),
                expect_tool="create_application",
            ),
        ),
    ),
    TrajectoryScenario(
        name="offline_application_to_email_events",
        policy="After reading an application, inspect its email events when requested.",
        context=_context(
            "看一下当前投递相关的邮件事件。",
            ConversationTaskState(tool_profile="application", active_application_id="app-1"),
        ),
        steps=(
            TrajectoryStep(expect_tool="get_application"),
            TrajectoryStep(
                observation=DecisionObservation(
                    tool_name="get_application", state="application_found",
                    message="投递记录已读取。", arguments={},
                ),
                expect_tool="list_email_events",
            ),
        ),
    ),
    TrajectoryScenario(
        name="offline_interview_to_calendar_sync",
        policy="A selected interview can lead to a calendar-sync proposal.",
        context=_context(
            "给当前这轮面试准备日历同步预览。",
            ConversationTaskState(tool_profile="interview", active_interview_round_id="round-1"),
        ),
        steps=(
            TrajectoryStep(expect_tool="get_interview"),
            TrajectoryStep(
                observation=DecisionObservation(
                    tool_name="get_interview", state="interview_found",
                    message="面试已读取。", arguments={},
                ),
                expect_tool="prepare_interview_calendar_sync",
            ),
        ),
    ),
    TrajectoryScenario(
        name="offline_match_to_resume_tailoring",
        policy="A completed job match can lead to a resume-tailoring draft.",
        context=_context("先比较岗位与简历，再按结果定制简历。", _MATCH_TASK),
        steps=(
            TrajectoryStep(expect_tool="match_resume_to_job"),
            TrajectoryStep(
                task_update={"domain_context": _MATCH_READY.domain_context},
                observation=DecisionObservation(
                    tool_name="match_resume_to_job", state="resume_job_match_ready",
                    message="匹配已完成。", arguments={},
                ),
                expect_tool="draft_resume_tailoring",
            ),
        ),
    ),
)
