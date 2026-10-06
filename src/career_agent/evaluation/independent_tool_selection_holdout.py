"""Blindly authored selection prompts, kept apart from selector development data.

The wording and intended next actions came from a separate agent that was given
only the case-writing rules, not catalog aliases, successor edges, or fixtures.
This module binds those proposals to valid runtime-shaped task snapshots.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from career_agent.agent.contracts.context import MainAgentContext
from career_agent.agent.contracts.observations import DecisionObservation
from career_agent.agent.contracts.profile import CareerProfileContext
from career_agent.agent.contracts.task_state import ConversationTaskState
from career_agent.evaluation.tool_selection import SelectionCase
from career_agent.evaluation.trajectory import TrajectoryScenario, TrajectoryStep


def _case(
    name: str,
    kind: Literal["single", "cross", "chain", "control"],
    message: str,
    task: ConversationTaskState,
    namespaces: tuple[str, ...],
    steps: tuple[TrajectoryStep, ...],
) -> SelectionCase:
    return SelectionCase(
        scenario=TrajectoryScenario(
            name=f"independent_holdout_{name}",
            policy="Use the next requested business action, respecting references and confirmation.",
            context=MainAgentContext(
                conversation_id=f"independent-holdout-{name}",
                profile=CareerProfileContext(user_id="independent-holdout"),
                user_message=message,
                task=task,
            ),
            steps=steps,
        ),
        split="holdout",
        kind=kind,
        namespaces=frozenset(namespaces),
    )


def _step(tool: str, *, previous: str | None = None,
          resulting_task: ConversationTaskState | None = None,
          user_message: str | None = None,
          forbid: frozenset[str] = frozenset()) -> TrajectoryStep:
    return TrajectoryStep(
        expect_tool=tool,
        observation=(DecisionObservation(
            tool_name=previous, state="completed", message="上一步已完成。", arguments={},
        ) if previous else None),
        task_update=({"domain_context": resulting_task.domain_context}
                     if resulting_task is not None else {}),
        user_message=user_message,
        forbid_tools=forbid,
    )


SELECTION_INDEPENDENT_HOLDOUT: tuple[SelectionCase, ...] = (
    _case("saved_remote_job", "single",
          "我上周收藏的那个温哥华远程数据分析岗找不到了，帮我翻出来。",
          ConversationTaskState(tool_profile="core"), ("job.library",),
          (_step("find_saved_jobs"),)),
    _case("saved_job_requirements", "single",
          "我存的 Northstar 产品经理岗到底要求几年经验？帮我看一下。",
          ConversationTaskState(tool_profile="core", active_job_posting_id="job-northstar"),
          ("job.library",), (_step("get_saved_job"),)),
    _case("resume_versions", "single",
          "我现在都有哪些简历版本？先给我列个名字就行。",
          ConversationTaskState(tool_profile="core"), ("resume.library",),
          (_step("list_resumes"),)),
    _case("compare_saved_jobs", "cross",
          "我收藏的 A 公司增长岗和 B 公司运营岗，哪个更贴我做过的用户留存？先横着比一比。",
          ConversationTaskState(tool_profile="core", saved_job_candidates=(
              {"job_posting_id": "job-a", "title": "增长", "company_name": "A"},
              {"job_posting_id": "job-b", "title": "运营", "company_name": "B"},
          )), ("job.library", "context"), (_step("compare_saved_jobs"),)),
    _case("resume_job_match", "cross",
          "用我那版数据简历看看能不能打这个商业分析岗，主要缺哪块。",
          ConversationTaskState(
              tool_profile="core", active_job_posting_id="job-analyst",
              active_jd_snapshot_id="jd-analyst", active_job_analysis_id="analysis-analyst",
              active_job_analysis_jd_snapshot_id="jd-analyst", job_analysis_status="ready",
              active_resume_version_id="resume-data",
          ), ("job.analysis", "resume.match"), (_step("match_resume_to_job"),)),
    _case("application_status", "single",
          "我刚刚自己投完 Bluebird 的岗位，投递记录里那条改成已投递吧。",
          ConversationTaskState(tool_profile="core", active_application_id="app-bluebird"),
          ("application.tracking",), (_step("update_application_status"),)),
    _case("email_to_interview", "chain",
          "邮箱刚来一封说下周二约面试的信，先跟我的投递对上，再把面试记下来。",
          ConversationTaskState(tool_profile="core", active_application_id="app-bluebird"),
          ("application.email", "interview.schedule"), (
              _step("sync_application_emails"),
              _step("resolve_email_event", previous="sync_application_emails",
                    resulting_task=ConversationTaskState(
                        active_application_id="app-bluebird", email_event_candidates=({
                        "email_event_id": "email-1", "event_type": "interview_invitation",
                        "status": "pending_confirmation", "summary": "下周二面试邀请",
                    },))),
              _step("create_interview", previous="resolve_email_event"),
          )),
    _case("confirmed_calendar", "chain",
          "这场面试时间已经和对方定好了，帮我排进日历。",
          ConversationTaskState(tool_profile="core", active_interview_round_id="round-1"),
          ("interview.calendar",), (
              _step("prepare_interview_calendar_sync"),
              _step("get_calendar_proposal", previous="prepare_interview_calendar_sync",
                    resulting_task=ConversationTaskState(
                        active_interview_round_id="round-1",
                        active_calendar_proposal_id="proposal-1",
                        active_calendar_proposal_expires_at=datetime(2026, 10, 6, tzinfo=timezone.utc),
                    )),
              _step("execute_calendar_proposal", previous="get_calendar_proposal",
                    user_message="这个日历预览没问题，确认加进去。"),
          )),
    _case("tailor_review_export", "chain",
          "按这个岗位把基础简历改一版，检查一遍再给我一个文件。",
          ConversationTaskState(tool_profile="core", active_job_posting_id="job-1",
                                active_resume_version_id="resume-1",
                                active_resume_job_match_id="match-1",
                                resume_job_match_status="ready"),
          ("resume.tailoring", "resume.library"), (
              _step("draft_resume_tailoring"),
              _step("review_resume_tailoring", previous="draft_resume_tailoring",
                    resulting_task=ConversationTaskState(
                        active_job_posting_id="job-1", active_resume_version_id="resume-1",
                        active_resume_job_match_id="match-1", resume_job_match_status="ready",
                        active_resume_tailoring_draft_id="draft-1",
                    )),
              _step("finalize_resume_tailoring", previous="review_resume_tailoring"),
              _step("export_resume_artifact", previous="finalize_resume_tailoring",
                    resulting_task=ConversationTaskState(
                        active_job_posting_id="job-1", active_resume_version_id="resume-final",
                        active_resume_tailoring_draft_id="draft-1",
                    )),
          )),
    _case("interview_from_experience", "cross",
          "明天面这个岗位，我想重点练他们写的‘跨团队协调’，从我的经历里找例子帮我准备。",
          ConversationTaskState(tool_profile="core", active_interview_round_id="round-1"),
          ("interview.prep", "context"), (_step("prepare_interview"),)),
    _case("uncertain_tableau", "control",
          "同事说我当年可能用过 Tableau，我自己不确定。先帮我查以前写过的经历，别记成我的技能。",
          ConversationTaskState(tool_profile="core"), ("context", "memory.search"),
          (_step("search_career_memory", forbid=frozenset({"propose_career_fact", "confirm_career_fact"})),)),
    _case("unconfirmed_email", "control",
          "那封招聘邮件只说‘我们会考虑安排下一轮’，还没给时间。先看看它具体怎么写的，投递状态别动。",
          ConversationTaskState(tool_profile="core", active_application_id="app-1"),
          ("application.email",),
          (_step("list_email_events", forbid=frozenset({"update_application_status", "create_interview"})),)),
)
