"""Offline-only tool-selection probes; no cassette or model recording needed."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from typing import Literal

from career_agent.agent.contracts.context import MainAgentContext
from career_agent.agent.contracts.observations import DecisionObservation
from career_agent.agent.contracts.profile import (
    CareerFactProposal,
    CareerProfileContext,
    JobIntentUpdate,
    MemoryAmendmentProposal,
)
from career_agent.agent.contracts.resources import CareerMemoryContext
from career_agent.agent.contracts.task_state import ConversationTaskState
from career_agent.evaluation.main_agent_scenarios import SCENARIOS, compacted_span_context
from career_agent.evaluation.tool_selection import SelectionCase
from career_agent.evaluation.trajectory import TrajectoryScenario, TrajectoryStep


FIXTURE_NOW = datetime(2026, 10, 5, 12, tzinfo=timezone.utc)


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


def _single(namespace: str, ordinal: str, message: str, tool: str, **state: object) -> SelectionCase:
    profile = namespace.split(".", 1)[0]
    if ordinal == "first" or profile in {"context", "actions"}:
        profile = "core"
    task = ConversationTaskState(tool_profile=profile, **state)
    context = _context(message, task)
    if tool == "load_skill":
        critique = next(
            scenario for scenario in SCENARIOS
            if scenario.name == "a_resume_critique_loads_its_skill"
        )
        context = critique.context.model_copy(update={
            "user_message": message,
            "task": critique.context.task.model_copy(update={"tool_profile": "core"}),
        })
    elif tool == "read_conversation_span":
        context = compacted_span_context(user_message=message, page_in=True)
    elif tool == "get_career_memory_detail":
        context = context.model_copy(update={
            "career_memory": CareerMemoryContext.model_validate({
                "records": ({
                    "record_type": "project", "title": "增长实验",
                    "confirmed_highlights": ({
                        "claim": "负责增长实验", "origin": "user_input",
                        "recorded_at": FIXTURE_NOW, "revision": 1,
                        "detail_ref": "detail_" + "a" * 24,
                    },),
                },),
            }),
        })
    scenario = TrajectoryScenario(
        name=f"selection_dev_{namespace.replace('.', '_')}_{ordinal}",
        policy="Offer the business tool needed for this user request and decision state.",
        context=context,
        steps=(TrajectoryStep(expect_tool=tool),),
    )
    return SelectionCase(scenario, "dev", "single", frozenset({namespace}))


_SINGLE_SPECS = {
    "context": (
        ("帮我点评一下这份简历，看看哪里写得不好、应该怎么改。", "load_skill", {}),
        ("我很早前说的那家目标公司叫什么？", "read_conversation_span", {}),
    ),
    "actions": (
        ("今天最该处理什么？", "get_daily_brief", {}),
        ("刚才提到的待办都列给我看看。", "list_action_items", {}),
    ),
    "job.library": (
        ("我之前存的产品经理岗位还有哪些？", "find_saved_jobs", {}),
        ("这个收藏的职位具体要求是什么？", "get_saved_job", {"active_job_posting_id": "job-1"}),
    ),
    "job.analysis": (
        ("这份 JD 的硬要求帮我拆一下。", "analyze_job", {"active_job_posting_id": "job-1", "active_jd_snapshot_id": "jd-1"}),
        ("刚才那条英语要求算加分项，不是必须项。", "correct_job_requirement_tier", {"active_job_posting_id": "job-1", "active_job_analysis_id": "analysis-1"}),
    ),
    "job.research": (
        ("这家公司最近在做什么，帮我查查。", "research_job", {"active_job_posting_id": "job-1"}),
        ("刚才做的那份公司分析给我打开。", "get_job_research", {"active_job_research_report_id": "report-1"}),
    ),
    "job.intent": (
        ("我现在只看上海的岗位，记一下。", "propose_job_intent", {}),
        ("刚才给我看的上海求职地点，确认保存。", "confirm_job_intent", {"pending_job_intent_update": JobIntentUpdate(city="上海"), "pending_proposed_at": {"pending_job_intent_update": FIXTURE_NOW}}),
    ),
    "resume.library": (
        ("我有哪几版简历？", "list_resumes", {}),
        ("刚才列表里第一份是什么版本？", "get_resume_metadata", {"resume_candidates": ({"resume_id": "resume-1", "target_role_id": "role-1", "name": "产品简历", "status": "active", "latest_version_id": "resume-version-1"},)}),
    ),
    "resume.match": (
        ("这份简历投这个岗位合适吗？", "match_resume_to_job", {"active_job_posting_id": "job-1", "active_jd_snapshot_id": "jd-1", "active_job_analysis_id": "analysis-1", "active_job_analysis_jd_snapshot_id": "jd-1", "job_analysis_status": "ready", "active_resume_version_id": "resume-version-1"}),
        ("刚才的匹配结论再给我看一眼。", "get_resume_job_match", {"active_resume_job_match_id": "match-1"}),
    ),
    "resume.tailoring": (
        ("照着匹配差距改一版简历吧。", "draft_resume_tailoring", {"active_resume_job_match_id": "match-1"}),
        ("刚生成的那版草稿打开看看。", "get_resume_tailoring_draft", {"active_resume_tailoring_draft_id": "draft-1"}),
    ),
    "application.tracking": (
        ("我最近投了哪些公司？", "list_applications", {}),
        ("这条投递现在到哪一步了？", "get_application", {"active_application_id": "app-1"}),
    ),
    "application.email": (
        ("邮箱里有没有新的投递回复？同步一下。", "sync_application_emails", {}),
        ("刚才同步到的招聘邮件有哪些？", "list_email_events", {"active_application_id": "app-1"}),
    ),
    "interview.schedule": (
        ("接下来有哪些面试？", "list_interviews", {}),
        ("这轮面试是几点、谁来面？", "get_interview", {"active_interview_round_id": "round-1"}),
    ),
    "interview.prep": (
        ("这轮面试帮我准备一下。", "prepare_interview", {"active_interview_round_id": "round-1"}),
        ("刚才那份面试准备材料给我看看。", "get_interview_preparation", {"active_interview_preparation_id": "prep-1"}),
    ),
    "interview.calendar": (
        ("我能用哪个日历记这场面试？", "list_calendar_accounts", {}),
        ("先让我看看这轮面试会怎么加进日历。", "prepare_interview_calendar_sync", {"active_interview_round_id": "round-1"}),
    ),
    "interview.mock": (
        ("现在来一场模拟面试吧。", "start_mock_interview", {}),
        ("这条投递上次模拟面试的反馈还有吗？", "get_mock_interview_result", {"active_application_id": "app-1"}),
    ),
    "memory.search": (
        ("我后来改过那段数据分析经历，原先怎么写的？", "search_career_history", {}),
        ("刚才列出的增长项目那条经历，具体记了什么？", "get_career_memory_detail", {}),
    ),
    "memory.proposals": (
        ("我那段经历是做增长分析的，帮我记下来。", "propose_career_fact", {}),
        ("刚才给我看的那条经历没问题，存上吧。", "confirm_career_fact", {"pending_career_fact": CareerFactProposal(career_evidence_id="career_evidence_" + "a" * 32, career_record_id="career_record_" + "b" * 32, claim="做过增长分析", reason="用户明确确认"), "pending_proposed_at": {"pending_career_fact": FIXTURE_NOW}}),
    ),
}

_SINGLE_DEV = tuple(
    _single(namespace, ordinal, message, tool, **state)
    for namespace, pair in _SINGLE_SPECS.items()
    for ordinal, (message, tool, state) in zip(("first", "next"), pair)
)


def _sequence(
    name: str,
    split: Literal["dev", "holdout"],
    kind: Literal["cross", "chain"],
    message: str,
    task: ConversationTaskState,
    tools: tuple[str, ...],
    namespaces: frozenset[str],
    *,
    resulting_states: tuple[ConversationTaskState | None, ...] = (),
    later_messages: tuple[str | None, ...] = (),
    observation_states: tuple[str, ...] = (),
) -> SelectionCase:
    steps = [TrajectoryStep(expect_tool=tools[0])]
    for index, tool in enumerate(tools[1:]):
        previous = tools[index]
        updated = resulting_states[index] if index < len(resulting_states) else None
        steps.append(TrajectoryStep(
            expect_tool=tool,
            observation=DecisionObservation(
                tool_name=previous,
                state=observation_states[index] if index < len(observation_states) else "completed",
                message="上一步已完成。",
                arguments={},
            ),
            task_update=(
                {"domain_context": updated.domain_context,
                 "pending_interaction": updated.pending_interaction}
                if updated is not None else {}
            ),
            user_message=later_messages[index] if index < len(later_messages) else None,
        ))
    return SelectionCase(
        TrajectoryScenario(
            name=name,
            policy="Offer the next business tool for each declared decision snapshot.",
            context=_context(message, task),
            steps=tuple(steps),
        ),
        split,
        kind,
        namespaces,
    )


_CROSS_SPECS = (
    # Four development examples; the remaining eight are frozen holdouts.
    ("dev", "compare_then_match", "我现在有哪些求职方向？这两个收藏岗位也比一下，再看简历跟更合适的那个匹不匹配。", "job", {"active_job_posting_id": "job-1", "saved_job_candidates": ({"job_posting_id": "job-1", "title": "产品经理", "company_name": "甲"}, {"job_posting_id": "job-2", "title": "产品经理", "company_name": "乙"}), "active_jd_snapshot_id": "jd-1", "active_job_analysis_id": "analysis-1", "active_job_analysis_jd_snapshot_id": "jd-1", "job_analysis_status": "ready", "active_resume_version_id": "resume-1"}, ("list_target_roles", "compare_saved_jobs", "match_resume_to_job"), ("job.library", "resume.match")),
    ("dev", "export_then_prepare", "刚改好的简历导出一份，然后帮我准备明天那场面试。", "resume", {"active_resume_version_id": "resume-1", "active_interview_round_id": "round-1"}, ("export_resume_artifact", "prepare_interview"), ("resume.library", "interview.prep")),
    ("dev", "read_job_then_track", "先看下这个收藏岗位，没问题就给它建一条投递记录。", "job", {"active_job_posting_id": "job-1"}, ("get_saved_job", "create_application"), ("job.library", "application.tracking")),
    ("dev", "application_then_email", "这条申请我看看；那封招聘邮件也帮我对上。", "application", {"active_application_id": "app-1", "email_event_candidates": ({"email_event_id": "email-1", "event_type": "interview_invitation", "status": "pending_confirmation", "summary": "面试邀请"},)}, ("get_application", "resolve_email_event"), ("application.tracking", "application.email")),
    ("holdout", "interview_then_links", "先确认这场面试是哪轮，时间改到下午三点，再看看日历里是不是已经有它了。", "interview", {"active_interview_round_id": "round-1"}, ("get_interview", "update_interview", "list_calendar_links"), ("interview.schedule", "interview.calendar")),
    ("holdout", "memory_then_job", "我之前说过的远程经历和那次面试的记录找一下，再拆拆这个岗位要求。", "core", {"active_job_posting_id": "job-1"}, ("search_career_memory", "search_career_episodes", "analyze_job"), ("context", "memory.search", "job.analysis")),
    ("holdout", "research_retry_then_application", "刚才那家公司资料没查完，再试一次；然后把这个职位加入我的投递清单。", "job", {"active_job_posting_id": "job-1", "active_job_research_run_id": "research-1"}, ("retry_job_research", "create_application"), ("job.research", "application.tracking")),
    ("holdout", "review_then_job", "改好的简历先检查，没问题就定稿，然后把这个职位原文给我看看。", "resume", {"active_resume_tailoring_draft_id": "draft-1", "active_job_posting_id": "job-1"}, ("review_resume_tailoring", "finalize_resume_tailoring", "get_saved_job"), ("resume.tailoring", "job.library")),
    ("holdout", "application_then_interview", "这条申请状态改成面试中，再给它登记一轮面试。", "application", {"active_application_id": "app-1"}, ("update_application_status", "create_interview"), ("application.tracking", "interview.schedule")),
    ("holdout", "retro_then_memory", "把刚才那场面试复盘记下来，我提到的项目成绩也留一条。", "interview", {"active_interview_round_id": "round-1"}, ("record_interview_retro", "propose_career_fact"), ("interview.schedule", "memory.proposals")),
    ("holdout", "constraints_then_actions", "先看看我之前定的限制，再把今天的待办列出来。", "core", {}, ("fetch_archived_constraints", "list_action_items"), ("context", "actions")),
    ("holdout", "source_then_tailor", "这条经历原文找出来，再照着岗位改简历。", "memory", {"active_resume_job_match_id": "match-1"}, ("resolve_claim_source", "draft_resume_tailoring"), ("memory.search", "resume.tailoring")),
)

_CROSS_CASES = tuple(
    _sequence(
        f"selection_{split}_cross_{name}", split, "cross", message,
        ConversationTaskState(tool_profile=profile, **state), tools,
        frozenset(namespaces),
    )
    for split, name, message, profile, state, tools, namespaces in _CROSS_SPECS
)


_job_chain_start = ConversationTaskState(tool_profile="job", active_resume_version_id="resume-1")
_job_chain_found = ConversationTaskState(
    tool_profile="job", active_job_posting_id="job-1", active_jd_snapshot_id="jd-1",
    active_resume_version_id="resume-1",
)
_job_chain_analyzed = _job_chain_found.update_job_context(
    active_analysis_id="analysis-1", active_analysis_jd_snapshot_id="jd-1",
    analysis_status="ready",
)
_job_chain_matched = _job_chain_analyzed.update_resume_context(
    active_job_match_id="match-1", job_match_status="ready",
)

_application_chain_start = ConversationTaskState(tool_profile="application")
_application_chain_found = _application_chain_start.update_application_context(active_id="app-1")
_application_chain_emailed = ConversationTaskState(
    tool_profile="application", active_application_id="app-1",
    email_event_candidates=({"email_event_id": "email-1", "event_type": "interview_invitation", "status": "pending_confirmation", "summary": "面试邀请"},),
)

_interview_chain_start = ConversationTaskState(tool_profile="interview", active_application_id="app-1")
_interview_chain_created = _interview_chain_start.update_interview_context(active_round_id="round-1")
_interview_chain_prepared = _interview_chain_created.update_interview_context(
    active_preparation_id="prep-1"
)

_intent_chain_start = ConversationTaskState(tool_profile="job")
_intent_chain_proposed = _intent_chain_start.with_pending_proposal(
    "pending_job_intent_update", JobIntentUpdate(city="上海"), proposed_at=FIXTURE_NOW
)
_amendment_chain_start = ConversationTaskState(tool_profile="memory")
_amendment_chain_proposed = _amendment_chain_start.with_pending_proposal(
    "pending_memory_amendment",
    MemoryAmendmentProposal(
        target_kind="career_evidence", detail_ref="detail_" + "a" * 24,
        new_claim="我负责过增长实验", reason="用户更正了原句",
    ),
    proposed_at=FIXTURE_NOW,
)
_calendar_chain_start = ConversationTaskState(
    tool_profile="interview", active_interview_round_id="round-1"
)
_calendar_chain_proposed = _calendar_chain_start.update_interview_context(
    active_calendar_proposal_id="proposal-1",
    calendar_proposal_expires_at=FIXTURE_NOW + timedelta(hours=1),
)

_CHAIN_HOLDOUT = (
    _sequence(
        "selection_holdout_chain_job_resume", "holdout", "chain",
        "找出我存的那条甲公司岗位，拆一下要求，再看刚才的简历匹不匹配，合适的话改一版。",
        _job_chain_start,
        ("find_saved_jobs", "analyze_job", "match_resume_to_job", "draft_resume_tailoring"),
        frozenset({"job.library", "job.analysis", "resume.match", "resume.tailoring"}),
        resulting_states=(_job_chain_found, _job_chain_analyzed, _job_chain_matched),
    ),
    _sequence(
        "selection_holdout_chain_application_email", "holdout", "chain",
        "找出我投甲公司的记录，看看它对应的邮件，把那封面试邀请对上。",
        _application_chain_start,
        ("list_applications", "list_email_events", "resolve_email_event"),
        frozenset({"application.tracking", "application.email"}),
        resulting_states=(_application_chain_found, _application_chain_emailed),
    ),
    _sequence(
        "selection_holdout_chain_interview_calendar", "holdout", "chain",
        "给这条投递登记明天的面试，准备一下，再让我预览要加到日历的内容。",
        _interview_chain_start,
        ("create_interview", "prepare_interview", "prepare_interview_calendar_sync"),
        frozenset({"interview.schedule", "interview.prep", "interview.calendar"}),
        resulting_states=(_interview_chain_created, _interview_chain_prepared),
    ),
    _sequence(
        "selection_holdout_chain_intent_confirmation", "holdout", "chain",
        "以后只看上海的机会，先给我确认一下再记住。",
        _intent_chain_start, ("propose_job_intent", "confirm_job_intent"),
        frozenset({"job.intent"}),
        resulting_states=(_intent_chain_proposed,), later_messages=("对，就这样。",),
        observation_states=("job_intent_proposed",),
    ),
    _sequence(
        "selection_holdout_chain_memory_amendment", "holdout", "chain",
        "上次那条经历写错了，应该是我负责增长实验；给我看看改法。",
        _amendment_chain_start,
        ("propose_memory_amendment", "confirm_memory_amendment"),
        frozenset({"memory.proposals"}),
        resulting_states=(_amendment_chain_proposed,), later_messages=("这句改得对，确认。",),
        observation_states=("memory_amendment_proposed",),
    ),
    _sequence(
        "selection_holdout_chain_calendar_approval", "holdout", "chain",
        "这轮面试先生成日历预览，我看完再决定要不要加。",
        _calendar_chain_start,
        ("prepare_interview_calendar_sync", "get_calendar_proposal", "execute_calendar_proposal"),
        frozenset({"interview.calendar"}),
        resulting_states=(_calendar_chain_proposed, _calendar_chain_proposed),
        later_messages=(None, "刚才这个日历预览没问题，执行吧。"),
        observation_states=("calendar_approval_required", "calendar_proposal_found"),
    ),
)


def _control(
    name: str,
    split: Literal["dev", "holdout"],
    message: str,
    task: ConversationTaskState,
    step: TrajectoryStep,
    namespaces: frozenset[str] = frozenset({"control"}),
) -> SelectionCase:
    return SelectionCase(
        TrajectoryScenario(
            name=f"selection_{split}_control_{name}",
            policy="Do not expose a write solely because a related read is relevant.",
            context=_context(message, task),
            steps=(step,),
        ),
        split,
        "control",
        namespaces,
    )


_CONTROL_DEV = (
    _control(
        "planning_to_apply", "dev", "这个岗位我准备投了，先放在心里。",
        ConversationTaskState(tool_profile="job", active_job_posting_id="job-1"),
        TrajectoryStep(expect_action="final", forbid_tools=frozenset({"create_application"})),
    ),
    _control(
        "interview_read_only", "dev", "看看这轮面试是几点，先别动日历。",
        ConversationTaskState(tool_profile="interview", active_interview_round_id="round-1"),
        TrajectoryStep(expect_tool="get_interview", forbid_tools=frozenset({"prepare_interview_calendar_sync"})),
        frozenset({"control", "interview.schedule"}),
    ),
    SelectionCase(
        TrajectoryScenario(
            name="selection_dev_control_note_derived_filter_waits",
            policy="Ask before reusing a filter inferred only from unconfirmed notes.",
            context=_context(
                "帮我找岗位，但别猜我想搜什么方向。",
                ConversationTaskState(tool_profile="job"),
            ).model_copy(update={
                "tool_observations": (DecisionObservation(
                    tool_name="find_saved_jobs",
                    state="working_notes_derived_argument",
                    message="Rust 只在未确认笔记里；请用户确认后再搜索。",
                    arguments={"query": "Rust"},
                ),),
            }),
            steps=(TrajectoryStep(
                expect_action="ask_user",
                forbid_tools=frozenset({"find_saved_jobs"}),
            ),),
        ),
        "dev", "control", frozenset({"control", "job.library"}),
    ),
)

_CONTROL_HOLDOUT = (
    _control(
        "job_maybe_later", "holdout", "这个职位我再想想，暂时别帮我投。",
        ConversationTaskState(tool_profile="job", active_job_posting_id="job-1"),
        TrajectoryStep(expect_action="final", forbid_tools=frozenset({"create_application"})),
    ),
    _control(
        "resume_pause", "holdout", "刚才那份简历先别改，我想缓缓。",
        ConversationTaskState(tool_profile="resume", active_resume_job_match_id="match-1"),
        TrajectoryStep(expect_action="final", forbid_tools=frozenset({"draft_resume_tailoring"})),
    ),
    _control(
        "today_read", "holdout", "今天有什么要紧的事？",
        ConversationTaskState(tool_profile="core"),
        TrajectoryStep(expect_tools=frozenset({"get_daily_brief", "list_action_items"}), forbid_tools=frozenset({"complete_action_item", "dismiss_action_item"})),
        frozenset({"control", "actions"}),
    ),
    _control(
        "no_calendar", "holdout", "这场面试怎么安排的？不用加到日历。",
        ConversationTaskState(tool_profile="interview", active_interview_round_id="round-1"),
        TrajectoryStep(expect_tool="get_interview", forbid_tools=frozenset({"prepare_interview_calendar_sync"})),
        frozenset({"control", "interview.schedule"}),
    ),
    _control(
        "proposal_not_approved", "holdout", "那个日历改动我还没答应，先别执行。",
        ConversationTaskState(tool_profile="interview", active_calendar_proposal_id="proposal-1"),
        TrajectoryStep(expect_action="final", forbid_tools=frozenset({"execute_calendar_proposal"})),
    ),
    SelectionCase(
        TrajectoryScenario(
            name="selection_holdout_control_waiting_for_confirmation",
            policy="A proposal awaiting user input does not authorize another call in the same turn.",
            context=_context("那段经历先让我看看你准备怎么改。", _amendment_chain_start),
            steps=(
                TrajectoryStep(expect_tool="propose_memory_amendment"),
                TrajectoryStep(
                    observation=DecisionObservation(
                        tool_name="propose_memory_amendment", state="memory_amendment_proposed",
                        message="等待用户确认。", arguments={},
                    ),
                    task_update={"pending_interaction": _amendment_chain_proposed.pending_interaction},
                    expect_action="ask_user",
                ),
            ),
        ),
        "holdout", "control", frozenset({"control", "memory.proposals"}),
    ),
)


_PROMOTED_CROSS = frozenset({
    "memory_then_job", "application_then_interview", "source_then_tailor",
})
_PROMOTED_CHAIN = frozenset({
    "job_resume", "application_email", "interview_calendar",
})
_DEMOTED_SINGLE = frozenset({
    "selection_dev_job_library_first",
    "selection_dev_application_tracking_first",
    "selection_dev_resume_library_next",
    "selection_dev_resume_match_next",
    "selection_dev_interview_schedule_first",
    "selection_dev_interview_prep_next",
    "selection_dev_interview_calendar_first",
})


def _in_split(case: SelectionCase, split: Literal["dev", "holdout"]) -> SelectionCase:
    old = f"selection_{case.split}_"
    new_name = case.scenario.name.replace(old, f"selection_{split}_", 1)
    return replace(case, split=split, scenario=replace(case.scenario, name=new_name))


SELECTION_DEV: tuple[SelectionCase, ...] = (
    *(SelectionCase(
        scenario=scenario,
        split="dev",
        kind="cross" if scenario.name in {
            "offline_resume_to_interview_preparation",
            "offline_job_to_application_creation",
        } else "chain",
        namespaces={
            "offline_resume_to_interview_preparation": frozenset({"resume.library", "interview.prep"}),
            "offline_job_to_application_creation": frozenset({"job.library", "application.tracking"}),
            "offline_application_to_email_events": frozenset({"application.tracking", "application.email"}),
            "offline_interview_to_calendar_sync": frozenset({"interview.schedule", "interview.calendar"}),
            "offline_match_to_resume_tailoring": frozenset({"resume.match", "resume.tailoring"}),
        }[scenario.name],
        raw_turn=False,
    ) for scenario in TOOL_SELECTION_SCENARIOS),
    *(case for case in _SINGLE_DEV if case.scenario.name not in _DEMOTED_SINGLE),
    *(case for case in _CROSS_CASES if case.split == "dev"),
    *(_in_split(case, "dev") for case in _CROSS_CASES if case.scenario.name.removeprefix("selection_holdout_cross_") in _PROMOTED_CROSS),
    *(_in_split(case, "dev") for case in _CHAIN_HOLDOUT if case.scenario.name.removeprefix("selection_holdout_chain_") in _PROMOTED_CHAIN),
    *_CONTROL_DEV,
)
SELECTION_HOLDOUT: tuple[SelectionCase, ...] = (
    *(_in_split(case, "holdout") for case in _SINGLE_DEV if case.scenario.name in _DEMOTED_SINGLE),
    *(case for case in _CROSS_CASES if case.split == "holdout" and case.scenario.name.removeprefix("selection_holdout_cross_") not in _PROMOTED_CROSS),
    *(case for case in _CHAIN_HOLDOUT if case.scenario.name.removeprefix("selection_holdout_chain_") not in _PROMOTED_CHAIN),
    *(case for case in _CONTROL_HOLDOUT if case.scenario.name != "selection_holdout_control_proposal_not_approved"),
)
