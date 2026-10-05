"""Frozen generalization probes, kept out of production prompts and policies."""
from datetime import datetime, timezone

from career_agent.agent.contracts.context import MainAgentContext
from career_agent.agent.contracts.profile import CareerProfileContext
from career_agent.agent.contracts.resources import ConversationMessageContext, ConversationResourceReference
from career_agent.agent.contracts.task_state import ConversationTaskState
from career_agent.evaluation.trajectory import TrajectoryScenario, TrajectoryStep

_AT = datetime(2026, 9, 18, 12, tzinfo=timezone.utc)


def paired(kind, user_message, descriptions, task):
    references = tuple(ConversationResourceReference(
        kind=kind, resource_id=f"holdout-{kind}-{index}", title="面试材料",
        description=description,
    ) for index, description in enumerate(descriptions))
    context = MainAgentContext(
        conversation_id="holdout-resources", profile=CareerProfileContext(user_id="eval-user"),
        user_message=user_message, task=task,
        recent_messages=tuple(ConversationMessageContext(
            role="assistant", content=description, created_at=_AT,
            resource_refs=(reference,),
        ) for description, reference in zip(descriptions, references)),
    )
    return context, context.reference_handle(references[0])


_PREP, _PREP_REFERENCE = paired(
    "interview_preparation", "把初面前那份准备材料再打开一下，终面那份先不用，也别重新生成。",
    ("初面之前生成的准备材料，覆盖团队协作和项目经历。", "后来终面之前生成的准备材料，覆盖架构设计和管理。"),
    ConversationTaskState(active_interview_preparation_id="holdout-interview_preparation-1"),
)
_MOCK, _MOCK_REFERENCE = paired(
    "mock_interview_report", "我想看练产品面试那一场的复盘；当前后端练习的先别读。",
    ("产品面试练习结束后的复盘报告。", "后端面试练习结束后的复盘报告。"),
    ConversationTaskState(active_application_id="current-backend-application"),
)
_MISSING = MainAgentContext(
    conversation_id="holdout-missing", profile=CareerProfileContext(user_id="eval-user"),
    user_message="之前群面前的准备材料还找得到吗？只想读那份，不要重新准备。",
    recent_messages=(ConversationMessageContext(
        role="assistant", content="当前只有一份公司调研。",
        created_at=_AT, resource_refs=(ConversationResourceReference(
            kind="job_research_report", resource_id="unrelated-research", title="公司调研",
            status_at_delivery="current", anchored_by_other_job=False,
        ),),
    ),),
)

RESOURCE_BINDING_HOLDOUTS = (
    TrajectoryScenario(
        name="holdout_preparation_with_duplicate_titles_uses_the_requested_resource",
        policy="A requested stored preparation is selected by its provenance, never by another active preparation or display-title equality.",
        context=_PREP, decisive_facts=("recent_messages.0.resources", "task.has_active_interview_preparation"),
        steps=(TrajectoryStep(expect_tool="get_interview_preparation", expect_arguments={"reference": _PREP_REFERENCE}, forbid_tools=frozenset({"prepare_interview"})),),
        recording_samples=3,
    ),
    TrajectoryScenario(
        name="holdout_mock_report_does_not_borrow_the_active_application",
        policy="Reading a specified past mock report binds its exact resource, not the current application's newest report.",
        context=_MOCK, decisive_facts=("recent_messages.0.resources", "task.has_active_application"),
        steps=(TrajectoryStep(expect_tool="get_mock_interview_result", expect_arguments={"reference": _MOCK_REFERENCE}),),
        recording_samples=3,
    ),
    TrajectoryScenario(
        name="holdout_missing_preparation_does_not_substitute_company_research",
        policy="A different kind of report cannot substitute for unavailable preparation evidence, and a read request does not authorize regenerating it.",
        context=_MISSING, decisive_facts=("recent_messages.0.resources", "user_message"),
        steps=(TrajectoryStep(forbid_tools=frozenset({"get_job_research", "get_interview_preparation", "research_job", "prepare_interview"})),),
        recording_samples=3,
    ),
    TrajectoryScenario(
        name="holdout_mixed_sources_keep_claim_search_when_preparation_is_missing",
        policy="An unavailable resource does not suppress an explicitly requested search of a different evidence source.",
        context=_MISSING.model_copy(update={"user_message": "先搜索我已确认的工作经历，再读取之前群面前的准备材料。材料找不到就说明，不要重新生成。"}),
        decisive_facts=("user_message", "recent_messages.0.resources"),
        steps=(TrajectoryStep(expect_tool="search_career_memory", forbid_tools=frozenset({"research_job", "prepare_interview"})),),
        recording_samples=3,
    ),
)
