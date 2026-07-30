import pytest

from career_agent.contracts import AgentRequest
from career_agent.jobs.grounding_guard import JobGroundingGuard
from career_agent.runtime.response_guard import ToolCallEvidence, TurnEvidence

pytestmark = pytest.mark.asyncio


async def test_search_request_requires_successful_search_tool() -> None:
    guard = JobGroundingGuard()
    request = AgentRequest(message="帮我找一些北京的 Agent 开发岗位")

    rejected = await guard.review(
        request=request,
        response_text="这里有几个岗位。",
        evidence=TurnEvidence(
            available_tool_names=("career_search_jobs", "career_get_job_detail"),
        ),
    )
    accepted = await guard.review(
        request=request,
        response_text="没有找到符合条件的岗位。",
        evidence=TurnEvidence(
            available_tool_names=("career_search_jobs", "career_get_job_detail"),
            tool_calls=(
                ToolCallEvidence(
                    name="career_search_jobs",
                    succeeded=True,
                    output='{"status":"no_results"}',
                ),
            ),
        ),
    )

    assert not rejected.accepted
    assert rejected.required_tool == "career_search_jobs"
    assert accepted.accepted


async def test_follow_up_reference_requires_detail_tool() -> None:
    guard = JobGroundingGuard()

    review = await guard.review(
        request=AgentRequest(message="我想看第 2 个岗位的完整 JD"),
        response_text="第二个岗位的职责是……",
        evidence=TurnEvidence(
            available_tool_names=("career_search_jobs", "career_get_job_detail"),
            tool_calls=(
                ToolCallEvidence(
                    name="career_search_jobs",
                    succeeded=True,
                    output='{"status":"published"}',
                ),
            ),
        ),
    )

    assert not review.accepted
    assert review.required_tool == "career_get_job_detail"


async def test_non_listing_advice_does_not_require_job_tool() -> None:
    review = await JobGroundingGuard().review(
        request=AgentRequest(message="怎么准备技术面试？"),
        response_text="可以先整理项目经历。",
        evidence=TurnEvidence(
            available_tool_names=("career_search_jobs", "career_get_job_detail"),
        ),
    )

    assert review.accepted


async def test_failed_tool_call_does_not_satisfy_grounding() -> None:
    review = await JobGroundingGuard().review(
        request=AgentRequest(message="搜索上海的后端开发职位"),
        response_text="我找到了几个职位。",
        evidence=TurnEvidence(
            available_tool_names=("career_search_jobs",),
            tool_calls=(
                ToolCallEvidence(
                    name="career_search_jobs",
                    succeeded=False,
                    output='{"ok":false,"error":{"code":"timeout"}}',
                ),
            ),
        ),
    )

    assert not review.accepted
