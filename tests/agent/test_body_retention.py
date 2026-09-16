from datetime import datetime, timezone

import pytest

from career_agent.agent.delivery_policy import DELIVERY_POLICIES
from career_agent.agent.main_agent_contracts import (
    ConversationResourceReference,
    ConversationSpanMessage,
    ConversationSpanView,
    ToolObservation,
    ToolResult,
)
from career_agent.agent.main_agent_runtime import MainAgentRuntime


def test_only_snapshots_and_entity_handles_are_retained() -> None:
    assert {
        state: policy.body_retention
        for state, policy in DELIVERY_POLICIES.items()
        if policy.body_retention != "none"
    } == {
        "daily_brief_ready": "snapshot",
        "saved_jobs_compared": "snapshot",
        "resume_analysis_ready": "source",
        "mock_interview_question_found": "source",
    }


@pytest.mark.parametrize(
    "state",
    [
        "conversation_span_found",
        "claim_source_found",
        "career_memory_detail_found",
        "career_memory_search_found",
        "career_episode_search_found",
        "career_history_found",
    ],
)
def test_none_retention_keeps_live_readback_without_a_second_persistent_copy(
    state,
) -> None:
    body = "这是本轮需要完整展示的回读内容。"
    payload = {"body": body, "source_quote": body}
    if state == "conversation_span_found":
        payload = ConversationSpanView(
            from_sequence=1,
            through_sequence=1,
            returned=1,
            total=1,
            messages=(
                ConversationSpanMessage(
                    sequence=1,
                    role="user",
                    content=body,
                    created_at=datetime.now(timezone.utc),
                ),
            ),
        ).model_dump(mode="json")
    output = ToolResult(
        tool_name="read", state=state, message="已回读。", payload=payload
    )
    assert body in MainAgentRuntime._undelivered_bodies((output,))
    assert MainAgentRuntime._delivered_bodies((output,)) == ()


def test_a_saved_jd_is_the_models_observation_and_the_readers_card_only() -> None:
    output = ToolObservation(
        tool_name="get_saved_job",
        state="saved_job_ready",
        message="已读取岗位。",
        payload={"jd_snapshot": {"id": "jds_1", "content": "LIVE JD BODY"}},
        resource_ref=ConversationResourceReference(
            kind="saved_job",
            resource_id="jds_1",
            job_posting_id="internal-job-id",
            title="示例公司｜工程师",
            description="JD 第 1 版 · test",
        ),
    )
    observation = MainAgentRuntime._tool_observation("get_saved_job", output)
    assert observation.body is not None and "LIVE JD BODY" in observation.body
    assert "LIVE JD BODY" not in MainAgentRuntime._undelivered_bodies((output,))
    assert MainAgentRuntime._delivered_bodies((output,)) == ()
    assert MainAgentRuntime._turn_is_card_backed((output,))
    assert MainAgentRuntime._turn_resource_refs((output,)) == (output.resource_ref,)
