from datetime import datetime, timezone

import pytest

from career_agent.agent.delivered_body_contracts import SavedJobBodySource
from career_agent.agent.delivery_policy import DELIVERY_POLICIES
from career_agent.agent.main_agent_contracts import (
    ConversationSpanMessage,
    ConversationSpanView,
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
        "saved_job_ready": "source",
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


def test_source_metadata_stays_internal_and_only_the_handle_is_persisted() -> None:
    output = ToolResult(
        tool_name="get_saved_job",
        state="saved_job_ready",
        message="已读取岗位。",
        payload={"jd_snapshot": {"content": "LIVE JD BODY"}},
        body_source=SavedJobBodySource(job_posting_id="internal-job-id"),
    )
    assert "internal-job-id" not in output.model_dump_json()
    assert "body_source" not in output.model_dump()
    assert "LIVE JD BODY" in MainAgentRuntime._undelivered_bodies((output,))
    bodies = MainAgentRuntime._delivered_bodies((output,))
    assert len(bodies) == 1 and bodies[0].body == ""
    assert bodies[0].source == output.body_source
    assert bodies[0].dependencies[0].resource_id == "internal-job-id"
