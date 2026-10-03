import pytest

from career_agent.agent.runtime_observability import RuntimeObservability
from career_agent.agent.turn_coordinator import STREAM_SINK


@pytest.mark.parametrize(
    ("tool_name", "capability"),
    [
        ("find_saved_jobs", "job_search"),
        ("retry_job_research", "job_research"),
        ("finalize_resume_tailoring", "resume"),
        ("create_interview", "interview"),
        ("unknown_capability", "career_task"),
    ],
)
def test_runtime_observability_exposes_only_public_capability_groups(
    tool_name: str, capability: str
) -> None:
    assert RuntimeObservability.public_capability(tool_name) == capability


def test_runtime_observability_maps_post_reply_failure_to_commit_failure() -> None:
    events = []
    token = STREAM_SINK.set(events.append)
    try:
        RuntimeObservability.emit_turn_failure(
            turn_id="turn-1",
            error=RuntimeError("database unavailable"),
            reply_delivered=True,
        )
    finally:
        STREAM_SINK.reset(token)

    assert len(events) == 1
    assert events[0].type == "turn_failed"
    assert events[0].code == "TURN_COMMIT_FAILED"
