from career_agent.evaluation.rederivation import (
    count_rederivations,
    summarize_rederivations,
    tool_call_fingerprint,
)


def test_same_tool_and_arguments_after_compaction_counts_as_rederivation() -> None:
    events = (
        {
            "event_type": "tool_call",
            "tool_name": "find_saved_jobs",
            "arguments": {"query": "X", "limit": 5},
        },
        {"event_type": "context_compacted"},
        {
            "event_type": "tool_call",
            "tool_name": "find_saved_jobs",
            "arguments": {"limit": 5, "query": "X"},
        },
    )

    assert count_rederivations(events) == 1


def test_page_in_after_compaction_is_not_rederivation() -> None:
    events = (
        {
            "event_type": "tool_call",
            "tool_name": "find_saved_jobs",
            "arguments": {"query": "X"},
        },
        {"event_type": "context_compacted"},
        {
            "event_type": "tool_call",
            "tool_name": "read_conversation_span",
            "arguments": {"from_sequence": 1, "through_sequence": 8},
        },
    )

    assert count_rederivations(events) == 0


def test_production_argument_digest_is_accepted_without_raw_arguments() -> None:
    fingerprint = tool_call_fingerprint("find_saved_jobs", {"query": "X"})
    events = (
        {
            "event_type": "model_succeeded",
            "details": {
                "tool_name": "find_saved_jobs",
                "tool_arguments_fingerprint": fingerprint,
            },
        },
        {"event_type": "context_compacted"},
        {
            "event_type": "model_succeeded",
            "details": {
                "tool_name": "find_saved_jobs",
                "tool_arguments_fingerprint": fingerprint,
            },
        },
    )

    assert count_rederivations(events) == 1
    assert summarize_rederivations(events).measurable is True


def test_trace_without_calls_on_both_sides_is_not_reported_as_zero() -> None:
    summary = summarize_rederivations(
        (
            {"event_type": "context_compacted"},
            {
                "event_type": "tool_call",
                "tool_name": "read_conversation_span",
                "arguments": {"from_sequence": 1, "through_sequence": 4},
            },
        )
    )

    assert summary.rederivation_count == 0
    assert summary.measurable is False
