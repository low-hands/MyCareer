from __future__ import annotations

import json
from pathlib import Path

from career_agent.agent.context_manager import ContextManager
from career_agent.agent.conversation_memory_contracts import (
    ConversationSummaryContent,
    SummaryMessage,
)
from career_agent.agent.main_agent_contracts import ConversationTaskState
from career_agent.agent.openai_compatible_client import AgentWorkerError, ProviderErrorMetadata
from career_agent.harness.observability import ACTIVE_TRACE_CONTEXT, InMemoryTraceRecorder
from career_agent.storage.context import CareerContextStore


class RejectedSummary:
    def summarize(
        self,
        *,
        previous: ConversationSummaryContent | None,
        messages: tuple[SummaryMessage, ...],
    ) -> ConversationSummaryContent:
        raise AgentWorkerError(
            "CONVERSATION_SUMMARY_REJECTED_400",
            "synthetic-private-provider-message",
            detail="synthetic-private-provider-body",
            provider=ProviderErrorMetadata(
                status=400, code="InvalidParameter", param="response_format.json_schema",
                type="synthetic_private_provider_type",
            ),
        )


def test_failed_summary_trace_preserves_safe_metadata_without_changing_backoff(
    tmp_path: Path,
) -> None:
    store = CareerContextStore(tmp_path / "trace.sqlite3")
    writer = ContextManager(store)
    for _ in range(12):
        context = writer.load_for_turn(user_id="u", conversation_id="c", user_message="synthetic")
        writer.commit_turn(context=context, task=ConversationTaskState(), assistant_message="ack")
    manager = ContextManager(store, summary_worker=RejectedSummary())
    manager.configure_request_token_estimator(lambda context: (16000, 32000))
    recorder = InMemoryTraceRecorder()
    token = ACTIVE_TRACE_CONTEXT.set((recorder, "synthetic-failure"))
    try:
        for _ in range(4):
            manager.load_for_turn(user_id="u", conversation_id="c", user_message="next")
    finally:
        ACTIVE_TRACE_CONTEXT.reset(token)
    events = [
        event for event in recorder.snapshot("synthetic-failure").events
        if event.event_type == "context_compaction_failed"
    ]
    assert len(events) == 3
    assert [event.details["compaction_suspended"] for event in events] == [False, False, True]
    for event in events:
        assert event.error_code == "CONVERSATION_SUMMARY_REJECTED_400"
        assert event.recoverable is False
        assert event.details["provider"] == {
            "status": 400, "code": "InvalidParameter", "param": "response_format.json_schema",
            "type": None, "category": "configuration", "retryable": False,
        }
        serialized = json.dumps(event.details)
        assert "synthetic-private" not in serialized
        assert "synthetic_private" not in serialized
    assert store.get_conversation_summary(user_id="u", conversation_id="c") is None
    assert len(store.list_message_records("u", "c", limit=24)) == 24
