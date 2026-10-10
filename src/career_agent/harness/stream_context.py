"""Request-local stream state shared by runtime and presentation."""
from contextvars import ContextVar

from career_agent.harness.streaming import StreamEventSink

STREAM_SINK: ContextVar[StreamEventSink | None] = ContextVar(
    "main_agent_stream_sink", default=None,
)
EARLY_RESOURCE_IDS: ContextVar[set[str] | None] = ContextVar(
    "main_agent_early_resource_ids", default=None,
)
