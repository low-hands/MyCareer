from __future__ import annotations

from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class ToolPermission(StrEnum):
    """The side-effect class a tool belongs to."""

    READ = "read"
    WRITE = "write"
    EXTERNAL = "external"


class AgentRequest(BaseModel):
    """Framework-neutral goal for one agent turn.

    ``message`` normally contains a user query, but an approved event or
    scheduled task may generate the goal as well.
    """

    message: str = Field(min_length=1)
    thread_id: str = Field(default_factory=lambda: uuid4().hex)
    user_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    allowed_permissions: set[ToolPermission] = Field(default_factory=lambda: {ToolPermission.READ})


class AgentResult(BaseModel):
    """Framework-neutral result for one agent turn."""

    thread_id: str
    trace_id: str
    output: str
    model_messages: int = 0
    tool_calls: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class PreparedContext(BaseModel):
    """Context selected for a single run."""

    system_prompt: str
    facts: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
