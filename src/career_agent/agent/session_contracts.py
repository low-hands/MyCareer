from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class AgentSession(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    user_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    status: Literal["active", "closed"] = "active"
    created_at: datetime
    last_active_at: datetime

    def touch(self) -> "AgentSession":
        return self.model_copy(update={"last_active_at": datetime.now(timezone.utc), "status": "active"})
