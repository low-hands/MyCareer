from __future__ import annotations

from datetime import datetime, timezone
import secrets
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class AgentSession(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    user_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    status: Literal["active", "closed"] = "active"
    created_at: datetime
    last_active_at: datetime
    # Persisted for byte-stable prompt prefixes. This is a spotlight delimiter,
    # not an authentication secret: reassess rotation before any future tool can
    # copy prompt delimiters into durable content that is later projected back.
    spotlight_nonce: str = Field(
        default_factory=lambda: secrets.token_hex(16), min_length=32, max_length=32
    )

    def touch(self) -> "AgentSession":
        return self.model_copy(update={"last_active_at": datetime.now(timezone.utc), "status": "active"})
