from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


AuthorizationRefusalKind = Literal[
    "not_offered",
    "preference_deny",
    "budget_exhausted",
    "duplicate_call",
    "retry_limit",
    "seal_unavailable",
]


@dataclass(frozen=True)
class AuthorizationRefusal:
    kind: AuthorizationRefusalKind
    reason: str
    next_action: str
