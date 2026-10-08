from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


AuthorizationRefusalKind = Literal[
    "not_offered",
    "precondition",
    "waiting_for_user",
    "preference_deny",
    "budget_exhausted",
    "duplicate_call",
    "retry_limit",
    "seal_unavailable",
    # Refused after the gates above, by argument binding rather than policy.
    # They count against projection_refusals, not authorization_refusals.
    "argument_projection",
    "working_notes",
]


@dataclass(frozen=True)
class AuthorizationRefusal:
    kind: AuthorizationRefusalKind
    reason: str
    next_action: str
