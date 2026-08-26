from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


ActionType = Literal[
    "application_follow_up",
    "email_event_confirmation",
    "interview_preparation",
    "material_submission",
    "interview_reminder",
    "interview_retro",
]
ActionSourceType = Literal["application", "email_event", "interview_round"]
# "completed" means the user did the thing; "obsolete" means the condition that
# generated the item went away on its own. Collapsing the two would let the
# history claim credit for work nobody did.
ActionStatus = Literal["open", "completed", "dismissed", "snoozed", "obsolete"]
# Statuses that carry a resolved_at timestamp.
RESOLVED_ACTION_STATUSES = frozenset({"completed", "dismissed", "obsolete"})
# Statuses that record a user decision and must therefore survive a refresh.
# "obsolete" is deliberately absent: it is a system observation, so the item
# reopens if the generating condition comes back.
USER_RESOLVED_ACTION_STATUSES = frozenset({"completed", "dismissed"})


class ActionCenterContract(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )


class ActionItem(ActionCenterContract):
    id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    stable_key: str = Field(min_length=1, max_length=1000)
    action_type: ActionType
    source_type: ActionSourceType
    source_id: str = Field(min_length=1)
    application_id: str | None = Field(default=None, min_length=1)
    title: str = Field(min_length=1, max_length=500)
    summary: str = Field(min_length=1, max_length=2000)
    due_at: datetime | None = None
    status: ActionStatus
    snoozed_until: datetime | None = None
    created_at: datetime
    updated_at: datetime
    resolved_at: datetime | None = None

    @model_validator(mode="after")
    def validate_lifecycle(self) -> "ActionItem":
        if self.status == "snoozed" and self.snoozed_until is None:
            raise ValueError("snoozed actions require snoozed_until")
        if self.status != "snoozed" and self.snoozed_until is not None:
            raise ValueError("only snoozed actions can have snoozed_until")
        if self.status in RESOLVED_ACTION_STATUSES and self.resolved_at is None:
            raise ValueError("resolved actions require resolved_at")
        if self.status in {"open", "snoozed"} and self.resolved_at is not None:
            raise ValueError("active actions cannot have resolved_at")
        return self


class ActionItemEvent(ActionCenterContract):
    id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    action_item_id: str = Field(min_length=1)
    event_type: Literal[
        "created",
        "refreshed",
        "completed",
        "dismissed",
        "snoozed",
        "reopened",
        "obsoleted",
    ]
    previous_status: ActionStatus | None = None
    new_status: ActionStatus
    occurred_at: datetime


class ActionCandidate(ActionCenterContract):
    stable_key: str = Field(min_length=1, max_length=1000)
    action_type: ActionType
    source_type: ActionSourceType
    source_id: str = Field(min_length=1)
    application_id: str | None = Field(default=None, min_length=1)
    title: str = Field(min_length=1, max_length=500)
    summary: str = Field(min_length=1, max_length=2000)
    due_at: datetime | None = None


class DailyBrief(ActionCenterContract):
    user_id: str = Field(min_length=1)
    timezone: str = Field(min_length=1)
    generated_at: datetime
    overdue: tuple[ActionItem, ...] = ()
    due_today: tuple[ActionItem, ...] = ()
    upcoming: tuple[ActionItem, ...] = ()
    no_due_date: tuple[ActionItem, ...] = ()

