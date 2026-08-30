"""Read-only HTTP surface for the user's own interface.

This is a different boundary from the agent's tools, and the difference is
deliberate. A tool returns an opaque observation because the model must not be
handed a complete JD or resume it could then paraphrase as its own conclusion.
The person looking at their own dashboard is under no such restriction: it is
their data, and withholding it from them would be a bug, not a safeguard.

What the two boundaries share is that neither may become a way around the
other. Nothing here is reachable by the model, and nothing here writes domain
state on the user's behalf: these endpoints answer "what do I have", and every
change still goes through the agent, where it gets confirmation and an audit
trail.

Identity is the caller's asserted ``user_id``, which is exactly as strong as the
rest of this deployment: a local single-user process with no CORS and no auth.
It is a scoping key, not an authorization boundary, and must not be treated as
one if this ever leaves localhost.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Query
from pydantic import BaseModel, ConfigDict, Field

from career_agent.domain.action_center import ActionItem, DailyBrief
from career_agent.services.action_center import ActionCenterService
from career_agent.services.applications import ApplicationService
from career_agent.services.email_tracking import EmailTrackingService
from career_agent.services.interviews import InterviewService
from career_agent.storage.action_center import SQLiteActionItemStore
from career_agent.storage.applications import SQLiteApplicationStore
from career_agent.storage.email_tracking import SQLiteEmailTrackingStore
from career_agent.storage.interviews import SQLiteInterviewStore
from career_agent.storage.jobs import SQLiteJobPostingRepository
from career_agent.storage.resumes import ResumeStore
from career_agent.connectors.email_accounts import EnvironmentEmailConnectorResolver


class ActionItemView(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    action_type: str
    source_type: str
    application_id: str | None = None
    title: str
    summary: str
    due_at: datetime | None = None
    status: str
    snoozed_until: datetime | None = None

    @classmethod
    def of(cls, item: ActionItem) -> "ActionItemView":
        # stable_key and source_id stay behind: they are derivation plumbing,
        # and echoing them would invite a client to key its own state on them.
        return cls(
            id=item.id,
            action_type=item.action_type,
            source_type=item.source_type,
            application_id=item.application_id,
            title=item.title,
            summary=item.summary,
            due_at=item.due_at,
            status=item.status,
            snoozed_until=item.snoozed_until,
        )


class DailyBriefResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    timezone: str
    generated_at: datetime
    overdue: tuple[ActionItemView, ...] = ()
    due_today: tuple[ActionItemView, ...] = ()
    upcoming: tuple[ActionItemView, ...] = ()
    no_due_date: tuple[ActionItemView, ...] = ()

    @classmethod
    def of(cls, brief: DailyBrief) -> "DailyBriefResponse":
        return cls(
            timezone=brief.timezone,
            generated_at=brief.generated_at,
            overdue=tuple(ActionItemView.of(item) for item in brief.overdue),
            due_today=tuple(ActionItemView.of(item) for item in brief.due_today),
            upcoming=tuple(ActionItemView.of(item) for item in brief.upcoming),
            no_due_date=tuple(ActionItemView.of(item) for item in brief.no_due_date),
        )


def build_action_center_service(args: argparse.Namespace) -> ActionCenterService:
    """Assemble the action centre without requiring any model configuration.

    Generated actions are derived from applications, interviews, and recorded
    email events, none of which needs a worker to read. EmailTrackingService
    falls back to its deterministic worker, so a dashboard keeps working on a
    machine where no API key is set.
    """
    job_repository = SQLiteJobPostingRepository(Path(args.job_store).expanduser())
    resume_store = ResumeStore(Path(args.resume_store).expanduser())
    application_service = ApplicationService(
        SQLiteApplicationStore(Path(args.application_store).expanduser()),
        job_repository,
        resume_store,
    )
    interview_service = InterviewService(
        SQLiteInterviewStore(Path(args.application_store).expanduser()),
        application_service,
    )
    email_tracking_service = EmailTrackingService(
        SQLiteEmailTrackingStore(Path(args.email_store).expanduser()),
        application_service,
        EnvironmentEmailConnectorResolver(),
        interview_service=interview_service,
    )
    return ActionCenterService(
        SQLiteActionItemStore(Path(args.action_store).expanduser()),
        application_service,
        email_tracking_service,
        interview_service,
    )


def build_read_router(
    action_center_factory: Callable[[], ActionCenterService],
) -> APIRouter:
    """Wire the read endpoints against a lazily built service.

    The factory runs on the first request rather than at wiring time, so
    creating an app never opens the local databases as a side effect.
    """
    router = APIRouter(prefix="/v1")
    cached: dict[str, ActionCenterService] = {}

    def action_center() -> ActionCenterService:
        if "service" not in cached:
            cached["service"] = action_center_factory()
        return cached["service"]

    @router.get("/daily-brief", response_model=DailyBriefResponse)
    async def daily_brief(
        user_id: str = Query(min_length=1, max_length=200),
        timezone: str = Query(default="Asia/Shanghai", min_length=1, max_length=100),
    ) -> DailyBriefResponse:
        # This regenerates derived action items before answering. That is a
        # write, which a GET would normally not do, but the items are a pure
        # function of the pipeline keyed by stable_key: reading a stale brief
        # would be the actual surprise. Nothing the user authored is touched.
        return DailyBriefResponse.of(
            action_center().daily_brief(user_id=user_id, timezone_name=timezone)
        )

    return router
