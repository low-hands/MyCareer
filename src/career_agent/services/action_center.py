from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from career_agent.domain.action_center import (
    ActionCandidate,
    ActionItem,
    ActionStatus,
    ActionType,
    DailyBrief,
)
from career_agent.services.applications import ApplicationService
from career_agent.services.email_tracking import EmailTrackingService
from career_agent.services.interviews import InterviewService
from career_agent.storage.action_center import SQLiteActionItemStore


class ActionItemNotFoundError(ValueError):
    pass


class InvalidActionTransitionError(ValueError):
    pass


class ActionCenterService:
    _MANAGED_TYPES: tuple[ActionType, ...] = (
        "application_follow_up",
        "email_event_confirmation",
        "interview_preparation",
        "material_submission",
        "interview_reminder",
        "interview_retro",
    )

    def __init__(
        self,
        store: SQLiteActionItemStore,
        application_service: ApplicationService,
        email_tracking_service: EmailTrackingService,
        interview_service: InterviewService,
        *,
        follow_up_days: int = 7,
        interview_window_days: int = 7,
    ) -> None:
        self._store = store
        self._application_service = application_service
        self._email_tracking_service = email_tracking_service
        self._interview_service = interview_service
        self._follow_up_days = follow_up_days
        self._interview_window_days = interview_window_days

    def refresh(
        self,
        *,
        user_id: str,
        now: datetime | None = None,
        timezone_name: str = "Asia/Shanghai",
    ) -> tuple[ActionItem, ...]:
        current = now or datetime.now(timezone.utc)
        local_zone = self._zone(timezone_name)
        candidates = self._candidates(
            user_id=user_id,
            now=current,
            local_zone=local_zone,
        )
        active_keys = frozenset(candidate.stable_key for candidate in candidates)
        for candidate in candidates:
            self._store.upsert_candidate(
                user_id=user_id,
                candidate=candidate,
                now=current,
            )
        self._store.resolve_missing(
            user_id=user_id,
            managed_types=self._MANAGED_TYPES,
            active_keys=active_keys,
            now=current,
        )
        return self._store.list(user_id=user_id, statuses=("open",), limit=200)

    def daily_brief(
        self,
        *,
        user_id: str,
        timezone_name: str = "Asia/Shanghai",
        now: datetime | None = None,
    ) -> DailyBrief:
        current = now or datetime.now(timezone.utc)
        zone = self._zone(timezone_name)
        items = self.refresh(
            user_id=user_id,
            now=current,
            timezone_name=timezone_name,
        )
        local_now = self._aware(current, zone).astimezone(zone)
        day_start = local_now.replace(hour=0, minute=0, second=0, microsecond=0)
        day_end = day_start + timedelta(days=1)
        upcoming_end = local_now + timedelta(days=self._interview_window_days)
        overdue = []
        due_today = []
        upcoming = []
        no_due = []
        for item in items:
            if item.due_at is None:
                no_due.append(item)
                continue
            due = self._aware(item.due_at, zone).astimezone(zone)
            if due < day_start:
                overdue.append(item)
            elif due < day_end:
                due_today.append(item)
            elif due <= upcoming_end:
                upcoming.append(item)
        return DailyBrief(
            user_id=user_id,
            timezone=timezone_name,
            generated_at=current,
            overdue=tuple(overdue),
            due_today=tuple(due_today),
            upcoming=tuple(upcoming),
            no_due_date=tuple(no_due),
        )

    def list_actions(
        self,
        *,
        user_id: str,
        statuses: tuple[ActionStatus, ...] = ("open", "snoozed"),
        limit: int = 100,
        refresh: bool = True,
        now: datetime | None = None,
        timezone_name: str = "Asia/Shanghai",
    ) -> tuple[ActionItem, ...]:
        if refresh:
            self.refresh(
                user_id=user_id,
                now=now,
                timezone_name=timezone_name,
            )
        return self._store.list(user_id=user_id, statuses=statuses, limit=limit)

    def complete_action(self, *, user_id: str, action_item_id: str) -> ActionItem:
        return self._transition(
            user_id=user_id,
            action_item_id=action_item_id,
            status="completed",
        )

    def dismiss_action(self, *, user_id: str, action_item_id: str) -> ActionItem:
        return self._transition(
            user_id=user_id,
            action_item_id=action_item_id,
            status="dismissed",
        )

    def snooze_action(
        self,
        *,
        user_id: str,
        action_item_id: str,
        snoozed_until: datetime,
        now: datetime | None = None,
    ) -> ActionItem:
        current = now or datetime.now(timezone.utc)
        if snoozed_until.tzinfo is None:
            raise InvalidActionTransitionError("snoozed_until requires a timezone")
        if snoozed_until <= current:
            raise InvalidActionTransitionError("snoozed_until must be in the future")
        item = self._store.transition(
            user_id=user_id,
            action_item_id=action_item_id,
            status="snoozed",
            snoozed_until=snoozed_until,
            now=current,
        )
        if item is None:
            raise ActionItemNotFoundError(action_item_id)
        return item

    def _transition(
        self,
        *,
        user_id: str,
        action_item_id: str,
        status: ActionStatus,
    ) -> ActionItem:
        item = self._store.transition(
            user_id=user_id,
            action_item_id=action_item_id,
            status=status,
        )
        if item is None:
            raise ActionItemNotFoundError(action_item_id)
        return item

    def _candidates(
        self,
        *,
        user_id: str,
        now: datetime,
        local_zone: ZoneInfo,
    ) -> tuple[ActionCandidate, ...]:
        candidates: list[ActionCandidate] = []
        interviews = self._interview_service.list_interviews(
            user_id=user_id,
            limit=200,
        )
        future_interview_apps: set[str] = set()
        for interview in interviews:
            if interview.status == "scheduled" and interview.scheduled_start is not None:
                starts_at = self._aware(interview.scheduled_start, local_zone)
                if now <= starts_at <= now + timedelta(days=self._interview_window_days):
                    future_interview_apps.add(interview.application_id)
                    label = interview.employer_label or f"内部第 {interview.sequence_number} 场"
                    candidates.extend(
                        (
                            ActionCandidate(
                                stable_key=f"interview_preparation:{interview.id}",
                                action_type="interview_preparation",
                                source_type="interview_round",
                                source_id=interview.id,
                                application_id=interview.application_id,
                                title=f"准备面试：{label}",
                                summary="复习岗位要求、简历证据并准备要询问的问题。",
                                due_at=starts_at - timedelta(days=2),
                            ),
                            ActionCandidate(
                                stable_key=f"interview_reminder:{interview.id}",
                                action_type="interview_reminder",
                                source_type="interview_round",
                                source_id=interview.id,
                                application_id=interview.application_id,
                                title=f"面试提醒：{label}",
                                summary="确认时间、时区、会议链接或线下地点。",
                                due_at=starts_at - timedelta(hours=2),
                            ),
                        )
                    )
            if interview.status == "completed" and interview.completed_at is not None:
                completed_at = self._aware(interview.completed_at, local_zone)
                if completed_at >= now - timedelta(days=7):
                    candidates.append(
                        ActionCandidate(
                            stable_key=f"interview_retro:{interview.id}",
                            action_type="interview_retro",
                            source_type="interview_round",
                            source_id=interview.id,
                            application_id=interview.application_id,
                            title="记录真实面试复盘",
                            summary="记录问题、回答卡点和下一轮需要加强的内容。",
                            due_at=completed_at + timedelta(hours=1),
                        )
                    )

        for event in self._email_tracking_service.list_events(
            user_id=user_id,
            limit=200,
        ):
            if event.status == "pending_confirmation":
                candidates.append(
                    ActionCandidate(
                        stable_key=f"email_event_confirmation:{event.id}",
                        action_type="email_event_confirmation",
                        source_type="email_event",
                        source_id=event.id,
                        application_id=event.application_id,
                        title="确认招聘邮件识别结果",
                        summary=event.summary,
                        due_at=event.occurred_at,
                    )
                )
            elif event.status == "applied" and event.event_type == "material_request":
                candidates.append(
                    ActionCandidate(
                        stable_key=f"material_submission:{event.id}",
                        action_type="material_submission",
                        source_type="email_event",
                        source_id=event.id,
                        application_id=event.application_id,
                        title="处理招聘方补充材料请求",
                        summary=event.summary,
                    )
                )

        for item in self._application_service.list_applications(
            user_id=user_id,
            limit=200,
        ):
            application = item.application
            if application.status in {"offer", "rejected", "withdrawn"}:
                continue
            if application.id in future_interview_apps:
                continue
            age = now - self._aware(application.updated_at, local_zone)
            if age < timedelta(days=self._follow_up_days):
                continue
            cycle = max(1, int(age // timedelta(days=self._follow_up_days)))
            candidates.append(
                ActionCandidate(
                    stable_key=(
                        f"application_follow_up:{application.id}:"
                        f"{application.updated_at.isoformat()}:{cycle}"
                    ),
                    action_type="application_follow_up",
                    source_type="application",
                    source_id=application.id,
                    application_id=application.id,
                    title=f"跟进投递：{item.job.posting.company_name}",
                    summary=f"{item.job.posting.title} 已有 {age.days} 天没有记录新进展。",
                    due_at=now,
                )
            )
        return tuple(candidates)

    @staticmethod
    def _zone(timezone_name: str) -> ZoneInfo:
        try:
            return ZoneInfo(timezone_name)
        except ZoneInfoNotFoundError as error:
            raise ValueError("unknown timezone") from error

    @staticmethod
    def _aware(value: datetime, fallback_zone: ZoneInfo) -> datetime:
        return value.replace(tzinfo=fallback_zone) if value.tzinfo is None else value
