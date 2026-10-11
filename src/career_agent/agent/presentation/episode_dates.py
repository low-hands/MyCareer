from datetime import datetime
from zoneinfo import ZoneInfo


def episode_date_reference(now: datetime | None = None) -> datetime:
    from career_agent.agent.runtime.decision_messages import runtime_clock

    clock = runtime_clock()
    reference = now if now is not None else datetime.fromisoformat(clock["now"])
    if reference.utcoffset() is None:
        raise ValueError("episode date labels require timezone-aware timestamps")
    return reference.astimezone(ZoneInfo(clock["timezone"]))


def relative_episode_date(occurred_at: datetime, *, now: datetime) -> str:
    """Label only the calendar year, using the reference clock's timezone."""
    if now.utcoffset() is None or occurred_at.utcoffset() is None:
        raise ValueError("episode date labels require timezone-aware timestamps")
    local = occurred_at.astimezone(now.tzinfo)
    difference = local.year - now.year
    if difference in {0, -1, 1}:
        return {0: "今年", -1: "去年", 1: "明年"}[difference]
    return f"{-difference} 年前" if difference < 0 else f"{difference} 年后"
