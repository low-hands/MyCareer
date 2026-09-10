from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sqlite3

from career_agent.storage.episodes import DecayPolicy


_NEAR_THRESHOLD_MULTIPLIER = 1.25


@dataclass(frozen=True, slots=True)
class EpisodeProjectionCounts:
    total: int
    projection_eligible: int
    near_threshold: int
    below_threshold: int


@dataclass(frozen=True, slots=True)
class PreferenceMaintenanceCounts:
    expired_situational: int
    stale_quarantine: int


@dataclass(frozen=True, slots=True)
class MemoryReport:
    user_id: str
    generated_at: datetime
    decay_policy: DecayPolicy
    quarantine_stale_days: int
    episodes: EpisodeProjectionCounts
    preferences: PreferenceMaintenanceCounts

    def as_payload(self) -> dict[str, object]:
        return {
            "state": "memory_report_ready",
            "user_id": self.user_id,
            "generated_at": self.generated_at.isoformat(),
            "decay_policy": {
                "half_life_days": self.decay_policy.half_life_days,
                "access_boost": self.decay_policy.access_boost,
                "projection_threshold": (
                    self.decay_policy.projection_threshold
                ),
            },
            "quarantine_stale_days": self.quarantine_stale_days,
            "episodes": {
                "total": self.episodes.total,
                "projection_eligible": self.episodes.projection_eligible,
                "near_threshold": self.episodes.near_threshold,
                "below_threshold": self.episodes.below_threshold,
            },
            "preferences": {
                "expired_situational": (
                    self.preferences.expired_situational
                ),
                "stale_quarantine": self.preferences.stale_quarantine,
            },
        }


def build_memory_report(
    path: Path,
    *,
    user_id: str,
    decay_policy: DecayPolicy | None = None,
    quarantine_stale_days: int = 14,
    now: datetime | None = None,
) -> MemoryReport:
    """Inspect memory freshness through a SQLite read-only connection."""

    if not user_id:
        raise ValueError("user_id must not be empty")
    if quarantine_stale_days < 1:
        raise ValueError("quarantine_stale_days must be positive")
    observed_at = now or datetime.now(timezone.utc)
    if observed_at.utcoffset() is None:
        raise ValueError("now must include a timezone offset")
    observed_at = observed_at.astimezone(timezone.utc)
    policy = decay_policy or DecayPolicy()
    database_path = path.expanduser().resolve()

    with sqlite3.connect(
        database_path.as_uri() + "?mode=ro",
        uri=True,
    ) as connection:
        connection.execute("PRAGMA query_only = ON")
        tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        episode_counts = _episode_counts(
            connection,
            tables=tables,
            user_id=user_id,
            policy=policy,
            now=observed_at,
        )
        preference_counts = _preference_counts(
            connection,
            tables=tables,
            user_id=user_id,
            quarantine_stale_days=quarantine_stale_days,
            now=observed_at,
        )

    return MemoryReport(
        user_id=user_id,
        generated_at=observed_at,
        decay_policy=policy,
        quarantine_stale_days=quarantine_stale_days,
        episodes=episode_counts,
        preferences=preference_counts,
    )


def _episode_counts(
    connection: sqlite3.Connection,
    *,
    tables: set[str],
    user_id: str,
    policy: DecayPolicy,
    now: datetime,
) -> EpisodeProjectionCounts:
    if "career_episodes" not in tables:
        return EpisodeProjectionCounts(0, 0, 0, 0)
    rows = connection.execute(
        """
        SELECT salience, last_accessed_at, created_at, occurred_at, access_count
        FROM career_episodes
        WHERE user_id = ?
        """,
        (user_id,),
    ).fetchall()
    effective = tuple(
        policy.effective_salience(
            salience=float(row[0]),
            accessed_or_created_at=_parse_timestamp(
                row[1] or row[2] or row[3]
            ),
            access_count=int(row[4]),
            now=now,
        )
        for row in rows
    )
    threshold = policy.projection_threshold
    near_upper = threshold * _NEAR_THRESHOLD_MULTIPLIER
    eligible = sum(value >= threshold for value in effective)
    near = sum(threshold <= value < near_upper for value in effective)
    return EpisodeProjectionCounts(
        total=len(effective),
        projection_eligible=eligible,
        near_threshold=near,
        below_threshold=len(effective) - eligible,
    )


def _preference_counts(
    connection: sqlite3.Connection,
    *,
    tables: set[str],
    user_id: str,
    quarantine_stale_days: int,
    now: datetime,
) -> PreferenceMaintenanceCounts:
    if "career_intent_versions" not in tables:
        return PreferenceMaintenanceCounts(0, 0)

    joins: list[str] = []
    visible_filters: list[str] = []
    if "memory_deleted_scopes" in tables:
        joins.append(
            """
            LEFT JOIN memory_deleted_scopes AS deleted
              ON deleted.user_id = intent.user_id
             AND deleted.scope_key = intent.scope_key
            """
        )
        visible_filters.append(
            "(deleted.deleted_at IS NULL OR intent.valid_from > deleted.deleted_at)"
        )
    if "intent_memory_tombstones" in tables:
        joins.append(
            """
            LEFT JOIN intent_memory_tombstones AS tombstone
              ON tombstone.user_id = intent.user_id
             AND tombstone.scope_key = intent.scope_key
             AND tombstone.pref_scope = intent.pref_scope
            """
        )
        visible_filters.append(
            "(tombstone.deleted_at IS NULL OR intent.valid_from > tombstone.deleted_at)"
        )
    where_visible = "".join(f" AND {item}" for item in visible_filters)
    rows = connection.execute(
        """
        SELECT intent.timescale, intent.valid_until, intent.admission_status,
               intent.valid_from
        FROM career_intent_versions AS intent
        """
        + " ".join(joins)
        + """
        WHERE intent.user_id = ?
          AND intent.superseded_at IS NULL
        """
        + where_visible,
        (user_id,),
    ).fetchall()

    stale_before = now - timedelta(days=quarantine_stale_days)
    expired_situational = 0
    stale_quarantine = 0
    for timescale, valid_until, admission_status, valid_from in rows:
        expiry = _parse_timestamp(valid_until) if valid_until else None
        if (
            timescale == "situational"
            and admission_status == "active"
            and expiry is not None
            and expiry <= now
        ):
            expired_situational += 1
        if (
            admission_status == "quarantined"
            and _parse_timestamp(valid_from) <= stale_before
            and (expiry is None or expiry > now)
        ):
            stale_quarantine += 1
    return PreferenceMaintenanceCounts(
        expired_situational=expired_situational,
        stale_quarantine=stale_quarantine,
    )


def _parse_timestamp(value: object) -> datetime:
    parsed = datetime.fromisoformat(str(value))
    if parsed.utcoffset() is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)
