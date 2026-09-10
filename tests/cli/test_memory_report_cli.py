from datetime import datetime, timedelta, timezone
from io import StringIO
import json
import sqlite3

from career_agent.cli import main
from career_agent.domain.episodes import CareerEpisodeDraft
from career_agent.storage.context import CareerContextStore
from career_agent.storage.episodes import SQLiteCareerEpisodeStore
from career_agent.storage.intent_versions import append_intent_version


def _episode(source_run_id: str) -> CareerEpisodeDraft:
    return CareerEpisodeDraft(
        user_id="u1",
        kind="application",
        source_run_id=source_run_id,
        occurred_at=datetime.now(timezone.utc),
        title="Memory report",
        summary="用于检查 episode 保鲜度。",
    )


def test_memory_report_is_read_only_and_counts_maintenance_states(tmp_path) -> None:
    path = tmp_path / "context.sqlite3"
    CareerContextStore(path)
    episodes = SQLiteCareerEpisodeStore(path)
    fresh = episodes.upsert(_episode("fresh"))
    near = episodes.upsert(_episode("near"))
    faded = episodes.upsert(_episode("faded"))
    now = datetime.now(timezone.utc)
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE career_episodes SET salience = 1, created_at = ? WHERE id = ?",
            ((now + timedelta(days=1)).isoformat(), fresh.id),
        )
        connection.execute(
            "UPDATE career_episodes SET salience = 0.11, created_at = ? WHERE id = ?",
            ((now + timedelta(days=1)).isoformat(), near.id),
        )
        connection.execute(
            "UPDATE career_episodes SET salience = 1, created_at = ? WHERE id = ?",
            ((now - timedelta(days=730)).isoformat(), faded.id),
        )
        append_intent_version(
            connection,
            user_id="u1",
            scope_key="person_intent/self/timing",
            value="年底前不动",
            source="test",
            valid_from=now - timedelta(days=30),
            valid_until=now - timedelta(days=1),
            pref_scope="freeform.person_situational",
            timescale="situational",
            layer="transient",
            admission_status="active",
        )
        append_intent_version(
            connection,
            user_id="u1",
            scope_key="person_intent/self/team_style",
            value="可能偏好小团队",
            source="test",
            valid_from=now - timedelta(days=20),
            pref_scope="freeform.person_default",
            timescale="permanent",
            layer="contextual",
            admission_status="quarantined",
        )
    modified_before = path.stat().st_mtime_ns
    output = StringIO()

    code = main(
        [
            "memory",
            "report",
            "--user-id",
            "u1",
            "--context-store",
            str(path),
            "--quarantine-stale-days",
            "14",
        ],
        stdout=output,
        stderr=StringIO(),
    )

    assert code == 0
    assert path.stat().st_mtime_ns == modified_before
    payload = json.loads(output.getvalue())
    assert payload["state"] == "memory_report_ready"
    assert payload["episodes"] == {
        "total": 3,
        "projection_eligible": 2,
        "near_threshold": 1,
        "below_threshold": 1,
    }
    assert payload["preferences"] == {
        "expired_situational": 1,
        "stale_quarantine": 1,
    }


def test_memory_report_rejects_invalid_policy_without_touching_the_store(
    tmp_path,
) -> None:
    path = tmp_path / "context.sqlite3"
    CareerContextStore(path)
    output = StringIO()

    code = main(
        [
            "memory",
            "report",
            "--user-id",
            "u1",
            "--context-store",
            str(path),
            "--half-life-days",
            "0",
        ],
        stdout=output,
        stderr=StringIO(),
    )

    assert code == 2
    assert json.loads(output.getvalue())["state"] == "failed"
