from __future__ import annotations

from io import StringIO
import json
import os
from pathlib import Path
import sqlite3

import pytest

from career_agent.api.single_worker import SingleWorkerLock, lock_path_for
from career_agent.cli import main
from career_agent.storage.backup import (
    BackupError,
    BackupPlan,
    create_backup,
    restore_backup,
    verify_backup,
)


def _make_db(path: Path, rows: list[str], *, wal: bool = False) -> None:
    connection = sqlite3.connect(path)
    if wal:
        connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("CREATE TABLE IF NOT EXISTS notes(body TEXT)")
    connection.executemany("INSERT INTO notes(body) VALUES (?)", [(row,) for row in rows])
    connection.commit()
    connection.close()


def _rows(path: Path) -> list[str]:
    connection = sqlite3.connect(path)
    try:
        return [row[0] for row in connection.execute("SELECT body FROM notes ORDER BY body")]
    finally:
        connection.close()


@pytest.fixture
def workspace(tmp_path: Path) -> tuple[Path, BackupPlan]:
    root = tmp_path / "workspace"
    root.mkdir()
    _make_db(root / "context.sqlite3", ["ctx-1"], wal=True)
    _make_db(root / "resumes.sqlite3", ["resume-1"])
    notes = root / "working-notes"
    notes.mkdir()
    (notes / "u1.md").write_text("# notes\n", encoding="utf-8")
    (notes / "u1.md.tmp").write_text("scratch", encoding="utf-8")
    plan = BackupPlan(
        databases=(root / "context.sqlite3", root / "resumes.sqlite3", root / "never-created.sqlite3"),
        directories=(notes,),
    )
    return root, plan


def test_create_backup_copies_every_store_with_a_checksummed_manifest(workspace, tmp_path):
    root, plan = workspace
    # Uncommitted WAL pages are exactly what a naive file copy misses.
    live = sqlite3.connect(root / "context.sqlite3")
    live.execute("INSERT INTO notes(body) VALUES ('ctx-2')")
    live.commit()

    destination = tmp_path / "backup"
    manifest = create_backup(plan, destination)
    live.close()

    names = [entry.name for entry in manifest.entries]
    assert names == [
        "databases/context.sqlite3",
        "databases/resumes.sqlite3",
        "directories/working-notes/u1.md",
    ]
    assert _rows(destination / "databases" / "context.sqlite3") == ["ctx-1", "ctx-2"]
    assert not (destination / "databases" / "context.sqlite3-wal").exists()
    payload = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    assert payload["version"] == 1
    assert {entry["name"] for entry in payload["entries"]} == set(names)
    assert oct(os.stat(destination).st_mode & 0o777) == "0o700"
    assert oct(os.stat(destination / "databases").st_mode & 0o777) == "0o700"
    assert verify_backup(destination).ok
    # Checking the copy must not litter the backup with WAL side files.
    assert sorted(path.name for path in (destination / "databases").iterdir()) == [
        "context.sqlite3",
        "resumes.sqlite3",
    ]


def test_create_backup_refuses_a_non_empty_destination(workspace, tmp_path):
    _, plan = workspace
    destination = tmp_path / "backup"
    destination.mkdir()
    (destination / "stray").write_text("x", encoding="utf-8")
    with pytest.raises(BackupError, match="not empty"):
        create_backup(plan, destination)


def test_verify_reports_tampering_missing_files_and_corruption(workspace, tmp_path):
    root, plan = workspace
    destination = tmp_path / "backup"
    create_backup(plan, destination)

    (destination / "directories" / "working-notes" / "u1.md").write_text("changed", encoding="utf-8")
    (destination / "databases" / "resumes.sqlite3").unlink()

    report = verify_backup(destination)

    assert not report.ok
    assert "databases/resumes.sqlite3: missing" in report.problems
    assert any(problem.startswith("directories/working-notes/u1.md: size") for problem in report.problems)


def test_verify_without_manifest_is_a_problem_not_a_crash(tmp_path):
    report = verify_backup(tmp_path)
    assert not report.ok
    assert report.manifest is None
    assert "manifest.json not found" in report.problems[0]


def test_restore_replaces_stores_drops_stale_wal_and_keeps_a_safety_copy(workspace, tmp_path):
    root, plan = workspace
    destination = tmp_path / "backup"
    create_backup(plan, destination)

    # Workspace moves on after the backup: new rows, a new note, a stray WAL.
    _make_db(root / "context.sqlite3", ["ctx-later"], wal=True)
    (root / "working-notes" / "u2.md").write_text("later", encoding="utf-8")
    (root / "resumes.sqlite3-wal").write_bytes(b"garbage")

    report = restore_backup(destination, plan, safety_copy_dir=tmp_path / "safety")

    assert _rows(root / "context.sqlite3") == ["ctx-1"]
    assert _rows(root / "resumes.sqlite3") == ["resume-1"]
    assert not (root / "resumes.sqlite3-wal").exists()
    assert not (root / "working-notes" / "u2.md").exists()
    assert (root / "working-notes" / "u1.md").read_text(encoding="utf-8") == "# notes\n"
    assert report.restored == (
        "databases/context.sqlite3",
        "databases/resumes.sqlite3",
        "directories/working-notes/u1.md",
    )
    assert report.safety_copy == tmp_path / "safety"
    safety = verify_backup(tmp_path / "safety")
    assert safety.ok
    assert _rows(tmp_path / "safety" / "databases" / "context.sqlite3") == ["ctx-1", "ctx-later"]
    assert (tmp_path / "safety" / "directories" / "working-notes" / "u2.md").exists()


def test_restore_refuses_a_backup_that_fails_verification(workspace, tmp_path):
    root, plan = workspace
    destination = tmp_path / "backup"
    create_backup(plan, destination)
    (destination / "databases" / "context.sqlite3").write_bytes(b"not a database")
    _make_db(root / "context.sqlite3", ["ctx-later"], wal=True)

    with pytest.raises(BackupError, match="nothing restored"):
        restore_backup(destination, plan, safety_copy_dir=None)

    assert _rows(root / "context.sqlite3") == ["ctx-1", "ctx-later"]


def test_restore_into_another_workspace_matches_by_file_name(workspace, tmp_path):
    root, plan = workspace
    destination = tmp_path / "backup"
    create_backup(plan, destination)

    other = tmp_path / "elsewhere"
    other_plan = BackupPlan(
        databases=(other / "context.sqlite3",),
        directories=(other / "working-notes",),
    )
    report = restore_backup(destination, other_plan, safety_copy_dir=None)

    assert _rows(other / "context.sqlite3") == ["ctx-1"]
    assert (other / "working-notes" / "u1.md").exists()
    assert report.skipped_missing_target == ("databases/resumes.sqlite3",)


def test_plan_rejects_stores_whose_file_names_collide(tmp_path):
    with pytest.raises(BackupError, match="distinct file names"):
        BackupPlan(
            databases=(tmp_path / "a" / "x.sqlite3", tmp_path / "b" / "x.sqlite3"),
            directories=(),
        )


def _run(argv: list[str]) -> tuple[int, dict]:
    output = StringIO()
    code = main(argv, stdout=output, stderr=StringIO())
    return code, json.loads(output.getvalue())


def _store_args(root: Path) -> list[str]:
    return [
        "--context-store", str(root / "context.sqlite3"),
        "--resume-store", str(root / "resumes.sqlite3"),
        "--application-store", str(root / "applications.sqlite3"),
        "--job-store", str(root / "jobs.sqlite3"),
        "--job-research-store", str(root / "job-research.sqlite3"),
        "--job-research-checkpoint-store", str(root / "job-research-checkpoints.sqlite3"),
        "--email-store", str(root / "email.sqlite3"),
        "--action-store", str(root / "actions.sqlite3"),
        "--calendar-store", str(root / "calendar.sqlite3"),
        "--mock-interview-store", str(root / "mock-interviews.sqlite3"),
        "--mock-interview-checkpoint-store", str(root / "mock-interview-checkpoints.sqlite3"),
        "--run-events-store", str(root / "run-events.sqlite3"),
        "--api-key-store", str(root / "api_keys.sqlite3"),
    ]


def test_cli_backup_create_verify_restore_round_trip(tmp_path, monkeypatch):
    root = tmp_path / "ws"
    root.mkdir()
    _make_db(root / "context.sqlite3", ["ctx-1"], wal=True)
    _make_db(root / "api_keys.sqlite3", ["key-1"])
    (root / "working-notes").mkdir()
    (root / "working-notes" / "u1.md").write_text("n", encoding="utf-8")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))

    code, created = _run(["backup", "create", "--dest", str(tmp_path / "b1"), *_store_args(root)])
    assert code == 0
    assert created["files"] == 3
    assert "databases/api_keys.sqlite3" in created["entries"]

    code, verified = _run(["backup", "verify", "--source", str(tmp_path / "b1")])
    assert (code, verified["ok"], verified["problems"]) == (0, True, [])

    _make_db(root / "context.sqlite3", ["ctx-2"], wal=True)
    code, refused = _run(["backup", "restore", "--source", str(tmp_path / "b1"), *_store_args(root)])
    assert code == 2
    assert "--yes" in refused["error"]
    assert _rows(root / "context.sqlite3") == ["ctx-1", "ctx-2"]

    code, restored = _run(
        ["backup", "restore", "--source", str(tmp_path / "b1"), "--yes", *_store_args(root)]
    )
    assert code == 0
    assert _rows(root / "context.sqlite3") == ["ctx-1"]
    safety = Path(restored["safety_copy"])
    assert safety.parent == tmp_path / "home" / ".career-agent-backups"
    assert _rows(safety / "databases" / "context.sqlite3") == ["ctx-1", "ctx-2"]


def test_cli_backup_create_defaults_to_a_timestamped_directory_under_home(tmp_path, monkeypatch):
    root = tmp_path / "ws"
    root.mkdir()
    _make_db(root / "context.sqlite3", ["ctx-1"])
    monkeypatch.setenv("HOME", str(tmp_path / "home"))

    code, created = _run(["backup", "create", *_store_args(root)])

    assert code == 0
    assert Path(created["backup"]).parent == tmp_path / "home" / ".career-agent-backups"
    assert verify_backup(Path(created["backup"])).ok


def test_cli_backup_verify_fails_loudly_on_a_broken_backup(tmp_path):
    code, verified = _run(["backup", "verify", "--source", str(tmp_path)])
    assert code == 5
    assert verified["ok"] is False


def test_cli_backup_restore_refuses_while_the_api_holds_the_workspace_lock(tmp_path):
    root = tmp_path / "ws"
    root.mkdir()
    _make_db(root / "context.sqlite3", ["ctx-1"])
    _, created = _run(["backup", "create", "--dest", str(tmp_path / "b1"), *_store_args(root)])
    _make_db(root / "context.sqlite3", ["ctx-2"])

    lock = SingleWorkerLock(lock_path_for(root))
    lock.acquire()
    try:
        code, refused = _run(
            [
                "backup", "restore", "--source", str(tmp_path / "b1"), "--yes",
                "--no-safety-copy", *_store_args(root),
            ]
        )
    finally:
        lock.release()

    assert code == 5
    assert "API 正在运行" in refused["error"]
    assert _rows(root / "context.sqlite3") == ["ctx-1", "ctx-2"]


def test_cli_backup_create_refuses_while_the_api_holds_the_workspace_lock(tmp_path):
    root = tmp_path / "ws"
    root.mkdir()
    _make_db(root / "context.sqlite3", ["ctx-1"])

    lock = SingleWorkerLock(lock_path_for(root))
    lock.acquire()
    try:
        code, refused = _run(
            ["backup", "create", "--dest", str(tmp_path / "b1"), *_store_args(root)]
        )
        assert code == 5
        assert "API 正在运行" in refused["error"]
        assert "--allow-running-api" in refused["error"]
        assert not (tmp_path / "b1" / "manifest.json").exists()

        code, created = _run(
            [
                "backup", "create", "--dest", str(tmp_path / "b2"),
                "--allow-running-api", *_store_args(root),
            ]
        )
    finally:
        lock.release()

    assert code == 0
    assert created["consistent_snapshot"] is False
    assert created["warnings"]
    manifest = json.loads((tmp_path / "b2" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["consistent_snapshot"] is False

    code, verified = _run(["backup", "verify", "--source", str(tmp_path / "b2")])
    assert (code, verified["ok"], verified["consistent_snapshot"]) == (0, True, False)
    assert "same moment" in verified["warnings"][0]

    _make_db(root / "context.sqlite3", ["ctx-2"])
    code, restored = _run(
        [
            "backup", "restore", "--source", str(tmp_path / "b2"), "--yes",
            "--no-safety-copy", *_store_args(root),
        ]
    )
    assert code == 0
    assert _rows(root / "context.sqlite3") == ["ctx-1"]
    assert "same moment" in restored["warnings"][0]


def test_cli_backup_create_holding_the_lock_is_a_consistent_snapshot(tmp_path):
    root = tmp_path / "ws"
    root.mkdir()
    _make_db(root / "context.sqlite3", ["ctx-1"])

    code, created = _run(["backup", "create", "--dest", str(tmp_path / "b1"), *_store_args(root)])

    assert code == 0
    assert (created["consistent_snapshot"], created["warnings"]) == (True, [])
    # The lock is released again once the copy is written.
    lock = SingleWorkerLock(lock_path_for(root))
    lock.acquire()
    lock.release()
    _, verified = _run(["backup", "verify", "--source", str(tmp_path / "b1")])
    assert (verified["consistent_snapshot"], verified["warnings"]) == (True, [])


def test_a_manifest_without_the_consistency_flag_is_unknown_and_warns(tmp_path):
    from career_agent.storage.backup import (
        BackupManifest,
        UNKNOWN_SNAPSHOT_WARNING,
        VerificationReport,
    )

    manifest = BackupManifest.from_json({"version": 1, "created_at": "2026-09-14T00:00:00+00:00", "entries": []})
    assert manifest.consistent_snapshot is None
    report = VerificationReport(directory=tmp_path, manifest=manifest, problems=())
    assert report.ok
    assert report.warnings == (UNKNOWN_SNAPSHOT_WARNING,)
    with pytest.raises(BackupError):
        BackupManifest.from_json(
            {"version": 1, "created_at": "2026-09-14T00:00:00+00:00", "entries": [], "consistent_snapshot": "no"}
        )


def test_cli_write_commands_refuse_while_the_api_holds_the_workspace_lock(tmp_path):
    root = tmp_path / "ws"
    root.mkdir()
    store = root / "context.sqlite3"

    lock = SingleWorkerLock(lock_path_for(root))
    lock.acquire()
    try:
        out = StringIO()
        code = main(
            ["settings", "set", "--user-id", "u1", "--boss-search", "allowed",
             "--context-store", str(store)],
            stdout=out,
            stderr=StringIO(),
        )
    finally:
        lock.release()

    assert code == 5
    assert "正在使用该工作区" in json.loads(out.getvalue())["error"]
    assert not store.exists()


def test_cli_read_commands_do_not_need_the_workspace_lock(tmp_path):
    root = tmp_path / "ws"
    root.mkdir()
    store = root / "context.sqlite3"

    lock = SingleWorkerLock(lock_path_for(root))
    lock.acquire()
    try:
        out = StringIO()
        code = main(
            ["settings", "show", "--user-id", "u1", "--context-store", str(store)],
            stdout=out,
            stderr=StringIO(),
        )
    finally:
        lock.release()

    assert code == 0
    assert json.loads(out.getvalue())["user_id"] == "u1"


def test_cli_write_commands_lock_beside_the_context_store_whatever_other_stores_are_given(tmp_path):
    """One lock for the workspace: a command writing a resume database in
    another directory still contends with the API and ``backup`` on the
    context store's lock, so spread-out databases cannot be written under a
    running backup."""
    from career_agent.api.single_worker import SingleWorkerLock, lock_path_for
    from career_agent.cli import main

    workspace = tmp_path / "ws"
    workspace.mkdir()
    other = tmp_path / "other"
    other.mkdir()
    holder = SingleWorkerLock(lock_path_for(workspace))
    holder.acquire()
    out = StringIO()
    try:
        code = main(
            [
                "target-role",
                "create",
                "--user-id",
                "u1",
                "--title",
                "Backend",
                "--resume-store",
                str(other / "resumes.sqlite3"),
                "--context-store",
                str(workspace / "context.sqlite3"),
            ],
            stdout=out,
            stderr=StringIO(),
        )
    finally:
        holder.release()
    assert code == 5
    assert not (other / "api-server.lock").exists()
    assert not (other / "resumes.sqlite3").exists()


def test_cli_api_key_writes_take_the_workspace_lock(tmp_path):
    """The key database is part of a backup, so issuing or revoking keys is a
    workspace write and waits for backup/restore like any other."""
    from career_agent.api.single_worker import SingleWorkerLock, lock_path_for
    from career_agent.cli import main

    workspace = tmp_path / "ws"
    workspace.mkdir()
    holder = SingleWorkerLock(lock_path_for(workspace))
    holder.acquire()
    out = StringIO()
    try:
        code = main(
            [
                "api-keys",
                "issue",
                "--user-id",
                "u1",
                "--name",
                "browser",
                "--scope",
                "chat:write",
                "--api-key-store",
                str(tmp_path / "api_keys.sqlite3"),
                "--context-store",
                str(workspace / "context.sqlite3"),
            ],
            stdout=out,
            stderr=StringIO(),
        )
    finally:
        holder.release()
    assert code == 5
    assert not (tmp_path / "api_keys.sqlite3").exists()


def test_cli_reports_a_lock_it_cannot_open_instead_of_crashing(tmp_path):
    from career_agent.cli import main

    blocker = tmp_path / "not-a-directory"
    blocker.write_text("")
    out = StringIO()
    code = main(
        [
            "settings",
            "set",
            "--user-id",
            "u1",
            "--boss-search",
            "allowed",
            "--context-store",
            str(blocker / "context.sqlite3"),
        ],
        stdout=out,
        stderr=StringIO(),
    )
    assert code == 3
    assert "工作区锁" in json.loads(out.getvalue())["error"]
