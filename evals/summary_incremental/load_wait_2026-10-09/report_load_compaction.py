"""Read-only summary of the frozen serial capture; never call providers."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sqlite3


def report(database: Path, manifest: Path, protocol: Path) -> dict:
    capture = json.loads(manifest.read_text())
    frozen = json.loads(protocol.read_text())
    with sqlite3.connect(f"file:{database.resolve()}?mode=ro", uri=True) as connection:
        connection.row_factory = sqlite3.Row
        events = [dict(row) for row in connection.execute(
            "SELECT * FROM run_events ORDER BY occurred_at, run_id, sequence"
        )]
    for event in events:
        event["details"] = json.loads(event.pop("details_json"))
    runs = {}
    for event in events:
        runs.setdefault(event["run_id"], []).append(event)

    labels = {}
    conversation_turns = {}
    for run_id, rows in runs.items():
        completed = next((e for e in rows if e["event_type"] == "turn_completed"), None)
        if completed is None:
            continue
        conversation_id = completed["details"].get("conversation_id")
        conversation_turns[conversation_id] = conversation_turns.get(conversation_id, 0) + 1
        labels[run_id] = {"conversation_id": conversation_id,
                          "turn": conversation_turns[conversation_id]}

    def timings(event_type, *, phase=None):
        return [
            {"run_id": e["run_id"], **labels.get(e["run_id"], {}),
             "occurred_at": e["occurred_at"], "duration_ms": e["duration_ms"],
             "outcome": e["outcome"], "details": e["details"]}
            for e in events if e["event_type"] == event_type
            and (phase is None or e["details"].get("phase") == phase)
        ]

    background = [e for e in events if e["event_type"] == "provider_request"
                  and e["stage"] == "conversation_summary"
                  and e["details"].get("background")]
    after_completion = []
    for run_id, rows in runs.items():
        completed = next((e for e in rows if e["event_type"] == "turn_completed"), None)
        if completed is None:
            continue
        end = datetime.fromisoformat(completed["occurred_at"])
        for event in rows:
            if event in background and datetime.fromisoformat(event["occurred_at"]) > end:
                after_completion.append({"run_id": run_id, "duration_ms": event["duration_ms"]})
    fields = ("source_sha256", "summary_prompt_sha256", "fixture_sha256")
    checks = {field: capture[field] == frozen[field] for field in fields}
    checks["final_source_matches_protocol"] = hashlib.sha256(b"".join(
        p.read_bytes() for p in sorted(Path("src/career_agent").rglob("*.py"))
    )).hexdigest() == frozen["source_sha256"]
    turns = capture["turns"]
    return {
        "fingerprints": checks, "attempted_turns": len(turns),
        "returned_turns": sum(t["status"] == "returned" for t in turns),
        "failed_turns": [t for t in turns if t["status"] != "returned"],
        "wall_total_ms": sum(t["wall_ms"] for t in turns),
        "wall_mean_ms": sum(t["wall_ms"] for t in turns) / len(turns) if turns else None,
        "load_waits": timings("context_compaction_waited", phase="load"),
        "load_sync_summaries": timings("context_compaction_sync_finished", phase="load"),
        "commit_sync_summaries": timings("context_compaction_sync_finished", phase="commit"),
        "skipped_jobs": timings("context_compaction_skipped"),
        "background_summaries": [
            {"run_id": e["run_id"], "duration_ms": e["duration_ms"], "outcome": e["outcome"]}
            for e in background
        ],
        "background_finished_after_turn_completed": after_completion,
        "compaction_triggers": [
            {"run_id": e["run_id"], "trigger": e["details"].get("trigger"),
             "occupancy": e["details"].get("occupancy"),
             "through_sequence": e["details"].get("through_sequence"),
             "background": e["details"].get("background")}
            for e in events if e["event_type"] == "context_compacted"
        ],
        "summary_failures": [
            {"run_id": e["run_id"], "error_code": e["error_code"], "details": e["details"]}
            for e in events if e["event_type"] == "context_compaction_failed"
        ],
        "turns": turns,
        "interpretation": "run_turn wall time includes load waits/synchronous summaries. Commit queues without awaiting a model; later background trace must not inflate TurnCompleted time. Immediate serial next-turn loads have no user thinking time; seeded history bypasses commit scheduling. Returned is not a model task-quality score.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(report(args.database, args.manifest, args.protocol), ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
