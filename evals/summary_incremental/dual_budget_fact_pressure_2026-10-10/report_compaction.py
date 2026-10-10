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
    checks["reading_gap_matches_protocol"] = (
        capture["inter_turn_delay_seconds"] == frozen["inter_turn_delay_seconds"]
    )
    checks["decision_model_matches_protocol"] = (
        capture["models"]["MAIN_AGENT_MODEL"] == frozen["decision_model"]
    )
    checks["summary_config_matches_protocol"] = capture["summary_config"] == {
        "model": frozen["summary_model"],
        "disable_thinking": frozen["summary_disable_thinking"],
        "timeout_seconds": frozen["summary_timeout_seconds"],
        "max_output_tokens": frozen["summary_max_output_tokens"],
    }
    turns = capture["turns"]
    pressure_peaks = {
        field: max(
            (event["details"][field] for event in events
             if isinstance(event["details"].get(field), (int, float))),
            default=None,
        )
        for field in ("request_occupancy", "recent_occupancy")
    }
    return {
        "fingerprints": checks, "pressure_peaks": pressure_peaks,
        "attempted_turns": len(turns),
        "returned_turns": sum(t["status"] == "returned" for t in turns),
        "planned_turns": frozen["maximum_turns"] * frozen["conversations"],
        "interrupted_turns": [t for t in turns if t.get("interrupted")],
        "capture_stopped_early": len(turns) < frozen["maximum_turns"] * frozen["conversations"],
        "failed_turns": [t for t in turns if t["status"] != "returned"],
        "inter_turn_delay_seconds": capture["inter_turn_delay_seconds"],
        "reading_gap_total_ms": sum(t.get("inter_turn_gap_ms", 0) for t in turns),
        "scheduled_jobs": timings("context_compaction_scheduled"),
        "load_request_estimates": [
            {"run_id": event["run_id"], **labels.get(event["run_id"], {}),
             "input_tokens": event["details"].get("input_occupancy_numerator"),
             "max_input_tokens": event["details"].get("input_occupancy_denominator"),
             "request_occupancy": event["details"].get("request_occupancy"),
             "recent_occupancy": event["details"].get("recent_occupancy")}
            for event in events if event["event_type"] == "context_estimated"
        ],
        "provider_failures": [
            {"run_id": event["run_id"], "stage": event["stage"],
             "error_code": event["error_code"]}
            for event in events if event["event_type"] == "provider_request"
            and event["outcome"] == "failed"
        ],
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
             "request_occupancy": e["details"].get("request_occupancy"),
             "recent_occupancy": e["details"].get("recent_occupancy"),
             "through_sequence": e["details"].get("through_sequence"),
             "background": e["details"].get("background")}
            for e in events if e["event_type"] == "context_compacted"
        ],
        "summary_failures": [
            {"run_id": e["run_id"], "error_code": e["error_code"], "details": e["details"]}
            for e in events if e["event_type"] == "context_compaction_failed"
        ],
        "turns": turns,
        "interpretation": "run_turn wall time includes load waits/synchronous summaries. Commit queues without awaiting a model; later background trace must not inflate TurnCompleted time. Report gap0 and gap20 separately; each gap is excluded from wall time. Seeded history bypasses commit scheduling. Zero waits must be read alongside actual background summary counts. Returned is not a model task-quality score.",
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
