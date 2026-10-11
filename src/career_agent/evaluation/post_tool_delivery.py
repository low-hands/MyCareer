"""Count the first model decision following each executed tool result."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from career_agent.evaluation.latency import distribution, phase, read_runs
from career_agent.agent.presentation.receipt_completion import _DIRECT_TOOLS


def summarize(paths: list[Path]) -> dict:
    groups = defaultdict(lambda: {"counts": Counter(), "reply_ms": []})
    runs_with_execution, legacy_runs = 0, 0
    for path in paths:
        for events in read_runs(path).values():
            if not any(e["event_type"] == "capability_executed" for e in events):
                legacy_runs += 1
                continue
            runs_with_execution += 1
            for index, event in enumerate(events):
                if event["event_type"] != "capability_executed":
                    continue
                key = (event["details"].get("tool_name", "unknown"), event["details"].get("result_state", "exception"))
                row = groups[key]
                row["counts"]["executions"] += 1
                following = next((e for e in events[index + 1:] if phase(e) == "decision"
                                  and e["event_type"] in {"model_succeeded", "model_failed"}), None)
                if following is None:
                    row["counts"]["no_following_model_terminal"] += 1
                    continue
                row["counts"]["following_model"] += 1
                action = following["details"].get("decision_action")
                if following["event_type"] == "model_failed":
                    row["counts"]["following_model_failed"] += 1
                elif action == "final":
                    row["counts"]["model_final"] += 1
                    if following["duration_ms"] is not None:
                        row["reply_ms"].append(following["duration_ms"])
                elif action == "tool_call":
                    row["counts"]["continued_tool_work"] += 1
                else:
                    row["counts"]["other_model_action"] += 1
    return {"version": 1, "runs_with_execution": runs_with_execution,
            "runs_without_execution_events": legacy_runs,
            "semantics": "The first subsequent decision is classified. A model final is a candidate, not proof that direct delivery preserves requested synthesis.",
            "results": [{"tool_name": name, "result_state": result_state, **dict(row["counts"]),
                         "final_decision_duration_ms": distribution(row["reply_ms"]),
                         "tool_default_direct_delivery": name in _DIRECT_TOOLS}
                        for (name, result_state), row in sorted(groups.items())]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("databases", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = summarize(args.databases)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"result_states": len(report["results"]), "runs": report["runs_with_execution"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
