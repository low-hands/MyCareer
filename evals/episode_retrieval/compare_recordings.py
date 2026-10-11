"""Compare recorded final decisions under today's scenario assertions.

This intentionally scores recorded decisions without replaying tool selection:
old cassettes have different prompt and schema fingerprints. It measures
behavioral assertions on the common scenarios, not prompt-cache validity.

Usage: python compare_recordings.py <before_root> <after_root>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from career_agent.agent.contracts.decisions import AgentDecision, ToolCall
from career_agent.evaluation.search_scenarios import SEARCH_SCENARIOS
from career_agent.evaluation.trajectory import (
    check_step,
    check_step_quality,
    load_cassette,
    quality_shortfall,
)


def score(root: Path, names: set[str]) -> dict[str, dict]:
    rows = {}
    for scenario in SEARCH_SCENARIOS:
        if scenario.name not in names:
            continue
        cassette = load_cassette(scenario.name, root=root)
        if cassette is None:
            continue
        hard_samples = []
        quality_samples = []
        for sample in cassette.recordings:
            hard = []
            quality = []
            for index, step in enumerate(scenario.steps):
                responses = [item for item in sample if item.get("scenario_step") == index]
                if not responses:
                    hard.append(f"{scenario.name}[{index}]: missing recorded decision")
                    continue
                item = responses[-1]
                try:
                    if item.get("tool_call") is not None:
                        call = item["tool_call"]
                        decision = AgentDecision(
                            action="tool_call",
                            tool_call=ToolCall(name=call["name"], arguments=call.get("arguments", {})),
                        )
                    else:
                        decision = AgentDecision.model_validate_json(item["content"])
                except (KeyError, ValueError) as error:
                    hard.append(f"{scenario.name}[{index}]: invalid recorded decision: {error}")
                    continue
                hard.extend(check_step(step, decision, scenario=scenario.name, index=index))
                quality.extend(check_step_quality(step, decision, scenario=scenario.name, index=index))
            hard_samples.append(tuple(hard))
            quality_samples.append(tuple(quality))
        rows[scenario.name] = {
            "model": cassette.model,
            "samples": len(hard_samples),
            "hard_failed": any(hard_samples),
            "quality_failed": bool(quality_shortfall(scenario, tuple(quality_samples)))
            if scenario.has_quality_assertions else False,
            "hard_failures": [list(failure) for failure in hard_samples],
        }
    return rows


def main() -> None:
    before_root, after_root = (Path(arg) for arg in sys.argv[1:3])
    before_names = {path.stem for path in before_root.glob("*.json")}
    after_names = {path.stem for path in after_root.glob("*.json")}
    common = before_names & after_names & {scenario.name for scenario in SEARCH_SCENARIOS}
    before = score(before_root, common)
    after = score(after_root, common)
    result = {
        "common_scenarios": len(common),
        "before": {
            "models": sorted({row["model"] for row in before.values()}),
            "hard_failed": sum(row["hard_failed"] for row in before.values()),
            "quality_failed": sum(row["quality_failed"] for row in before.values()),
        },
        "after": {
            "models": sorted({row["model"] for row in after.values()}),
            "hard_failed": sum(row["hard_failed"] for row in after.values()),
            "quality_failed": sum(row["quality_failed"] for row in after.values()),
        },
        "changed": [
            {
                "scenario": name,
                "before_hard_failed": before[name]["hard_failed"],
                "after_hard_failed": after[name]["hard_failed"],
                "before_quality_failed": before[name]["quality_failed"],
                "after_quality_failed": after[name]["quality_failed"],
            }
            for name in sorted(common)
            if (before[name]["hard_failed"], before[name]["quality_failed"])
            != (after[name]["hard_failed"], after[name]["quality_failed"])
        ],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
