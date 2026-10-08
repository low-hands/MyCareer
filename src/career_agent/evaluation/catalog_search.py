"""Direct model-style catalogue queries; never scores holdout by default.

python -m career_agent.evaluation.catalog_search --output evals/catalog_search/baseline.json
Add --check to enforce recall only; action exposure is diagnostic.
Use --final-holdout --output <new-file> once, after retrieval rules are finalized.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from career_agent.agent.capabilities.catalog import CAPABILITIES
from career_agent.agent.capabilities.search import search_catalog

FIXTURE = Path(__file__).resolve().parents[3] / "evals/catalog_search/model_queries.json"
MODEL_QUERY_SCORES = FIXTURE.with_name("model_query_semantic_scores.json")


def workspace_snapshot(*, require_clean=False):
    root = FIXTURE.parents[2]
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"], cwd=root, text=True,
    )
    if require_clean and status:
        raise ValueError("Final holdout requires a clean git worktree (including untracked files); no queries were scored.")
    # HEAD alone cannot identify uncommitted source/fixture changes. Record
    # content hashes without reading unrelated untracked files such as .env.
    paths = sorted((root / "src").rglob("*.py")) + [
        FIXTURE, MODEL_QUERY_SCORES, root / "evals/capability_semantic_scores.json",
    ]
    return {
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "git_status_porcelain": status,
        "worktree_clean": not bool(status),
        "source_sha256": {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in paths if path.exists()},
    }


def evaluate_queries(cases, *, search=search_catalog):
    rows = []
    hits = {k: 0 for k in (1, 3, 5)}
    demand = 0
    for case in cases:
        offered = search(query=case["query"], limit=5)
        expected = set(case["expected_tools"])
        if expected:
            demand += 1
            for k in hits:
                hits[k] += expected <= set(offered[:k])
        is_write_query = any(CAPABILITIES[name].effect == "WRITE" for name in expected)
        # Diagnostic only: an explicit write may retrieve additional writes.
        # Keep these separate from forbidden results and pass/fail contracts.
        non_target_writes = [name for name in offered if (
            is_write_query and name not in expected and CAPABILITIES[name].effect == "WRITE"
        )]
        missing = sorted(expected - set(offered[:case["required_k"]]))
        writes = [name for name in offered if CAPABILITIES[name].effect == "WRITE"]
        proposals = [name for name in offered if name.startswith("propose_")]
        actions = [name for name in offered if name in writes or name in proposals]
        rows.append({**case, "offered": list(offered), "missing": missing,
                     "is_write_query": is_write_query,
                     "write_offered": writes, "propose_offered": proposals,
                     "action_offered": actions,
                     "non_target_write_offered": non_target_writes,
                     "passed": not missing if expected else None})
    result = {
        "query_count": len(rows), "demand_count": demand,
        "recall": {str(k): {"hits": hits[k], "total": demand,
                            "rate": hits[k] / demand if demand else None} for k in hits},
        "write_query_count": sum(row["is_write_query"] for row in rows),
        "non_target_write_offer_count": sum(len(row["non_target_write_offered"]) for row in rows),
        "observation_only_count": sum(row["passed"] is None for row in rows),
        "failed_case_count": sum(row["passed"] is False for row in rows), "rows": rows,
    }
    for category in ("write", "propose", "action"):
        result[f"{category}_offer_count"] = sum(len(row[f"{category}_offered"]) for row in rows)
        result[f"non_write_query_{category}_offer_count"] = sum(
            len(row[f"{category}_offered"]) for row in rows if not row["is_write_query"]
        )
    return result


def load_model_query_scores(path: Path):
    """Validate recording provenance without ranking or evaluating holdout."""
    from career_agent.evaluation.capability_semantic_fixture import RecordedCapabilityScores, catalogue_digest

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("catalogue_digest") != catalogue_digest():
        raise ValueError("semantic score catalogue changed; re-record scores")
    if payload.get("fixture_sha256") != hashlib.sha256(FIXTURE.read_bytes()).hexdigest():
        raise ValueError("semantic score query fixture changed; re-record scores")
    queries = {case["query"] for case in json.loads(FIXTURE.read_text())["cases"]}
    if set(payload["scores"]) != queries:
        raise ValueError("semantic scores must cover all 26 frozen queries exactly")
    return RecordedCapabilityScores(payload["scores"])


def existing_dev_baseline(*, include_hybrid=False):
    from career_agent.evaluation.capability_semantic_fixture import recorded_capability_scores
    from career_agent.evaluation.tool_selection import RuntimeSearchSelector, evaluate_tool_selection
    from career_agent.evaluation.tool_selection_scenarios import SELECTION_DEV
    from career_agent.evaluation.trajectory import trajectory_tool_specs

    cases = []
    for case in SELECTION_DEV:
        step = case.scenario.steps[0]
        expected = step.expect_tools or ({step.expect_tool} if step.expect_tool else set())
        cases.append({"id": case.scenario.name, "query": case.scenario.context.user_message,
                      "expected_tools": sorted(expected), "required_k": 5, "kind": case.kind})
    direct = evaluate_queries(cases)
    report = evaluate_tool_selection(SELECTION_DEV, selector=RuntimeSearchSelector(trajectory_tool_specs()))
    result = {"user_sentence_search_catalog": direct, "runtime_selection": {
        "semantic_mode": "recorded" if recorded_capability_scores() else "lexical_fallback",
        "steps": len(report.steps), "covered_steps": report.covered_steps,
        "demand_steps": report.demand_steps,
        "unrequested_write_offer_count": report.unrequested_write_offer_count,
        "unreachable_offer_count": report.unreachable_offer_count,
        "waiting_reoffer_count": report.waiting_reoffer_count,
        "schema_tokens_proxy_total": sum(step.schema_tokens_proxy for step in report.steps),
    }}

    if include_hybrid:
        scorer = recorded_capability_scores()
        if scorer is None:
            raise ValueError("user-sentence hybrid evaluation needs matching recorded scores")
        result["user_sentence_search_catalog_hybrid"] = evaluate_queries(
            cases, search=lambda **kwargs: search_catalog(
                **kwargs, semantic_scores=scorer.scores(kwargs["query"]),
            ),
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--final-holdout", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--semantic-scores", type=Path, help="Compare lexical and hybrid using recorded embedding scores")
    args = parser.parse_args()
    try:
        snapshot = workspace_snapshot(require_clean=args.final_holdout)
        scorer = load_model_query_scores(args.semantic_scores) if args.semantic_scores else None
    except ValueError as error:
        parser.error(str(error))
    raw = FIXTURE.read_bytes()
    fixture = json.loads(raw)
    split = "holdout" if args.final_holdout else "dev"
    # Exclusive creation prevents accidentally overwriting a final evaluation.
    with args.output.open("x", encoding="utf-8") as output:
        result = {
            **snapshot,
            "fixture_sha256": hashlib.sha256(raw).hexdigest(),
            "scoring_policy": "recall_only; WRITE/propose_* exposure is diagnostic",
            "search_mode": ("search_catalog lexical and hybrid; recorded semantic scores"
                            if scorer is not None else "search_catalog lexical; no semantic scores injected"),
            "split": split,
            "holdout_status": "scored_final" if args.final_holdout else "frozen_unscored",
            "model_queries": evaluate_queries([c for c in fixture["cases"] if c["split"] == split]),
        }
        if scorer is not None:
            result["semantic_recording"] = {
                "path": str(args.semantic_scores),
                "sha256": hashlib.sha256(args.semantic_scores.read_bytes()).hexdigest(),
            }
            result["model_queries_hybrid"] = evaluate_queries(
                [c for c in fixture["cases"] if c["split"] == split],
                search=lambda **kwargs: search_catalog(
                    **kwargs, semantic_scores=scorer.scores(kwargs["query"]),
                ),
            )
        if not args.final_holdout:
            result["existing_dev"] = existing_dev_baseline(include_hybrid=scorer is not None)
        output.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), **{k: v for k, v in result["model_queries"].items() if k != "rows"}}, ensure_ascii=False))
    checked = result.get("model_queries_hybrid", result["model_queries"])
    if scorer is not None:
        print(json.dumps({"hybrid": {k: v for k, v in checked.items() if k != "rows"}}, ensure_ascii=False))
    return int(args.check and checked["failed_case_count"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
