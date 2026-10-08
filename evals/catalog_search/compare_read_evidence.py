"""Archived v3 read-eligibility experiment; not a current v4 evaluator.

Run from repository root with .venv/bin/python evals/catalog_search/compare_read_evidence.py
"""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

from career_agent.agent.capabilities import search
from career_agent.evaluation.catalog_search import (
    FIXTURE, evaluate_queries, existing_dev_baseline, workspace_snapshot,
)


def historical_v3_experiment():
    original = search._has_retrieval_evidence
    entries = search.searchable_capabilities()
    common = search._common_example_terms(entries)
    metadata = {
        item.name: {token for field in (item.name, " ".join(item.aliases_zh), item.summary or "")
                    for token in search._tokens(field)} - common
        for item in entries
    }
    frequency = Counter(token for tokens in metadata.values() for token in tokens)
    weights = {token: math.log(1 + (len(entries) - count + 0.5) / (count + 0.5))
               for token, count in frequency.items()}

    def idf_majority(query, query_terms, descriptor, common_terms, effective_query_terms):
        normalized = query.strip().lower()
        if (descriptor.effect != "READ" or descriptor.name.startswith("propose_")
                or normalized in (descriptor.name, descriptor.namespace)
                or normalized in (alias.strip().lower() for alias in descriptor.aliases_zh)):
            return original(query, query_terms, descriptor, common_terms, effective_query_terms)
        denominator = sum(weights[token] for token in effective_query_terms)
        matched = effective_query_terms.intersection(search._indexed_terms(descriptor))
        return denominator > 0 and sum(weights[token] for token in matched) / denominator >= 0.5

    raw = FIXTURE.read_bytes()
    cases = [case for case in json.loads(raw)["cases"] if case["split"] == "dev"]
    report = {
        **workspace_snapshot(), "fixture_sha256": hashlib.sha256(raw).hexdigest(),
        "experiment_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "holdout_status": "frozen_unscored",
        "idf_definition": "BM25 IDF over name/aliases/summary; effective matched weight / all effective query weight >= 0.5; nonempty evidence; exact and action policies unchanged.",
        "variants": {},
    }
    try:
        for label, gate in (("metadata_count_2_1_2", original), ("metadata_idf_majority", idf_majority)):
            search._has_retrieval_evidence = gate
            report["variants"][label] = {
                "model_queries": evaluate_queries(cases),
                "existing_dev": existing_dev_baseline(),
            }
    finally:
        search._has_retrieval_evidence = original
    output = Path(__file__).with_name("read_evidence_comparison_2026-10-08.json")
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    for name, result in report["variants"].items():
        print(name, {key: value for key, value in result["model_queries"].items() if key != "rows"})


if __name__ == "__main__":
    raise SystemExit("Archived v3 experiment: the retrieval and scoring policies changed. "
                     "Use python -m career_agent.evaluation.catalog_search for v4 dev results. "
                     "Saved v3 comparisons remain in read_evidence_comparison_2026-10-08.json.")
