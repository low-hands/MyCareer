"""Dev-only comparison: current catalogue search vs an industry-standard baseline.

The baseline keeps tokenisation, field-weighted BM25, the semantic threshold,
exact-name precedence and RRF (k=60). It replaces the project's own rules with
the published patterns they were adapted from:

- example queries are appended to each tool document before indexing
  (doc2query) instead of being scored separately behind overlap gates;
- no evidence gate: every tool with a positive BM25 score is a lexical
  candidate (Anthropic's BM25 tool search returns top-k).

Never touches the first holdout split; --holdout-v2 scores the second one once. Run from the repository root:

    .venv/bin/python evals/catalog_search/compare_simplified.py --output <new-file>
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path

from career_agent.agent.capabilities import search, selection_strategy
from career_agent.evaluation.catalog_search import (
    FIXTURE, MODEL_QUERY_SCORES, evaluate_queries, load_model_query_scores,
    workspace_snapshot,
)


def _baseline_index():
    entries = search._DEFAULT_INDEX.entries
    documents = {}
    for item in entries:
        document = Counter(search._document(item))
        for example in item.example_queries:
            document.update(search._tokens(example))
        documents[item.name] = document
    lengths = {name: sum(tokens.values()) for name, tokens in documents.items()}
    average = sum(lengths.values()) / len(lengths)
    frequency = Counter(token for tokens in documents.values() for token in tokens)
    idf = {term: math.log(1 + (len(entries) - count + 0.5) / (count + 0.5))
           for term, count in frequency.items()}
    return entries, documents, lengths, average, idf


_ENTRIES, _DOCUMENTS, _LENGTHS, _AVERAGE, _IDF = _baseline_index()


def baseline_lexical_scores(query: str) -> dict[str, float]:
    terms = set(search._tokens(query))
    scores = {}
    for item in _ENTRIES:
        tokens = _DOCUMENTS[item.name]
        score = 0.0
        for term in terms:
            tf = tokens.get(term, 0)
            if tf:
                score += _IDF[term] * (tf * 2.2) / (
                    tf + 1.2 * (0.25 + 0.75 * _LENGTHS[item.name] / _AVERAGE)
                )
        if score:
            scores[item.name] = score
    return scores


def baseline_search(*, query=None, names=None, limit=5, semantic_scores=None, descriptors=None):
    if names is not None or descriptors is not None:
        return search.search_catalog(
            query=query, names=names, limit=limit,
            semantic_scores=semantic_scores, descriptors=descriptors,
        )
    index = search._DEFAULT_INDEX
    normalized = query.strip().lower()
    exact = tuple(item.name for item in _ENTRIES if normalized in index.exact_queries[item.name])
    lexical = search._rank(baseline_lexical_scores(query), index.order)
    semantic = search._rank(
        {name: score for name, score in (semantic_scores or {}).items()
         if name in index.order and score >= search.MIN_SEMANTIC_SIMILARITY},
        index.order,
    )[:search.MAX_SEMANTIC_CANDIDATES]
    combined: dict[str, float] = {}
    for ranking in (lexical, semantic):
        for position, name in enumerate(ranking, start=1):
            combined[name] = combined.get(name, 0.0) + 1 / (60 + position)
    ranked = sorted(combined, key=lambda name: (-combined[name], index.order[name]))
    return tuple(dict.fromkeys((*exact, *ranked)))[:limit]


def _summary(result):
    keep = ("demand_count", "failed_case_count", "write_offer_count",
            "propose_offer_count", "non_write_query_action_offer_count",
            "non_target_write_offer_count")
    return {
        "recall": {k: [v["hits"], v["total"]] for k, v in result["recall"].items()},
        **{k: result[k] for k in keep},
        "missing": {row["query"]: row["missing"] for row in result["rows"] if row["missing"]},
    }


def _user_sentence_cases():
    from career_agent.evaluation.tool_selection_scenarios import SELECTION_DEV

    cases = []
    for case in SELECTION_DEV:
        step = case.scenario.steps[0]
        expected = step.expect_tools or ({step.expect_tool} if step.expect_tool else set())
        cases.append({"id": case.scenario.name, "query": case.scenario.context.user_message,
                      "expected_tools": sorted(expected), "required_k": 5, "kind": case.kind})
    return cases


def _runtime(search_fn):
    """Runtime offers use the intent ranking inside SearchStrategy."""
    from career_agent.evaluation.tool_selection import RuntimeSearchSelector, evaluate_tool_selection
    from career_agent.evaluation.tool_selection_scenarios import SELECTION_DEV
    from career_agent.evaluation.trajectory import trajectory_tool_specs

    original = selection_strategy.search_catalog
    selection_strategy.search_catalog = search_fn
    try:
        report = evaluate_tool_selection(
            SELECTION_DEV, selector=RuntimeSearchSelector(trajectory_tool_specs()),
        )
    finally:
        selection_strategy.search_catalog = original
    return {
        "covered_steps": [report.covered_steps, report.demand_steps],
        "unrequested_write_offer_count": report.unrequested_write_offer_count,
        "unreachable_offer_count": report.unreachable_offer_count,
        "waiting_reoffer_count": report.waiting_reoffer_count,
        "schema_tokens_proxy_total": sum(step.schema_tokens_proxy for step in report.steps),
    }


def evaluate(search_fn):
    from career_agent.evaluation.capability_semantic_fixture import recorded_capability_scores

    dev = [case for case in json.loads(FIXTURE.read_text(encoding="utf-8"))["cases"]
           if case["split"] == "dev"]
    model_scores = load_model_query_scores(MODEL_QUERY_SCORES)
    sentence_scores = recorded_capability_scores()
    if sentence_scores is None:
        raise ValueError("user-sentence hybrid evaluation needs matching recorded scores")
    sentences = _user_sentence_cases()

    def hybrid(scorer):
        return lambda **kw: search_fn(**kw, semantic_scores=scorer.scores(kw["query"]))

    return {
        "model_query_lexical": _summary(evaluate_queries(dev, search=search_fn)),
        "model_query_hybrid": _summary(evaluate_queries(dev, search=hybrid(model_scores))),
        "user_sentence_lexical": _summary(evaluate_queries(sentences, search=search_fn)),
        "user_sentence_hybrid": _summary(evaluate_queries(sentences, search=hybrid(sentence_scores))),
        "runtime_selection": _runtime(search_fn),
    }


HOLDOUT_V2 = FIXTURE.with_name("holdout_v2_queries.json")
HOLDOUT_V2_SCORES = FIXTURE.with_name("holdout_v2_semantic_scores.json")


def _holdout_v2_scorer():
    import hashlib
    from career_agent.evaluation.capability_semantic_fixture import (
        RecordedCapabilityScores, catalogue_digest,
    )

    payload = json.loads(HOLDOUT_V2_SCORES.read_text(encoding="utf-8"))
    if payload.get("catalogue_digest") != catalogue_digest():
        raise ValueError("semantic score catalogue changed; re-record scores")
    if payload.get("fixture_sha256") != hashlib.sha256(HOLDOUT_V2.read_bytes()).hexdigest():
        raise ValueError("holdout v2 changed after its scores were recorded")
    return RecordedCapabilityScores(payload["scores"])


def evaluate_holdout_v2():
    """Score the second holdout once, for both rule sets, in a clean worktree."""
    snapshot = workspace_snapshot(require_clean=True)
    cases = json.loads(HOLDOUT_V2.read_text(encoding="utf-8"))["cases"]
    scorer = _holdout_v2_scorer()
    result = {**snapshot, "split": "holdout_v2"}
    for name, search_fn in (("current", search.search_catalog), ("baseline", baseline_search)):
        result[name] = {
            "lexical": _summary(evaluate_queries(cases, search=search_fn)),
            "hybrid": _summary(evaluate_queries(
                cases, search=lambda fn=search_fn, **kw: fn(**kw, semantic_scores=scorer.scores(kw["query"])),
            )),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--holdout-v2", action="store_true",
                        help="Score the second holdout once; requires a clean worktree.")
    args = parser.parse_args()
    if args.holdout_v2:
        result = evaluate_holdout_v2()
        with args.output.open("x", encoding="utf-8") as output:
            output.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        for name in ("current", "baseline"):
            print(name, {k: v["recall"] for k, v in result[name].items()})
        return
    result = {
        **workspace_snapshot(),
        "split": "dev",
        "current": evaluate(search.search_catalog),
        "baseline": evaluate(baseline_search),
    }
    with args.output.open("x", encoding="utf-8") as output:
        output.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    for name in ("current", "baseline"):
        print(name, json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "missing"}
                                for k, v in result[name].items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
