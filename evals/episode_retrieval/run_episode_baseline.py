"""Lexical baseline for the Chinese episode retrieval query set.

Usage: python run_episode_baseline.py <branch_src_dir> <queryset.json> <work_dir>
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

src_dir, queryset_path, work_dir = (Path(arg) for arg in sys.argv[1:4])
sys.path.insert(0, str(src_dir))

from career_agent.domain.episodes.models import CareerEpisodeDraft  # noqa: E402
from career_agent.storage.episodes import SQLiteCareerEpisodeStore  # noqa: E402

data = json.loads(queryset_path.read_text(encoding="utf-8"))
work_dir.mkdir(parents=True, exist_ok=True)
db = work_dir / "episodes.sqlite3"
for suffix in ("", "-wal", "-shm"):
    Path(str(db) + suffix).unlink(missing_ok=True)

store = SQLiteCareerEpisodeStore(db)
USER = "eval_user"
id_map: dict[str, str] = {}
for item in data["corpus"]:
    episode = store.upsert(
        CareerEpisodeDraft(
            user_id=USER,
            kind=item["kind"],
            source_run_id=item["id"],
            occurred_at=datetime.fromisoformat(item["occurred_at"]),
            title=item["title"],
            summary=item["summary"],
        )
    )
    assert episode is not None, item["id"]
    id_map[episode.id] = item["id"]


def run(mode: str, query: dict) -> tuple[list[str], str | None, list[str], list[str], list[str]]:
    filters = query.get("filters", {})
    text = query["user_message"] if mode != "tool_keywords" else query["tool_query"]
    matched, partial, unmatched = store.term_matches(
        user_id=USER,
        query=text,
        start_datetime=(datetime.fromisoformat(filters["start"]) if "start" in filters else None),
        end_datetime=(datetime.fromisoformat(filters["end"]) if "end" in filters else None),
        kinds=tuple(filters.get("kinds", ())),
    )
    try:
        if mode == "projection":
            hits = store.project_relevant(
                user_id=USER, query=query["user_message"], limit=5
            )
        else:
            hits = store.search(
                user_id=USER,
                query=text,
                limit=8,
                start_datetime=(
                    datetime.fromisoformat(filters["start"]) if "start" in filters else None
                ),
                end_datetime=(
                    datetime.fromisoformat(filters["end"]) if "end" in filters else None
                ),
                kinds=tuple(filters.get("kinds", ())),
            )
    except Exception as error:  # report, do not hide
        return [], f"{type(error).__name__}: {error}", list(matched), list(partial), list(unmatched)
    return [id_map[hit.id] for hit in hits], None, list(matched), list(partial), list(unmatched)


def ndcg(ranked: list[str], relevant: dict[str, int], k: int) -> float:
    dcg = sum(
        relevant.get(doc, 0) / math.log2(rank + 2)
        for rank, doc in enumerate(ranked[:k])
    )
    ideal = sorted(relevant.values(), reverse=True)[:k]
    idcg = sum(grade / math.log2(rank + 2) for rank, grade in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


modes = ("projection", "tool_keywords", "tool_sentence")
rows = []
for query in data["queries"]:
    relevant = query["relevant"]
    k = 8 if query["id"] == "Q38" else 5
    for mode in modes:
        ranked, error, matched_terms, partially_matched_terms, unmatched_terms = run(mode, query)
        targets = {doc for doc, grade in relevant.items() if grade == 2}
        if relevant:
            recall = len(targets & set(ranked[:k])) / len(targets) if targets else (
                1.0 if set(relevant) & set(ranked[:k]) else 0.0
            )
            score = {"recall": recall, "ndcg": ndcg(ranked, relevant, k)}
        else:
            top1 = ranked[0] if ranked else None
            score = {
                "returned": len(ranked),
                "top1": top1,
                "top1_near_miss": top1 in query.get("near_miss", []),
            }
        rows.append(
            {
                "id": query["id"],
                "category": query["category"],
                "mode": mode,
                "ranked": ranked,
                "error": error,
                "matched_terms": matched_terms,
                "partially_matched_terms": partially_matched_terms,
                "unmatched_terms": unmatched_terms,
                **score,
            }
        )

(work_dir / "baseline_results.json").write_text(
    json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8"
)

# Summary per mode and category (answerable queries only for recall/ndcg).
summary: dict = defaultdict(lambda: defaultdict(list))
for row in rows:
    if "recall" in row:
        summary[row["mode"]][row["category"]].append(row)
        summary[row["mode"]]["ALL"].append(row)
for mode in modes:
    print(f"\n== {mode}")
    for category, items in sorted(summary[mode].items()):
        recall = sum(item["recall"] for item in items) / len(items)
        score = sum(item["ndcg"] for item in items) / len(items)
        errors = sum(1 for item in items if item["error"])
        print(f"  {category:20s} n={len(items):2d} recall={recall:.2f} ndcg={score:.2f} errors={errors}")
    unanswerable = [row for row in rows if row["mode"] == mode and "returned" in row]
    print(
        "  unanswerable: "
        + ", ".join(
            f"{row['id']}:returned={row['returned']},top1={row['top1']},"
            f"unmatched={row['unmatched_terms']}"
            f"{'(near-miss)' if row['top1_near_miss'] else ''}"
            for row in unanswerable
        )
    )

print("\n== per-query (recall/ndcg; ranked ids)")
for row in rows:
    if "recall" in row:
        print(
            f"{row['id']} {row['mode']:13s} r={row['recall']:.2f} n={row['ndcg']:.2f} "
            f"{row['ranked'][:5]}{' ERR ' + row['error'] if row['error'] else ''}"
        )
