"""Record embeddings for all frozen queries without evaluating either split."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from career_agent.agent.capabilities.search import MAX_SEMANTIC_CANDIDATES, MIN_SEMANTIC_SIMILARITY
from career_agent.agent.context.semantic_retrieval import EmbeddingClient
from career_agent.evaluation.capability_semantic_fixture import record_capability_scores
from career_agent.evaluation.catalog_search import FIXTURE


def record_model_query_scores(
    client: EmbeddingClient, *, model: str, output: Path, fixture: Path = FIXTURE,
) -> Path:
    if output.exists():
        raise FileExistsError("choose a new semantic-score output path")
    raw = fixture.read_bytes()
    queries = tuple(case["query"] for case in json.loads(raw)["cases"])
    record_capability_scores(client, model=model, queries=queries, path=output)
    payload = json.loads(output.read_text(encoding="utf-8"))
    payload.update({
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "fixture_sha256": hashlib.sha256(raw).hexdigest(),
        "model_id": client.model_id,
        "query_count": len(queries),
        "min_similarity": MIN_SEMANTIC_SIMILARITY,
        "max_candidates": MAX_SEMANTIC_CANDIDATES,
        "holdout_status": "embeddings_and_similarity_recorded_only; no search ranking or recall evaluation",
        "score_scope": "runtime SemanticCapabilityIndex: similarity >= 0.55, top 10 candidates",
    })
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output


def main() -> int:
    from dotenv import load_dotenv
    from career_agent.agent.context.semantic_retrieval import CareerEmbeddingConfig, OpenAICompatibleEmbeddingClient

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, default=FIXTURE)
    args = parser.parse_args()
    load_dotenv(FIXTURE.parents[2] / ".env")
    config = CareerEmbeddingConfig.optional_from_env()
    if config is None:
        parser.error("CAREER_EMBEDDING_BASE_URL, _API_KEY and _MODEL are required")
    path = record_model_query_scores(
        OpenAICompatibleEmbeddingClient(config), model=config.model, output=args.output,
        fixture=args.fixture,
    )
    count = len(json.loads(args.fixture.read_text(encoding="utf-8"))["cases"])
    print(f"recorded {count} query similarity maps to {path}; holdout not evaluated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
