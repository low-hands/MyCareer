"""Recorded capability embedding scores, so evaluation stays offline.

The runtime ranks turn intent with lexical and embedding scores. Evaluation
replays the same ranking from scores recorded once per catalogue, instead of
calling the embedding provider on every replay. Refresh after changing a
capability's name, namespace, summary or aliases, or an evaluation message:

    uv run python -m career_agent.evaluation.capability_semantic_fixture
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import lru_cache
import hashlib
import json
from pathlib import Path

from career_agent.agent.capabilities.search import (
    SemanticCapabilityIndex, capability_embedding_texts, searchable_capabilities,
)
from career_agent.agent.capabilities.selection_strategy import SearchStrategy
from career_agent.agent.context.semantic_retrieval import EmbeddingClient


SEMANTIC_FIXTURE_PATH = (
    Path(__file__).resolve().parents[3] / "evals" / "capability_semantic_scores.json"
)
_INTENT_QUERY_CHARS = 200


def catalogue_digest() -> str:
    texts = capability_embedding_texts(searchable_capabilities())
    return hashlib.sha256(
        json.dumps(texts, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def evaluation_queries() -> tuple[str, ...]:
    """Every user message an evaluation strategy ranks as turn intent."""
    from career_agent.evaluation.independent_tool_selection_holdout import (
        SELECTION_INDEPENDENT_HOLDOUT,
    )
    from career_agent.evaluation.main_agent_scenarios import SCENARIOS
    from career_agent.evaluation.tool_selection_scenarios import (
        SELECTION_DEV, SELECTION_HOLDOUT,
    )

    scenarios = (
        *SCENARIOS,
        *(case.scenario for case in (
            *SELECTION_DEV, *SELECTION_HOLDOUT, *SELECTION_INDEPENDENT_HOLDOUT,
        )),
    )
    messages = {
        message[:_INTENT_QUERY_CHARS]
        for scenario in scenarios
        for message in (
            scenario.context.user_message,
            *(step.user_message for step in scenario.steps),
        )
        if message is not None and message.strip()
    }
    return tuple(sorted(messages))


class RecordedCapabilityScores:
    def __init__(self, scores: Mapping[str, Mapping[str, float]]) -> None:
        self._scores = scores

    def scores(self, query: str) -> Mapping[str, float]:
        if query not in self._scores:
            raise KeyError(
                f"no recorded capability scores for this query; refresh {SEMANTIC_FIXTURE_PATH.name}"
            )
        return self._scores[query]


@lru_cache(maxsize=1)
def recorded_capability_scores() -> RecordedCapabilityScores | None:
    """None when absent or recorded for another catalogue; ranking is then lexical."""
    if not SEMANTIC_FIXTURE_PATH.exists():
        return None
    payload = json.loads(SEMANTIC_FIXTURE_PATH.read_text(encoding="utf-8"))
    if payload.get("catalogue_digest") != catalogue_digest():
        return None
    return RecordedCapabilityScores(payload["scores"])


def evaluation_strategy() -> SearchStrategy:
    return SearchStrategy(recorded_capability_scores())


def record_capability_scores(
    client: EmbeddingClient, *, model: str,
    queries: Sequence[str] | None = None, path: Path = SEMANTIC_FIXTURE_PATH,
) -> Path:
    """Score every evaluation query with the runtime index and write the fixture."""
    queries = evaluation_queries() if queries is None else tuple(queries)
    vectors = dict(zip(queries, client.embed(queries), strict=True))

    class _Recorded:
        model_id = client.model_id

        def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
            return tuple(vectors[text] for text in texts)

    index = SemanticCapabilityIndex(client, _Recorded())
    index.warm()
    payload = {
        "model": model,
        "catalogue_digest": catalogue_digest(),
        "scores": {
            query: {name: round(score, 6) for name, score in index.scores(query).items()}
            for query in queries
        },
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    recorded_capability_scores.cache_clear()
    return path


def main() -> int:
    from dotenv import load_dotenv

    from career_agent.agent.context.semantic_retrieval import (
        CareerEmbeddingConfig, OpenAICompatibleEmbeddingClient,
    )

    load_dotenv()
    config = CareerEmbeddingConfig.optional_from_env()
    if config is None:
        print("CAREER_EMBEDDING_BASE_URL, _API_KEY and _MODEL are required")
        return 2
    path = record_capability_scores(
        OpenAICompatibleEmbeddingClient(config, timeout_seconds=30.0),
        model=config.model,
    )
    print(f"recorded {len(evaluation_queries())} queries to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
