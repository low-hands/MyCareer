from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import sqlite3
from typing import Protocol
from urllib.parse import urlsplit

from openai import OpenAI

from career_agent.storage.career_history import CareerHistoryStore

_EMBED_TIMEOUT_SECONDS = 30.0
_LOGGER = logging.getLogger(__name__)


class EmbeddingClient(Protocol):
    model_id: str

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]: ...


class SemanticEvidenceCache(Protocol):
    """Derived vector rows that tombstone cleanup must be able to reach."""

    def forget_evidence_ids(self, evidence_ids: Sequence[str]) -> int: ...


@dataclass(frozen=True)
class CareerEmbeddingConfig:
    base_url: str
    api_key: str
    model: str
    batch_size: int = 64

    def __post_init__(self) -> None:
        parts = urlsplit(self.base_url)
        if parts.scheme != "https" or not parts.netloc:
            raise ValueError("career embedding base URL must be HTTPS")
        if not self.api_key.strip() or not self.model.strip():
            raise ValueError("career embedding API key and model are required")
        if not 1 <= self.batch_size <= 256:
            raise ValueError("career embedding batch size must be between 1 and 256")

    @classmethod
    def optional_from_env(
        cls,
        *,
        environ: Mapping[str, str] | None = None,
    ) -> CareerEmbeddingConfig | None:
        source = os.environ if environ is None else environ
        values = {
            "base_url": source.get("CAREER_EMBEDDING_BASE_URL", "").strip(),
            "api_key": source.get("CAREER_EMBEDDING_API_KEY", "").strip(),
            "model": source.get("CAREER_EMBEDDING_MODEL", "").strip(),
        }
        if not any(values.values()):
            return None
        if not all(values.values()):
            raise ValueError(
                "CAREER_EMBEDDING_BASE_URL, CAREER_EMBEDDING_API_KEY, and "
                "CAREER_EMBEDDING_MODEL must be configured together"
            )
        raw_batch_size = source.get("CAREER_EMBEDDING_BATCH_SIZE", "64").strip()
        try:
            batch_size = int(raw_batch_size)
        except ValueError as error:
            raise ValueError(
                "CAREER_EMBEDDING_BATCH_SIZE must be an integer"
            ) from error
        return cls(**values, batch_size=batch_size)


class OpenAICompatibleEmbeddingClient:
    def __init__(self, config: CareerEmbeddingConfig) -> None:
        namespace = f"{config.base_url.rstrip('/')}\n{config.model}"
        self.model_id = "openai-compatible:" + hashlib.sha256(
            namespace.encode("utf-8")
        ).hexdigest()
        self._model = config.model
        self._client = OpenAI(
            base_url=config.base_url.rstrip("/"),
            api_key=config.api_key,
            timeout=_EMBED_TIMEOUT_SECONDS,
            max_retries=0,
        )

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        if not texts:
            return ()
        response = self._client.embeddings.create(
            model=self._model,
            input=list(texts),
        )
        ordered = sorted(response.data, key=lambda item: item.index)
        return tuple(tuple(item.embedding) for item in ordered)


class SQLiteCareerEvidenceSemanticRetriever:
    """Cosine retrieval over current confirmed evidence with a derived cache.

    The cache is not authoritative memory. Evidence ids are returned to the
    projector, which re-reads and revalidates them against the typed career
    store before projection. This keeps embedding providers out of the write,
    provenance, correction, and tombstone paths.
    """

    def __init__(
        self,
        *,
        career_history: CareerHistoryStore,
        cache_path: Path,
        client: EmbeddingClient,
        batch_size: int = 64,
    ) -> None:
        if not 1 <= batch_size <= 256:
            raise ValueError("embedding batch size must be between 1 and 256")
        self._career_history = career_history
        self._cache_path = cache_path.expanduser()
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._client = client
        self._batch_size = batch_size
        self._initialize()

    def rank_current_evidence_ids(
        self,
        *,
        user_id: str,
        query: str,
        limit: int,
    ) -> Sequence[str]:
        if not 1 <= limit <= 100:
            raise ValueError("semantic evidence limit must be between 1 and 100")
        normalized_query = query.strip()
        if not normalized_query:
            return ()
        evidence = self._career_history.list_evidence(
            user_id=user_id,
            verification_status="confirmed",
            include_historical=False,
        )
        self.forget_evidence_ids(
            tuple(
                evidence_id
                for tombstone in self._career_history.list_evidence_tombstones(
                    user_id=user_id
                )
                for evidence_id in tombstone.evidence_ids
            )
        )
        if not evidence:
            return ()

        try:
            vectors = self._ensure_vectors(evidence)
            query_vector = self._validated_vectors(
                self._client.embed((normalized_query,)),
                expected=1,
            )[0]
        except Exception:
            _LOGGER.exception(
                "semantic retrieval failed; falling back to lexical ranking"
            )
            return ()
        ranked = sorted(
            (
                (self._cosine(query_vector, vectors[item.id]), item.id)
                for item in evidence
                if item.id in vectors
            ),
            key=lambda item: (-item[0], item[1]),
        )
        return tuple(evidence_id for _, evidence_id in ranked[:limit])

    def forget_evidence_ids(self, evidence_ids: Sequence[str]) -> int:
        """Drop derived vectors for redacted evidence. Safe to call repeatedly."""

        selected = tuple(dict.fromkeys(evidence_ids))
        if not selected:
            return 0
        placeholders = ",".join("?" for _ in selected)
        with sqlite3.connect(self._cache_path) as connection:
            cursor = connection.execute(
                f"""
                DELETE FROM career_evidence_embeddings
                WHERE evidence_id IN ({placeholders})
                """,
                selected,
            )
        return max(0, cursor.rowcount)

    def _ensure_vectors(
        self, evidence: Sequence[object]
    ) -> dict[str, tuple[float, ...]]:
        vectors = self._cached_vectors(evidence)
        missing = [item for item in evidence if item.id not in vectors]
        for start in range(0, len(missing), self._batch_size):
            batch = missing[start : start + self._batch_size]
            embedded = self._validated_vectors(
                self._client.embed([item.claim for item in batch]),
                expected=len(batch),
            )
            self._store_vectors(tuple(zip(batch, embedded, strict=True)))
            vectors.update(
                (item.id, vector)
                for item, vector in zip(batch, embedded, strict=True)
            )
        return vectors

    def _initialize(self) -> None:
        with sqlite3.connect(self._cache_path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS career_evidence_embeddings(
                    evidence_id TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    content_digest TEXT NOT NULL,
                    dimensions INTEGER NOT NULL CHECK(dimensions > 0),
                    vector_json TEXT NOT NULL,
                    PRIMARY KEY(evidence_id, model_id)
                )
                """
            )

    def _cached_vectors(self, evidence: Sequence[object]) -> dict[str, tuple[float, ...]]:
        expected = {
            item.id: self._content_digest(item.claim) for item in evidence
        }
        placeholders = ",".join("?" for _ in expected)
        if not placeholders:
            return {}
        with sqlite3.connect(self._cache_path) as connection:
            rows = connection.execute(
                f"""
                SELECT evidence_id, content_digest, dimensions, vector_json
                FROM career_evidence_embeddings
                WHERE model_id = ? AND evidence_id IN ({placeholders})
                """,
                (self._client.model_id, *expected),
            ).fetchall()
        vectors = {}
        for evidence_id, digest, dimensions, payload in rows:
            if expected.get(evidence_id) != digest:
                continue
            raw = json.loads(payload)
            vector = self._validated_vectors((raw,), expected=1)[0]
            if len(vector) == dimensions:
                vectors[evidence_id] = vector
        return vectors

    def _store_vectors(self, rows: Sequence[tuple[object, tuple[float, ...]]]) -> None:
        with sqlite3.connect(self._cache_path) as connection:
            connection.executemany(
                """
                INSERT INTO career_evidence_embeddings(
                    evidence_id, model_id, content_digest,
                    dimensions, vector_json
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(evidence_id, model_id) DO UPDATE SET
                    content_digest = excluded.content_digest,
                    dimensions = excluded.dimensions,
                    vector_json = excluded.vector_json
                """,
                (
                    (
                        item.id,
                        self._client.model_id,
                        self._content_digest(item.claim),
                        len(vector),
                        json.dumps(vector, separators=(",", ":")),
                    )
                    for item, vector in rows
                ),
            )

    @staticmethod
    def _content_digest(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    @staticmethod
    def _validated_vectors(
        vectors: Sequence[Sequence[float]],
        *,
        expected: int,
    ) -> tuple[tuple[float, ...], ...]:
        if len(vectors) != expected:
            raise ValueError("embedding provider returned the wrong vector count")
        normalized = tuple(tuple(float(value) for value in row) for row in vectors)
        dimensions = {len(row) for row in normalized}
        if dimensions == {0} or len(dimensions) != 1:
            raise ValueError("embedding vectors must share one nonzero dimension")
        if any(not math.isfinite(value) for row in normalized for value in row):
            raise ValueError("embedding vectors must contain finite values")
        return normalized

    @staticmethod
    def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
        if len(left) != len(right):
            raise ValueError("query and evidence embedding dimensions differ")
        left_norm = math.sqrt(math.fsum(value * value for value in left))
        right_norm = math.sqrt(math.fsum(value * value for value in right))
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        return math.fsum(a * b for a, b in zip(left, right, strict=True)) / (
            left_norm * right_norm
        )


def optional_semantic_retriever(
    *,
    career_history: CareerHistoryStore,
    cache_path: Path,
    environ: Mapping[str, str] | None = None,
) -> SQLiteCareerEvidenceSemanticRetriever | None:
    config = CareerEmbeddingConfig.optional_from_env(environ=environ)
    if config is None:
        return None
    return SQLiteCareerEvidenceSemanticRetriever(
        career_history=career_history,
        cache_path=cache_path,
        client=OpenAICompatibleEmbeddingClient(config),
        batch_size=config.batch_size,
    )
