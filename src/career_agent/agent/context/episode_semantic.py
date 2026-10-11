"""Derived episode vectors; authoritative candidates always come from the store."""
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import ContextVar, copy_context
from dataclasses import dataclass
import heapq
import logging
import os
from pathlib import Path
from threading import Event, RLock
from collections.abc import Sequence

from career_agent.storage.sqlite import open_connection
from career_agent.agent.context.semantic_retrieval import (
    CareerEmbeddingConfig, EmbeddingClient, OpenAICompatibleEmbeddingClient,
    SQLiteCareerEvidenceSemanticRetriever,
)
from career_agent.harness.observability import record_active_trace

_LOGGER = logging.getLogger(__name__)


class _TurnQueries:
    def __init__(self):
        self.lock = RLock()
        self.vectors: dict[tuple[str, str], Future] = {}


_TURN_QUERIES: ContextVar[_TurnQueries | None] = ContextVar("episode_query_vectors", default=None)


@contextmanager
def episode_query_vector_scope():
    """Cache queries only for this runtime turn, including provider failures."""
    token = _TURN_QUERIES.set(_TurnQueries())
    try:
        yield
    finally:
        _TURN_QUERIES.reset(token)


@dataclass(frozen=True)
class _EmbeddingText:
    id: str
    claim: str


class SQLiteCareerEpisodeSemanticIndex(SQLiteCareerEvidenceSemanticRetriever):
    _table = "career_episode_embeddings"
    _id_column = "episode_id"

    def __init__(self, *, episode_path: Path, cache_path: Path,
                 client: EmbeddingClient, batch_size: int = 64, candidate_limit: int = 20) -> None:
        if candidate_limit < 1:
            raise ValueError("candidate_limit must be positive")
        super().__init__(career_history=None, cache_path=cache_path,
                         client=client, batch_size=batch_size)
        os.chmod(self._cache_path, 0o600)
        self._episode_path = episode_path.expanduser()
        self._lock = RLock()
        self._epoch = 0
        self._pending: set[tuple[str, str, str]] = set()
        self._candidate_limit = candidate_limit
        self._backfilling = Event()
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="episode-embedding")

    def forget_episode_ids(self, episode_ids: Sequence[str]) -> int:
        with self._lock:
            self._epoch += 1
            return self.forget_evidence_ids(episode_ids)

    def prune_orphans(self) -> int:
        # The cache is a separate database, hence eventual, not atomic consistency.
        with self._lock, open_connection(self._cache_path) as connection:
            connection.execute("ATTACH DATABASE ? AS episodes", (str(self._episode_path),))
            cursor = connection.execute("""
                DELETE FROM career_episode_embeddings
                WHERE NOT EXISTS (SELECT 1 FROM episodes.career_episodes e
                                  WHERE e.id = career_episode_embeddings.episode_id)
            """)
            return max(0, cursor.rowcount)

    @staticmethod
    def _texts(episodes):
        return tuple(_EmbeddingText(item.id, f"{item.title}\n{item.summary}") for item in episodes)

    def _ensure_vectors(self, evidence):
        # Claim missing work briefly; a search never waits for a background embed.
        with self._lock:
            vectors = self._cached_vectors(evidence)
            missing = []
            keys = []
            epoch = self._epoch
            for item in evidence:
                key = (self._client.model_id, item.id, self._content_digest(item.claim))
                if item.id not in vectors and key not in self._pending:
                    self._pending.add(key)
                    keys.append(key)
                    missing.append(item)
        try:
            for start in range(0, len(missing), self._batch_size):
                batch = missing[start:start + self._batch_size]
                embedded = self._validated_vectors(
                    self._client.embed([item.claim for item in batch]), expected=len(batch))
                # Recheck committed text and deletion epoch before storing network results.
                with self._lock:
                    current = {item.id: item.claim for item in self._committed_texts([item.id for item in batch])}
                    rows = tuple((item, vector) for item, vector in zip(batch, embedded, strict=True)
                                 if epoch == self._epoch and current.get(item.id) == item.claim)
                    if rows:
                        self._store_vectors(rows)
                        vectors.update((item.id, vector) for item, vector in rows)
            return vectors
        finally:
            with self._lock:
                self._pending.difference_update(keys)

    def _committed_texts(self, episode_ids):
        if not episode_ids:
            return ()
        placeholders = ",".join("?" for _ in episode_ids)
        with open_connection(self._episode_path) as connection:
            rows = connection.execute(
                f"SELECT id,title,summary FROM career_episodes WHERE id IN ({placeholders})",
                tuple(episode_ids)).fetchall()
        return tuple(_EmbeddingText(id, f"{title}\n{summary}") for id, title, summary in rows)

    def _query_vector(self, query):
        scope = _TURN_QUERIES.get()
        if scope is None:
            return self._validated_vectors(self._client.embed((query,)), expected=1)[0]
        key = (self._client.model_id, query)
        with scope.lock:
            creator = key not in scope.vectors
            if creator:
                scope.vectors[key] = Future()
            future = scope.vectors[key]
        if creator:
            try:
                future.set_result(self._validated_vectors(self._client.embed((query,)), expected=1)[0])
            except Exception as error:
                future.set_exception(error)
        return future.result()

    def rank_episode_ids(self, *, query: str, candidates: Sequence[object]) -> Sequence[str]:
        try:
            if not query.strip() or not candidates:
                return ()
            if self._backfilling.is_set():
                # Startup owns cold misses. The first user turn does not embed the backlog.
                with self._lock:
                    vectors = self._cached_vectors(self._texts(candidates))
            else:
                vectors = self._ensure_vectors(self._texts(candidates))
            ready = [item for item in candidates if item.id in vectors]
            if not ready:
                return ()
            query_vector = self._query_vector(query)
            ranked = heapq.nsmallest(self._candidate_limit, ready, key=lambda item: (
                -self._cosine(query_vector, vectors[item.id]), item.id))
            return tuple(item.id for item in ranked)
        except Exception:
            self._fallback()
            return ()

    @staticmethod
    def _fallback() -> None:
        # No title, summary, query or provider exception text in trace.
        _LOGGER.warning("episode embedding failed; using lexical retrieval")
        record_active_trace("episode_semantic_fallback", "episode_retrieval",
                            outcome="failed", recoverable=True,
                            error_code="EPISODE_EMBEDDING_FAILED")

    def schedule_episode_ids(self, episode_ids: Sequence[str]) -> None:
        selected = tuple(dict.fromkeys(episode_ids))
        if selected:
            context = copy_context()
            self._worker.submit(context.run, self._warm, selected)

    def schedule_backfill(self) -> None:
        """Fill existing missing/stale vectors asynchronously, in bounded batches."""
        with self._lock:
            if self._backfilling.is_set():
                return
            self._backfilling.set()
        try:
            self._worker.submit(self._backfill)
        except Exception:
            self._backfilling.clear()
            raise

    def _backfill(self) -> None:
        try:
            self._cleanup()
            last_id = ""
            while True:
                with open_connection(self._episode_path) as connection:
                    rows = connection.execute(
                        "SELECT id,title,summary FROM career_episodes WHERE id > ? ORDER BY id LIMIT ?",
                        (last_id, self._batch_size)).fetchall()
                if not rows:
                    break
                last_id = rows[-1][0]
                texts = tuple(_EmbeddingText(id, f"{title}\n{summary}") for id, title, summary in rows)
                try:
                    self._ensure_vectors(texts)
                except Exception:
                    self._fallback()
        except Exception:
            self._fallback()
        finally:
            self._cleanup()
            self._backfilling.clear()

    def schedule_cleanup(self) -> None:
        self._worker.submit(self._cleanup)

    def _cleanup(self) -> None:
        try:
            self.prune_orphans()
        except Exception:
            self._fallback()

    def _warm(self, episode_ids) -> None:
        try:
            self._ensure_vectors(self._committed_texts(episode_ids))
        except Exception:
            self._fallback()
        finally:
            self._cleanup()

    def close(self) -> None:
        self._worker.shutdown(wait=True)


def optional_episode_semantic_index(*, episode_path: Path, cache_path: Path,
                                    environ=None) -> SQLiteCareerEpisodeSemanticIndex | None:
    config = CareerEmbeddingConfig.optional_from_env(environ=environ)
    if config is None:
        return None
    index = SQLiteCareerEpisodeSemanticIndex(episode_path=episode_path, cache_path=cache_path,
        client=OpenAICompatibleEmbeddingClient(config), batch_size=config.batch_size)
    index.schedule_backfill()  # restart repair and cold-cache fill, outside retrieval
    return index
