"""Deterministic capability discovery over catalogue metadata."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol
import hashlib
import json
import math
import re

from career_agent.agent.capabilities.catalog import CAPABILITIES, CapabilityDescriptor
from career_agent.agent.context.semantic_retrieval import EmbeddingClient, MAX_EMBEDDING_BATCH_SIZE


EXCLUDED = frozenset({"search_capabilities"})
MIN_SEMANTIC_SIMILARITY = 0.55
MAX_SEMANTIC_CANDIDATES = 10
_WORDS = re.compile(r"[a-zA-Z0-9]+|[\u3400-\u9fff]+")
# Stop words for unsolicited intent ranking only; explicit model searches keep
# every term. English is Lucene's default English stop set. Chinese text is
# split into bigrams, so cross-word fragments such as \u6211\u770b cannot be listed;
# instead a CJK token is dropped when every character is a function character
# (pronouns, particles, demonstratives, interrogatives and light request verbs
# from common Chinese stop-word lists). \u67e5\u770b, \u4e0b\u8f7d and \u8981\u6c42 keep a content
# character and survive.
ENGLISH_STOP_WORDS = frozenset((
    "a an and are as at be but by for if in into is it no not of on or such "
    "that the their then there these they this to was will with"
).split())
CJK_FUNCTION_CHARACTERS = frozenset(
    "\u6211\u4f60\u60a8\u4ed6\u5979\u5b83\u4eec\u7684\u5730\u5f97\u4e86\u7740\u8fc7\u5417\u5462\u5427\u554a\u5440\u54e6\u561b\u4e48\u8fd9\u90a3\u54ea\u4e9b\u4e2a\u4e00\u4e0b\u5e2e\u8bf7\u7ed9\u8ba9\u628a\u88ab"
    "\u662f\u6709\u5728\u548c\u4e0e\u53ca\u6216\u5c31\u90fd\u4e5f\u8fd8\u53c8\u518d\u5f88\u592a\u8981\u60f3\u80fd\u4f1a\u53ef\u4ee5\u770b\u600e\u6837\u4ec0\u5982\u4f55\u4e3a\u5565"
)


def searchable_capabilities() -> tuple[CapabilityDescriptor, ...]:
    return _DEFAULT_INDEX.entries


def _tokens(text: str) -> tuple[str, ...]:
    result: list[str] = []
    for match in _WORDS.finditer(text.lower().replace("_", " ")):
        word = match.group()
        if "\u3400" <= word[0] <= "\u9fff":
            if len(word) == 1:
                result.append(word)
            else:
                result.extend(word[index:index + 2] for index in range(len(word) - 1))
        else:
            # Conservative regular plurals; retain singular endings such as
            # status, analysis and process. Both queries and metadata use this.
            if (word.isalpha() and len(word) > 3 and word.endswith("s")
                    and not word.endswith(("ss", "us", "is"))):
                if word.endswith(("sses", "shes", "ches", "xes", "zes")):
                    word = word[:-2]
                else:
                    word = word[:-1]
            result.append(word)
    return tuple(result)


def _is_stop_word(token: str) -> bool:
    if "\u3400" <= token[0] <= "\u9fff":
        return all(character in CJK_FUNCTION_CHARACTERS for character in token)
    return token in ENGLISH_STOP_WORDS


def _parameter_names(descriptor: CapabilityDescriptor) -> str:
    names: list[str] = []

    def visit(node: object) -> None:
        if isinstance(node, dict):
            properties = node.get("properties")
            if isinstance(properties, dict):
                names.extend(str(name) for name in properties)
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)

    visit(descriptor.tool_schema()["function"]["parameters"])
    return " ".join(names)


def _document(descriptor: CapabilityDescriptor) -> Counter[str]:
    """Field-weighted BM25 document.

    Example queries are appended as ordinary text before indexing, as in
    doc2query document expansion, rather than scored as separate documents.
    """
    weighted: Counter[str] = Counter()
    for value, weight in (
        (descriptor.name, 5),
        (" ".join(descriptor.aliases_zh), 5),
        (descriptor.summary or "", 2),
        (_parameter_names(descriptor), 1),
        (" ".join(descriptor.example_queries), 1),
    ):
        weighted.update({token: weight * count for token, count in Counter(_tokens(value)).items()})
    return weighted


@dataclass(frozen=True)
class _CatalogIndex:
    entries: tuple[CapabilityDescriptor, ...]
    order: Mapping[str, int]
    namespaces: frozenset[str | None]
    documents: Mapping[str, Counter[str]]
    lengths: Mapping[str, int]
    average: float
    idf: Mapping[str, float]
    exact_queries: Mapping[str, frozenset[str | None]]
    embedding_texts: tuple[str, ...]


def _build_index(entries: tuple[CapabilityDescriptor, ...]) -> _CatalogIndex:
    documents = {item.name: _document(item) for item in entries}
    lengths = {name: sum(tokens.values()) for name, tokens in documents.items()}
    frequency = Counter(token for tokens in documents.values() for token in tokens)
    return _CatalogIndex(
        entries=entries,
        order={item.name: index for index, item in enumerate(entries)},
        namespaces=frozenset(item.namespace for item in entries),
        documents=documents, lengths=lengths,
        average=sum(lengths.values()) / len(lengths) if lengths else 0.0,
        idf={term: math.log(1 + (len(entries) - count + 0.5) / (count + 0.5))
             for term, count in frequency.items()},
        exact_queries={item.name: frozenset((
            item.name, item.namespace, *(alias.strip().lower() for alias in item.aliases_zh),
        )) for item in entries},
        embedding_texts=capability_embedding_texts(entries),
    )


def lexical_scores(
    query: str, descriptors: Sequence[CapabilityDescriptor] | None = None,
) -> dict[str, float]:
    """BM25 over weighted catalogue fields; zero-overlap entries are omitted.

    Every tool with a positive score is a lexical candidate. There is no
    separate evidence gate: ranking and the top-k cut decide, as in plain
    BM25 tool search, and write safety belongs to execution approval.
    """
    index = _DEFAULT_INDEX if descriptors is None else _build_index(tuple(descriptors))
    return _lexical_scores(query, index)


def _lexical_scores(
    query: str, index: _CatalogIndex, *, drop_stop_words: bool = False,
) -> dict[str, float]:
    terms = set(_tokens(query))
    if drop_stop_words:
        terms = {term for term in terms if not _is_stop_word(term)}
    if not terms or not index.entries:
        return {}
    scores: dict[str, float] = {}
    for item in index.entries:
        tokens = index.documents[item.name]
        score = 0.0
        for term in terms:
            tf = tokens.get(term, 0)
            if not tf:
                continue
            score += index.idf[term] * (tf * 2.2) / (
                tf + 1.2 * (0.25 + 0.75 * index.lengths[item.name] / index.average)
            )
        if score:
            scores[item.name] = score
    return scores


def _rank(scores: Mapping[str, float], order: Mapping[str, int]) -> tuple[str, ...]:
    return tuple(sorted(scores, key=lambda name: (-scores[name], order[name])))


def search_catalog(
    *, query: str | None = None, names: Sequence[str] | None = None,
    limit: int = 5, semantic_scores: Mapping[str, float] | None = None,
    descriptors: Sequence[CapabilityDescriptor] | None = None,
    drop_stop_words: bool = False,
) -> tuple[str, ...]:
    """Return catalogue names in stable order; exact names/namespaces lead.

    ``drop_stop_words`` is for intent ranking that nobody requested: a filler
    sentence must not offer tools. Model-written searches keep every term.
    """
    if (query is None) == (names is None):
        raise ValueError("provide exactly one of query or names")
    if query is not None and not 1 <= limit <= 10:
        raise ValueError("limit must be between 1 and 10")
    if names is not None and not 1 <= len(names) <= 10:
        raise ValueError("names must contain between 1 and 10 entries")
    if query is not None and not 1 <= len(query) <= 200:
        raise ValueError("query must contain between 1 and 200 characters")
    index = _DEFAULT_INDEX if descriptors is None else _build_index(tuple(descriptors))
    entries, order, namespaces = index.entries, index.order, index.namespaces
    if names is not None:
        unknown = [name for name in names if name not in order and name not in namespaces]
        if unknown:
            raise ValueError(f"unknown capability or namespace: {unknown[0]}")
        selected = {
            item.name for item in entries
            if item.name in names or item.namespace in names
        }
        return tuple(name for name in order if name in selected)
    assert query is not None
    normalized = query.strip().lower()
    exact = tuple(
        item.name for item in entries
        if normalized in index.exact_queries[item.name]
    )
    lexical = _rank(_lexical_scores(query, index, drop_stop_words=drop_stop_words), order)
    semantic = _rank(
        {
            name: score for name, score in (semantic_scores or {}).items()
            if name in order and score >= MIN_SEMANTIC_SIMILARITY
        },
        order,
    )[:MAX_SEMANTIC_CANDIDATES]
    combined: dict[str, float] = {}
    for ranking in (lexical, semantic):
        for position, name in enumerate(ranking, start=1):
            combined[name] = combined.get(name, 0.0) + 1 / (60 + position)
    ranked = sorted(combined, key=lambda name: (-combined[name], order[name]))
    return tuple(dict.fromkeys((*exact, *ranked)))[:limit]


def capability_embedding_texts(
    entries: Sequence[CapabilityDescriptor],
) -> tuple[str, ...]:
    return tuple(
        " ".join((item.name, item.namespace or "", item.summary or "", *item.aliases_zh))
        for item in entries
    )


# Catalogue descriptors are fixed after import. Custom descriptor sequences
# get an independent index per call, so test and caller-specific catalogues
# cannot contaminate the default index.
_DEFAULT_INDEX = _build_index(tuple(
    descriptor for descriptor in CAPABILITIES.values()
    if descriptor.model_callable and descriptor.name not in EXCLUDED
))


class CapabilityScorer(Protocol):
    def scores(self, query: str) -> Mapping[str, float]: ...


@dataclass
class SemanticCapabilityIndex:
    """Optional in-memory embedding cache keyed by model and catalogue content."""

    client: EmbeddingClient
    query_client: EmbeddingClient | None = None
    _key: str | None = None
    _vectors: tuple[tuple[float, ...], ...] = ()
    _vector_norms: tuple[float, ...] = ()
    _catalogue_model: str | None = None
    _catalogue_digest: str | None = None

    def _catalogue(self) -> tuple[tuple[CapabilityDescriptor, ...], tuple[str, ...], str]:
        if self._catalogue_model != self.client.model_id or self._catalogue_digest is None:
            self._catalogue_digest = hashlib.sha256(json.dumps(
                (self.client.model_id, _DEFAULT_INDEX.embedding_texts),
                ensure_ascii=False, separators=(",", ":")
            ).encode("utf-8")).hexdigest()
            self._catalogue_model = self.client.model_id
        return _DEFAULT_INDEX.entries, _DEFAULT_INDEX.embedding_texts, self._catalogue_digest

    def warm(self) -> None:
        """Build catalogue vectors before handling any search request."""
        entries, texts, digest = self._catalogue()
        if digest == self._key:
            return
        vectors = tuple(
            tuple(row)
            for start in range(0, len(texts), MAX_EMBEDDING_BATCH_SIZE)
            for row in self.client.embed(texts[start:start + MAX_EMBEDDING_BATCH_SIZE])
        )
        if len(vectors) != len(entries):
            raise ValueError("embedding provider returned the wrong number of vectors")
        self._vectors = vectors
        self._vector_norms = tuple(math.sqrt(sum(value * value for value in vector)) for vector in vectors)
        self._key = digest

    def scores(self, query: str) -> dict[str, float]:
        entries, _, digest = self._catalogue()
        if digest != self._key:
            return {}
        query_vectors = (self.query_client or self.client).embed((query,))
        if len(query_vectors) != 1:
            raise ValueError("embedding provider returned no query vector")
        needle = tuple(query_vectors[0])
        needle_norm = math.sqrt(sum(value * value for value in needle))
        if not needle_norm:
            return {}
        scores: dict[str, float] = {}
        for item, vector, norm in zip(entries, self._vectors, self._vector_norms):
            if norm and len(vector) == len(needle):
                score = sum(left * right for left, right in zip(needle, vector)) / (needle_norm * norm)
                if score >= MIN_SEMANTIC_SIMILARITY:
                    scores[item.name] = score
        order = _DEFAULT_INDEX.order
        ranked = sorted(scores, key=lambda name: (-scores[name], order[name]))[
            :MAX_SEMANTIC_CANDIDATES
        ]
        return {name: scores[name] for name in ranked}
