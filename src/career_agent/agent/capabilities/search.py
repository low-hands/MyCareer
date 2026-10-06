"""Deterministic capability discovery over catalogue metadata."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import re

from career_agent.agent.capabilities.catalog import CAPABILITIES, CapabilityDescriptor
from career_agent.agent.context.semantic_retrieval import EmbeddingClient


EXCLUDED = frozenset({"search_capabilities", "route_to_capability"})
MIN_SEMANTIC_SIMILARITY = 0.55
MAX_SEMANTIC_CANDIDATES = 10
COMMON_EXAMPLE_TERM_FRACTION = 0.15
COMMON_EXAMPLE_NAMESPACE_FRACTION = 0.60
_WORDS = re.compile(r"[a-zA-Z0-9]+|[\u3400-\u9fff]+")


def searchable_capabilities() -> tuple[CapabilityDescriptor, ...]:
    return tuple(
        descriptor for descriptor in CAPABILITIES.values()
        if descriptor.model_callable and descriptor.name not in EXCLUDED
    )


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
            result.append(word)
    return tuple(result)


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
    weighted: Counter[str] = Counter()
    for value, weight in (
        (descriptor.name, 5),
        (" ".join(descriptor.aliases_zh), 5),
        (descriptor.summary or "", 2),
        (_parameter_names(descriptor), 1),
    ):
        weighted.update({token: weight * count for token, count in Counter(_tokens(value)).items()})
    return weighted


def _common_example_terms(entries: Sequence[CapabilityDescriptor]) -> frozenset[str]:
    """Find frequent, widely dispersed CJK bigrams in query metadata."""
    if not entries:
        return frozenset()
    tool_frequency: Counter[str] = Counter()
    namespaces: dict[str, set[str]] = {}
    for item in entries:
        terms = {
            token
            for phrase in (*item.aliases_zh, *item.example_queries)
            for token in _tokens(phrase)
            if len(token) == 2 and "\u3400" <= token[0] <= "\u9fff"
        }
        tool_frequency.update(terms)
        for token in terms:
            namespaces.setdefault(token, set()).add(item.namespace or "")
    namespace_count = len({item.namespace for item in entries})
    return frozenset(
        token for token, count in tool_frequency.items()
        if count / len(entries) > COMMON_EXAMPLE_TERM_FRACTION
        and len(namespaces[token]) / namespace_count > COMMON_EXAMPLE_NAMESPACE_FRACTION
    )


def lexical_scores(
    query: str, descriptors: Sequence[CapabilityDescriptor] | None = None,
) -> dict[str, float]:
    """BM25 over weighted catalogue fields; zero-overlap entries are omitted."""
    entries = tuple(searchable_capabilities() if descriptors is None else descriptors)
    terms = set(_tokens(query))
    if not terms or not entries:
        return {}
    documents = {item.name: _document(item) for item in entries}
    lengths = {name: sum(tokens.values()) for name, tokens in documents.items()}
    average = sum(lengths.values()) / len(lengths)
    frequency = Counter(token for tokens in documents.values() for token in tokens)
    scores: dict[str, float] = {}
    for item in entries:
        tokens = documents[item.name]
        score = 0.0
        for term in terms:
            tf = tokens.get(term, 0)
            if not tf:
                continue
            idf = math.log(1 + (len(entries) - frequency[term] + 0.5) / (frequency[term] + 0.5))
            score += idf * (tf * 2.2) / (tf + 1.2 * (0.25 + 0.75 * lengths[item.name] / average))
        if score:
            scores[item.name] = score
    # Score each example as its own short document, then keep the strongest
    # match per tool. More examples expand vocabulary without increasing the
    # BM25 document length or adding repeated votes for the same tool.
    example_documents = {
        item.name: tuple(Counter(_tokens(query)) for query in item.example_queries)
        for item in entries
    }
    common_example_terms = _common_example_terms(entries)
    all_examples = [tokens for rows in example_documents.values() for tokens in rows]
    if all_examples:
        example_average = sum(sum(tokens.values()) for tokens in all_examples) / len(all_examples)
        example_frequency = Counter(
            token for rows in example_documents.values()
            for token in set().union(*(set(tokens) for tokens in rows))
        )
        for item in entries:
            best = 0.0
            for tokens in example_documents[item.name]:
                length = sum(tokens.values())
                example_score = 0.0
                # Frequent phrases spread across many namespaces do not count
                # as evidence, but still contribute their normal BM25 IDF.
                overlap = len(terms.intersection(tokens).difference(common_example_terms))
                required_coverage = 0.5 if item.effect == "WRITE" else 0.25
                if overlap < 2 or overlap / len(tokens) < required_coverage:
                    continue
                for term in terms:
                    tf = tokens.get(term, 0)
                    if not tf:
                        continue
                    idf = math.log(
                        1 + (len(entries) - example_frequency[term] + 0.5)
                        / (example_frequency[term] + 0.5)
                    )
                    example_score += idf * (tf * 2.2) / (
                        tf + 1.2 * (0.25 + 0.75 * length / example_average)
                    )
                best = max(best, example_score)
            if best:
                scores[item.name] = scores.get(item.name, 0.0) + 3 * best
    return scores


def _rank(scores: Mapping[str, float], order: Mapping[str, int]) -> tuple[str, ...]:
    return tuple(sorted(scores, key=lambda name: (-scores[name], order[name])))


def search_catalog(
    *, query: str | None = None, names: Sequence[str] | None = None,
    limit: int = 5, semantic_scores: Mapping[str, float] | None = None,
    descriptors: Sequence[CapabilityDescriptor] | None = None,
) -> tuple[str, ...]:
    """Return catalogue names in stable order; exact names/namespaces lead."""
    if (query is None) == (names is None):
        raise ValueError("provide exactly one of query or names")
    if query is not None and not 1 <= limit <= 10:
        raise ValueError("limit must be between 1 and 10")
    if names is not None and not 1 <= len(names) <= 10:
        raise ValueError("names must contain between 1 and 10 entries")
    if query is not None and not 1 <= len(query) <= 200:
        raise ValueError("query must contain between 1 and 200 characters")
    entries = tuple(searchable_capabilities() if descriptors is None else descriptors)
    order = {item.name: index for index, item in enumerate(entries)}
    namespaces = {item.namespace for item in entries}
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
        if item.name == normalized or item.namespace == normalized
    )
    lexical = _rank(lexical_scores(query, entries), order)
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


@dataclass
class SemanticCapabilityIndex:
    """Optional in-memory embedding cache keyed by model and catalogue content."""

    client: EmbeddingClient
    query_client: EmbeddingClient | None = None
    _key: str | None = None
    _vectors: tuple[tuple[float, ...], ...] = ()

    def _catalogue(self) -> tuple[tuple[CapabilityDescriptor, ...], tuple[str, ...], str]:
        entries = searchable_capabilities()
        texts = tuple(
            " ".join((item.name, item.namespace or "", item.summary or "", *item.aliases_zh))
            for item in entries
        )
        digest = hashlib.sha256(json.dumps(
            (self.client.model_id, texts), ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")).hexdigest()
        return entries, texts, digest

    def warm(self) -> None:
        """Build catalogue vectors before handling any search request."""
        entries, texts, digest = self._catalogue()
        if digest == self._key:
            return
        vectors = tuple(tuple(row) for row in self.client.embed(texts))
        if len(vectors) != len(entries):
            raise ValueError("embedding provider returned the wrong number of vectors")
        self._vectors = vectors
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
        for item, vector in zip(entries, self._vectors):
            norm = math.sqrt(sum(value * value for value in vector))
            if norm and len(vector) == len(needle):
                score = sum(left * right for left, right in zip(needle, vector)) / (needle_norm * norm)
                if score >= MIN_SEMANTIC_SIMILARITY:
                    scores[item.name] = score
        order = {item.name: index for index, item in enumerate(entries)}
        ranked = sorted(scores, key=lambda name: (-scores[name], order[name]))[
            :MAX_SEMANTIC_CANDIDATES
        ]
        return {name: scores[name] for name in ranked}
