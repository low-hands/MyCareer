"""Deterministic identities and merge for the bounded fact projection."""
from __future__ import annotations
import hashlib
import re
import unicodedata
from collections.abc import Callable, Sequence
from career_agent.agent.contracts.memory import SummaryKeyFact


def fact_id(fact: SummaryKeyFact) -> str:
    return 'fact_' + hashlib.sha256(f'{fact.source_sequence}\0{fact.fact}'.encode()).hexdigest()[:32]


def merge_facts(previous: Sequence[SummaryKeyFact], additions: Sequence[SummaryKeyFact], superseded: Sequence[str] = ()) -> tuple[SummaryKeyFact, ...]:
    known = {fact_id(fact): fact for fact in previous}
    if any(identity not in known for identity in superseded):
        raise ValueError('Cannot supersede an undisclosed fact')
    for identity in superseded:
        known.pop(identity)
    for fact in additions:
        known[fact_id(fact)] = fact
    return tuple(sorted(known.values(), key=lambda fact: (fact.source_sequence, fact_id(fact))))


# Keep retrieval independent from ledger size and leave room for source messages.
ARCHIVED_FACT_MAX_ITEMS = 24
ARCHIVED_FACT_MAX_TOKENS = 2048


def _terms(text: str) -> set[str]:
    text = unicodedata.normalize("NFKC", text).casefold()
    terms = set(re.findall(r"[a-z0-9_]+", text))
    for run in re.findall(r"[\u4e00-\u9fff]+", text):
        terms.update(run[i:i + 2] for i in range(len(run) - 1))
    return terms


def select_worker_facts(
    active: Sequence[SummaryKeyFact],
    ledger: Sequence[SummaryKeyFact],
    message_text: str,
    *,
    fits: Callable[[tuple[SummaryKeyFact, ...]], bool] | None = None,
) -> tuple[SummaryKeyFact, ...]:
    """Whole relevant archive rows, bounded separately and by request headroom.

    Active facts have priority. Sources and facts are never truncated to fit.
    Stop at the first request-budget refusal, even if a later fact is shorter:
    each fit check serializes and counts the complete request.
    """
    from career_agent.agent.providers.token_budget import serialized_token_count

    selected = tuple(active[-12:])
    active_ids = {fact_id(fact) for fact in selected}
    query = _terms(message_text)
    ranked = []
    for fact in ledger:
        if fact_id(fact) in active_ids:
            continue
        overlap = len(query & _terms(fact.fact))
        if overlap:
            ranked.append((overlap, fact))
    ranked.sort(key=lambda item: (-item[0], -item[1].source_sequence, fact_id(item[1])))
    recalled: list[SummaryKeyFact] = []
    for _, fact in ranked:
        if len(recalled) >= ARCHIVED_FACT_MAX_ITEMS:
            break
        candidate = (*recalled, fact)
        wire = [dict(item.model_dump(), fact_id=fact_id(item)) for item in candidate]
        if serialized_token_count(wire) > ARCHIVED_FACT_MAX_TOKENS:
            continue
        if fits is not None and not fits((*selected, *candidate)):
            break
        recalled.append(fact)
    return (*selected, *recalled)
