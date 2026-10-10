"""Deterministic identities and merge for the bounded fact projection."""
from __future__ import annotations
import hashlib
from collections.abc import Sequence
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
