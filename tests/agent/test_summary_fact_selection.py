import pytest

from career_agent.agent.context.summary_facts import (
    ARCHIVED_FACT_MAX_ITEMS,
    ARCHIVED_FACT_MAX_TOKENS,
    fact_id,
    select_worker_facts,
)
from career_agent.agent.contracts.memory import SummaryKeyFact
from career_agent.agent.providers.token_budget import serialized_token_count


def test_selection_caps_active_and_ranked_archive_without_clipping_facts():
    active = tuple(SummaryKeyFact(fact=f"活动{i}", source_sequence=i + 1) for i in range(15))
    archived = tuple(SummaryKeyFact(fact=f"美团 HR 联系人{i}", source_sequence=i + 1) for i in range(500))
    selected = select_worker_facts(active, archived, "美团 HR 联系人更正")
    assert selected[:12] == active[-12:]
    assert len(selected) <= 12 + ARCHIVED_FACT_MAX_ITEMS
    recalled = selected[12:]
    assert recalled
    assert all(fact in archived for fact in recalled)
    assert serialized_token_count([dict(f.model_dump(), fact_id=fact_id(f)) for f in recalled]) <= ARCHIVED_FACT_MAX_TOKENS
    assert selected == select_worker_facts(active, tuple(reversed(archived)), "美团 HR 联系人更正")


def test_unrelated_archive_and_zero_request_headroom_add_nothing():
    active = (SummaryKeyFact(fact="当前状态", source_sequence=2),)
    archived = (SummaryKeyFact(fact="美团 HR 叫林晓雯", source_sequence=1),)
    assert select_worker_facts(active, archived, "天气预报") == active
    assert select_worker_facts(active, archived, "美团 HR 更正", fits=lambda facts: False) == active
    # CJK entity overlap still recalls a fact whose old value changed entirely.
    assert select_worker_facts(active, archived, "美团联系人改成张敏了") == (*active, *archived)


@pytest.mark.parametrize("ledger_size", [500, 2000])
@pytest.mark.parametrize("capacity", [0, 2, 24])
def test_tight_request_headroom_bounds_full_request_checks(ledger_size, capacity):
    active = (SummaryKeyFact(fact="当前状态", source_sequence=1),)
    ledger = tuple(
        SummaryKeyFact(fact=f"美团 HR 联系人{i}", source_sequence=i + 2)
        for i in range(ledger_size)
    )
    checks = 0

    def fits(facts):
        nonlocal checks
        checks += 1
        return len(facts) <= len(active) + capacity

    selected = select_worker_facts(active, ledger, "美团 HR 联系人更正", fits=fits)
    assert len(selected) == len(active) + capacity
    assert checks == min(capacity + 1, ARCHIVED_FACT_MAX_ITEMS)
