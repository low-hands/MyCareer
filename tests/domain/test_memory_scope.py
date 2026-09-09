import pytest
from pydantic import ValidationError

from career_agent.domain.memory_scope import ScopeProposal
from career_agent.services.canonical_scope import CanonicalScopeResolver


def _proposal(**changes) -> ScopeProposal:
    values = {
        "user_id": "u1",
        "family": "person_intent",
        "subject_id": "self",
        "relation": "default_city",
        "proposed_value": "杭州",
    }
    values.update(changes)
    return ScopeProposal(**values)


def test_known_typed_intent_resolves_before_a_key_is_written() -> None:
    resolution = CanonicalScopeResolver().resolve(_proposal())

    assert resolution.resolved is True
    assert resolution.canonical_scope is not None
    assert resolution.canonical_scope.scope_key == (
        "person_intent/self/default_city"
    )


def test_free_text_evidence_does_not_get_a_claim_hash_disguised_as_a_key() -> None:
    """Evidence keys come from ``career_evidence_scope_key``, not this resolver.

    Rejecting the family outright is stronger than resolving it to an unresolved
    outcome: an empty relation registry would have made every evidence proposal
    unresolvable with no candidate to offer, which is a clarification that cannot
    be answered rather than a field that cannot be named.
    """

    with pytest.raises(ValidationError):
        _proposal(
            family="career_evidence",
            subject_id="career_record_abc",
            relation="我擅长端到端搭建 RAG",
            proposed_value="我擅长端到端搭建 RAG",
        )


def test_person_scope_cannot_smuggle_a_raw_subject_identifier() -> None:
    resolution = CanonicalScopeResolver().resolve(
        _proposal(subject_id="u1", relation="default_city")
    )

    assert resolution.canonical_scope is None
    assert "'self'" in resolution.reason
