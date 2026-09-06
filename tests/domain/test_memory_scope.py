from career_agent.domain.memory_scope import ScopeProposal
from career_agent.services.canonical_scope import CanonicalScopeResolver


def _proposal(**changes) -> ScopeProposal:
    values = {
        "user_id": "u1",
        "family": "person_intent",
        "subject_id": "self",
        "relation": "default_city",
        "proposed_value": "杭州",
        "source_kind": "job_intent",
        "source_id": "self:default_city",
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
    resolution = CanonicalScopeResolver().resolve(
        _proposal(
            family="career_evidence",
            subject_id="career_record_abc",
            relation="我擅长端到端搭建 RAG",
            proposed_value="我擅长端到端搭建 RAG",
            source_kind="career_evidence",
            source_id="career_evidence_abc",
        )
    )

    assert resolution.resolved is False
    assert resolution.canonical_scope is None
    assert resolution.reason


def test_semantic_normalizer_can_only_select_from_the_closed_registry() -> None:
    class Normalizer:
        def __init__(self, answer: str) -> None:
            self.answer = answer

        def normalize(self, **_kwargs) -> str:
            return self.answer

    proposal = _proposal(
        family="career_evidence",
        subject_id="career_record_abc",
        relation="主要工作地点",
        proposed_value="杭州",
        source_kind="career_evidence",
        source_id="career_evidence_abc",
    )
    accepted = CanonicalScopeResolver(
        evidence_relations=frozenset({"work_city"}),
        semantic_normalizer=Normalizer("work_city"),
    ).resolve(proposal)
    rejected = CanonicalScopeResolver(
        evidence_relations=frozenset({"work_city"}),
        semantic_normalizer=Normalizer("model_invented_key"),
    ).resolve(proposal)

    assert accepted.canonical_scope is not None
    assert accepted.canonical_scope.scope_key.endswith("/work_city")
    assert rejected.canonical_scope is None


def test_person_scope_cannot_smuggle_a_raw_subject_identifier() -> None:
    resolution = CanonicalScopeResolver().resolve(
        _proposal(subject_id="u1", relation="default_city")
    )

    assert resolution.canonical_scope is None
    assert "'self'" in resolution.reason
