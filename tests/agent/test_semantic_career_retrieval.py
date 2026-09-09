from __future__ import annotations

from career_agent.agent.semantic_career_retrieval import (
    CareerEmbeddingConfig,
    OpenAICompatibleEmbeddingClient,
    SQLiteCareerEvidenceSemanticRetriever,
)
from career_agent.storage.career_history import CareerHistoryStore


class FakeEmbeddingClient:
    model_id = "fake-embedding-v1"

    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []

    def embed(self, texts):
        values = tuple(texts)
        self.calls.append(values)
        return tuple(self._vector(value) for value in values)

    @staticmethod
    def _vector(value: str):
        if "推荐" in value or "candidate" in value:
            return (1.0, 0.0)
        return (0.0, 1.0)


def _confirmed(store, *, record_id: str, claim: str):
    pending = store.create_evidence(
        user_id="u1",
        career_record_id=record_id,
        claim=claim,
        origin="user_input",
    )
    return store.confirm_evidence(
        user_id="u1",
        career_evidence_id=pending.id,
    )


def test_embedding_configuration_is_opt_in_and_fails_on_partial_values() -> None:
    assert CareerEmbeddingConfig.optional_from_env(environ={}) is None

    try:
        CareerEmbeddingConfig.optional_from_env(
            environ={"CAREER_EMBEDDING_MODEL": "embedding-model"}
        )
    except ValueError as error:
        assert "must be configured together" in str(error)
    else:
        raise AssertionError("partial embedding configuration must fail closed")


def test_semantic_retriever_ranks_by_cosine_and_reuses_derived_cache(
    tmp_path,
) -> None:
    store = CareerHistoryStore(tmp_path / "career.sqlite3")
    record = store.create_record(
        user_id="u1",
        record_type="project",
        title="Evidence",
    )
    recommendation = _confirmed(
        store,
        record_id=record.id,
        claim="负责推荐系统召回与排序",
    )
    payments = _confirmed(
        store,
        record_id=record.id,
        claim="维护支付清算服务",
    )
    client = FakeEmbeddingClient()
    cache_path = tmp_path / "embeddings.sqlite3"
    retriever = SQLiteCareerEvidenceSemanticRetriever(
        career_history=store,
        cache_path=cache_path,
        client=client,
        batch_size=8,
    )

    first = retriever.rank_current_evidence_ids(
        user_id="u1",
        query="candidate generation",
        limit=2,
    )
    second = retriever.rank_current_evidence_ids(
        user_id="u1",
        query="candidate generation",
        limit=2,
    )

    assert first == second == (recommendation.id, payments.id)
    assert client.calls[0] == (
        "负责推荐系统召回与排序",
        "维护支付清算服务",
    )
    assert client.calls[1:] == [
        ("candidate generation",),
        ("candidate generation",),
    ]


def test_semantic_cache_does_not_resurrect_superseded_evidence(tmp_path) -> None:
    store = CareerHistoryStore(tmp_path / "career.sqlite3")
    record = store.create_record(
        user_id="u1",
        record_type="project",
        title="Evidence",
    )
    original = _confirmed(
        store,
        record_id=record.id,
        claim="负责推荐系统召回",
    )
    client = FakeEmbeddingClient()
    retriever = SQLiteCareerEvidenceSemanticRetriever(
        career_history=store,
        cache_path=tmp_path / "embeddings.sqlite3",
        client=client,
    )
    assert original.id in retriever.rank_current_evidence_ids(
        user_id="u1", query="candidate generation", limit=5
    )

    corrected = store.correct_evidence(
        user_id="u1",
        career_evidence_id=original.id,
        new_claim="负责支付清算",
        reason="Correct the project",
    ).current
    ranked = retriever.rank_current_evidence_ids(
        user_id="u1", query="candidate generation", limit=5
    )

    assert original.id not in ranked
    assert ranked == (corrected.id,)


def test_revised_lineage_does_not_crowd_the_semantic_population(tmp_path) -> None:
    store = CareerHistoryStore(tmp_path / "career.sqlite3")
    record = store.create_record(
        user_id="u1",
        record_type="project",
        title="Evidence",
    )
    current = _confirmed(
        store,
        record_id=record.id,
        claim="负责推荐系统召回",
    )
    for claim in (
        "负责推荐系统召回与粗排",
        "负责推荐系统召回粗排与精排",
        "负责推荐系统召回粗排精排与重排",
        "负责推荐系统全链路排序",
        "负责推荐系统全链路排序与评估",
    ):
        current = store.correct_evidence(
            user_id="u1",
            career_evidence_id=current.id,
            new_claim=claim,
            reason="Tighten the claim",
        ).current

    client = FakeEmbeddingClient()
    ranked = SQLiteCareerEvidenceSemanticRetriever(
        career_history=store,
        cache_path=tmp_path / "embeddings.sqlite3",
        client=client,
    ).rank_current_evidence_ids(
        user_id="u1",
        query="candidate generation",
        limit=5,
    )

    assert ranked == (current.id,)
    assert client.calls[0] == (current.claim,)


def test_embedding_failure_degrades_to_an_empty_semantic_channel(tmp_path) -> None:
    store = CareerHistoryStore(tmp_path / "career.sqlite3")
    record = store.create_record(
        user_id="u1",
        record_type="project",
        title="Evidence",
    )
    _confirmed(
        store,
        record_id=record.id,
        claim="负责推荐系统召回与排序",
    )

    class FailingClient:
        model_id = "failing-embedding-v1"

        def embed(self, texts):
            raise TimeoutError("embedding provider unavailable")

    ranked = SQLiteCareerEvidenceSemanticRetriever(
        career_history=store,
        cache_path=tmp_path / "embeddings.sqlite3",
        client=FailingClient(),
    ).rank_current_evidence_ids(
        user_id="u1",
        query="candidate generation",
        limit=5,
    )

    assert ranked == ()


def test_tombstone_can_delete_derived_vectors(tmp_path) -> None:
    store = CareerHistoryStore(tmp_path / "career.sqlite3")
    record = store.create_record(
        user_id="u1",
        record_type="project",
        title="Evidence",
    )
    evidence = _confirmed(
        store,
        record_id=record.id,
        claim="负责推荐系统召回与排序",
    )
    cache_path = tmp_path / "embeddings.sqlite3"
    retriever = SQLiteCareerEvidenceSemanticRetriever(
        career_history=store,
        cache_path=cache_path,
        client=FakeEmbeddingClient(),
    )
    retriever.rank_current_evidence_ids(
        user_id="u1",
        query="candidate generation",
        limit=5,
    )
    store.tombstone_evidence(
        user_id="u1",
        career_evidence_id=evidence.id,
        reason="Remove this internship detail permanently.",
    )

    assert retriever.forget_evidence_ids((evidence.id,)) == 1
    assert retriever.forget_evidence_ids((evidence.id,)) == 0


def test_the_next_rank_sweeps_tombstoned_vectors(tmp_path) -> None:
    store = CareerHistoryStore(tmp_path / "career.sqlite3")
    record = store.create_record(
        user_id="u1",
        record_type="project",
        title="Evidence",
    )
    evidence = _confirmed(
        store,
        record_id=record.id,
        claim="负责推荐系统召回与排序",
    )
    retriever = SQLiteCareerEvidenceSemanticRetriever(
        career_history=store,
        cache_path=tmp_path / "embeddings.sqlite3",
        client=FakeEmbeddingClient(),
    )
    retriever.rank_current_evidence_ids(
        user_id="u1",
        query="candidate generation",
        limit=5,
    )
    store.tombstone_evidence(
        user_id="u1",
        career_evidence_id=evidence.id,
        reason="Remove this internship detail permanently.",
    )

    assert retriever.rank_current_evidence_ids(
        user_id="u1",
        query="candidate generation",
        limit=5,
    ) == ()
    assert retriever.forget_evidence_ids((evidence.id,)) == 0


def test_embedding_client_bounds_timeout_and_does_not_retry(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(
        "career_agent.agent.semantic_career_retrieval.OpenAI",
        FakeOpenAI,
    )
    OpenAICompatibleEmbeddingClient(
        CareerEmbeddingConfig(
            base_url="https://api.example.com/v1",
            api_key="test-key",
            model="embedding-model",
        )
    )

    assert captured["timeout"] == 30.0
    assert captured["max_retries"] == 0
