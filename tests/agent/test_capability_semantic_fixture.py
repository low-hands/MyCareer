"""Recorded capability scores keep evaluation offline and in step with the catalogue."""

from __future__ import annotations

import json

import pytest

from career_agent.agent.capabilities.search import (
    MIN_SEMANTIC_SIMILARITY, searchable_capabilities,
)
from career_agent.evaluation import capability_semantic_fixture as fixture


def test_recorded_scores_cover_the_catalogue_and_nine_paired_scenarios() -> None:
    recorded = fixture.recorded_capability_scores()
    assert recorded is not None, (
        "capability catalogue changed; run "
        "`uv run python -m career_agent.evaluation.capability_semantic_fixture`"
    )
    from career_agent.evaluation.search_scenarios import SEARCH_SCENARIOS
    for query in {
        message[:200]
        for scenario in SEARCH_SCENARIOS
        if scenario.name in {
            "a_core_request_routes_before_job_analysis",
            "a_questionnaire_answer_is_proposed_with_user_input_provenance",
            "research_is_not_started_as_part_of_matching",
            "a_repeated_call_is_not_reissued_after_an_observation",
            "a_missing_city_is_asked_for_not_guessed",
            "a_calendar_write_is_never_executed_in_the_turn_that_prepared_it",
            "planning_to_apply_does_not_create_an_application",
            "interview_completion_is_not_inferred_from_the_clock",
            "an_unseen_result_is_delivered_rather_than_characterized",
        }
        for message in (scenario.context.user_message, *(step.user_message for step in scenario.steps))
        if message
    }:
        recorded.scores(query)


def test_an_unrecorded_query_is_an_error_not_a_network_call() -> None:
    recorded = fixture.RecordedCapabilityScores({"known": {"list_resumes": 0.8}})
    assert recorded.scores("known") == {"list_resumes": 0.8}
    with pytest.raises(KeyError, match="refresh"):
        recorded.scores("unknown")


def test_replay_uses_recorded_scores_without_embedding_requests(monkeypatch) -> None:
    from career_agent.agent.context.semantic_retrieval import OpenAICompatibleEmbeddingClient
    from career_agent.evaluation.search_scenarios import SEARCH_SCENARIOS
    from career_agent.evaluation.search_trajectory import replay_search_sample
    from career_agent.evaluation.trajectory import load_cassette, trajectory_tool_specs

    def fail_if_called(*args, **kwargs):
        raise AssertionError("replay attempted an embedding request")

    monkeypatch.setattr(OpenAICompatibleEmbeddingClient, "embed", fail_if_called)
    scenario = next(item for item in SEARCH_SCENARIOS if item.name == "a_core_request_routes_before_job_analysis")
    cassette = load_cassette(scenario.name)
    assert cassette is not None
    replay_search_sample(
        scenario, tool_specs=trajectory_tool_specs(), responses=cassette.recordings[0],
    )


def test_recording_scores_batches_catalogue_embeddings(tmp_path, monkeypatch) -> None:
    names = [item.name for item in searchable_capabilities()]

    class Client:
        model_id = "fake-embedding"

        def __init__(self) -> None:
            self.calls: list[tuple[str, ...]] = []

        def embed(self, texts):
            self.calls.append(tuple(texts))
            # Every text points the same way, so every capability scores 1.0.
            return [(1.0, 0.0) for _ in texts]

    client = Client()
    path = tmp_path / "scores.json"
    monkeypatch.setattr(fixture, "SEMANTIC_FIXTURE_PATH", path)
    fixture.record_capability_scores(
        client, model="fake-model", queries=("简历", "岗位"), path=path,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["model"] == "fake-model"
    assert payload["catalogue_digest"] == fixture.catalogue_digest()
    assert set(payload["scores"]) == {"简历", "岗位"}
    assert all(
        set(scores) <= set(names) and min(scores.values()) >= MIN_SEMANTIC_SIMILARITY
        for scores in payload["scores"].values()
    )
    # One pass for the queries, one for the catalogue; no call per query.
    assert [len(call) for call in client.calls] == [2, 20, 20, 20, len(names) - 60]
    fixture.recorded_capability_scores.cache_clear()
