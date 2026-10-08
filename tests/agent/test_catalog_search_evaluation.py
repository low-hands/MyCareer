"""Model-query contracts, separate from user-sentence trajectory evaluation.

Known baseline regressions are strict xfails, so fixing a contract requires
removing its marker. No test executes the frozen holdout queries.
"""
import hashlib
import json

import pytest

from career_agent.agent.capabilities.catalog import CAPABILITIES
from career_agent.evaluation.catalog_search import FIXTURE, evaluate_queries

FIXTURE_SHA256 = "dca9a3dae15fdcdefa1df50d1d4e38052ab0072ff1c02cfebf0b00d60f85a621"
CASES = json.loads(FIXTURE.read_text(encoding="utf-8"))["cases"]
KNOWN_FAILURES = {"dev_view_resumes", "dev_control_experience"}


def test_fixture_is_frozen_and_splits_are_disjoint():
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == FIXTURE_SHA256
    assert len({case["id"] for case in CASES}) == len(CASES)
    dev = [case for case in CASES if case["split"] == "dev"]
    holdout = [case for case in CASES if case["split"] == "holdout"]
    assert (len(dev), len(holdout)) == (15, 11)
    assert {c["query"] for c in dev}.isdisjoint(c["query"] for c in holdout)
    for case in CASES:
        assert case["required_k"] == 5
        assert set(case["expected_tools"]) <= CAPABILITIES.keys()
        assert "forbidden_effects" not in case
        assert "forbid_action_tools" not in case
        assert "forbidden_name_prefixes" not in case


def test_each_split_has_non_alias_write_demands_and_read_recall():
    for split in ("dev", "holdout"):
        cases = [c for c in CASES if c["split"] == split]
        writes = [c for c in cases if c.get("non_exact_write")]
        assert len(writes) == 3
        assert {c["expected_tools"][0] for c in writes} == {
            "create_application", "draft_resume_tailoring", "export_resume_artifact",
        }
        for case in writes:
            descriptor = CAPABILITIES[case["expected_tools"][0]]
            assert descriptor.effect == "WRITE"
            assert case["query"].strip().lower() not in {
                value.strip().lower() for value in (
                    descriptor.name, descriptor.namespace, *descriptor.aliases_zh,
                )
            }
        assert sum(bool(c["expected_tools"]) and all(
            CAPABILITIES[name].effect == "READ" for name in c["expected_tools"]
        ) for c in cases) >= 6
    by_id = {c["id"]: c for c in CASES}
    assert by_id["dev_control_interview"]["expected_tools"] == ["list_interviews"]
    assert by_id["dev_control_application"]["expected_tools"] == ["list_applications"]
    assert by_id["holdout_control"]["expected_tools"] == ["search_career_memory"]
    assert by_id["dev_control_experience"]["expected_tools"] == by_id["holdout_control"]["expected_tools"]


@pytest.mark.parametrize("case", [
    pytest.param(case, id=case["id"], marks=(
        pytest.mark.xfail(strict=True, reason="Remaining lexical recall regressions: resume ranking and experience rank 6")
        if case["id"] in KNOWN_FAILURES else ()
    ))
    for case in CASES if case["split"] == "dev"
])
def test_model_query_contract(case):
    row = evaluate_queries([case])["rows"][0]
    assert not row["missing"], row
    assert row["passed"] is (True if case["expected_tools"] else None), row


@pytest.mark.parametrize("case", [
    case for case in CASES if case.get("source") == "64aea67"
], ids=lambda case: case["id"])
def test_controls_with_semantic_action_pressure_only_check_recall(case):
    from career_agent.agent.capabilities.search import search_catalog

    result = evaluate_queries([case], search=lambda **kwargs: search_catalog(
        **kwargs, semantic_scores={"propose_memory_tombstone": 1.0,
                                  "confirm_career_fact": 1.0, "create_application": 1.0},
    ))
    row = result["rows"][0]
    assert row["passed"] is (not row["missing"] if case["expected_tools"] else None)
    assert result["action_offer_count"] == len(row["action_offered"])


@pytest.mark.parametrize("status", ["", " M src/example.py\n", "?? untracked.txt\n"])
def test_workspace_snapshot_clean_gate(monkeypatch, status):
    from career_agent.evaluation import catalog_search

    def git_output(command, **kwargs):
        return status if command[1] == "status" else "example-head\n"

    monkeypatch.setattr(catalog_search.subprocess, "check_output", git_output)
    snapshot = catalog_search.workspace_snapshot()
    assert snapshot["worktree_clean"] == (not status)
    assert snapshot["git_status_porcelain"] == status
    assert snapshot["source_sha256"][str(FIXTURE.relative_to(FIXTURE.parents[2]))] == FIXTURE_SHA256
    if status:
        with pytest.raises(ValueError, match="clean git worktree"):
            catalog_search.workspace_snapshot(require_clean=True)
    else:
        assert catalog_search.workspace_snapshot(require_clean=True)["worktree_clean"]


def test_dirty_final_holdout_is_rejected_before_scoring_or_output(monkeypatch, tmp_path):
    from career_agent.evaluation import catalog_search

    def dirty_snapshot(**kwargs):
        assert kwargs["require_clean"]
        raise ValueError("Final holdout requires a clean git worktree")

    def forbidden_scoring(*args, **kwargs):
        pytest.fail("Holdout was scored despite dirty workspace")

    output = tmp_path / "holdout.json"
    monkeypatch.setattr("sys.argv", ["catalog_search", "--final-holdout", "--output", str(output)])
    monkeypatch.setattr(catalog_search, "workspace_snapshot", dirty_snapshot)
    monkeypatch.setattr(catalog_search, "evaluate_queries", forbidden_scoring)
    with pytest.raises(SystemExit) as error:
        catalog_search.main()
    assert error.value.code == 2
    assert not output.exists()


def test_evaluator_counts_write_exposure_without_failing_read_recall():
    case = next(c for c in CASES if c["id"] == "dev_resume_zh")
    result = evaluate_queries([case], search=lambda **_: (
        "export_resume_artifact", "list_resumes",
    ))
    assert [result["recall"][str(k)]["hits"] for k in (1, 3, 5)] == [0, 1, 1]
    assert result["write_offer_count"] == 1
    assert result["failed_case_count"] == 0
    assert result["rows"][0]["missing"] == []
    assert result["rows"][0]["write_offered"] == ["export_resume_artifact"]


def test_empty_results_cannot_pass_read_or_non_alias_write_demands():
    result = evaluate_queries([c for c in CASES if c["split"] == "dev"],
                              search=lambda **_: ())
    assert result["demand_count"] == 14
    assert result["failed_case_count"] == 14
    assert all(row["missing"] for row in result["rows"] if row["expected_tools"])
    assert result["action_offer_count"] == 0


def test_non_target_writes_are_counted_without_failing_write_request():
    case = next(c for c in CASES if c["id"] == "dev_application_write")
    result = evaluate_queries([case], search=lambda **_: (
        "list_applications", "create_application", "create_interview",
        "update_application_status", "propose_memory_amendment",
    ))
    assert result["write_query_count"] == 1
    assert result["non_target_write_offer_count"] == 2
    assert result["rows"][0]["non_target_write_offered"] == [
        "create_interview", "update_application_status",
    ]
    assert result["rows"][0]["passed"]
    assert result["failed_case_count"] == 0


def test_read_queries_count_writes_and_read_proposals_as_observations():
    case = next(c for c in CASES if c["id"] == "dev_control_experience")
    result = evaluate_queries([case], search=lambda **_: (
        "search_career_memory", "propose_memory_amendment", "create_application",
    ))
    assert result["write_offer_count"] == result["propose_offer_count"] == 1
    assert result["action_offer_count"] == 2
    assert result["non_write_query_action_offer_count"] == 2
    assert result["failed_case_count"] == 0
    assert result["write_query_count"] == result["non_target_write_offer_count"] == 0


def test_action_exposure_union_does_not_double_count_write_proposals():
    name = next(name for name, d in CAPABILITIES.items()
                if name.startswith("propose_") and d.effect == "WRITE")
    case = next(c for c in CASES if c["id"] == "dev_resume_zh")
    result = evaluate_queries([case], search=lambda **_: ("list_resumes", name))
    assert result["write_offer_count"] == result["propose_offer_count"] == result["action_offer_count"] == 1
    assert result["failed_case_count"] == 0


def test_policy_amendment_preserves_all_query_targets_and_splits():
    prior = json.loads(FIXTURE.with_name("model_queries_v3_recall_and_effects.json").read_text())
    fields = ("id", "query", "split", "expected_tools", "required_k")
    assert [{k: c[k] for k in fields} for c in prior["cases"]] == [
        {k: c[k] for k in fields} for c in CASES
    ]


@pytest.mark.parametrize("case", [c for c in CASES if c["split"] == "dev"], ids=lambda c: c["id"])
def test_recorded_model_query_hybrid_dev_recall(case):
    from career_agent.agent.capabilities.search import search_catalog
    from career_agent.evaluation.catalog_search import MODEL_QUERY_SCORES, load_model_query_scores

    scorer = load_model_query_scores(MODEL_QUERY_SCORES)
    result = evaluate_queries([case], search=lambda **kwargs: search_catalog(
        **kwargs, semantic_scores=scorer.scores(kwargs["query"]),
    ))
    assert not result["rows"][0]["missing"], result["rows"][0]


def test_hybrid_dev_meets_step1_recall_floor():
    from career_agent.agent.capabilities.search import search_catalog
    from career_agent.evaluation.catalog_search import MODEL_QUERY_SCORES, load_model_query_scores

    scorer = load_model_query_scores(MODEL_QUERY_SCORES)
    result = evaluate_queries([c for c in CASES if c["split"] == "dev"],
                              search=lambda **kwargs: search_catalog(
                                  **kwargs, semantic_scores=scorer.scores(kwargs["query"]),
                              ))
    assert result["demand_count"] == 14
    assert result["recall"]["5"]["hits"] >= 12


@pytest.mark.parametrize("mismatch", ["catalogue_digest", "fixture_sha256", "missing_query"])
def test_recorded_scores_reject_stale_or_incomplete_provenance(tmp_path, mismatch):
    from career_agent.evaluation.catalog_search import MODEL_QUERY_SCORES, load_model_query_scores

    payload = json.loads(MODEL_QUERY_SCORES.read_text())
    if mismatch == "missing_query":
        payload["scores"].pop("resume")
    else:
        payload[mismatch] = "stale"
    path = tmp_path / "scores.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="changed|26 frozen queries"):
        load_model_query_scores(path)


def test_plural_normalization_preserves_prior_dev_recall():
    from career_agent.agent.capabilities.search import search_catalog

    baseline = json.loads(FIXTURE.with_name("uniform_gate_dev_2026-10-08.json").read_text())
    # Both saved suites are dev. Never execute a holdout query for this check.
    for suite in (baseline["model_queries"], baseline["existing_dev"]["user_sentence_search_catalog"]):
        for row in suite["rows"]:
            if not row["expected_tools"]:
                continue
            offered = search_catalog(query=row["query"], limit=5)
            expected = set(row["expected_tools"])
            for k in (1, 3, 5):
                if expected <= set(row["offered"][:k]):
                    assert expected <= set(offered[:k]), (row["query"], k, offered)


def test_score_recorder_covers_both_splits_without_search_or_evaluation(tmp_path, monkeypatch):
    from career_agent.evaluation import catalog_search
    from career_agent.evaluation.catalog_search_embeddings import record_model_query_scores

    def forbidden_evaluation(*args, **kwargs):
        pytest.fail("recording similarities must never evaluate holdout")

    class Client:
        model_id = "fixture-client"

        def embed(self, texts):
            return [(1.0, 0.0) for _ in texts]

    monkeypatch.setattr(catalog_search, "evaluate_queries", forbidden_evaluation)
    path = record_model_query_scores(Client(), model="fixture-model", output=tmp_path / "scores.json")
    payload = json.loads(path.read_text())
    assert payload["query_count"] == len(payload["scores"]) == 26
    assert set(payload["scores"]) == {c["query"] for c in CASES}
    assert payload["fixture_sha256"] == FIXTURE_SHA256
    assert "no search ranking or recall evaluation" in payload["holdout_status"]
    with pytest.raises(FileExistsError):
        record_model_query_scores(Client(), model="fixture-model", output=path)
