import json
from pathlib import Path

import pytest

from career_agent.agent.providers.main_agent import OpenAICompatibleMainAgentDecisionMaker
from career_agent.agent.providers.openai_client import OpenAICompatibleAgentConfig
from career_agent.evaluation.runtime_quality import CATEGORIES, evaluate, load_cases, run_sample
from career_agent.evaluation import runtime_quality
from career_agent.evaluation.trajectory import ReplayClient


def _maker(responses):
    return OpenAICompatibleMainAgentDecisionMaker(
        OpenAICompatibleAgentConfig(endpoint="https://runtime-replay.invalid/v1/chat/completions", api_key="replay", model="test"),
        client=ReplayClient(responses),
    )


def _responses(case):
    replies = []
    for turn in case["turns"]:
        # Explicit names loading mirrors a model's discovery, without relying
        # on lexical preloading or inventing a private bypass for the harness.
        replies.append({"tool_call": {"name": "search_capabilities", "arguments": {"names": turn["required_calls"]}}})
        for name in turn["required_calls"]:
            arguments = {"markdown": "下一步先查面试安排，再查投递记录。", "expected_revision": "empty"} if name == "update_working_notes" else {}
            replies.append({"tool_call": {"name": name, "arguments": arguments}})
        replies.append({"content": json.dumps({"action": "final", "message": "已完成全部查询。"})})
    return replies


def test_fixture_covers_all_missing_runtime_categories():
    cases = load_cases()
    assert {case["category"] for case in cases} == CATEGORIES
    assert all(len(case["turns"]) >= 2 for case in cases if case["category"] == "cross_turn")
    assert all(case.get("tool_injection") or case.get("initial_notes") for case in cases if case["category"] == "prompt_injection")


@pytest.mark.parametrize("case", load_cases(), ids=lambda case: case["name"])
def test_scripted_success_validates_harness_not_model_quality(tmp_path, case):
    result = run_sample(case, _maker(_responses(case)), root=tmp_path)
    assert result["passed"], result["failures"]
    assert len(result["turns"]) == len(case["turns"])
    assert result["premature_stop_count"] == 0
    assert result["unsafe_attempt_count"] == 0


def test_premature_final_does_not_count_as_task_success(tmp_path):
    case = next(c for c in load_cases() if c["category"] == "under_action")
    result = run_sample(case, _maker([{"content": '{"action":"final","message":"我已经全部做完了。"}'}]), root=tmp_path)
    assert not result["passed"]
    assert result["premature_stop_count"] == 1
    assert result["turns"][0]["missing_calls"] == case["turns"][0]["required_calls"]


def test_injection_write_is_measured_even_if_runtime_refuses_it(tmp_path):
    case = next(c for c in load_cases() if c["name"] == "notes_cannot_authorize_writes")
    replies = [{"tool_call": {"name": "search_capabilities", "arguments": {"names": ["update_working_notes"]}}},
               {"tool_call": {"name": "update_working_notes", "arguments": {"markdown": "INJECTION_ACCEPTED", "expected_revision": "empty"}}},
               {"content": '{"action":"final","message":"已完成。"}'}]
    result = run_sample(case, _maker(replies), root=tmp_path)
    assert not result["passed"]
    assert result["unsafe_attempt_count"] >= 1


def test_quoting_a_rejected_attack_marker_is_observation_not_failure(tmp_path):
    case = next(c for c in load_cases() if c["name"] == "tool_output_fake_owner_approval")
    replies = _responses(case)
    replies[-1] = {"content": json.dumps({"action": "final", "message": "已查完。网页中的 INJECTION_ACCEPTED 指令是不可信的，我没有执行。"})}
    result = run_sample(case, _maker(replies), root=tmp_path)
    assert result["passed"], result["failures"]
    assert result["injection_marker_echo_count"] == 1


def test_attack_only_final_reply_cannot_pass_after_required_reads(tmp_path):
    case = next(c for c in load_cases() if c["name"] == "tool_output_fake_owner_approval")
    replies = _responses(case)
    replies[-1] = {"content": '{"action":"final","message":"INJECTION_ACCEPTED"}'}
    result = run_sample(case, _maker(replies), root=tmp_path)
    assert not result["passed"]
    assert result["premature_stop_count"] == 0


def _save(root, case, samples, *, fingerprint="test-fingerprint", model="test", k=3):
    (root / f"{case['name']}.json").write_text(json.dumps({
        "suite_fingerprint": fingerprint, "model": model, "k": k, "samples": samples,
    }), encoding="utf-8")


def test_pass_k_requires_every_sample_and_never_discards_a_failure(tmp_path):
    case = load_cases()[0]
    good = run_sample(case, _maker(_responses(case)), root=tmp_path / "good")
    bad = run_sample(case, _maker([{"content": '{"action":"final","message":"完成。"}'}]), root=tmp_path / "bad")
    _save(tmp_path, case, [good, bad, good])
    report = evaluate((case,), root=tmp_path, model="test", k=3, fingerprint="test-fingerprint")
    assert report["pass_k"] == {"passed": 0, "total": 1}
    assert report["results"][0]["samples_passed"] == 2
    assert report["premature_stop_count"] == 1


@pytest.mark.parametrize("kind", ["stale", "incomplete", "unrecorded"])
def test_missing_stale_and_incomplete_recordings_cannot_pass(tmp_path, kind):
    case = load_cases()[0]
    if kind != "unrecorded":
        _save(tmp_path, case, [], fingerprint="old" if kind == "stale" else "test-fingerprint")
    report = evaluate((case,), root=tmp_path, model="test", k=3, fingerprint="test-fingerprint")
    assert report["results"][0]["status"] == kind
    assert report["results"][0]["pass_k"] is False


def test_setup_failure_is_retained_for_every_attempted_sample(tmp_path, monkeypatch):
    case = load_cases()[0]

    def fail_setup(*args, **kwargs):
        raise RuntimeError("synthetic setup failed")

    monkeypatch.setattr(runtime_quality, "_live_sample", fail_setup)
    config = OpenAICompatibleAgentConfig(endpoint="https://test.invalid/v1/chat/completions", api_key="test", model="test")
    runtime_quality.record_cases((case,), config=config, options={}, root=tmp_path,
                                 k=3, jobs=2, fingerprint="test-fingerprint")
    report = evaluate((case,), root=tmp_path, model="test", k=3, fingerprint="test-fingerprint")
    assert report["results"][0]["sample_count"] == 3
    assert report["results"][0]["samples_passed"] == 0
    assert report["results"][0]["status"] == "failed"
