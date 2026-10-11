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
        "version": 2, "suite_fingerprint": fingerprint, "model": model, "k": k, "samples": samples,
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


def test_summary_guard_rejects_a_receipt_even_after_all_requested_tools_ran(tmp_path, monkeypatch):
    import career_agent.harness.graph_routing as routing
    case = next(case for case in load_cases() if case["name"] == "empty_receipt_still_needs_interpretation")
    # Simulate an overly broad receipt shortcut: one requested read completes,
    # but the explicit interpretation is never written by the model.
    monkeypatch.setattr(routing, "completes_tool_delivery", lambda state: True)
    result = run_sample(case, _maker([{"tool_call": {"name": "list_interviews", "arguments": {}}}]), root=tmp_path)
    assert result["turns"][0]["missing_calls"] == []
    assert not result["passed"]
    assert "requested model summary missing" in result["failures"][0]


def test_complete_brief_requires_a_final_model_call(tmp_path):
    case = next(case for case in load_cases() if case["name"] == "single_brief_body_is_delivered")
    result = run_sample(case, _maker(_responses(case)), root=tmp_path)
    assert result["passed"]
    assert result["turns"][0]["decisions"][-1]["action"] == "final"
    assert result["turns"][0]["model_summary_delivered"]


def test_summary_with_optional_followup_is_still_a_model_authored_reply(tmp_path):
    case = next(case for case in load_cases() if case["name"] == "empty_receipt_still_needs_interpretation")
    replies = _responses(case)
    replies[-1] = {"content": json.dumps({"action": "ask_user", "message": "没有找到面试安排。你可以先整理自我介绍、复习目标岗位的核心技能，并通过模拟面试练习表达。你想先准备哪个方向？"})}
    result = run_sample(case, _maker(replies), root=tmp_path)
    assert result["passed"], result["failures"]
    assert result["turns"][0]["model_summary_delivered"]


def test_recorded_normal_completion_survives_replay(tmp_path):
    case = next(c for c in load_cases() if c["name"] == "single_empty_resume_receipt")
    sample = run_sample(case, _maker(_responses(case)), root=tmp_path / "live")
    assert sample["passed"]
    replies = [d["response"] for t in sample["turns"] for d in t["decisions"]]
    assert "tool_call" not in replies[-1]
    replay = run_sample(case, _maker(replies), root=tmp_path / "replay")
    assert replay["passed"]
    assert len(replay["turns"][0]["decisions"]) == len(sample["turns"][0]["decisions"])


def test_paraphrased_answer_is_not_a_wording_failure(tmp_path):
    case = next(c for c in load_cases() if c["name"] == "single_empty_resume_receipt")
    responses = [
        {"tool_call": {"name": "search_capabilities", "arguments": {"names": ["list_resumes"]}}},
        {"tool_call": {"name": "list_resumes", "arguments": {}}},
        {"content": json.dumps({"action": "final", "message": "当前没有找到任何简历。"})},
    ]
    sample = run_sample(case, _maker(responses), root=tmp_path)
    assert sample["passed"], sample["failures"]
    assert not sample["turns"][0]["direct_delivery"]


def test_score_tape_replay_never_initializes_embedding(tmp_path, monkeypatch):
    case = load_cases()[0]
    sample = run_sample(case, _maker(_responses(case)), root=tmp_path / "live")
    def forbidden():
        raise AssertionError("offline replay attempted embedding initialization")
    monkeypatch.setattr(runtime_quality, "default_capability_scorer", forbidden)
    replay = run_sample(case, _maker(_responses(case)), root=tmp_path / "replay",
                        semantic_recording=sample["semantic_recording"])
    assert replay["passed"]
    assert replay["semantic_recording"] == sample["semantic_recording"]


def test_missing_semantic_query_cannot_silently_fallback(tmp_path):
    from career_agent.evaluation.runtime_semantic import SemanticTape
    case = load_cases()[0]
    with pytest.raises(ValueError, match="missing from recording"):
        run_sample(case, _maker(_responses(case)), root=tmp_path,
                   semantic_recording=SemanticTape().snapshot())


def test_actual_semantic_scores_are_replayed_as_selection_inputs(tmp_path, monkeypatch):
    case = next(c for c in load_cases() if c["name"] == "single_empty_resume_receipt")
    class Scorer:
        def scores(self, query):
            return {"list_resumes": 0.9123456789, "list_interviews": 0.01}
    sample = run_sample(case, _maker(_responses(case)), root=tmp_path / "live", scorer=Scorer())
    assert sample["semantic_recording"]["queries"][case["turns"][0]["message"]]["list_resumes"] == 0.9123456789
    monkeypatch.setattr(runtime_quality, "default_capability_scorer", lambda: pytest.fail("embedding initialized"))
    replay = run_sample(case, _maker(_responses(case)), root=tmp_path / "replay",
                        semantic_recording=json.loads(json.dumps(sample["semantic_recording"])))
    assert replay["passed"]
    assert replay["turns"][0]["decisions"] == sample["turns"][0]["decisions"]


def test_request_change_is_stale_before_recorded_response_is_used(tmp_path, monkeypatch):
    case = load_cases()[0]
    sample = run_sample(case, _maker(_responses(case)), root=tmp_path / "live")
    _save(tmp_path, case, [sample], k=1)
    original = OpenAICompatibleMainAgentDecisionMaker._system_prompt
    monkeypatch.setattr(OpenAICompatibleMainAgentDecisionMaker, "_system_prompt",
                        staticmethod(lambda *args, **kwargs: original(*args, **kwargs) + " Changed instruction."))
    report = evaluate((case,), root=tmp_path, model="test", k=1, fingerprint="test-fingerprint")
    assert report["results"][0]["status"] == "stale"
    assert "model request 1" in report["results"][0]["validity_error"]
    assert report["results"][0]["sample_count"] == 0


def test_legacy_recording_without_request_provenance_is_not_assumed_valid(tmp_path):
    case = load_cases()[0]
    _save(tmp_path, case, [{}], k=1)
    path = tmp_path / f"{case['name']}.json"
    payload = json.loads(path.read_text())
    payload["version"] = 1
    path.write_text(json.dumps(payload))
    assert evaluate((case,), root=tmp_path, model="test", k=1,
                    fingerprint="test-fingerprint")["results"][0]["status"] == "missing_request_recording"


def test_request_identity_checks_content_and_ignores_transport_only_changes(tmp_path):
    from career_agent.evaluation.runtime_requests import request_fingerprint
    request = {"messages": [{"role": "user", "content": "证据原文"}], "tools": [],
               "timeout": 30, "extra_body": {"prompt_cache_key": "one", "enable_thinking": False}}
    same = dict(request, timeout=1, extra_body={"prompt_cache_key": "two", "enable_thinking": False})
    assert request_fingerprint(request) == request_fingerprint(same)
    assert request_fingerprint(request) != request_fingerprint(dict(request, tools=[{"name": "other"}]))
    assert request_fingerprint(request) != request_fingerprint(dict(request, messages=[{"role": "user", "content": "另一条证据"}]))
    assert request_fingerprint(request) != request_fingerprint(dict(request, extra_body={"enable_thinking": True}))


def test_delivery_changes_reuse_responses_when_requests_are_unchanged(tmp_path, monkeypatch):
    from career_agent.agent.presentation.presenter import TurnPresenter
    case = next(c for c in load_cases() if c["name"] == "single_brief_body_is_delivered")
    sample = run_sample(case, _maker(_responses(case)), root=tmp_path / "live")
    _save(tmp_path, case, [sample], k=1)
    original = TurnPresenter._undelivered_bodies
    monkeypatch.setattr(TurnPresenter, "_undelivered_bodies",
                        lambda self, *args, **kwargs: original(self, *args, **kwargs) + "\n展示说明。")
    report = evaluate((case,), root=tmp_path, model="test", k=1, fingerprint="test-fingerprint")
    assert report["results"][0]["status"] == "passed"
    assert report["results"][0]["samples_passed"] == 1


def test_model_visible_observation_changes_are_stale(tmp_path, monkeypatch):
    case = load_cases()[0]
    sample = run_sample(case, _maker(_responses(case)), root=tmp_path / "live")
    _save(tmp_path, case, [sample], k=1)
    original = runtime_quality._SyntheticRegistry.invoke_atomic_tool
    def changed(self, name, arguments):
        result = original(self, name, arguments)
        return result.model_copy(update={"message": "工具结果发生变化。"}) if name in runtime_quality._READ_RESULTS else result
    monkeypatch.setattr(runtime_quality._SyntheticRegistry, "invoke_atomic_tool", changed)
    report = evaluate((case,), root=tmp_path, model="test", k=1, fingerprint="test-fingerprint")
    assert report["results"][0]["status"] == "stale"


def test_provider_retries_are_recorded_and_matched_individually(tmp_path):
    from career_agent.agent.providers.openai_client import AgentWorkerError
    case = load_cases()[0]
    client = ReplayClient(_responses(case))
    original = client._next
    failed = False
    def first_request_fails(kwargs):
        nonlocal failed
        if not failed:
            failed = True
            raise AgentWorkerError("MAIN_AGENT_TRANSPORT_ERROR", "synthetic", retryable=True)
        return original(kwargs)
    client._next = first_request_fails
    maker = OpenAICompatibleMainAgentDecisionMaker(
        OpenAICompatibleAgentConfig(endpoint="https://offline.invalid/v1/chat/completions", api_key="replay", model="test"),
        client=client, sleep=lambda _: None)
    sample = run_sample(case, maker, root=tmp_path / "live")
    assert "error" in sample["request_recording"][0]
    assert sample["request_recording"][0]["fingerprint"] == sample["request_recording"][1]["fingerprint"]
    _save(tmp_path, case, [sample], k=1)
    assert evaluate((case,), root=tmp_path, model="test", k=1,
                    fingerprint="test-fingerprint")["results"][0]["status"] == "passed"


@pytest.mark.parametrize("vendor,prompt_cache", [("qwen", "implicit"), ("qwen", "explicit"), ("deepseek", "implicit")])
def test_v2_recording_replays_the_recorded_generation_options(tmp_path, monkeypatch, vendor, prompt_cache):
    from career_agent.agent.providers.main_agent import main_model_options
    case = load_cases()[0]
    options = main_model_options({"MAIN_AGENT_VENDOR": vendor, "MAIN_AGENT_ENABLE_THINKING": "false"})
    options["max_output_tokens"] = 2048
    monkeypatch.setattr(runtime_quality, "default_capability_scorer", lambda: None)
    def scripted_record(case, config, options, scorer):
        maker = OpenAICompatibleMainAgentDecisionMaker(config, client=ReplayClient(_responses(case)), **options)
        return run_sample(case, maker, root=tmp_path / "live", scorer=scorer)
    monkeypatch.setattr(runtime_quality, "_live_sample", scripted_record)
    config = OpenAICompatibleAgentConfig(endpoint="https://offline.invalid/v1/chat/completions", api_key="replay", model="test", prompt_cache=prompt_cache)
    runtime_quality.record_cases((case,), config=config, options=options, root=tmp_path, k=1, jobs=1, fingerprint="test")
    assert evaluate((case,), root=tmp_path, model="test", k=1, fingerprint="test")["results"][0]["status"] == "passed"


def test_terminal_model_protocol_failure_is_kept_and_replayed(tmp_path):
    from career_agent.evaluation.runtime_requests import _response_object
    case = load_cases()[0]
    client = ReplayClient(())
    client._next = lambda kwargs: _response_object({"choices": [{"message": {"content": "unfinished plain text", "tool_calls": []}}]})
    maker = OpenAICompatibleMainAgentDecisionMaker(
        OpenAICompatibleAgentConfig(endpoint="https://offline.invalid/v1/chat/completions", api_key="replay", model="test"),
        client=client)
    sample = run_sample(case, maker, root=tmp_path / "live")
    assert not sample["passed"]
    assert sample["turns"][0]["error"]
    assert len(sample["request_recording"]) > 1
    _save(tmp_path, case, [sample], k=1)
    row = evaluate((case,), root=tmp_path, model="test", k=1, fingerprint="test-fingerprint")["results"][0]
    assert row["status"] == "failed"
    assert row["failures"] == [sample["failures"]]
    assert row["sample_count"] == 1


@pytest.mark.parametrize("failure_type", [ValueError, RuntimeError])
def test_unrelated_replay_bug_is_not_disguised_as_stale(tmp_path, monkeypatch, failure_type):
    case = load_cases()[0]
    sample = run_sample(case, _maker(_responses(case)), root=tmp_path / "live")
    _save(tmp_path, case, [sample], k=1)
    def broken_replay(*args, **kwargs):
        raise failure_type("production regression")
    monkeypatch.setattr(runtime_quality, "run_sample", broken_replay)
    with pytest.raises(failure_type, match="production regression"):
        evaluate((case,), root=tmp_path, model="test", k=1, fingerprint="test-fingerprint")


def test_recorded_semantic_mismatch_is_stale_without_initializing_embedding(tmp_path, monkeypatch):
    case = load_cases()[0]
    sample = run_sample(case, _maker(_responses(case)), root=tmp_path / "live")
    sample["semantic_recording"]["catalogue_digest"] = "old catalogue"
    _save(tmp_path, case, [sample], k=1)
    monkeypatch.setattr(runtime_quality, "default_capability_scorer",
                        lambda: pytest.fail("offline replay initialized embedding"))
    row = evaluate((case,), root=tmp_path, model="test", k=1,
                   fingerprint="test-fingerprint")["results"][0]
    assert row["status"] == "stale"
    assert row["validity_error"] == "recorded semantic catalogue changed"


def test_runtime_value_error_remains_a_failure_despite_unused_recorded_responses(tmp_path, monkeypatch):
    case = load_cases()[0]
    sample = run_sample(case, _maker(_responses(case)), root=tmp_path / "live")
    _save(tmp_path, case, [sample], k=1)
    def broken_turn(*args, **kwargs):
        raise ValueError("production turn regression")
    monkeypatch.setattr(runtime_quality.MainAgentRuntime, "run_turn", broken_turn)
    row = evaluate((case,), root=tmp_path, model="test", k=1,
                   fingerprint="test-fingerprint")["results"][0]
    assert row["status"] == "failed"
    assert row["sample_count"] == 1
    assert row["samples_passed"] == 0
    assert row["failures"][0][0] == "turn 1: runtime error ValueError"
