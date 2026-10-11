"""Reconstruct legacy request baselines from their verified frozen source.

Original cassettes remain byte-for-byte unchanged. This is an audited derived
baseline, not a claim that the legacy format captured raw HTTP exchanges.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
from tempfile import TemporaryDirectory

from career_agent.agent.providers.tiktoken_assets import bundled_tiktoken_cache_dir

# Executed in a separate interpreter whose imports come from the frozen tree.
_RECONSTRUCT = r'''
import hashlib, importlib.util, json, sys
from pathlib import Path
from tempfile import TemporaryDirectory
from career_agent.evaluation import runtime_quality as frozen
from career_agent.evaluation.trajectory import ReplayClient
from career_agent.agent.providers.main_agent import OpenAICompatibleMainAgentDecisionMaker, main_model_options
from career_agent.agent.providers.openai_client import OpenAICompatibleAgentConfig
spec = importlib.util.spec_from_file_location("request_tape", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
root, output = Path(sys.argv[2]), Path(sys.argv[3])
clock = {"now": "2026-10-11T12:00:00+08:00", "timezone": "Asia/Shanghai"}
fingerprint = frozen.suite_fingerprint()
result = {"version": 1, "source_revision": sys.argv[4], "source_suite_fingerprint": fingerprint,
          "provenance": "reconstructed_from_frozen_source", "clock_policy": "fixed_synthetic_clock",
          "cases": {}}
for case in frozen.load_cases():
    path = root / (case["name"] + ".json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["suite_fingerprint"] != fingerprint:
        raise ValueError("cassette does not match the supplied frozen source: " + case["name"])
    options_env = {"MAIN_AGENT_VENDOR": sys.argv[5], "MAIN_AGENT_ENABLE_THINKING": "false"}
    derived = []
    for sample in payload["samples"]:
        if any(t["error"] for t in sample["turns"]):
            raise ValueError("legacy terminal errors cannot be reconstructed without raw exchanges")
        responses = [d["response"] for t in sample["turns"] for d in t["decisions"]]
        client = ReplayClient(responses)
        maker = OpenAICompatibleMainAgentDecisionMaker(
            OpenAICompatibleAgentConfig(endpoint="https://offline.invalid/v1/chat/completions", api_key="replay", model=payload["model"]),
            client=client, sleep=lambda _: None, **main_model_options(options_env))
        tape = module.RequestTape(clock=clock)
        tape.attach(maker)
        with TemporaryDirectory() as workspace:
            replay = frozen.run_sample(case, maker, root=Path(workspace), semantic_recording=sample["semantic_recording"])
        if client._responses or replay["failures"] != sample["failures"]:
            raise ValueError("frozen replay does not reproduce recorded outcomes: " + case["name"])
        before = [d for t in sample["turns"] for d in t["decisions"]]
        after = [d for t in replay["turns"] for d in t["decisions"]]
        if before != after:
            raise ValueError("frozen decisions changed: " + case["name"])
        derived.append({"request_recording": tape.entries, "clock": clock})
    result["cases"][case["name"]] = {
        "cassette_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "case": case, "model_options": options_env, "samples": derived}
output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
'''


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--vendor", choices=("qwen", "deepseek"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output already exists; preserve historical derivations")
    revision = subprocess.check_output(["git", "rev-parse", args.revision + "^{commit}"], text=True).strip()
    with TemporaryDirectory(prefix="career-frozen-runtime-") as directory:
        frozen = Path(directory)
        archive = frozen / "source.tar"
        with archive.open("wb") as stream:
            subprocess.run(["git", "archive", revision, "src", "evals/runtime_quality/scenarios.json"], stdout=stream, check=True)
        with tarfile.open(archive, encoding="utf-8") as source:
            source.extractall(frozen, filter="data")
        # -I ignores ambient PYTHONPATH; prepend only this verified archive.
        runner = "import sys; sys.path.insert(0, sys.argv.pop(1)); " + _RECONSTRUCT
        subprocess.run([sys.executable, "-I", "-c", runner, str(frozen / "src"),
                        str(Path(__file__).with_name("runtime_requests.py")), str(args.root.resolve()),
                        str(args.output.resolve()), revision, args.vendor], check=True,
                       env=dict(os.environ, TIKTOKEN_CACHE_DIR=str(bundled_tiktoken_cache_dir())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
