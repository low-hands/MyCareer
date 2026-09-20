"""Opt-in synthetic probe near the declared Main Agent context window."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from time import perf_counter

from openai import OpenAI

from career_agent.agent.context_deployment_config import ContextDeploymentConfig
from career_agent.agent.openai_compatible_client import OpenAICompatibleAgentConfig, provider_error_metadata
from career_agent.agent.openai_compatible_main_agent import max_output_tokens_from_env
from career_agent.agent.token_budget import serialized_token_count


_MARKERS = ("NORTHSTAR-0217", "RIVERSTONE-5834", "SKYLINE-9462")
_WORDS = "amber bridge cedar delta ember forest granite harbor island juniper kernel lantern meadow orbit pine quartz river silver timber valley".split()


def _messages(target_tokens: int) -> tuple[list[dict[str, str]], int]:
    system = "Read the synthetic context. Reply with only the three CHECK markers in their original order, separated by |."
    prefix = f"CHECK {_MARKERS[0]}\n"
    suffix = f"\nCHECK {_MARKERS[2]}\nReply with the CHECK markers only."
    lines: list[str] = []
    while True:
        first = lines[: len(lines) // 2]
        second = lines[len(lines) // 2 :]
        content = prefix + "".join(first) + f"CHECK {_MARKERS[1]}\n" + "".join(second) + suffix
        messages = [{"role": "system", "content": system}, {"role": "user", "content": content}]
        measured = serialized_token_count(messages)
        if measured >= target_tokens:
            return messages, measured
        batch = min(200, max(1, (target_tokens - measured) // 12))
        for _ in range(batch):
            number = len(lines)
            words = " ".join(_WORDS[(number + offset * 7) % len(_WORDS)] for offset in range(12))
            lines.append(f"Synthetic context row {number:05d}: {words}.\n")


def run_probe(*, target_fraction: float = 0.97,
              target_input_tokens: int | None = None,
              output_budget_override: int | None = None) -> dict[str, object]:
    config = OpenAICompatibleAgentConfig.from_env(prefix="MAIN_AGENT")
    deployment = ContextDeploymentConfig.from_env()
    window = deployment.main_context_window_tokens
    output_budget = (
        output_budget_override
        if output_budget_override is not None
        else max_output_tokens_from_env()
    )
    target_input = (
        target_input_tokens
        if target_input_tokens is not None
        else int(window * target_fraction) - output_budget
    )
    if target_input < 1024 or target_input + output_budget > window:
        raise ValueError("probe_budget_outside_declared_window")
    messages, local_input = _messages(target_input)
    result: dict[str, object] = {
        "declared_window_tokens": window,
        "local_cl100k_input_tokens": local_input,
        "requested_output_tokens": output_budget,
        "target_fraction": target_fraction if target_input_tokens is None else None,
        "local_reserved_occupancy": round((local_input + output_budget) / window, 4),
        "passed": False,
    }
    started = perf_counter()
    try:
        with OpenAI(
            api_key=config.api_key, base_url=config.endpoint.removesuffix("/chat/completions"),
            timeout=config.timeout_seconds, max_retries=0,
        ) as client:
            response = client.chat.completions.create(
                model=config.model, messages=messages, max_tokens=output_budget,
            )
        usage = response.usage
        reply = response.choices[0].message.content or ""
        positions = [reply.find(marker) for marker in _MARKERS]
        markers_retained = all(position >= 0 for position in positions) and positions == sorted(positions)
        result.update(
            provider_prompt_tokens=usage.prompt_tokens if usage else None,
            provider_completion_tokens=usage.completion_tokens if usage else None,
            finish_reason=response.choices[0].finish_reason,
            marker_retention=markers_retained,
            marker_presence=[position >= 0 for position in positions],
            reply_characters=len(reply),
        )
        if usage is not None:
            result["reported_reserved_occupancy"] = round(
                (usage.prompt_tokens + output_budget) / window, 4
            )
        result["passed"] = (
            usage is not None and result["marker_retention"]
            and response.choices[0].finish_reason == "stop"
            and result["reported_reserved_occupancy"] >= 0.9
        )
    except Exception as error:
        result["error_type"] = type(error).__name__
        metadata = provider_error_metadata(error)
        if metadata is not None:
            result["provider"] = metadata.as_dict()
    result["elapsed_seconds"] = round(perf_counter() - started, 4)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--target-fraction", type=float, default=0.97)
    parser.add_argument("--target-input-tokens", type=int)
    parser.add_argument("--output-budget", type=int)
    args = parser.parse_args()
    if not 0.9 <= args.target_fraction <= 0.99:
        parser.error("--target-fraction must be between 0.90 and 0.99")
    if args.output_budget is not None and not 256 <= args.output_budget <= 16384:
        parser.error("--output-budget must be between 256 and 16384")
    if args.report.exists() or not args.report.parent.is_dir():
        parser.error("--report must be a new file in an existing directory")
    logging.disable(logging.CRITICAL)
    try:
        result = run_probe(target_fraction=args.target_fraction,
                           target_input_tokens=args.target_input_tokens,
                           output_budget_override=args.output_budget)
    except Exception as error:
        result = {"passed": False, "configuration_available": False,
                  "error_type": type(error).__name__}
    args.report.write_text(json.dumps(result, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=True), flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
