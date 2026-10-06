"""Search-mode trajectory recording with intermediate capability discovery.

The legacy recorder has one response per declared step. Here a declared step
is the business decision; discovery decisions are recorded ahead of it and
replayed through the same selection strategy.
"""

from __future__ import annotations

import json
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from career_agent.agent.capabilities.search import search_catalog
from career_agent.agent.runtime.decision_messages import project_decision_messages
from career_agent.agent.capabilities.selection_strategy import SearchStrategy
from career_agent.agent.contracts.context import MainAgentContext
from career_agent.agent.contracts.observations import (
    DecisionObservation, append_decision_observation,
)
from career_agent.agent.contracts.tools.core_memory import SearchCapabilitiesToolArguments
from career_agent.agent.providers.openai_client import AgentWorkerError, OpenAICompatibleAgentConfig
from career_agent.agent.providers.main_agent import OpenAICompatibleMainAgentDecisionMaker, main_model_options
from career_agent.evaluation.trajectory import (
    ReplayClient, TrajectoryScenario, TrajectoryStep, _has_path,
    _recording_can_retry, _retry_wait, advance_trajectory_context,
    cassette_path, check_step, check_step_quality, context_shape_fingerprint,
    load_cassette, prompt_fingerprint, trajectory_prompt_fingerprint,
)


SEARCH_CASSETTE_ROOT = Path(__file__).resolve().parents[3] / "evals" / "main_agent_search"
MAX_SEARCH_DECISIONS = 5
_FINGERPRINT_CLOCK = {"now": "2000-01-01T00:00:00+08:00", "timezone": "Asia/Shanghai"}


def _selected_context(context: MainAgentContext, strategy: SearchStrategy,
                      specs: tuple[dict[str, Any], ...]):
    selection = strategy.select(context, specs)
    return context.model_copy(update={"capability_selection": selection}), selection


def _decision_shape_fingerprint(context: MainAgentContext) -> str:
    projected = project_decision_messages(context, clock=_FINGERPRINT_CLOCK)
    messages = projected.messages(system_prompt="[policy]", spotlight_nonce="0" * 32)
    encoded = json.dumps(messages, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _search_result(context: MainAgentContext, arguments: Mapping[str, Any]) -> MainAgentContext:
    request = SearchCapabilitiesToolArguments.model_validate(arguments)
    found = search_catalog(query=request.query, names=request.names, limit=request.limit)
    loaded = tuple(name for name in found if name not in context.task.loaded_capabilities)
    observation = DecisionObservation(
        tool_name="search_capabilities",
        state="capabilities_found" if found else "no_capabilities_found",
        message=f"找到 {len(found)} 个相关能力。" if found else "没有找到匹配的能力。",
        arguments=dict(arguments),
    )
    return context.model_copy(update={
        "task": context.task.add_loaded_capabilities(loaded),
        "tool_observations": append_decision_observation(
            context.tool_observations, observation,
        ),
    })


def check_search_contract(scenario: TrajectoryScenario,
                          *, tool_specs: tuple[dict[str, Any], ...]) -> tuple[str, ...]:
    """Verify every declared business tool can be offered after exact-name discovery."""
    failures: list[str] = []
    strategy = SearchStrategy()
    context = scenario.context
    projection = _selected_context(context, strategy, tool_specs)[0].model_context()
    for path in scenario.decisive_facts:
        if not _has_path(projection, path):
            failures.append(f"{scenario.name}: the projection has no '{path}'")
    for index, step in enumerate(scenario.steps):
        if step.expect_user_input and step.expect_action is not None:
            failures.append(f"{scenario.name}[{index}]: conflicting action expectations")
        context = advance_trajectory_context(context, step)
        expected = set(step.expect_tools)
        if step.expect_tool:
            expected.add(step.expect_tool)
        for name in sorted(expected):
            _, selection = _selected_context(context, strategy, tool_specs)
            if name in selection.offered_names:
                continue
            try:
                context = _search_result(context, {"names": [name]})
            except ValueError as error:
                failures.append(f"{scenario.name}[{index}]: {name}: {error}")
                continue
            _, selection = _selected_context(context, strategy, tool_specs)
            if name not in selection.offered_names:
                failures.append(f"{scenario.name}[{index}]: expected tool '{name}' unreachable after search")
    return tuple(failures)


def _response(decision, *, scenario_step: int, selection, context: MainAgentContext,
              elapsed_ms: float,
              model_request_count: int, retry_events: list[dict[str, Any]],
              recording_retry_events: list[dict[str, Any]]) -> dict[str, Any]:
    item = (
        {"tool_call": {"name": decision.tool_call.name,
                        "arguments": decision.tool_call.arguments}}
        if decision.tool_call is not None else
        {"content": decision.model_dump_json(exclude_none=True)}
    )
    item.update({
        "scenario_step": scenario_step,
        "selected_schema_fingerprint": prompt_fingerprint(selection.schemas, mode="search"),
        "decision_shape_fingerprint": _decision_shape_fingerprint(context),
        "offered_tools": list(selection.offered_names),
        "elapsed_ms": round(elapsed_ms, 1),
        "model_request_count": model_request_count,
        "decision_retry_telemetry_version": 1,
        "decision_retry_events": retry_events,
        "recording_retry_events": recording_retry_events,
    })
    return item


def _record_sample(scenario: TrajectoryScenario,
                   *, tool_specs: tuple[dict[str, Any], ...],
                   config: OpenAICompatibleAgentConfig,
                   max_attempts: int = 3) -> dict[str, Any]:
    maker = OpenAICompatibleMainAgentDecisionMaker(config, **main_model_options())
    strategy = SearchStrategy()
    maker.configure_tool_selection(strategy)
    context = scenario.context.model_copy(update={"received_at": datetime.now(timezone.utc)})
    recorded: list[dict[str, Any]] = []
    started = time.perf_counter()
    search_count = 0
    for index, step in enumerate(scenario.steps):
        if step.user_message is not None:
            search_count = 0
        context = advance_trajectory_context(context, step)
        while True:
            decision_context, selection = _selected_context(context, strategy, tool_specs)
            attempts = 0
            requests = 0
            retries: list[dict[str, Any]] = []
            recording_retries: list[dict[str, Any]] = []
            decision_started = time.perf_counter()
            while True:
                attempts += 1
                try:
                    decision = maker.decide(decision_context, selection.schemas)
                    error = None
                except AgentWorkerError as caught:
                    decision = None
                    error = caught
                metrics = maker.consume_cache_metrics()
                requests += int(metrics.get("attempt_count", 0))
                retry_metrics = maker.consume_decision_retry_metrics()
                retries.extend({**event, "decision_invocation": attempts}
                               for event in retry_metrics.get("decision_retry_events", ()))
                if error is None:
                    break
                if attempts >= max_attempts or not _recording_can_retry(error):
                    error.recording_trace = {
                        "step": index, "completed_steps": recorded,
                        "offered_tools": list(selection.offered_names),
                        "error_code": error.code,
                        "model_request_count": requests,
                        "decision_retry_events": retries,
                        "recording_retry_events": recording_retries,
                    }
                    raise error
                recording_retries.append({"error_code": error.code,
                                          "decision_invocation": attempts})
                time.sleep(_retry_wait(1.0, attempts, jitter=True))
            assert decision is not None
            item = _response(
                decision, scenario_step=index, selection=selection,
                context=decision_context,
                elapsed_ms=(time.perf_counter() - decision_started) * 1000,
                model_request_count=requests, retry_events=retries,
                recording_retry_events=recording_retries,
            )
            item["decision_invocations"] = attempts
            recorded.append(item)
            if decision.tool_call is None or decision.tool_call.name != "search_capabilities":
                break
            search_count += 1
            if search_count > MAX_SEARCH_DECISIONS:
                item["search_budget_exhausted"] = True
                break
            try:
                context = _search_result(context, decision.tool_call.arguments)
            except ValueError as error:
                item["search_error"] = str(error)
                break
    return {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
        "steps": recorded,
    }


def record_search_catalogue(scenarios: Sequence[TrajectoryScenario],
                            *, tool_specs: tuple[dict[str, Any], ...],
                            config: OpenAICompatibleAgentConfig,
                            root: Path = SEARCH_CASSETTE_ROOT,
                            sample_count: int | None = None,
                            jobs: int = 8, force: bool = False) -> tuple[Path, ...]:
    if not 1 <= jobs <= 32:
        raise ValueError("trajectory record jobs must be between 1 and 32")
    work = []
    for scenario in scenarios:
        existing = load_cassette(scenario.name, root=root)
        if existing is not None and not force and search_cassette_staleness(
            existing, scenario=scenario, tool_specs=tool_specs, expected_model=config.model,
        ) is None:
            continue
        count = scenario.recording_samples if sample_count is None else sample_count
        if not 1 <= count <= 5:
            raise ValueError("trajectory sample count must be between 1 and 5")
        work.extend((scenario, index) for index in range(count))
    if not work:
        return ()
    completed: dict[str, dict[int, dict[str, Any]]] = {}
    written: list[Path] = []
    failed: set[str] = set()
    recording_failures: list[dict[str, Any]] = []
    first_error: AgentWorkerError | None = None
    with ThreadPoolExecutor(max_workers=min(jobs, len(work))) as pool:
        futures = {
            pool.submit(_record_sample, scenario, tool_specs=tool_specs, config=config):
            (scenario, index) for scenario, index in work
        }
        for future in as_completed(futures):
            scenario, index = futures[future]
            try:
                sample = future.result()
            except AgentWorkerError as error:
                failed.add(scenario.name)
                recording_failures.append({
                    "scenario": scenario.name, "sample": index + 1,
                    "error_code": error.code,
                    "trace": getattr(error, "recording_trace", None),
                })
                if first_error is None:
                    first_error = error
                continue
            if scenario.name in failed:
                continue
            completed.setdefault(scenario.name, {})[index] = sample
            count = sum(item.name == scenario.name for item, _ in work)
            if len(completed[scenario.name]) != count:
                continue
            samples = [completed[scenario.name][number] for number in range(count)]
            path = cassette_path(scenario.name, root=root)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({
                "scenario": scenario.name,
                "policy": scenario.policy,
                "selection_mode": "search",
                "model": config.model,
                "prompt_fingerprint": trajectory_prompt_fingerprint(
                    scenario, tool_specs, mode="search"),
                "context_shape_fingerprint": context_shape_fingerprint(scenario),
                "recorded_at": samples[-1]["recorded_at"],
                "steps": samples[0]["steps"],
                "sample_count": count,
                "samples": samples,
            }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            written.append(path)
    if first_error is not None:
        first_error.recording_failures = recording_failures
        raise first_error
    return tuple(written)


def search_cassette_staleness(cassette, *, scenario: TrajectoryScenario,
                              tool_specs: tuple[dict[str, Any], ...],
                              expected_model: str | None) -> str | None:
    if cassette.prompt_fingerprint != trajectory_prompt_fingerprint(
        scenario, tool_specs, mode="search",
    ):
        return "search prompt or schema changed; re-record it"
    if cassette.context_shape_fingerprint != context_shape_fingerprint(scenario):
        return "scenario context changed; re-record it"
    if not expected_model or cassette.model != expected_model:
        return "cassette model differs from current model; re-record it"
    if cassette.sample_count < scenario.recording_samples:
        return "cassette has too few samples; re-record it"
    return None


def replay_search_sample(scenario: TrajectoryScenario,
                         *, tool_specs: tuple[dict[str, Any], ...],
                         responses: Sequence[Mapping[str, Any]],
                         quality: bool = False) -> tuple[str, ...]:
    strategy = SearchStrategy()
    maker = OpenAICompatibleMainAgentDecisionMaker(
        OpenAICompatibleAgentConfig(
            endpoint="https://replay.invalid/v1/chat/completions",
            api_key="replay", model="replay",
        ), client=ReplayClient(responses),
    )
    maker.configure_tool_selection(strategy)
    context = scenario.context
    failures: list[str] = []
    response_index = 0
    search_count = 0
    for index, step in enumerate(scenario.steps):
        if step.user_message is not None:
            search_count = 0
        context = advance_trajectory_context(context, step)
        while response_index < len(responses):
            item = responses[response_index]
            if item.get("scenario_step") != index:
                failures.append(f"{scenario.name}[{index}]: missing terminal decision")
                break
            decision_context, selection = _selected_context(context, strategy, tool_specs)
            if item.get("selected_schema_fingerprint") != prompt_fingerprint(
                selection.schemas, mode="search",
            ):
                failures.append(f"{scenario.name}[{index}]: selected schema changed during replay")
            if (item.get("decision_shape_fingerprint") is not None and
                    item["decision_shape_fingerprint"] != _decision_shape_fingerprint(decision_context)):
                failures.append(f"{scenario.name}[{index}]: model context changed during replay")
            call = item.get("tool_call")
            if call is not None and call["name"] not in selection.offered_names:
                failures.append(f"{scenario.name}[{index}]: called unavailable tool '{call['name']}'")
                response_index += 1
                break
            decision = maker.decide(decision_context, selection.schemas)
            response_index += 1
            if decision.tool_call is not None and decision.tool_call.name == "search_capabilities":
                search_count += 1
                if search_count > MAX_SEARCH_DECISIONS:
                    failures.append(f"{scenario.name}[{index}]: search budget exhausted")
                    break
                try:
                    context = _search_result(context, decision.tool_call.arguments)
                except ValueError as error:
                    failures.append(f"{scenario.name}[{index}]: invalid search: {error}")
                    break
                continue
            if quality:
                failures.extend(check_step_quality(step, decision,
                    scenario=scenario.name, index=index))
            else:
                failures.extend(check_step(step, decision,
                    scenario=scenario.name, index=index))
            break
        else:
            failures.append(f"{scenario.name}[{index}]: no recorded decision")
    if response_index != len(responses):
        failures.append(f"{scenario.name}: {len(responses) - response_index} extra decisions")
    return tuple(failures)
