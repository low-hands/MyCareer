"""Opt-in six-stage synthetic provider matrix. Prints no request/response values."""
from __future__ import annotations

import argparse
from contextlib import closing
from datetime import datetime, timezone
import json
import logging
import math
from pathlib import Path
from time import perf_counter
from typing import Callable, Literal
from uuid import uuid4

import httpx
from langgraph.checkpoint.sqlite import SqliteSaver
from openai import OpenAI
from openai.types.responses import Response
from openai.types.responses.response_create_params import ResponseCreateParamsNonStreaming

from career_agent.agent.deepagent_job_research_worker import (
    DeepAgentJobResearchWorker,
    _base_url,
)
from career_agent.agent.job_research_provider_diagnostics import (
    ProviderRequestObserver, ProviderRequestStructure, sdk_versions,
)
from career_agent.agent.openai_compatible_client import (
    AgentWorkerError, OpenAICompatibleAgentConfig, provider_error_metadata,
)
from career_agent.domain.job_discovery import JobDetail, Provenance
from career_agent.domain.job_research import company_key
from career_agent.services.job_research import JobResearchService
from career_agent.storage.job_research import SQLiteJobResearchStore
from career_agent.storage.jobs import SQLiteJobPostingRepository


MatrixStage = Literal[
    "responses_basic", "web_search", "function_tools", "structured",
    "web_search_structured", "deepagents_full",
]
_STAGES: tuple[MatrixStage, ...] = (
    "responses_basic", "web_search", "function_tools", "structured",
    "web_search_structured", "deepagents_full",
)
_OUTPUT_TYPES = frozenset({"message", "reasoning", "function_call", "web_search_call"})
_PUBLIC_QUESTION = (
    "Research Microsoft's Azure cloud product from current official public sources. "
    "Use web search and cite an official source. This is a synthetic protocol smoke."
)
_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {"ok": {"type": "boolean"}},
    "required": ["ok"], "additionalProperties": False,
}


class SmokeValidationError(ValueError):
    """A fixed, content-free check label, never a model or provider message."""

    def __init__(self, check: str) -> None:
        super().__init__(check)
        self.check = check


def _require(condition: bool, check: str) -> None:
    if not condition:
        raise SmokeValidationError(check)


def _full_agent(
    config: OpenAICompatibleAgentConfig,
    sink: Callable[[ProviderRequestStructure], None],
    *,
    skills_root: Path = Path("skills"),
    work_root: Path | None = None,
) -> dict[str, object]:
    # Keep synthetic artifacts for diagnosis/readback, never in OS temp or a
    # live business store. A fresh directory prevents accidental reuse.
    if work_root is None or not work_root.is_dir():
        raise SmokeValidationError("synthetic_work_root_required")
    root = work_root / f"research093-smoke-{uuid4().hex}"
    root.mkdir(mode=0o700)
    with SqliteSaver.from_conn_string(str(root / "checkpoints.sqlite3")) as saver:
        with closing(
            DeepAgentJobResearchWorker(
                config, skills_root=skills_root, checkpointer=saver,
                diagnostics_sink=sink, recursion_limit=12,
            )
        ) as worker:
            jobs = SQLiteJobPostingRepository(root / "jobs.sqlite3")
            now = datetime.now(timezone.utc)
            saved = jobs.save_detail(
                user_id="synthetic-smoke", run_id="synthetic-discovery",
                result_ref="synthetic-result", selection_index=1,
                detail=JobDetail(
                    source_name="synthetic", source_job_id="synthetic-job",
                    source_url="https://www.microsoft.com/", title="Cloud Engineer",
                    company_name="Microsoft", description="Synthetic role: Azure cloud services.",
                    captured_at=now, provenance=Provenance(
                        source_name="synthetic", source_job_id="synthetic-job",
                        source_url="https://www.microsoft.com/", captured_at=now,
                        operation="detail", adapter_version="synthetic-093",
                    ),
                ),
            )
            store = SQLiteJobResearchStore(root / "research.sqlite3")
            service = JobResearchService(jobs=jobs, store=store, worker=worker)
            result = service.research(
                user_id="synthetic-smoke", job_posting_id=saved.posting.id,
                jd_snapshot_id=saved.snapshot.id,
                focus="Azure public product context; use 1-2 official sources, concise findings.",
                max_sources=2,
            )
            source_keys = {source.source_key for source in result.sources}
            cited = {key for finding in result.report.findings for key in finding.source_keys}
            _require(bool(source_keys) and cited == source_keys, "citation_closure")
            _require(result.run.status == "completed" and bool(result.report.summary), "completed_report")
            _require(
                result.report.company_key == result.run.company_key == company_key("Microsoft")
                and result.report.jd_snapshot_id == result.run.jd_snapshot_id == saved.snapshot.id
                and result.report.job_posting_id == result.run.job_posting_id == saved.posting.id
                and result.report.user_id == result.run.user_id == "synthetic-smoke",
                "exact_resource_binding",
            )
            # Reconstruct both store and service, as an application refresh does.
            reopened = SQLiteJobResearchStore(root / "research.sqlite3")
            refreshed = JobResearchService(
                jobs=SQLiteJobPostingRepository(root / "jobs.sqlite3"),
                store=reopened, worker=worker,
            ).get_report(user_id="synthetic-smoke", report_id=result.report.id)
            _require(refreshed.report == result.report, "report_refresh")
            _require(set(refreshed.sources) == set(result.sources), "source_refresh")
            _require(refreshed.run == result.run, "run_refresh")
            _require(
                reopened.get_report(user_id="another-owner", report_id=result.report.id) is None
                and not reopened.list_sources(user_id="another-owner", report_id=result.report.id),
                "owner_isolation",
            )
            return {
                "report_persisted": True, "sources": len(result.sources),
                "findings": len(result.report.findings), "citation_closure": True,
                "company_binding": True, "posting_binding": True, "jd_binding": True,
                "refresh_readback": True, "owner_isolation": True,
            }


def _probe_request(stage: MatrixStage, model: str) -> ResponseCreateParamsNonStreaming:
    request: ResponseCreateParamsNonStreaming = {
        "model": model, "input": "Reply with the word OK.", "store": False,
    }
    if stage in {"web_search", "web_search_structured"}:
        request["tools"] = [{"type": "web_search"}]
        # An auto-selected tool may be skipped. Force the native capability so
        # a 200 with only a memorized answer cannot masquerade as web research.
        request["tool_choice"] = {"type": "web_search"}
        request["input"] = _PUBLIC_QUESTION
    if stage == "function_tools":
        request["tools"] = [{
            "type": "function", "name": "smoke_marker",
            "description": "Return a synthetic health marker.",
            "parameters": _SCHEMA, "strict": True,
        }]
        # The configured specialist endpoint accepted auto-selected functions,
        # but rejected the forced-function selector in the measured matrix.
        request["tool_choice"] = "auto"
        request["input"] = "Call smoke_marker with ok true."
    if stage in {"structured", "web_search_structured"}:
        request["text"] = {"format": {
            "type": "json_schema", "name": "smoke_marker", "schema": _SCHEMA, "strict": True,
        }}
        request["input"] = (
            (_PUBLIC_QUESTION if stage == "web_search_structured" else "")
            + " Return JSON with ok true after completing the task."
        )
    return request


def _validate_response(stage: MatrixStage, response: Response, result: dict[str, object]) -> None:
    output_types = {item.type for item in response.output}
    result["output_item_types"] = sorted(
        item if item in _OUTPUT_TYPES else "other" for item in output_types
    )
    result["url_citations"] = sum(
        annotation.type == "url_citation"
        for item in response.output if item.type == "message"
        for content in item.content if content.type == "output_text"
        for annotation in content.annotations
    )
    result["native_search_observed"] = "web_search_call" in output_types
    result["native_search_completed"] = any(
        item.type == "web_search_call" and item.status == "completed"
        for item in response.output
    )
    _require(response.status == "completed", "response_completed")
    if stage in {"web_search", "web_search_structured"}:
        _require("web_search_call" in output_types, "native_web_search_missing")
        _require(bool(result["native_search_completed"]), "native_web_search_incomplete")
        if stage == "web_search":
            _require(bool(result["url_citations"]), "search_citations_missing")
    if stage == "function_tools":
        _require(any(
            item.type == "function_call" and item.name == "smoke_marker"
            and _marker_is_valid(item.arguments)
            for item in response.output
        ), "function_arguments")
    elif stage in {"structured", "web_search_structured"}:
        _require(_marker_is_valid(response.output_text), "structured_output")
    else:
        _require(bool(response.output_text.strip()), "nonempty_output")
    result["output_validated"] = True


def _marker_is_valid(text: str) -> bool:
    try:
        value = json.loads(text)
    except ValueError:
        return False
    return isinstance(value, dict) and set(value) == {"ok"} and value["ok"] is True


def _failure_metadata(error: BaseException) -> dict[str, object]:
    result: dict[str, object] = {"error_type": type(error).__name__}
    current: BaseException | None = error
    chain: list[str] = []
    seen: set[int] = set()
    while current is not None and id(current) not in seen and len(chain) < 8:
        seen.add(id(current))
        chain.append(type(current).__name__)
        metadata = provider_error_metadata(current)
        if metadata is not None:
            result["provider"] = metadata.as_dict()
        if isinstance(current, SmokeValidationError):
            result["failed_check"] = current.check
        if isinstance(current, AgentWorkerError) and current.code in {
            "JOB_RESEARCH_INVALID_RESPONSE", "JOB_RESEARCH_STEP_LIMIT",
            "JOB_RESEARCH_SEARCH_UNVERIFIED",
        }:
            result["capability_error_code"] = current.code
        current = current.__cause__
    result["error_chain"] = chain
    return result


def run_matrix(
    config: OpenAICompatibleAgentConfig,
    *,
    skills_root: Path = Path("skills"),
    work_root: Path | None = None,
    report_path: Path | None = None,
) -> list[dict[str, object]]:
    structures: list[ProviderRequestStructure] = []
    results: list[dict[str, object]] = []
    with httpx.Client(event_hooks={"request": [ProviderRequestObserver(structures.append)]}) as http:
        with OpenAI(
            api_key=config.api_key, base_url=_base_url(config.endpoint),
            timeout=config.timeout_seconds, max_retries=0, http_client=http,
        ) as client:
            for stage in _STAGES:
                structures.clear()
                started = perf_counter()
                result: dict[str, object] = {"stage": stage}
                try:
                    if stage == "deepagents_full":
                        result.update(_full_agent(
                            config, structures.append, skills_root=skills_root, work_root=work_root,
                        ))
                    else:
                        response = client.responses.create(**_probe_request(stage, config.model))
                        _validate_response(stage, response, result)
                    result["passed"] = True
                except Exception as error:
                    result["passed"] = False
                    result.update(_failure_metadata(error))
                result["duration_ms"] = int((perf_counter() - started) * 1000)
                result["request_count"] = len(structures)
                result["requests"] = [item.as_dict() for item in structures]
                results.append(result)
                if report_path is not None:
                    # Incremental durability also preserves earlier stages if a
                    # later process is interrupted. Only safe metadata is saved.
                    report_path.write_text(
                        json.dumps(results, ensure_ascii=True, indent=2) + "\n",
                        encoding="utf-8",
                    )
                print(json.dumps(result, ensure_ascii=True), flush=True)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", default="RESUME_ANALYSIS_AGENT")
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--skills-root", type=Path, default=Path("skills"))
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or not 1 <= args.timeout <= 120:
        parser.error("--timeout must be between 1 and 120 seconds")
    if not args.work_root.is_dir() or not args.report.parent.is_dir():
        parser.error("artifact directories must already exist")
    if args.report.exists():
        parser.error("--report must be a new file")
    # SDK/transport loggers can emit model responses; this CLI reports only the
    # allowlisted summaries above, including on exceptions.
    logging.disable(logging.CRITICAL)
    try:
        configured = OpenAICompatibleAgentConfig.from_env(prefix=args.prefix)
        config = OpenAICompatibleAgentConfig(
            endpoint=configured.endpoint, api_key=configured.api_key,
            model=configured.model, timeout_seconds=args.timeout,
        )
    except Exception as error:
        print(json.dumps({"configuration_available": False, "error_type": type(error).__name__}))
        return 2
    print(json.dumps({"configuration_available": True, "sdk_versions": sdk_versions()}), flush=True)
    results = run_matrix(
        config, skills_root=args.skills_root, work_root=args.work_root, report_path=args.report,
    )
    return 0 if all(item["passed"] for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
