"""Synthetic-only bounded summary comparison. Emits aggregate metrics, never content.

Run on the server with --work-dir pointing to an existing synthetic scratch root.
Add --live for the real Summary worker, or --provider-probe for three bounded calls.
The fixed tool probes are not an autonomous Main Agent tool-choice evaluation.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
from dataclasses import dataclass, replace
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from dotenv import load_dotenv
from openai import APIConnectionError, APIStatusError, OpenAI
from openai.types.chat import ChatCompletionMessageParam
from openai.types.shared_params import ResponseFormatJSONSchema

from career_agent.agent.context_deployment_config import (
    ContextDeploymentConfig,
    ConversationSummaryAgentConfig,
)
from career_agent.agent.context_manager import ContextManager
from career_agent.agent.conversation_memory_contracts import (
    ConversationSummaryContent,
    ConversationSummaryWorker,
    SummaryMessage,
)
from career_agent.agent.main_agent_contracts import ConversationTaskState
from career_agent.agent.openai_compatible_client import (
    AgentWorkerError,
    OpenAICompatibleAgentConfig,
    provider_worker_error,
)
from career_agent.agent.openai_conversation_summary_worker import (
    OpenAIConversationSummaryWorker,
    summary_response_format,
)

from career_agent.storage.context import CareerContextStore

KNOWN_FACTS = ("Northstar", "Python", "SQLite", "Taipei")
SYNTHETIC_MESSAGES = (
    "The selected synthetic employer is Northstar, not a similarly named company.",
    "The chosen language for this synthetic review is Python.",
    "We decided to use SQLite for the synthetic tracking database.",
    "The selected synthetic role is located in Taipei.",
    *(
        f"Continue the same synthetic saved-role review, step {index}."
        for index in range(4, 15)
    ),
)


class FactSummaryWorker:
    def summarize(
        self,
        *,
        previous: ConversationSummaryContent | None,
        messages: tuple[SummaryMessage, ...],
    ) -> ConversationSummaryContent:
        seen = previous.confirmed_decisions if previous else ()
        additions = tuple(
            fact
            for fact in KNOWN_FACTS
            if fact not in seen and any(fact in item.content for item in messages)
        )
        return ConversationSummaryContent(confirmed_decisions=(*seen, *additions))


class MeteredSummaryWorker:
    def __init__(self, worker: ConversationSummaryWorker, max_calls: int = 6) -> None:
        self.worker = worker
        self.max_calls = max_calls
        self.sequences: list[tuple[int, ...]] = []
        self.elapsed_seconds = 0.0
        self.failures: list[str] = []
        self.failure_metadata: list[dict[str, object]] = []
        self.successes = 0

    def summarize(
        self,
        *,
        previous: ConversationSummaryContent | None,
        messages: tuple[SummaryMessage, ...],
    ) -> ConversationSummaryContent:
        if len(self.sequences) >= self.max_calls:
            raise RuntimeError("synthetic summary smoke call bound exceeded")
        self.sequences.append(tuple(item.sequence for item in messages))
        started = perf_counter()
        try:
            result = self.worker.summarize(previous=previous, messages=messages)
            self.successes += 1
            return result
        except AgentWorkerError as error:
            self.failures.append(error.code)
            if error.provider is not None:
                self.failure_metadata.append(error.provider.as_dict())
            raise
        finally:
            self.elapsed_seconds += perf_counter() - started


@dataclass(frozen=True)
class SyntheticMetrics:
    recent: int
    batch: int
    compaction_calls: int
    compaction_attempts: int
    summary_failure_codes: tuple[str, ...]
    summary_failure_metadata: tuple[dict[str, object], ...]
    compaction_seconds: float
    retained_known_facts: int
    known_fact_count: int
    page_in_calls: int
    duplicate_tool_calls: int
    duplicate_summary_calls: int
    watermark: int
    history_reachable: bool
    projection_contiguous: bool
    summary_available: bool
    passed: bool


def run_trajectory(
    path: Path,
    *,
    recent: int,
    batch: int,
    worker: ConversationSummaryWorker,
) -> SyntheticMetrics:
    store = CareerContextStore(path)
    metered = MeteredSummaryWorker(worker)
    manager = ContextManager(
        store,
        summary_worker=metered,
        recent_message_limit=recent,
        summary_batch_size=batch,
    )
    manager.configure_request_token_estimator(
        lambda context: (16000, 32000),
        static_input_tokens=15000,
        max_input_tokens=32000,
    )
    contiguous = True
    for index, user_message in enumerate(SYNTHETIC_MESSAGES):
        context = manager.load_for_turn(
            user_id="synthetic-owner",
            conversation_id="synthetic-093",
            user_message=user_message,
        )
        if context.recent_messages:
            contiguous = (
                contiguous
                and context.recent_from_sequence == context.through_sequence + 1
            )
            contiguous = (
                contiguous
                and context.through_sequence + len(context.recent_messages) == index * 2
            )
        manager.commit_turn(
            context=context,
            task=ConversationTaskState(),
            assistant_message=f"Continue the agreed synthetic plan, step {index}.",
        )
    final = manager.load_for_turn(
        user_id="synthetic-owner",
        conversation_id="synthetic-093",
        user_message="Check the agreed synthetic plan.",
    )
    summary = store.get_conversation_summary(
        user_id="synthetic-owner",
        conversation_id="synthetic-093",
    )
    watermark = summary.through_sequence if summary else 0
    text = summary.content.model_dump_json().casefold() if summary else ""
    retained = sum(fact.casefold() in text for fact in KNOWN_FACTS)
    contiguous = contiguous and final.recent_from_sequence == watermark + 1
    contiguous = contiguous and watermark + len(final.recent_messages) == 30
    retained_records = store.list_message_records(
        "synthetic-owner", "synthetic-093", limit=30
    )
    reachable = [record.sequence for record in retained_records] == list(range(1, 31))
    # One exact historical-span probe proves that original text is still
    # recoverable. A missing known fact adds a separate exact-span probe, making
    # fidelity loss visible as additional scripted page-in work, not tool reuse.
    tool_calls: list[tuple[int, int]] = [(1, 2)]
    for index, fact in enumerate(KNOWN_FACTS):
        if fact.casefold() not in text:
            tool_calls.append((index * 2 + 1, index * 2 + 1))
    for start, end in tool_calls:
        span = store.read_conversation_span(
            user_id="synthetic-owner",
            conversation_id="synthetic-093",
            from_sequence=start,
            through_sequence=end,
        )
        expected = list(range(start, end + 1))
        reachable = (
            reachable and [message.sequence for message in span.messages] == expected
        )
        reachable = reachable and all(
            message.content
            == (
                SYNTHETIC_MESSAGES[(message.sequence - 1) // 2]
                if message.sequence % 2
                else f"Continue the agreed synthetic plan, step {message.sequence // 2 - 1}."
            )
            for message in span.messages
        )
    duplicate_summary = len(metered.sequences) - len(set(metered.sequences))
    duplicates = len(tool_calls) - len(set(tool_calls))
    count = metered.successes
    return SyntheticMetrics(
        recent=recent,
        batch=batch,
        compaction_calls=count,
        compaction_attempts=len(metered.sequences),
        summary_failure_codes=tuple(metered.failures),
        summary_failure_metadata=tuple(metered.failure_metadata),
        compaction_seconds=round(metered.elapsed_seconds, 4),
        retained_known_facts=retained,
        known_fact_count=len(KNOWN_FACTS),
        page_in_calls=len(tool_calls),
        duplicate_tool_calls=duplicates,
        duplicate_summary_calls=duplicate_summary,
        watermark=watermark,
        history_reachable=reachable,
        projection_contiguous=contiguous,
        summary_available=summary is not None,
        passed=(
            not metered.failures
            and retained == len(KNOWN_FACTS)
            and reachable
            and contiguous
            and duplicates == duplicate_summary == 0
            and (count <= 2 if (recent, batch) == (16, 8) else count == 5)
        ),
    )


def recorded_trace_presence() -> dict[str, int | bool]:
    """Read only existence/counts in known local deployment database roots."""
    conversation = "conversation-bdc448be-ffb8-436e-8c86-27775b247993"
    paths = set()
    for root in (Path.home() / ".career-agent", Path.cwd() / "data"):
        if root.is_dir():
            paths.update(root.rglob("*.sqlite3"))
            paths.update(root.rglob("*.db"))
    matched = 0
    checked = 0
    unreadable = 0
    for path in sorted(paths):
        try:
            with sqlite3.connect(
                f"file:{path}?mode=ro", uri=True, timeout=2
            ) as connection:
                tables = connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='conversation_messages'"
                ).fetchall()
                if not tables:
                    continue
                checked += 1
                matched += connection.execute(
                    "SELECT COUNT(*) FROM conversation_messages WHERE conversation_id=?",
                    (conversation,),
                ).fetchone()[0]
        except sqlite3.Error:
            unreadable += 1
    return {
        "checked_transcript_databases": checked,
        "unreadable_databases": unreadable,
        "recorded_message_count": matched,
        "exact_trajectory_available": matched == 30,
    }


def probe_summary_provider(config: ConversationSummaryAgentConfig) -> dict[str, object]:
    """Three synthetic calls on the configured connection, no fallback or bodies."""
    provider = replace(
        config.provider, timeout_seconds=min(config.provider.timeout_seconds, 20.0)
    )
    endpoint = provider.endpoint.removesuffix("/chat/completions")
    messages: list[ChatCompletionMessageParam] = [
        {"role": "user", "content": 'Return only the JSON object {"ok":true}.'}
    ]
    minimal: ResponseFormatJSONSchema = {
        "type": "json_schema",
        "json_schema": {
            "name": "synthetic_probe",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {"ok": {"type": "boolean"}},
                "required": ["ok"],
                "additionalProperties": False,
            },
        },
    }
    results: list[dict[str, object]] = []
    with OpenAI(api_key=provider.api_key, base_url=endpoint, max_retries=0) as client:
        for probe in ("basic_chat", "minimal_json_schema", "summary_json_schema"):
            started = perf_counter()
            result: dict[str, object] = {"probe": probe, "passed": False}
            except_error: AgentWorkerError | None = None
            try:
                if probe == "summary_json_schema":
                    OpenAIConversationSummaryWorker(
                        provider,
                        client=client,
                        max_output_tokens=config.max_output_tokens,
                        disable_thinking=config.disable_thinking,
                    ).summarize(
                        previous=None,
                        messages=(
                            SummaryMessage(
                                sequence=1, role="user", content="Use SQLite for the synthetic plan."
                            ),
                        ),
                    )
                    result.update(status=200, passed=True)
                else:
                    response = client.chat.completions.create(
                        model=provider.model,
                        messages=messages,
                        max_tokens=256,
                        timeout=provider.timeout_seconds,
                        **(
                            {"extra_body": {"enable_thinking": False}}
                            if config.disable_thinking
                            else {}
                        ),
                        **({"response_format": minimal} if probe == "minimal_json_schema" else {}),
                    )
                    choice = response.choices[0] if response.choices else None
                    content = choice.message.content if choice else None
                    result.update(status=200, completed=bool(choice and choice.finish_reason == "stop"))
                    if choice and choice.finish_reason == "stop" and content:
                        if probe == "basic_chat":
                            result["passed"] = True
                        else:
                            try:
                                decoded = json.loads(content)
                                result["passed"] = (
                                    isinstance(decoded, dict)
                                    and set(decoded) == {"ok"}
                                    and decoded["ok"] is True
                                )
                            except ValueError:
                                pass
            except (APIConnectionError, APIStatusError) as error:
                except_error = provider_worker_error("CONVERSATION_SUMMARY", error)
            except AgentWorkerError as error:
                except_error = error
            if except_error is not None:
                result["error_code"] = except_error.code
                if except_error.provider is not None:
                    result.update(except_error.provider.as_dict())
            result["elapsed_seconds"] = round(perf_counter() - started, 4)
            results.append(result)
    return {
        "synthetic": True,
        "independent_summary_namespace": any(
            key.startswith("CONVERSATION_SUMMARY_AGENT_") for key in os.environ
        ),
        "automatic_fallback_used": False,
        "max_retries": 0,
        "timeout_seconds_per_request": provider.timeout_seconds,
        "provider_bodies_or_model_content_recorded": False,
        "results": results,
        "passed": all(item["passed"] for item in results),
    }


def main() -> int:
    from dataclasses import asdict

    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--trace-presence", action="store_true")
    parser.add_argument("--provider-probe", action="store_true")
    parser.add_argument(
        "--work-dir",
        type=Path,
        help="Required existing server scratch directory; synthetic databases are retained.",
    )
    parser.add_argument(
        "--output", type=Path, help="Write only safe aggregate JSON to this path."
    )
    args = parser.parse_args()
    logging.disable(logging.CRITICAL)
    if args.trace_presence:
        report = json.dumps(recorded_trace_presence(), sort_keys=True)
        if args.output is not None:
            args.output.write_text(report + "\n", encoding="utf-8")
        print(report)
        return 0
    if not args.provider_probe and (args.work_dir is None or not args.work_dir.is_dir()):
        parser.error("--work-dir must name an existing server scratch directory")
    try:
        worker: ConversationSummaryWorker = FactSummaryWorker()
        if args.live or args.provider_probe:
            load_dotenv()
            main_config = OpenAICompatibleAgentConfig.from_env(prefix="MAIN_AGENT")
            context_config = ContextDeploymentConfig.from_env()
            summary_config = ConversationSummaryAgentConfig.from_env(
                main_config=main_config,
                main_context_window_tokens=context_config.main_context_window_tokens,
            )
            if args.provider_probe:
                probe = probe_summary_provider(summary_config)
                report = json.dumps(probe, sort_keys=True)
                if args.output is not None:
                    args.output.write_text(report + "\n", encoding="utf-8")
                print(report)
                return 0 if probe["passed"] else 1
            worker = OpenAIConversationSummaryWorker(
                summary_config.provider,
                max_output_tokens=summary_config.max_output_tokens,
                disable_thinking=summary_config.disable_thinking,
            )
        directory = args.work_dir / f"compaction-093-synthetic-{uuid4().hex}"
        directory.mkdir()
        results = [
            run_trajectory(
                directory / f"synthetic-{recent}-{batch}.sqlite3",
                recent=recent,
                batch=batch,
                worker=worker,
            )
            for recent, batch in ((8, 4), (16, 8))
        ]
        report = json.dumps(
            {
                "synthetic": True,
                "live_summary_model": args.live,
                "exact_private_trajectory_replay": False,
                "faithfulness_measurement": "four_known_fact_keywords_not_semantic_equivalence",
                "occupancy_measurement": "fixed_16000_of_32000_complete_request_estimator_plus_reply",
                "actual_provider_context_window_measured": False,
                "tool_measurement": "scripted_exact_history_probes_not_main_agent_decisions",
                "openai_version": version("openai"),
                "schema_bytes": len(json.dumps(summary_response_format()).encode()),
                "results": [asdict(item) for item in results],
                "passed": all(item.passed for item in results),
            },
            sort_keys=True,
        )
        if args.output is not None:
            args.output.write_text(report + "\n", encoding="utf-8")
        print(report)
        return 0 if all(item.passed for item in results) else 1
    except AgentWorkerError as error:
        print(json.dumps({"passed": False, "code": error.code}, sort_keys=True))
        return 1
    except Exception as error:
        print(
            json.dumps({"passed": False, "type": type(error).__name__}, sort_keys=True)
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
