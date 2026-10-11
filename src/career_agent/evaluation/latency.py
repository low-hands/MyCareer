"""Read-only phase latency summaries from production run-events telemetry."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sqlite3
from typing import Any

from career_agent.storage.sqlite import open_connection

PHASES = ("decision", "specialist_worker", "embedding", "tool_execution", "summary_compaction")


def distribution(values: list[int | float]) -> dict[str, Any]:
    """Nearest-rank percentiles, including real zeroes; absent data stays null."""
    ordered = sorted(values)
    def percentile(p):
        return ordered[max(0, math.ceil(len(ordered) * p) - 1)] if ordered else None
    return {"samples": len(ordered), "total": round(sum(ordered), 3),
            "mean": round(sum(ordered) / len(ordered), 3) if ordered else None,
            "p50": percentile(.5), "p95": percentile(.95),
            "max": ordered[-1] if ordered else None}


def phase(event: dict[str, Any]) -> str | None:
    kind, stage = event["event_type"], event["stage"]
    category = event.get("model_call_category")
    if kind in {"model_succeeded", "model_failed", "model_attempt"}:
        if category == "orchestrator_decision" or (category is None and stage == "main_agent_decide"):
            return "decision"
        if category is not None:
            return "specialist_worker"
    if kind == "capability_executed":
        return "tool_execution"
    if kind == "provider_request":
        if stage == "embedding":
            return "embedding"
        if stage == "conversation_summary":
            return "summary_compaction"
    return None


def _utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timestamps must include a UTC offset")
    return parsed.astimezone(timezone.utc)


def read_runs(path: Path) -> dict[str, list[dict[str, Any]]]:
    # mode=ro cannot create a missing file, migrate schema or touch permissions.
    with open_connection(path.expanduser().resolve().as_uri() + "?mode=ro", uri=True) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute("SELECT * FROM run_events ORDER BY run_id, sequence")
        runs: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            event = dict(row)
            event["details"] = json.loads(event.pop("details_json"))
            runs[event["run_id"]].append(event)
    return dict(runs)


def _stats(events: list[dict[str, Any]], run_ids: list[str]) -> dict[str, Any]:
    starts = [e for e in events if e["event_type"] == "model_attempt"]
    terminals = [e for e in events if e["event_type"] != "model_attempt"]
    timed = [e for e in terminals if e["duration_ms"] is not None]
    totals = Counter()
    counts = Counter(e["run_id"] for e in terminals)
    for event in timed:
        totals[event["run_id"]] += event["duration_ms"]
    # A run containing untimed events has an unknown phase total, not zero.
    untimed_runs = {e["run_id"] for e in terminals if e["duration_ms"] is None}
    start_counts = Counter((e["run_id"], e["stage"]) for e in starts)
    end_counts = Counter((e["run_id"], e["stage"]) for e in terminals)
    unfinished = start_counts - end_counts
    untimed_runs.update(run_id for run_id, _ in unfinished)
    return {"started_count": len(starts), "completed_count": len(terminals),
            "unfinished_model_calls": sum(unfinished.values()),
            "outcomes": dict(Counter(e["outcome"] for e in terminals)),
            "missing_duration_count": len(terminals) - len(timed),
            "runs_with_events": len({e["run_id"] for e in events}),
            "operation_duration_ms": distribution([e["duration_ms"] for e in timed]),
            "per_run_count": distribution([counts[r] for r in run_ids]),
            "per_run_started_count": distribution([sum(e["run_id"] == r for e in starts) for r in run_ids]) if starts else None,
            "per_run_duration_ms": distribution([totals[r] for r in run_ids if r not in untimed_runs] if terminals else []),
            "successful_duration_ms": distribution([e["duration_ms"] for e in timed if e["outcome"] == "succeeded"]),
            "failed_duration_ms": distribution([e["duration_ms"] for e in timed if e["outcome"] == "failed"])}


def post_tool_reply(runs: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    finals, clarifications, continued, failures, other = [], [], [], [], []
    rows = []
    unknown_actions = 0
    for run_id, events in runs.items():
        last_tool = None
        reply_events = []
        for event in events:
            if event["event_type"] == "capability_executed":
                last_tool = event
            if (last_tool is None or phase(event) != "decision"
                    or event["event_type"] not in {"model_succeeded", "model_failed"}):
                continue
            action = event["details"].get("decision_action")
            if event["event_type"] == "model_failed":
                failures.append(event)
            elif action == "final":
                finals.append(event)
                reply_events.append(event)
            elif action == "ask_user":
                clarifications.append(event)
            elif action == "tool_call":
                continued.append(event)
            else:
                other.append(event)
                if action is None:
                    unknown_actions += 1
        completed = any(e["event_type"] == "turn_completed" for e in events)
        span = (max(_utc(e["occurred_at"]) for e in events) - min(_utc(e["occurred_at"]) for e in events)).total_seconds() * 1000
        known = bool(reply_events) and all(e["duration_ms"] is not None for e in reply_events)
        reply_ms = sum(e["duration_ms"] for e in reply_events) if known else None
        fraction = reply_ms / span if completed and known and span > 0 and reply_ms <= span else None
        rows.append({"run_id": run_id, "completed": completed,
                     "post_tool_final_calls": len(reply_events), "reply_ms": reply_ms,
                     "observed_span_ms": round(span, 3),
                     "reply_share_of_observed_span": fraction})
    shares = [r["reply_share_of_observed_span"] for r in rows if r["reply_share_of_observed_span"] is not None]
    eligible = [r for r in rows if r["reply_share_of_observed_span"] is not None]
    return {"final_reply": _stats(finals, list(runs)),
            "ask_user_reply": _stats(clarifications, list(runs)),
            "continued_tool_selection": _stats(continued, list(runs)),
            "failed_post_tool_decision": _stats(failures, list(runs)),
            "other_post_tool_decision": _stats(other, list(runs)),
            "unknown_action_count": unknown_actions,
            "per_completed_run_reply_share": distribution(shares),
            "weighted_reply_share": (sum(r["reply_ms"] for r in eligible) / sum(r["observed_span_ms"] for r in eligible)) if eligible else None,
            "runs": rows,
            "semantics": "Only successful decision_action=final after an executed tool is reply writing. ask_user/questionnaire and further tool selection are separate. Shares use completed runs with known reply timing and the observed trace span, not user-perceived latency; failed runs and absent timings are excluded from shares, retained in counts."}


def summarize(path: Path, *, since: str | None = None, until: str | None = None,
              min_context_chars: int = 0, require_tool_trace: bool = False) -> dict[str, Any]:
    runs = read_runs(path)
    lower, upper = _utc(since) if since else None, _utc(until) if until else None
    if lower and upper and lower >= upper:
        raise ValueError("since must precede until")
    selected, contexts = {}, {}
    for run_id, events in runs.items():
        start = min(_utc(e["occurred_at"]) for e in events)
        decisions = [e for e in events if phase(e) == "decision"]
        context = next((e["details"].get("context_chars") for e in decisions
                        if e["details"].get("context_chars") is not None), None)
        if lower and start < lower or upper and start >= upper:
            continue
        if min_context_chars and (context is None or context < min_context_chars):
            continue
        if require_tool_trace and not any(e["event_type"] == "capability_executed" for e in events):
            continue
        selected[run_id] = events
        contexts[run_id] = context
    events = [e for group in selected.values() for e in group]
    ids = list(selected)
    by_phase = {name: [e for e in events if phase(e) == name] for name in PHASES}
    # Older traces only recorded compaction outcomes. Use them for counts and
    # missing-coverage reporting, never alongside a timed summary operation.
    for group in selected.values():
        if not any(phase(e) == "summary_compaction" for e in group):
            by_phase["summary_compaction"].extend(e for e in group if e["event_type"] in
                                                 {"context_compacted", "context_compaction_failed"})
    phases = {name: _stats(group, ids) for name, group in by_phase.items()}
    phases["summary_compaction"]["execution_mode"] = {
        label: _stats([e for e in by_phase["summary_compaction"]
            if bool(e["details"].get("background")) == background], ids)
        for label, background in (("foreground_or_legacy", False), ("background", True))
    }
    decision_requests = Counter()
    request_count_unknown = 0
    for event in by_phase["decision"]:
        if event["event_type"] == "model_attempt":
            continue
        attempts = event["details"].get("attempt_count")
        if isinstance(attempts, int) and attempts > 0:
            decision_requests[event["run_id"]] += attempts
        else:
            request_count_unknown += 1
    phases["decision"]["provider_request_count"] = sum(decision_requests.values()) if request_count_unknown == 0 else None
    phases["decision"]["unknown_request_count_decisions"] = request_count_unknown
    phases["decision"]["per_run_provider_request_count"] = distribution([decision_requests[r] for r in ids]) if request_count_unknown == 0 else None
    for name in ("specialist_worker", "tool_execution"):
        groups = defaultdict(list)
        for event in by_phase[name]:
            label = (event["details"].get("tool_name", "unknown") if name == "tool_execution"
                     else event["stage"])
            groups[label].append(event)
        phases[name]["breakdown"] = {label: _stats(group, ids) for label, group in sorted(groups.items())}
    cohorts = {}
    for label, low, high in (("short_lt_4000", 0, 4000), ("medium_4000_11999", 4000, 12000),
                             ("long_gte_12000", 12000, math.inf)):
        cohort_ids = [r for r, c in contexts.items() if c is not None and low <= c < high]
        cohorts[label] = {"runs": len(cohort_ids),
                          "decision": _stats([e for e in by_phase["decision"] if e["run_id"] in cohort_ids], cohort_ids)}
    warnings = []
    if not selected:
        warnings.append("No runs match the selection.")
    if not by_phase["tool_execution"]:
        warnings.append("No capability_executed events: tool execution is unobserved.")
    for name in PHASES:
        if not by_phase[name]:
            warnings.append(f"{name}: no observed operations; this does not establish zero latency.")
        elif phases[name]["missing_duration_count"]:
            warnings.append(f"{name}: some operations have no duration; totals are incomplete.")
        if phases[name]["unfinished_model_calls"]:
            warnings.append(f"{name}: model starts without terminal events; their latency is unknown.")
        samples = phases[name]["operation_duration_ms"]["samples"]
        if 0 < samples < 20:
            warnings.append(f"{name}: only {samples} timed operations; p95 is a descriptive small-sample statistic.")
        if samples and not phases[name]["successful_duration_ms"]["samples"]:
            warnings.append(f"{name}: no successful timed operations; a successful-operation baseline is unavailable.")
    return {"version": 2, "generated_at": datetime.now(timezone.utc).isoformat(),
            "selection": {"since": since, "until": until, "min_context_chars": min_context_chars,
                          "require_tool_trace": require_tool_trace},
            "run_count": len(ids), "excluded_run_count": len(runs) - len(ids), "event_count": len(events),
            "time_range": {"first": min((e["occurred_at"] for e in events), default=None),
                           "last": max((e["occurred_at"] for e in events), default=None)},
            "terminal_events": dict(Counter(e["event_type"] for e in events if e["event_type"] in
                                            {"turn_completed", "turn_failed", "run_interrupted", "turn_rejected"})),
            "first_decision_context_chars": distribution([c for c in contexts.values() if c is not None]),
            "unknown_context_runs": sum(c is None for c in contexts.values()),
            "observed_trace_span_ms": distribution([
                round((max(_utc(e["occurred_at"]) for e in group) - min(_utc(e["occurred_at"]) for e in group)).total_seconds() * 1000)
                for group in selected.values()]),
            "post_tool_reply": post_tool_reply(selected),
            "phases": phases, "context_cohorts": cohorts, "warnings": warnings,
            "semantics": ["Durations include failures and internal retries; model_retry is not an extra decision.",
                          "Decision counts are logical decide() calls; provider_request_count uses attempt_count to include internal retries when available.",
                          "Tool execution is inclusive of nested workers/embedding; phase totals must not be added.",
                          "Embedding measures one embed() operation including its batches; startup warmup outside a turn is excluded.",
                          "Summary timing covers summarize() including validation, not the entire context load/commit.",
                          "Observed trace span excludes time before the first event; it is not end-to-end response latency.",
                          "Per-run counts include zero-use runs; duration totals exclude runs with missing phase timings.",
                          "Context cohorts use the first decision context_chars, not the full serialized request token count."]}


def markdown(report: dict[str, Any]) -> str:
    def number(value):
        return "—" if value is None else str(round(value, 2))
    lines = ["# 分阶段延迟基线", "", f"运行数：{report['run_count']}；事件数：{report['event_count']}。",
             f"时间范围（UTC）：{report['time_range']['first']} 至 {report['time_range']['last']}。", "",
             "| 阶段 | 模型开始次数 | 完成次数 | 失败 | 缺失计时 | 单次 p50 ms | 单次 p95 ms | 每轮次数均值 | 每轮耗时均值 ms | 每轮 p95 ms |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, stats in report["phases"].items():
        lines.append(f"| {name} | {stats['started_count'] if name in {'decision', 'specialist_worker'} else '—'} | {stats['completed_count']} | {stats['outcomes'].get('failed', 0)} | {stats['missing_duration_count']} | {number(stats['operation_duration_ms']['p50'])} | {number(stats['operation_duration_ms']['p95'])} | {number(stats['per_run_count']['mean'])} | {number(stats['per_run_duration_ms']['mean']) if stats['completed_count'] else '—'} | {number(stats['per_run_duration_ms']['p95'])} |")
    lines += ["", "首个决策上下文字符数：" + json.dumps(report["first_decision_context_chars"], ensure_ascii=False), ""]
    lines += [f"决策模型请求数（包含内部重试）：{number(report['phases']['decision']['provider_request_count'])}。", ""]
    modes = report["phases"]["summary_compaction"].get("execution_mode", {})
    if modes:
        lines += ["摘要执行方式（后台耗时不能计作当前轮等待）：", "", "| 方式 | 次数 | 总耗时 ms |", "|---|---:|---:|"]
        for label, stats in modes.items():
            lines.append(f"| {label} | {stats['completed_count']} | {number(stats['operation_duration_ms']['total'])} |")
        lines += ["", "后台事件可能晚于 turn_completed；trace 跨度不能替代整轮 wall time。", ""]
    tail = report["post_tool_reply"]
    lines += ["## 工具后模型动作（final 与 ask_user 分开）", "",
              "ask_user 次数及耗时（ms）：" + json.dumps(tail["ask_user_reply"]["operation_duration_ms"], ensure_ascii=False),
              "最终回复模型耗时（ms）：" + json.dumps(tail["final_reply"]["operation_duration_ms"], ensure_ascii=False),
              "完成轮次中的耗时占 trace 跨度比例：" + json.dumps(tail["per_completed_run_reply_share"], ensure_ascii=False),
              "按时间加权的占比：" + number(tail["weighted_reply_share"]),
              "", tail["semantics"], ""]
    for name in ("specialist_worker", "tool_execution"):
        lines += [f"## {name}", "", "| 名称 | 次数 | p50 ms | p95 ms |", "|---|---:|---:|---:|"]
        for label, stats in report["phases"][name]["breakdown"].items():
            lines.append(f"| {label} | {stats['completed_count']} | {number(stats['operation_duration_ms']['p50'])} | {number(stats['operation_duration_ms']['p95'])} |")
        lines.append("")
    lines += ["## 测量口径与缺口", ""] + [f"- {note}" for note in report["semantics"] + report["warnings"]]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path)
    parser.add_argument("--since", help="Inclusive run start, ISO 8601 with offset")
    parser.add_argument("--until", help="Exclusive run start, ISO 8601 with offset")
    parser.add_argument("--min-context-chars", type=int, default=0)
    parser.add_argument("--require-tool-trace", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args(argv)
    if args.min_context_chars < 0:
        parser.error("min-context-chars must be nonnegative")
    try:
        report = summarize(args.database, since=args.since, until=args.until,
                           min_context_chars=args.min_context_chars, require_tool_trace=args.require_tool_trace)
    except (OSError, sqlite3.Error, ValueError) as error:
        parser.error(str(error))
    content = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content, encoding="utf-8")
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(markdown(report), encoding="utf-8")
    print(content if args.output is None else json.dumps({"runs": report["run_count"], "output": str(args.output), "warnings": report["warnings"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
