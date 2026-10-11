"""Phase baseline plus post-tool prose/clarification time, without model calls.

The core latency report keeps strict final decisions separate. This view also
counts ask_user: the production model often uses it for a complete answer with
an optional follow-up question. Neither category establishes task success.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from career_agent.evaluation.latency import _stats, _utc, distribution, markdown, phase, read_runs, summarize


def analyze(path: Path, **selection) -> dict:
    report = summarize(path, **selection)
    selected = {r['run_id'] for r in report['post_tool_reply']['runs']}
    runs = {key: events for key, events in read_runs(path).items() if key in selected}
    by_action = {name: [] for name in ('final', 'ask_user')}
    rows = []
    for run_id, events in runs.items():
        tool_seen = False
        replies = []
        for event in events:
            tool_seen |= event['event_type'] == 'capability_executed'
            action = event['details'].get('decision_action')
            if tool_seen and phase(event) == 'decision' and event['event_type'] == 'model_succeeded' and action in by_action:
                by_action[action].append(event)
                replies.append(event)
        span = (max(_utc(e['occurred_at']) for e in events) - min(_utc(e['occurred_at']) for e in events)).total_seconds() * 1000
        completed = any(e['event_type'] == 'turn_completed' for e in events)
        decisions = [e for e in events if phase(e) == 'decision']
        terminals = [e for e in decisions if e['event_type'] in {'model_succeeded', 'model_failed'}]
        complete_coverage = (sum(e['event_type'] == 'model_attempt' for e in decisions) == len(terminals)
                             and bool(terminals)
                             and all(e['details'].get('decision_action') is not None for e in terminals))
        if replies:
            reply_ms = sum(e['duration_ms'] for e in replies) if all(e['duration_ms'] is not None for e in replies) else None
        else:
            reply_ms = 0 if tool_seen and completed and complete_coverage else None
        share = reply_ms / span if completed and reply_ms is not None and 0 < span and reply_ms <= span else None
        action_ms = {action: (sum(e['duration_ms'] for e in replies if e['details']['decision_action'] == action) if reply_ms is not None else None) for action in by_action}
        rows.append({'action_ms': action_ms, 'run_id': run_id, 'completed': completed, 'has_tool_execution': tool_seen, 'response_calls': len(replies), 'response_ms': reply_ms, 'trace_span_ms': span, 'share': share})
    eligible = [r for r in rows if r['share'] is not None]
    total_events = by_action['final'] + by_action['ask_user']
    report['post_tool_response_message'] = {
        'by_action': {key: _stats(events, list(runs)) for key, events in by_action.items()},
        'combined': _stats(total_events, list(runs)),
        'per_completed_run_share': distribution([r['share'] for r in eligible]),
        'share_eligible_runs': len(eligible),
        'completed_runs_without_tool_trace': len([r for r in rows if r['completed'] and not r['has_tool_execution']]),
        'runs_with_reply': len([r for r in rows if r['response_calls']]),
        'completed_runs_without_reply': len([r for r in rows if r['completed'] and r['response_calls'] == 0 and r['response_ms'] == 0]),
        'weighted_share_among_reply_runs': (sum(r['response_ms'] for r in eligible if r['response_calls']) / sum(r['trace_span_ms'] for r in eligible if r['response_calls'])) if any(r['response_calls'] for r in eligible) else None,
        'weighted_share': sum(r['response_ms'] for r in eligible) / sum(r['trace_span_ms'] for r in eligible) if eligible else None,
        'runs': rows,
        'semantics': 'Successful final or ask_user after an executed tool. ask_user can be an answer with a follow-up or a needed clarification; it is not necessarily successful task completion. Further tool selection, questionnaire and failed decisions are separate. Shares include only completed runs with known response timings; denominator is trace span, not first-card or end-to-end user latency. Completed turns with fully paired decision telemetry and no response call contribute a true zero. Incomplete or unknown telemetry is not imputed as zero; conditional shares among reply runs are also reported.'}
    return report


def attach_capture_wall(report: dict, database: Path, manifest: Path) -> None:
    """Join sequential authored captures by conversation and chronological turn.

    Reject ambiguous/incomplete joins rather than assigning another turn's wall
    time. The production trace carries conversation IDs but not fixture indices.
    """
    from collections import defaultdict
    capture = json.loads(manifest.read_text())
    if not capture.get('finished_at'):
        raise ValueError('capture must be finished before joining wall times')
    by_conversation = defaultdict(list)
    events_by_id = read_runs(database)
    for run_id, events in events_by_id.items():
        conversation_ids = {e['details']['conversation_id'] for e in events if 'conversation_id' in e['details']}
        if len(conversation_ids) != 1:
            raise ValueError('run has missing or ambiguous conversation identity')
        by_conversation[next(iter(conversation_ids))].append((run_id, events))
    attempted = defaultdict(list)
    for turn in capture['turns']:
        attempted[f"latency-{turn['conversation']}"].append(turn)
    if set(attempted) != set(by_conversation):
        raise ValueError('capture/trace conversations differ')
    wall_by_id = {}
    for conversation, runs in by_conversation.items():
        runs.sort(key=lambda pair: min(_utc(e['occurred_at']) for e in pair[1]))
        turns = sorted(attempted[conversation], key=lambda turn: turn['turn'])
        if len(runs) != len(turns):
            raise ValueError('capture/trace turn counts differ')
        for previous, following in zip(runs, runs[1:]):
            if max(_utc(e['occurred_at']) for e in previous[1]) >= min(_utc(e['occurred_at']) for e in following[1]):
                raise ValueError('capture turns overlap; sequential join is invalid')
        for (run_id, _), turn in zip(runs, turns):
            wall_by_id[run_id] = turn
    rows = []
    for row in report['post_tool_response_message']['runs']:
        turn = wall_by_id[row['run_id']]
        response_ms = row['response_ms']
        if response_ms is None and not row['has_tool_execution']:
            events = events_by_id[row['run_id']]
            terminal = next((e for e in events if e['event_type'] == 'turn_completed'), None)
            decisions = [e for e in events if phase(e) == 'decision']
            ends = [e for e in decisions if e['event_type'] in {'model_succeeded', 'model_failed'}]
            paired = sum(e['event_type'] == 'model_attempt' for e in decisions) == len(ends)
            # A completed runtime refusal has neither execution nor a post-tool
            # writer. This is a proven zero, not missing legacy tool telemetry.
            if (terminal and terminal['details'].get('tool_call_count') == 0
                    and terminal['details'].get('final_state') == 'model:tool_call'
                    and paired and ends and ends[-1]['details'].get('decision_action') == 'tool_call'):
                response_ms = 0
        action_ms = row['action_ms'] if row['response_ms'] is not None else {action: (0 if response_ms == 0 else None) for action in ('final', 'ask_user')}
        known = row['completed'] and turn['status'] == 'returned' and response_ms is not None and 0 <= response_ms <= turn['wall_ms']
        rows.append({**row, 'action_ms': action_ms, 'conversation': turn['conversation'], 'turn': turn['turn'], 'wall_ms': turn['wall_ms'], 'response_ms': response_ms,
                     'wall_share': response_ms / turn['wall_ms'] if known and turn['wall_ms'] else None})
    eligible = [r for r in rows if r['wall_share'] is not None]
    reply_runs = [r for r in eligible if r['response_calls']]
    report['capture_wall_time'] = {
        'manifest': str(manifest), 'attempted_turns': len(capture['turns']),
        'wall_ms': distribution([t['wall_ms'] for t in capture['turns']]),
        'share_eligible_turns': len(eligible), 'turns_with_reply': len(reply_runs),
        'known_zero_reply_turns': sum(r['response_ms'] == 0 for r in eligible),
        'per_turn_reply_share': distribution([r['wall_share'] for r in eligible]),
        'weighted_reply_share': sum(r['response_ms'] for r in eligible) / sum(r['wall_ms'] for r in eligible) if eligible else None,
        'weighted_reply_share_among_reply_turns': sum(r['response_ms'] for r in reply_runs) / sum(r['wall_ms'] for r in reply_runs) if reply_runs else None,
        'by_action': {action: {
            'total_ms': sum(r['action_ms'][action] for r in eligible),
            'mean_ms_per_eligible_turn': sum(r['action_ms'][action] for r in eligible) / len(eligible) if eligible else None,
            'weighted_wall_share': sum(r['action_ms'][action] for r in eligible) / sum(r['wall_ms'] for r in eligible) if eligible else None,
        } for action in ('final', 'ask_user')},
        'runs': rows,
        'semantics': 'Denominator is capture run_turn() wall time, including context load/commit. Only exact sequential conversation joins are accepted. Runtime completion is not task-success grading. Failed/unknown timing is excluded from shares; proven no-writer runtime refusals are retained as zero.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('database', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--markdown', type=Path, required=True)
    parser.add_argument('--min-context-chars', type=int, default=0)
    parser.add_argument('--require-tool-trace', action='store_true')
    parser.add_argument('--capture-manifest', type=Path)
    args = parser.parse_args()
    if args.min_context_chars < 0:
        parser.error('min-context-chars must be nonnegative')
    report = analyze(args.database, min_context_chars=args.min_context_chars, require_tool_trace=args.require_tool_trace)
    if args.capture_manifest:
        attach_capture_wall(report, args.database, args.capture_manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    tail = report['post_tool_response_message']
    text = markdown(report) + '\n## 工具后回复正文与澄清\n\n'
    text += '| 模型动作 | 次数 | p50 ms | p95 ms |\n|---|---:|---:|---:|\n'
    for action, stats in tail['by_action'].items():
        times = stats['operation_duration_ms']
        text += f"| {action} | {stats['completed_count']} | {times['p50']} | {times['p95']} |\n"
    text += '\n以下合计仅表示 final + ask_user，不能称为最终总结或用于推断纯写回复的优化上限。\n'
    text += '\n合计耗时（ms）：' + json.dumps(tail['combined']['operation_duration_ms']) + '\n'
    text += '\n逐轮占比：' + json.dumps(tail['per_completed_run_share']) + '\n'
    text += '\n时间加权占比：' + str(tail['weighted_share']) + '\n\n' + tail['semantics'] + '\n'
    if 'capture_wall_time' in report:
        wall = report['capture_wall_time']
        text += '\n## 实测每轮 wall time 的回复占比\n\n' + json.dumps({key: value for key, value in wall.items() if key != 'runs'}, ensure_ascii=False, indent=2) + '\n'
    args.markdown.write_text(text)
    print(json.dumps({'runs': report['run_count'], 'response': tail['combined']['operation_duration_ms'], 'trace_span_weighted_share': tail['weighted_share'], 'wall_weighted_share': report.get('capture_wall_time', {}).get('weighted_reply_share')}))


if __name__ == '__main__':
    main()
