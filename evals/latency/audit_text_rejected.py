"""Read-only rejection audit; never calls a provider or edits recordings."""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sqlite3


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--cassette-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output exists; preserve earlier audits')
    groups = defaultdict(lambda: {'decisions': 0, 'duration_ms': 0, 'provider_requests': 0, 'text_rejection_events': 0})
    baseline_raw = 0
    with sqlite3.connect(f'{args.database.resolve().as_uri()}?mode=ro', uri=True) as db:
        for duration, encoded in db.execute("SELECT duration_ms, details_json FROM run_events WHERE event_type='model_succeeded' AND stage='main_agent_decide'"):
            if duration is None:
                raise ValueError('Missing decision timing; cannot treat unknown as zero')
            details = json.loads(encoded)
            events = [e for e in details.get('decision_retry_events', []) if e.get('reason') == 'text_rejected']
            baseline_raw += sum(bool(e.get('raw_output')) for e in events)
            key = (details.get('decision_action', 'unknown'), bool(events))
            g = groups[key]
            g['decisions'] += 1
            g['duration_ms'] += duration
            g['provider_requests'] += details.get('attempt_count', 1)
            g['text_rejection_events'] += len(events)
    manifest = json.loads(args.manifest.read_text())
    if not manifest.get('finished_at'):
        raise ValueError('Capture has not finished')
    # Capture rows store wall times, including context loading and committing.
    rows = manifest.get('turns', [])
    if not rows:
        rows = [r for conv in manifest.get('conversations', []) for r in conv.get('turns', [])]
    wall_ms = sum(r['wall_ms'] for r in rows)
    result = {'capture_wall_ms': wall_ms, 'baseline_rejected_bodies_available': baseline_raw,
              'groups': [{ 'action': action, 'has_text_rejection': rejected, **g,
                           'mean_ms': round(g['duration_ms'] / g['decisions'], 3),
                           'share_capture_wall': g['duration_ms'] / wall_ms }
                         for (action, rejected), g in sorted(groups.items())],
              'cassette_examples': []}
    for path in sorted(args.cassette_root.glob('*.json')):
        cassette = json.loads(path.read_text())
        samples = cassette.get('samples') or [{'steps': cassette.get('steps', [])}]
        for sample_index, sample in enumerate(samples):
            for step_index, step in enumerate(sample['steps']):
                for event in step.get('decision_retry_events', []):
                    raw = event.get('raw_output', {})
                    if event.get('reason') == 'text_rejected' and raw.get('content'):
                        result['cassette_examples'].append({
                            'path': str(path), 'sample': sample_index + 1, 'step': step_index,
                            'accepted_tool_call': step.get('tool_call'), 'rejected_text': raw['content']})
    result['limits'] = [
        'Duration is for the entire logical decision, including useful output and retries; not measured wasted time.',
        'Baseline trace stores rejection reasons, not rejected bodies or separate per-attempt durations.',
        'Cassette bodies come from different authored evaluation scenarios, not the 24-turn latency capture.',
        'No automatic quality grade: inspect source scenario and observations before calling text usable.'
    ]
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'cassette_examples'}, ensure_ascii=False, indent=2))
    print('cassette_text_samples:', len(result['cassette_examples']))

if __name__ == '__main__':
    main()
