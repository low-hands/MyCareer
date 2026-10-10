"""Repeat the predefined Qwen batch, with safe error diagnostics and progress.

Observers only; production decisions, contracts and grading are unchanged.
"""
from collections import Counter
from pathlib import Path
import json
from threading import Lock
from career_agent.evaluation import runtime_quality as suite
from career_agent.agent.providers.main_agent import OpenAICompatibleMainAgentDecisionMaker
from career_agent.agent.providers.openai_client import provider_error_metadata

root = Path(__file__).resolve().parent / 'qwen_retry_2026-10-09'
preflight = json.loads((root / 'preflight_report.json').read_text())
if any(row['errors'] for row in preflight):
    raise SystemExit('Preflight has provider errors; batch was not started.')
if (root / 'recordings').exists():
    raise SystemExit('Recordings already exist; do not overwrite a previous batch.')
lock = Lock()
errors = []
completed = 0
sample_errors = Counter()
original_request = OpenAICompatibleMainAgentDecisionMaker._request
original_sample = suite._live_sample


def observed_request(self, **kwargs):
    try:
        return original_request(self, **kwargs)
    except Exception as error:
        cause = error.__cause__ or error
        metadata = provider_error_metadata(cause)
        body = getattr(cause, 'body', None)
        fields = body.get('error', body) if isinstance(body, dict) else {}
        message = str(fields.get('message', '')) if isinstance(fields, dict) else str(body or '')
        lower = message.lower()
        record = {
            'worker_code': getattr(error, 'code', None),
            'provider': metadata.as_dict() if metadata else None,
            'body_kind': type(body).__name__,
            'hints': {label: any(word in lower for word in words) for label, words in {
                'quota': ('quota', 'balance', 'credit', '余额', '额度', '欠费'),
                'rate': ('rate limit', 'too many', 'throttl'),
                'auth': ('api key', 'unauthorized', 'authentication', '鉴权'),
                'gateway': ('cisco', 'umbrella', 'opendns', 'cloudflare', 'firewall'),
                'schema': ('schema', 'parameter', 'invalid request'),
            }.items()},
        }
        with lock:
            errors.append(record)
        raise


def observed_sample(*args, **kwargs):
    global completed
    result = original_sample(*args, **kwargs)
    with lock:
        completed += 1
        sample_errors.update(t['error'] for t in result['turns'] if t['error'])
        if completed % 10 == 0:
            print(json.dumps({'completed': completed, 'total': 190,
                              'sample_errors': dict(sample_errors)}, ensure_ascii=False), flush=True)
    return result


OpenAICompatibleMainAgentDecisionMaker._request = observed_request
suite._live_sample = observed_sample
try:
    status = suite.main([
        '--record', '--env-prefix', 'RESUME_ANALYSIS_AGENT', '--vendor', 'qwen',
        '--root', str(root / 'recordings'), '--output', str(root / 'runtime_report.json'),
        '--k', '10', '--jobs', '2',
    ])
finally:
    (root / 'request_diagnostics.json').write_text(json.dumps({
        'completed_live_samples': completed, 'sample_errors': dict(sample_errors),
        'provider_request_errors': errors,
        'measurement_note': 'Diagnostics contain no raw bodies, messages, credentials or headers. Observers do not alter decisions or grading.',
    }, ensure_ascii=False, indent=2) + '\n')
raise SystemExit(status)
