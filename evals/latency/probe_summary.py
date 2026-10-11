"""Fixed summary calls over the first batch of the frozen long fixture."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter

from dotenv import load_dotenv
from career_agent.agent.context.deployment import ConversationSummaryAgentConfig
from career_agent.agent.contracts.memory import SummaryMessage, ConversationSummaryDelta
from career_agent.agent.providers.openai_client import OpenAICompatibleAgentConfig
from career_agent.agent.providers.conversation_summary import OpenAIConversationSummaryWorker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=5)
    parser.add_argument('--diagnostic', action='store_true')
    parser.add_argument('--cumulative-fixture', type=Path, help='Frozen rounds; samples are dependent consecutive compactions')
    args = parser.parse_args()
    if not 1 <= args.samples <= 10:
        parser.error('samples must be between 1 and 10')
    if args.output.exists():
        parser.error('output must not exist; preserve every attempt')
    root = Path(__file__).resolve().parents[2]
    load_dotenv(root / '.env')
    for suffix in ('BASE_URL', 'API_KEY', 'MODEL'):
        os.environ['MAIN_AGENT_' + suffix] = os.environ['RESUME_ANALYSIS_AGENT_' + suffix]
    main_config = OpenAICompatibleAgentConfig.from_env(prefix='MAIN_AGENT')
    config = ConversationSummaryAgentConfig.from_env(main_config=main_config)
    fixture = root / 'evals/latency/scenarios.json'
    data = json.loads(fixture.read_text())
    # Summary requests cap each source message at 4000 characters. The first
    # compaction uses the oldest eight messages, preserving original sequences.
    messages = tuple(SummaryMessage(sequence=i + 1, role=role, content=text[:4000])
                     for i, (role, text) in enumerate(
                         (role, text.format(resume=data['resume'], jd=data['jd']))
                         for pair in data['history'][:4]
                         for role, text in zip(('user', 'assistant'), pair)))
    worker = OpenAIConversationSummaryWorker(config.provider, max_output_tokens=config.max_output_tokens,
                                              disable_thinking=config.disable_thinking)
    report = {'fixed_samples': args.samples, 'fixture_sha256': hashlib.sha256(fixture.read_bytes()).hexdigest(),
              'input_sha256': hashlib.sha256(json.dumps([m.model_dump() for m in messages], sort_keys=True).encode()).hexdigest(),
              'summary_prompt_sha256': hashlib.sha256(worker._budgeted_system_prompt(config.max_output_tokens).encode()).hexdigest(),
              'source_characters': sum(len(m.content) for m in messages),
              'model': config.provider.model, 'disable_thinking': config.disable_thinking,
              'timeout_seconds': config.provider.timeout_seconds, 'max_output_tokens': config.max_output_tokens,
              'attempts': []}
    if args.cumulative_fixture:
        parser.error('Legacy rolling probe is historical; use probe_incremental_summary.py for ledger-backed accumulation')
    diagnostics = []
    if args.diagnostic:
        from types import SimpleNamespace
        from career_agent.agent.contracts.memory import ConversationSummaryContent, HARNESS_SUMMARY_COUNTER_FIELDS
        client = worker._client
        def create(**kwargs):
            response = client.chat.completions.create(**kwargs)
            choice = response.choices[0] if response.choices else None
            usage = getattr(response, 'usage', None)
            diagnostic = {'finish_reason': choice.finish_reason if choice else None,
                          'completion_tokens': getattr(usage, 'completion_tokens', None),
                          'reasoning_tokens': getattr(getattr(usage, 'completion_tokens_details', None), 'reasoning_tokens', None),
                          'content_characters': len(choice.message.content or '') if choice else 0}
            try:
                decoded = json.loads(choice.message.content or '')
                required = set(ConversationSummaryDelta.model_fields)
                diagnostic['missing_fields'] = sorted(required - set(decoded))
                diagnostic['extra_field_count'] = len(set(decoded) - required)
                diagnostic['overlong_fact_count'] = sum(isinstance(f, dict) and isinstance(f.get('fact'), str) and len(f['fact']) > 200 for f in decoded.get('new_facts', []))
                try:
                    ConversationSummaryDelta.model_validate(decoded)
                    diagnostic['raw_pydantic_valid'] = True
                except Exception as error:
                    diagnostic['raw_pydantic_valid'] = False
                    diagnostic['validation_errors'] = [{'location': e['loc'], 'type': e['type']} for e in error.errors()]
            except (ValueError, TypeError, AttributeError):
                diagnostic['invalid_json'] = True
            diagnostics.append(diagnostic)
            return response
        worker._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for index in range(report['fixed_samples']):
        start = perf_counter()
        row = {'sample': index + 1}
        try:
            result = worker.summarize(previous=None, messages=messages)
            row.update(status='succeeded', new_fact_count=len(result.new_facts), summary_characters=len(result.model_dump_json()))
        except Exception as error:
            row.update(status='failed', error_code=getattr(error, 'code', type(error).__name__))
        row['wall_ms'] = round((perf_counter() - start) * 1000)
        if diagnostics:
            row['diagnostic'] = diagnostics.pop(0)
        report['attempts'].append(row)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
