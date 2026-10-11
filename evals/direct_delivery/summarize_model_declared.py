"""Summarize all attempted samples, separating task scores from provider errors."""
from pathlib import Path
from collections import Counter
import json
from career_agent.evaluation.runtime_quality import load_cases, suite_fingerprint

root=Path(__file__).resolve().parents[2]
rows=[]
models={}
for vendor in ('deepseek','qwen'):
    items=[]
    errors=Counter()
    for case in load_cases():
        payload=json.loads((root/'evals/direct_delivery/model_declared_v2'/vendor/(case['name']+'.json')).read_text())
        samples=payload['samples']
        error_count=sum(any(t['error'] for t in s['turns']) for s in samples)
        errors.update(t['error'] for s in samples for t in s['turns'] if t['error'])
        # Merely ending on tool_call is not proof that the new guard fired:
        # legacy authorization/budget exits can also leave that final action.
        eligible_direct=0
        for s in samples:
            if len(s['turns']) != 1 or not s['passed']:
                continue
            t=s['turns'][0]
            if t['error'] or not t['decisions'] or t['decisions'][-1]['action']!='tool_call':
                continue
            call=t['decisions'][-1]['response'].get('tool_call',{})
            business=[c for c in t['executed_calls'] if c['name']!='search_capabilities']
            declared=call.get('arguments',{}).get('deliver_direct')
            eligible_direct+=int(len(business)==1 and (declared is True or (declared is None and call.get('name') in {'get_daily_brief','export_resume_artifact'})))
        items.append({'scenario':case['name'],'attempts':len(samples),'live_task_passed':sum(s['passed'] for s in samples),
                      'live_task_success_rate':sum(s['passed'] for s in samples)/len(samples),
                      'provider_error_samples':error_count,'eligible_single_receipt_endings':eligible_direct,
                      'suite_fingerprint_matches_current':payload['suite_fingerprint']==suite_fingerprint()})
    report=json.loads((root/f'evals/direct_delivery/{vendor}_model_declared_runtime_report.json').read_text())
    models[vendor]={'attempts':sum(i['attempts'] for i in items),'provider_errors':dict(errors),
                    'replay_task_passed':sum(i['samples_passed'] for i in report['results']),
                    'eligible_single_receipt_endings':sum(i['eligible_single_receipt_endings'] for i in items),
                    'all_cassettes_fresh':all(i['suite_fingerprint_matches_current'] for i in items), 'scenarios':items}
result={'implementation':'tool_defaults_and_model_declaration','status':'quality_not_established_provider_failures',
        'deterministic_tests_passed':361,'baseline_files_restored':24,
        'baseline_preservation':'baseline_preservation.json','historical_decision_audit':'frozen_runtime_route_audit.json',
        'legacy_47_status':'stale_for_both_models_due_to_prompt_and_schema_change','models':models,
        'latency_savings_claim':'unmeasured_in_real_usage'}
(root/'evals/direct_delivery/model_declared_guard_report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
for vendor in models:
    print(vendor,{k:v for k,v in models[vendor].items() if k!='scenarios'})
lines=['\n## 当前采样结果\n','两模型各 190 个尝试完整保留。请求失败计入端到端失败分母，但这些错误不能用于估计交付策略的质量影响。下表为 live 任务成功数 / 10，正式报告另重放同一批决策。\n',
       '| 场景 | DeepSeek 成功 | Qwen 成功 | DeepSeek 请求错误 | Qwen 请求错误 |',
       '|---|---:|---:|---:|---:|']
for ds,qw in zip(models['deepseek']['scenarios'],models['qwen']['scenarios']):
    lines.append(f"| {ds['scenario']} | {ds['live_task_passed']}/10 | {qw['live_task_passed']}/10 | {ds['provider_error_samples']} | {qw['provider_error_samples']} |")
lines+=['\n[当前完整报告](model_declared_guard_report.json) 的状态为 `quality_not_established_provider_failures`。DeepSeek 190 个样本全部为 transport error；Qwen 148 个样本包含 HTTP 403。随后极小 Qwen 请求及带当前工具 schema 的请求都成功，故未把 403 擅自归因于密钥、额度或策略。没有持续补采刷绿。\n',
        '原 47 条轨迹当前 stale，新模型声明的实际直交付覆盖和质量尚未验证，真实延迟收益也未测得。确定性测试和历史路由排除不能替代这些结论。\n']
# Experiment write-ups are local records under the gitignored docs/ tree.
p=root/'docs/experiments/evals/direct_delivery/README.md'
s=p.read_text().split('\n## 当前采样结果')[0]
p.write_text(s+'\n'.join(lines))
