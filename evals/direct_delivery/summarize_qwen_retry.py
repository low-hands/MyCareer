"""Summarize the complete retry without conflating wording, work and routing."""
from collections import Counter
from pathlib import Path
import json
from career_agent.evaluation.runtime_quality import load_cases,suite_fingerprint,_READ_RESULTS
root=Path(__file__).resolve().parent/'qwen_retry_2026-10-09'
report=json.loads((root/'runtime_report.json').read_text())
rows=[];errors=Counter()
for case in load_cases():
 p=json.loads((root/'recordings'/(case['name']+'.json')).read_text())
 samples=p['samples']
 assert p['suite_fingerprint']==suite_fingerprint(), 'stale source fingerprint'
 assert len(samples)==p['k']==10, 'incomplete sample batch'
 direct=0;premature_direct=0
 for sample in samples:
  for t in sample['turns']:
   if t['error']:errors[t['error']]+=1;continue
   if not t['decisions'] or t['decisions'][-1]['action']!='tool_call':continue
   call=t['decisions'][-1]['response'].get('tool_call',{})
   name=call.get('name');declared=call.get('arguments',{}).get('deliver_direct')
   business=[c for c in t['executed_calls'] if c['name']!='search_capabilities']
   if name in _READ_RESULTS and len(business)==1 and business[0]['name']==name and (declared is True or (declared is None and name=='get_daily_brief')):
    direct+=1
    premature_direct+=bool(t['missing_calls'] or (t['requires_model_summary'] and not t['model_summary_delivered']))
 rows.append({'scenario':case['name'],'attempts':10,
              'live_strict_passed':sum(s['passed'] for s in samples),
              'replay_strict_passed':next(r['samples_passed'] for r in report['results'] if r['scenario']==case['name']),
              'requested_calls_completed_without_error':sum(all(not t['error'] and not t['missing_calls'] for t in s['turns']) and len(s['turns'])==len(case['turns']) for s in samples),
              'fixed_text_failure_samples':sum(any('required delivery text missing' in f for f in s['failures']) for s in samples),
              'missing_model_summary_samples':sum(any('requested model summary missing' in f for f in s['failures']) for s in samples),
              'provider_error_samples':sum(any(t['error'] for t in s['turns']) for s in samples),
              'direct_read_receipt_endings':direct,'premature_direct_read_receipt_endings':premature_direct})
summary={'measurement':'unchanged_production_runtime_with_synthetic_business_io','attempts':190,
         'k':10,'jobs':2,'suite_fingerprint':suite_fingerprint(),'all_cassettes_fresh':True,
         'provider_errors':dict(errors),'strict_passed':sum(r['replay_strict_passed'] for r in rows),
         'direct_read_receipt_endings':sum(r['direct_read_receipt_endings'] for r in rows),
         'premature_direct_read_receipt_endings':sum(r['premature_direct_read_receipt_endings'] for r in rows),
         'interpretation':'Strict grade includes fixed receipt text. Work completion and receipt endings are structural diagnostics, not semantic quality or real-world latency estimates.',
         'scenarios':rows}
(root/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k!='scenarios'},ensure_ascii=False))
lines=['\n## 完整结果\n','| 场景 | 严格检查成功 / 10 | 请求工具完成 / 10 | 固定文字失败 | 请求错误 | 直接回执结束 |', '|---|---:|---:|---:|---:|---:|']
for r in rows:
 lines.append(f"| {r['scenario']} | {r['replay_strict_passed']} | {r['requested_calls_completed_without_error']} | {r['fixed_text_failure_samples']} | {r['provider_error_samples']} | {r['direct_read_receipt_endings']} |")
lines+=['\n逐场景明细见 [summary.json](summary.json)，严格检查报告见 [runtime_report.json](runtime_report.json)。直接回执计数只覆盖本夹具已知完整的 read 回执、单业务结果以及符合声明/默认值条件的末尾调用，不将所有 tool_call 终止算成直接交付。\n']
# Experiment write-ups are local records under the gitignored docs/ tree.
p=root.parents[2]/'docs/experiments'/root.relative_to(root.parents[2])/'README.md';p.write_text(p.read_text().split('\n## 完整结果')[0]+'\n'.join(lines))
