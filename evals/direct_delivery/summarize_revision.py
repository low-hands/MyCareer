"""Separate task, delivery, continuation and provider measurements."""
import json
from pathlib import Path
from collections import Counter
from career_agent.evaluation.runtime_quality import load_cases
root=Path(__file__).resolve().parent/'revision_v3/qwen'
report=json.loads((root/'compatible_replay_report.json').read_text())
rows=[];errors=Counter();texts=[]
for case,row in zip(load_cases(),report['results']):
    payload=json.loads((root/'runtime'/(case['name']+'.json')).read_text())
    endings=0;stops=0;summary_missing=0
    for i,sample in enumerate(payload['samples']):
        for turn in sample['turns']:
            if turn['error']:errors[turn['error']]+=1
            if turn['requires_model_summary'] and not turn['model_summary_delivered']:summary_missing+=1
            if turn['decisions']:
                call=turn['decisions'][-1]['response'].get('tool_call',{})
                business=[c for c in turn['executed_calls'] if c['name'] not in {'search_capabilities','load_skill'}]
                declared=call.get('arguments',{}).get('deliver_direct')
                eligible=(call.get('name') in {'list_resumes','list_interviews','list_applications','find_saved_jobs','get_daily_brief'} and
                          len(business)==1 and (declared is True or declared is None and call.get('name')=='get_daily_brief') and not turn['error'])
                endings+=eligible
                stops+=int(eligible and (turn['missing_calls'] or turn['requires_model_summary'] and not turn['model_summary_delivered']))
        if case['name'] in {'single_empty_resume_receipt','single_empty_interview_receipt','single_brief_body_is_delivered','colloquial_empty_resume_receipt','count_empty_resume_receipt','english_empty_resume_receipt'}:
            texts.append({'scenario':case['name'],'sample':i+1,'assistant':sample['turns'][-1]['assistant_message']})
    rows.append({'scenario':case['name'],'passed':row['samples_passed'],'total':row['sample_count'],'success_rate':row['success_rate'],'eligible_direct_deliveries':endings,'premature_direct_deliveries':stops,'missing_requested_summaries':summary_missing})
result={'passed':sum(r['passed'] for r in rows),'total':sum(r['total'] for r in rows),'eligible_direct_deliveries':sum(r['eligible_direct_deliveries'] for r in rows),'premature_direct_deliveries':sum(r['premature_direct_deliveries'] for r in rows),'errors':dict(errors),'embedding_replay_requests':0,'source_provenance':report['source_provenance'],'results':rows}
(root/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
(root/'key_content_review_inputs.json').write_text(json.dumps(texts,ensure_ascii=False,indent=2)+'\n')
lines=['\n## Qwen runtime完整批次\n',f"任务结构成功 {result['passed']}/{result['total']}；符合声明/默认值的单项完整读回执结束 {result['eligible_direct_deliveries']} 次；其中漏步骤或要求汇总未交付 {result['premature_direct_deliveries']} 次。请求错误 {json.dumps(result['errors'],ensure_ascii=False)}。\n",'| 场景 | 成功/10 | 直接交付 | 提前直接结束 |','|---|---:|---:|---:|']
for r in rows:lines.append(f"| {r['scenario']} | {r['passed']} | {r['eligible_direct_deliveries']} | {r['premature_direct_deliveries']} |")
lines+=['\n当前代码兼容重放使用录制语义分数，embedding请求=0。源文件并非完全相同，窄范围差异重构证明、录制/重放源指纹及逐场景报告见 [兼容重放](qwen/compatible_replay_report.json)。判分仅证明结构护栏，关键内容另审，成功率不能当作真实使用延迟收益。']
# Experiment write-ups are local records under the gitignored docs/ tree.
p=root.parents[3]/'docs/experiments'/root.parent.relative_to(root.parents[3])/'README.md';s=p.read_text().split('\n## Qwen runtime完整批次')[0];p.write_text(s+'\n'.join(lines)+'\n')
print(json.dumps(result,ensure_ascii=False))
