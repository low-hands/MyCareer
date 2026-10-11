"""Report frozen experiments without treating a nonsignificant delta as equivalence."""
import hashlib,json,math
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parent

def wilson(passed,total):
    z=1.959963984540054
    p=passed/total;d=1+z*z/total
    c=(p+z*z/(2*total))/d
    w=z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/d
    return [c-w,c+w]

def fisher(a,b):
    # Two-sided exact comparison of two independent binomial counts (n=30).
    success=a+b;den=math.comb(60,success)
    def probability(x):return math.comb(30,x)*math.comb(30,success-x)/den
    possible=range(max(0,success-30),min(30,success)+1)
    observed=probability(a)
    return min(1.0,sum(probability(x) for x in possible if probability(x)<=observed+1e-12))

manifest=json.loads((ROOT/'preregistration.json').read_text())
assert hashlib.sha256((ROOT/'holdout.json').read_bytes()).hexdigest()==manifest['fixture_sha256']
models={arm:json.loads((ROOT/f'regression_{arm}/report.json').read_text()) for arm in ('baseline','current')}
provider_codes={'MAIN_AGENT_TRANSPORT_ERROR','MAIN_AGENT_TIMEOUT'}
def is_provider(code):return code in provider_codes or str(code).startswith('MAIN_AGENT_REJECTED_')
comparisons=[]
for name in manifest['regression_cases']:
    counts={}
    for arm,report in models.items():
        samples=report['cases'][name];assert len(samples)==30
        passed=sum(s['passed'] for s in samples)
        errors=Counter(s['recording']['terminal_error'] for s in samples if s['recording'].get('terminal_error'))
        provider=sum(n for code,n in errors.items() if is_provider(code))
        counts[arm]={'passed':passed,'total':30,'success_rate':passed/30,'wilson_95':wilson(passed,30),'terminal_errors':dict(errors),'provider_error_samples':provider,'success_rate_excluding_provider_errors':passed/(30-provider) if provider<30 else None}
    old=counts['baseline'];new=counts['current']
    comparisons.append({'scenario':name,**counts,'difference':new['success_rate']-old['success_rate'],'fisher_two_sided_p':fisher(old['passed'],new['passed'])})
report=json.loads((ROOT/'holdout_qwen/runtime_report.json').read_text())
assert report['suite_fingerprint']==manifest['runtime_suite_fingerprint']
rows=[];answers=[];provider_errors=Counter()
for row in report['results']:
    payload=json.loads((ROOT/'holdout_qwen/recordings'/(row['scenario']+'.json')).read_text())
    assert len(payload['samples'])==10
    case=next(c for c in json.loads((ROOT/'holdout.json').read_text())['cases'] if c['name']==row['scenario'])
    direct=0;premature=0;summary_failures=0
    for number,sample in enumerate(payload['samples'],1):
        for index,turn in enumerate(sample['turns']):
            if turn['error']:provider_errors[turn['error']]+=1
            last=turn['decisions'][-1]['response'].get('tool_call',{}) if turn['decisions'] else {}
            business=[c for c in turn['executed_calls'] if c['name'] not in {'load_skill','search_capabilities'}]
            declared=last.get('arguments',{}).get('deliver_direct')
            ended=bool(not turn['error'] and len(business)==1 and last.get('name') in {'list_resumes','list_interviews','get_daily_brief'} and (declared is True or declared is None and last.get('name')=='get_daily_brief'))
            direct+=ended
            premature+=bool(ended and (turn['missing_calls'] or turn['requires_model_summary'] and not turn['model_summary_delivered']))
            summary_failures+=bool(not turn['error'] and turn['requires_model_summary'] and not turn['model_summary_delivered'])
            answers.append({'scenario':row['scenario'],'sample':number,'turn':index+1,'message':turn['message'],'assistant':turn['assistant_message'],'error':turn['error'],'model_summary_delivered':turn['model_summary_delivered']})
    rows.append({'scenario':row['scenario'],'passed':row['samples_passed'],'total':10,'direct_receipt_endings':direct,'premature_direct_endings':premature,'missing_requested_summaries':summary_failures})
result={'measurement':'preregistered_new_utterance_holdout_and_contemporary_baseline_regression','holdout_passed':sum(r['passed'] for r in rows),'holdout_total':140,'holdout_single_query_direct':sum(r['direct_receipt_endings'] for r in rows[:8]),'holdout_single_query_total':80,'premature_direct_endings':sum(r['premature_direct_endings'] for r in rows),'runtime_errors':dict(provider_errors),'regression_comparisons':comparisons,'holdout_results':rows,'interpretation_limit':'These are short synthetic workspaces with complete empty read receipts. Language generalization within this fixture does not establish real-use frequency, latency savings or arbitrary multi-step safety. Baseline comparison is observational, not a proof of equivalence or isolation of the schema versus prompt effect.'}
(ROOT/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
(ROOT/'key_content_review_inputs.json').write_text(json.dumps(answers,ensure_ascii=False,indent=2)+'\n')
lines=['# 抽象提示与新holdout验证\n','上一批提示含原评估原话，60/60仅为已见措辞的开发集成绩，撤回其泛化解释。当前提示只含抽象原则；新holdout先固定后采样，未按结果再改提示。\n','[预注册](preregistration.json) 固定14场景×10次及三个回退场景每个版本×30次。原样git archive 18158e1与当前源码在独立进程中运行，均用同一Qwen配置、相同场景轮换顺序、并发1，时段交错；全部失败保留。\n',f"新holdout任务结构成功 **{result['holdout_passed']}/140**；8个单项查询/简报直接回执结束 **{result['holdout_single_query_direct']}/80**。提前直接结束 {result['premature_direct_endings']} 次。runtime错误 {json.dumps(result['runtime_errors'],ensure_ascii=False)}。\n",'| 回退场景 | 18158e1 / 30 | 抽象提示当前 / 30 | 差值 | Fisher双侧p（未作多重比较校正） |','|---|---:|---:|---:|---:|']
for c in comparisons:lines.append(f"| {c['scenario']} | {c['baseline']['passed']} | {c['current']['passed']} | {c['difference']:+.1%} | {c['fisher_two_sided_p']:.4f} |")
lines+=['\n不能把p不显著当作无回退的证明；每组30次仍有较宽的不确定范围，Wilson区间、全部逐样本结果和终止错误见 [汇总](summary.json)。本实验同时比较提示和schema变化，不能分离两者各自的因果影响。\n','| holdout | 任务成功 / 10 | 直接回执结束（跨轮可超过10） | 提前直接结束 |','|---|---:|---:|---:|']
for r in rows:lines.append(f"| {r['scenario']} | {r['passed']} | {r['direct_receipt_endings']} | {r['premature_direct_endings']} |")
lines+=['\n业务I/O仍为短合成工作区和完整空结果；这个holdout检验未见措辞下的边界，不能当作真实使用覆盖率或延迟收益。任务结构与回复关键内容分开，关键内容另审。旧批次均保留，不覆盖evaluation_restoration历史报告。']
# Experiment write-ups are local records under the gitignored docs/ tree.
REPORT=ROOT.parents[2]/'docs/experiments'/ROOT.relative_to(ROOT.parents[2])/'README.md'
REPORT.parent.mkdir(parents=True,exist_ok=True)
REPORT.write_text('\n'.join(lines)+'\n')
print(json.dumps(result,ensure_ascii=False))
