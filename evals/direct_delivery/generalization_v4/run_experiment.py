"""Frozen holdout and contemporary old/new regression measurements.

Run from the main checkout with its interpreter. Baseline imports are isolated
in a child process using an untouched git archive, never patched into production.
"""
import argparse,json,os,sys,hashlib,random,subprocess,tarfile,tempfile,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
from dataclasses import replace
from datetime import datetime,timezone
from dotenv import load_dotenv
ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[2]
NAMES=('research_is_not_started_as_part_of_matching','saved_jd_body_without_language_requirement_finishes','tool_selection_combines_job_analysis_and_resume_match')
parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['freeze','holdout','baseline','current','trajectories']);parser.add_argument('--snapshot');args=parser.parse_args()
if args.mode=='baseline' and not args.snapshot:
    snapshot=ROOT/'baseline_18158e1'
    if not snapshot.exists():
        snapshot.mkdir()
        data=subprocess.check_output(['git','archive','18158e1'],cwd=REPO)
        import io
        with tarfile.open(fileobj=io.BytesIO(data)) as archive:archive.extractall(snapshot,filter='data')
    os.execve(sys.executable,[sys.executable,str(Path(__file__).resolve()),'baseline','--snapshot',str(snapshot)],{**os.environ,'PYTHONPATH':str(snapshot/'src')})
# Force baseline imports before any installed editable package can resolve.
if args.snapshot:sys.path.insert(0,str(Path(args.snapshot)/'src'))
from career_agent.evaluation.trajectory import trajectory_tool_specs,prompt_fingerprint,trajectory_prompt_fingerprint
from career_agent.evaluation.search_scenarios import SEARCH_SCENARIOS
from career_agent.evaluation.search_trajectory import _record_sample,replay_search_sample,check_search_contract
from career_agent.agent.providers.openai_client import OpenAICompatibleAgentConfig,AgentWorkerError
from career_agent.agent.providers.main_agent import main_model_options
load_dotenv(REPO/'.env')
config=replace(OpenAICompatibleAgentConfig.from_env(prefix='RESUME_ANALYSIS_AGENT'),timeout_seconds=45)
for suffix in ('BASE_URL','API_KEY','MODEL'):
    value=os.environ.get('RESUME_ANALYSIS_AGENT_'+suffix)
    if value is not None:os.environ['MAIN_AGENT_'+suffix]=value
os.environ['MAIN_AGENT_VENDOR']='qwen';os.environ['MAIN_AGENT_ENABLE_THINKING']='false'
specs=trajectory_tool_specs()
if args.mode=='freeze':
    from career_agent.evaluation.runtime_quality import suite_fingerprint,load_cases
    holdout=ROOT/'holdout.json';cases=load_cases(holdout)
    from career_agent.agent.providers.main_agent import OpenAICompatibleMainAgentDecisionMaker
    prompt=OpenAICompatibleMainAgentDecisionMaker._system_prompt()
    assert all(t['message'] not in prompt for c in cases for t in c['turns'])
    manifest={'frozen_at':datetime.now(timezone.utc).isoformat(),'fixture_sha256':hashlib.sha256(holdout.read_bytes()).hexdigest(),'runtime_suite_fingerprint':suite_fingerprint(holdout),'prompt_fingerprint':prompt_fingerprint(specs),'holdout_k':10,'holdout_scenarios':len(cases),'regression_cases':NAMES,'samples_per_regression_case_per_arm':30,'baseline_ref':'18158e1','schedule_seed':3110,'regression_workers_per_arm':1,'holdout_workers':2,'model':config.model,'prompt_contains_test_utterances':False,'policy':'One frozen prompt and one complete batch. No selective retries, no tuning on holdout. Report provider errors separately and retain them in end-to-end denominator.'}
    if (ROOT/'preregistration.json').exists():raise SystemExit('Freeze already exists')
    (ROOT/'preregistration.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    (ROOT/'production_main_agent_snapshot.py').write_bytes((REPO/'src/career_agent/agent/providers/main_agent.py').read_bytes())
    print(json.dumps(manifest,ensure_ascii=False));raise SystemExit()
manifest=json.loads((ROOT/'preregistration.json').read_text())
if args.mode=='holdout':
    from career_agent.evaluation import runtime_quality as suite
    path=ROOT/'holdout.json'
    assert hashlib.sha256(path.read_bytes()).hexdigest()==manifest['fixture_sha256']
    assert suite.suite_fingerprint(path)==manifest['runtime_suite_fingerprint']
    out=ROOT/'holdout_qwen';out.mkdir(exist_ok=False)
    original=suite._live_sample
    from threading import Lock
    lock=Lock();count=0
    def observed(case,*a,**kw):
        global count
        result=original(case,*a,**kw)
        with lock:
            count+=1
            (out/f"attempt_{count:03}.json").write_text(json.dumps({'scenario':case['name'],'sample':result},ensure_ascii=False,indent=2)+'\n')
            print(json.dumps({'completed':count,'total':140,'scenario':case['name']},ensure_ascii=False),flush=True)
        return result
    suite._live_sample=observed
    cases=suite.load_cases(path)
    suite.record_cases(cases,config=config,options=main_model_options(),root=out/'recordings',k=10,jobs=2,fingerprint=manifest['runtime_suite_fingerprint'])
    report=suite.evaluate(cases,root=out/'recordings',model=config.model,k=10,fingerprint=manifest['runtime_suite_fingerprint'])
    (out/'runtime_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'complete':140,'passed':sum(r['samples_passed'] for r in report['results'])}),flush=True);raise SystemExit()
if args.mode=='trajectories':
    from career_agent.cli import main
    import io
    out=ROOT/'qwen_47';out.mkdir(exist_ok=False);output=io.StringIO()
    status=main(['eval','trajectories','--record','--cassette-root',str(out/'recordings'),'--jobs','2','--main-agent-timeout-seconds','45'],stdout=output)
    (out/'report.json').write_text(output.getvalue());print(json.dumps({'status':status}),flush=True);raise SystemExit(status)
if args.mode=='current':assert prompt_fingerprint(specs)==manifest['prompt_fingerprint']
assert config.model==manifest['model']
out=ROOT/f'regression_{args.mode}';out.mkdir(exist_ok=False)
selected={s.name:s for s in SEARCH_SCENARIOS if s.name in NAMES}
assert len(selected)==3
for scenario in selected.values():assert not check_search_contract(scenario,tool_specs=specs)
rng=random.Random(manifest['schedule_seed']);schedule=[]
for number in range(30):
    order=list(NAMES);rng.shuffle(order);schedule.extend((name,number) for name in order)
samples={name:{} for name in NAMES}
for completed,(name,number) in enumerate(schedule,1):
    scenario=selected[name];start=datetime.now(timezone.utc).isoformat()
    try:
        sample=_record_sample(scenario,tool_specs=specs,config=config)
    except AgentWorkerError as error:
        trace=getattr(error,'recording_trace',None) or {}
        sample={'terminal_error':error.code,'recorded_at':datetime.now(timezone.utc).isoformat(),'steps':[*trace.get('completed_steps',()),{'scenario_step':trace.get('step',0),'recording_error':error.code}]}
    response=sample['steps']
    failures=replay_search_sample(scenario,tool_specs=specs,responses=response)
    quality=replay_search_sample(scenario,tool_specs=specs,responses=response,quality=True)
    row={'scenario':name,'sample':number+1,'started_at':start,'recording':sample,'passed':not failures,'failures':failures,'quality_failures':quality,'prompt_fingerprint':trajectory_prompt_fingerprint(scenario,specs),'model':config.model,'source_arm':args.mode}
    samples[name][number]=row
    (out/f'{name}_{number+1:02}.json').write_text(json.dumps(row,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'arm':args.mode,'completed':completed,'total':90,'scenario':name,'passed':not failures}),flush=True)
(out/'report.json').write_text(json.dumps({'arm':args.mode,'model':config.model,'sample_count':90,'cases':{n:[samples[n][i] for i in range(30)] for n in NAMES}},ensure_ascii=False,indent=2)+'\n')
