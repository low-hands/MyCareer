"""Record the entire fixed suite; preserve previous batches and baselines."""
import argparse, json, os, io
from pathlib import Path
from dotenv import load_dotenv
from career_agent.cli import main as cli
from career_agent.evaluation import runtime_quality as runtime
from career_agent.agent.providers.openai_client import OpenAICompatibleAgentConfig
from career_agent.agent.providers.main_agent import OpenAICompatibleMainAgentDecisionMaker, main_model_options
from career_agent.evaluation.trajectory import trajectory_tool_specs
parser=argparse.ArgumentParser()
parser.add_argument('vendor',choices=['qwen','deepseek'])
parser.add_argument('kind',choices=['probe','trajectories','runtime'])
args=parser.parse_args()
load_dotenv()
prefix='RESUME_ANALYSIS_AGENT' if args.vendor=='qwen' else 'MAIN_AGENT'
config=OpenAICompatibleAgentConfig.from_env(prefix=prefix)
root=Path(__file__).resolve().parent/'revision_v3'/args.vendor
root.mkdir(parents=True,exist_ok=True)
if args.kind=='probe':
    from career_agent.evaluation.search_scenarios import SEARCH_SCENARIOS
    maker=OpenAICompatibleMainAgentDecisionMaker(config, **main_model_options({'MAIN_AGENT_VENDOR':args.vendor,'MAIN_AGENT_ENABLE_THINKING':'false'}))
    from career_agent.evaluation.capability_semantic_fixture import evaluation_strategy
    scenario=SEARCH_SCENARIOS[0];strategy=evaluation_strategy()
    maker.configure_tool_selection(strategy)
    try:
        maker.decide(scenario.context,trajectory_tool_specs())
        report={'status':'success'}
    except Exception as error:
        chain=[];cause=error
        while cause is not None:
            chain.append({'type':type(cause).__name__,'code':getattr(cause,'code',None)})
            cause=cause.__cause__
        report={'status':'failed','exception_chain':chain}
    finally:maker._client.close()
    (root/'probe.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)
    raise SystemExit(report['status']!='success')
if args.kind=='runtime':
    if (root/'runtime').exists():raise SystemExit('Refusing to overwrite recordings')
    raise SystemExit(runtime.main(['--record','--env-prefix',prefix,'--vendor',args.vendor,
        '--root',str(root/'runtime'),'--output',str(root/'runtime_report.json'),'--k','10','--jobs','2']))
for suffix in ('BASE_URL','API_KEY','MODEL'):
    value=os.environ.get(prefix+'_'+suffix)
    if value is not None:os.environ['MAIN_AGENT_'+suffix]=value
os.environ['MAIN_AGENT_VENDOR']=args.vendor
os.environ['MAIN_AGENT_ENABLE_THINKING']='false'
if (root/'trajectories').exists():raise SystemExit('Refusing to overwrite recordings')
output=io.StringIO()
status=cli(['eval','trajectories','--record','--cassette-root',str(root/'trajectories'),
            '--jobs','3','--main-agent-timeout-seconds','45'],stdout=output)
(root/'trajectory_report.json').write_text(output.getvalue())
print(json.dumps({'status':status,'report':str(root/'trajectory_report.json')}),flush=True)
raise SystemExit(status)
