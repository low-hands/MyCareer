"""Fixed targeted experiment; request override is confined to this process."""
from __future__ import annotations
import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
from dotenv import load_dotenv
from career_agent.agent.providers.main_agent import OpenAICompatibleMainAgentDecisionMaker
from career_agent.agent.providers.openai_client import OpenAICompatibleAgentConfig
from career_agent.evaluation.main_agent_scenarios import SCENARIOS
from career_agent.evaluation import search_trajectory as suite
from career_agent.evaluation.trajectory import trajectory_tool_specs, check_contract, trajectory_prompt_fingerprint

NAMES = (
    'a_card_backed_report_is_answered_without_reproducing_it',
    'five_missing_resume_facts_use_one_questionnaire',
    'a_report_made_this_turn_is_read_back_by_its_index',
    'a_report_made_this_turn_without_an_index_cannot_be_named',
)
class RequiredMaker(OpenAICompatibleMainAgentDecisionMaker):
    def _request(self, *, request_options, **kwargs):
        options = dict(request_options)
        if options.get('tool_choice', 'auto') == 'auto':
            options['tool_choice'] = 'required'
        return super()._request(request_options=options, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--compatibility-only', action='store_true')
    args = parser.parse_args()
    specs = trajectory_tool_specs()
    cases = [next(c for c in SCENARIOS if c.name == name) for name in NAMES]
    for case in cases:
        assert not check_contract(case, tool_specs=specs)
    protocol = {'fixed_samples_per_scenario':10, 'scenarios':list(NAMES),
        'compatibility_preflight_required':True, 'model':'qwen3.7-plus', 'mode':'required_on_initial_auto_requests',
        'grading':'Unchanged existing scenario behavioural and quality rules; all errors count, no replacement samples.',
        'comparison':'Existing three-sample auto recordings are historical, not concurrent matched A/B.',
        'prompt_fingerprints':{c.name:trajectory_prompt_fingerprint(c,tool_specs=specs) for c in cases}}
    if args.compatibility_only:
        protocol['fixed_samples_per_scenario'] = 1
        protocol['scenarios'] = list(NAMES[:1])
        protocol['mode'] = 'single_provider_compatibility_probe'
    if args.preflight:
        args.root.mkdir(parents=True, exist_ok=True)
        path=args.root/'protocol.json'
        if path.exists():raise ValueError('Do not overwrite protocol')
        path.write_text(json.dumps(protocol,indent=2)+'\n')
        print('Frozen 4 authored repository scenarios, 10 samples each; no provider calls.')
        return
    assert json.loads((args.root/'protocol.json').read_text()) == protocol
    output=args.root/'attempts.json'
    if output.exists():raise ValueError('Do not replace existing attempts')
    load_dotenv(Path(__file__).resolve().parents[2]/'.env')
    for key in ('BASE_URL','API_KEY','MODEL'):
        os.environ['MAIN_AGENT_'+key] = os.environ['RESUME_ANALYSIS_AGENT_'+key]
    os.environ['MAIN_AGENT_VENDOR']='qwen'
    os.environ['MAIN_AGENT_ENABLE_THINKING']='false'
    config=replace(OpenAICompatibleAgentConfig.from_env(prefix='MAIN_AGENT'),timeout_seconds=45)
    assert config.model == protocol['model']
    suite.OpenAICompatibleMainAgentDecisionMaker=RequiredMaker
    rows=[]
    # Round-robin serial sampling avoids separate before/after load blocks.
    for index in range(1 if args.compatibility_only else 10):
        for case in (cases[:1] if args.compatibility_only else cases):
            row={'scenario':case.name,'sample':index+1}
            try:
                sample=suite._record_sample(case,tool_specs=specs,config=config,max_attempts=1)
                row['recording']=sample
                row['failures']=list(suite.replay_search_sample(case,tool_specs=specs,responses=sample['steps']))
                row['status']='passed' if not row['failures'] else 'behaviour_failed'
            except Exception as error:
                row.update(status='failed',error_code=getattr(error,'code',type(error).__name__),recording=getattr(error,'recording_trace',{}))
            rows.append(row)
            output.write_text(json.dumps({'protocol':protocol,'attempts':rows},ensure_ascii=False,indent=2)+'\n')
            print(json.dumps({k:v for k,v in row.items() if k!='recording'},ensure_ascii=False),flush=True)
if __name__ == '__main__':main()
