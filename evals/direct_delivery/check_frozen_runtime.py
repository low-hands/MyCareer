"""Audit new routing against historical decisions, without resampling models.

This is route exclusion evidence, not current model quality or schema freshness.
"""
from pathlib import Path
from types import SimpleNamespace
import json
from career_agent.evaluation.runtime_quality import load_cases, _READ_RESULTS
from career_agent.agent.contracts.decisions import AgentDecision, ToolCall
from career_agent.agent.contracts.observations import ToolObservation
from career_agent.harness.graph_routing import GraphRoutingPolicy

root = Path(__file__).resolve().parents[2]
rows = []
for vendor in ('qwen', 'deepseek'):
    for case in load_cases()[:11]:
        payload = json.loads((root / 'output/direct-delivery-before' / vendor / (case['name'] + '.json')).read_text())
        for number, sample in enumerate(payload['samples']):
            for turn in sample['turns']:
                for item in turn['decisions']:
                    call = item['response'].get('tool_call')
                    if call is None or call['name'] == 'search_capabilities':
                        continue
                    name = call['name']
                    result_state = _READ_RESULTS.get(name, ('working_notes_updated', 'updated'))[0]
                    result = ToolObservation(tool_name=name, state=result_state, message='receipt', execution_outcome='committed')
                    arguments = dict(call['arguments'])
                    direct = arguments.pop('deliver_direct', None)
                    state = {'decision': AgentDecision(action='tool_call', tool_call=ToolCall(name=name, arguments=arguments, deliver_direct=direct)),
                             'pending': {'name': name, 'effect': 'WRITE' if name == 'update_working_notes' else 'READ', 'result': result},
                             'context': SimpleNamespace(turn_continuation_capability=None), 'tool_results': (result,)}
                    # Give the shortcut its most permissive integrity conditions:
                    # one successful committed result with no continuation.
                    rows.append({'vendor':vendor,'scenario':case['name'],'sample':number+1,
                                 'tool':name,'route':GraphRoutingPolicy.after_observe(state)})
legacy_create_calls = {}
for vendor, folder in (('qwen','main_agent_search'),('deepseek','main_agent_deepseek_search')):
    count = 0
    for path in (root / 'evals' / folder).glob('*.json'):
        payload=json.loads(path.read_text())
        def count_calls(value):
            if isinstance(value,dict):
                return int(isinstance(value.get('tool_call'),dict) and value['tool_call'].get('name')=='create_interview') + sum(count_calls(v) for v in value.values())
            if isinstance(value,list): return sum(count_calls(v) for v in value)
            return 0
        count += count_calls(payload)
    legacy_create_calls[vendor] = count
report={'measurement':'historical_decision_route_exclusion_not_live_quality',
        'business_decisions':len(rows),'shortcut_routes':sum(r['route']=='present' for r in rows),
        'runtime_create_interview_calls':sum(r['tool']=='create_interview' for r in rows),
        'legacy_47_create_interview_calls':legacy_create_calls,'rows':rows}
(root / 'evals/direct_delivery/frozen_runtime_route_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='rows'}))
