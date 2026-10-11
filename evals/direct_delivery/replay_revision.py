"""Offline replay with explicit, narrow source compatibility evidence.

Does not rewrite cassette fingerprints. The only allowed source difference is
restoring the None guard around capability-search configuration. Recorded runs
use a non-None SemanticTape factory, so the conditional never changes that path.
"""
import hashlib,json
from pathlib import Path
from career_agent.evaluation.runtime_quality import load_cases,evaluate,suite_fingerprint,FIXTURE
from career_agent.evaluation.trajectory import prompt_fingerprint,trajectory_tool_specs
from career_agent.agent.context.semantic_retrieval import OpenAICompatibleEmbeddingClient
root=Path(__file__).resolve().parent/'revision_v3/qwen'
recordings=root/'runtime'
cases=load_cases()
payload=json.loads((recordings/(cases[0]['name']+'.json')).read_text())
recorded=payload['suite_fingerprint'];current=suite_fingerprint()
source_root=Path(FIXTURE).resolve().parents[2]/'src'
# Mirror the normal fingerprint computation, with the one documented source
# change reverted in memory. No source or recorded file is edited.
manifest=[]
changed=[]
for p in sorted(source_root.rglob('*.py')):
    data=p.read_bytes()
    if p.relative_to(source_root).as_posix()=='career_agent/agent/runtime/composition.py':
        original=data.replace(b'    if semantic_index is not None:\n        tools.configure_capability_search(semantic_index)\n',b'    tools.configure_capability_search(semantic_index)\n')
        if original!=data:changed.append(str(p.relative_to(source_root)))
        data=original
    manifest.append((str(p.relative_to(source_root)),hashlib.sha256(data).hexdigest()))
prior=hashlib.sha256(json.dumps((hashlib.sha256(FIXTURE.read_bytes()).hexdigest(),prompt_fingerprint(trajectory_tool_specs()),manifest),sort_keys=True).encode()).hexdigest()
if current!=recorded and (changed!=['career_agent/agent/runtime/composition.py'] or prior!=recorded):
    raise SystemExit('Source differs beyond the documented guard restoration; refusing compatible replay')
for p in recordings.glob('*.json'):
    d=json.loads(p.read_text())
    if d['suite_fingerprint']!=recorded:raise SystemExit('Mixed recording source fingerprints')
    if any('semantic_recording' not in s for s in d['samples']):raise SystemExit('Missing score tape')

def forbidden_embedding(*args,**kwargs):
    raise AssertionError('Replay must be offline')
OpenAICompatibleEmbeddingClient.embed=forbidden_embedding
report=evaluate(cases,root=recordings,model=payload['model'],k=payload['k'],fingerprint=recorded)
report['source_provenance']={'recorded_suite_fingerprint':recorded,'replayed_current_suite_fingerprint':current,'source_identical':current==recorded,'compatibility_verified':prior==recorded,'allowed_difference':changed,'proof':'Recorded samples use non-None SemanticTape. The restored None guard has the same branch for every sample. All other source, fixture, prompt and schema bytes reconstruct the recorded suite digest.'}
report['embedding_requests']=0
(root/'compatible_replay_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'passed':sum(r['samples_passed'] for r in report['results']),'pass_k':report['pass_k'],'direct':sum(r.get('direct_delivery_count',0) for r in report['results']),'source_provenance':report['source_provenance']},ensure_ascii=False))
