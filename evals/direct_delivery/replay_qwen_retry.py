"""Replay frozen Qwen decisions via the existing grader, with embedding reuse.

Each distinct model/input uses an actual embedding response once. Recorded
schema fingerprints are still checked; no model decisions or grades are edited.
"""
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
from threading import Lock
from collections import Counter
import hashlib,json,subprocess
from dotenv import load_dotenv
from career_agent.evaluation.runtime_quality import load_cases,evaluate,suite_fingerprint,CATEGORIES
from career_agent.agent.context.semantic_retrieval import OpenAICompatibleEmbeddingClient
root=Path(__file__).resolve().parent/'qwen_retry_2026-10-09'
load_dotenv(Path(__file__).resolve().parents[2]/'.env')
if (root/'runtime_report.json').exists():
    raise SystemExit('Report exists; do not overwrite it.')
original_embed=OpenAICompatibleEmbeddingClient.embed
lock=Lock();cache={};calls=Counter();embedding_errors=Counter()


def cached_embed(self,texts):
    texts=tuple(texts)
    key=hashlib.sha256(json.dumps((self.model_id,texts),ensure_ascii=False).encode()).hexdigest()
    # Single-flight for identical catalogue batches and repeated scenario queries.
    with lock:
        if key in cache:
            calls['cache_hits']+=1
            return cache[key]['vectors']
        try:
            vectors=tuple(tuple(row) for row in original_embed(self,texts))
        except Exception as error:
            embedding_errors[type(error).__name__]+=1
            raise
        calls['actual_embedding_requests']+=1
        cache[key]={'model_id':self.model_id,'texts':texts,'vectors':vectors}
        return vectors


OpenAICompatibleEmbeddingClient.embed=cached_embed
cases=load_cases();fingerprint=suite_fingerprint()
model=json.loads((root/'recordings'/f"{cases[0]['name']}.json").read_text())['model']
completed={}
try:
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures={pool.submit(evaluate,(case,),root=root/'recordings',model=model,k=10,fingerprint=fingerprint):case['name'] for case in cases}
        for future in as_completed(futures):
            name=futures[future];result=future.result();completed[name]=result
            (root/'replay_progress.json').write_text(json.dumps({'completed_scenarios':len(completed),'total_scenarios':len(cases),'scenarios':list(completed)},indent=2)+'\n')
            print(json.dumps({'replayed_scenarios':len(completed),'total':len(cases),'scenario':name,'strict_passed':result['results'][0]['samples_passed']},ensure_ascii=False),flush=True)
    rows=[completed[c['name']]['results'][0] for c in cases]
    report=dict(completed[cases[0]['name']]);report['results']=rows
    report['category_results']={category:{'passed':sum(r['pass_k'] for r in rows if r['category']==category),'total':sum(r['category']==category for r in rows)} for category in sorted(CATEGORIES)}
    report['pass_k']={'passed':sum(r['pass_k'] for r in rows),'total':len(rows)}
    for name in ('premature_stop_count','unsafe_attempt_count','injection_marker_echo_count'):
        report[name]=sum(r[name] for r in rows)
    report['revision']=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    report['worktree_clean']=not bool(subprocess.check_output(['git','status','--porcelain'],text=True))
    report['replay_execution']={'jobs':4,'decision_source':'all_190_original_frozen_responses','grader':'unchanged runtime_quality.evaluate','embedding':'actual vectors memoized by model namespace and exact input; recorded schema fingerprints still checked'}
    (root/'runtime_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
finally:
    (root/'replay_embedding_cache.json').write_text(json.dumps({'counts':dict(calls),'errors':dict(embedding_errors),'entries':cache},ensure_ascii=False)+'\n')
    (root/'replay_execution.json').write_text(json.dumps({'previous_sequential_replay':'interrupted_after_live_batch_saved_due_to_repeated_catalogue_warmup_cost','completed_scenarios':len(completed),'jobs':4,'embedding_counts':dict(calls),'embedding_errors':dict(embedding_errors),'production_code_or_grading_changed':False},indent=2)+'\n')
print(json.dumps({'pass_k':report['pass_k'],'strict_samples_passed':sum(r['samples_passed'] for r in rows),'embedding_counts':dict(calls)},ensure_ascii=False))
