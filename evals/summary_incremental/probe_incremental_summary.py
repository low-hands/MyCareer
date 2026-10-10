"""Frozen marker audit through the actual manager and SQLite fact ledger."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter
from dotenv import load_dotenv
from career_agent.agent.context.deployment import ConversationSummaryAgentConfig
from career_agent.agent.context.manager import ContextManager
from career_agent.agent.contracts.task_state import ConversationTaskState
from career_agent.agent.providers.conversation_summary import OpenAIConversationSummaryWorker
from career_agent.agent.providers.openai_client import OpenAICompatibleAgentConfig
from career_agent.storage.context import CareerContextStore


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--preflight',action='store_true')
    args=parser.parse_args()
    repo=Path(__file__).resolve().parents[2]
    data=json.loads((repo/'evals/latency/scenarios.json').read_text())
    rounds=[]
    for index in range(1,11):
        facts=[{'marker':f'F{index:02}{letter}','value':index*100+offset} for offset,letter in enumerate('ABC',1)]
        content='新增独立项目事实：'+ '；'.join(f"项目 {f['marker']} 的日事件量是 {f['value']} 条" for f in facts)+'。'
        if index==1:content+='背景资料供核对职业目标：\n'+data['resume']+'\n'+data['jd']
        if index==10:
            facts[-1]={'marker':'F01A','value':9999}
            content='新增独立项目事实：项目 F10A 的日事件量是 1001 条；项目 F10B 的日事件量是 1002 条。更正第 1 轮的项目 F01A：日事件量为 9999 条，原来的 101 条是错误值。其余项目事实不变。'
        rounds.append({'content':content,'facts':facts})
    worker_path=repo/'src/career_agent/agent/providers/conversation_summary.py'
    protocol={'fixed_rounds':10,'independent_chains':1,'model':'qwen3.7-plus','disable_thinking':True,'max_output_tokens':1200,
        'prompt_sha256':hashlib.sha256(OpenAIConversationSummaryWorker._budgeted_system_prompt(1200).encode()).hexdigest(),
        'worker_source_sha256':hashlib.sha256(worker_path.read_bytes()).hexdigest(),
        'rounds':rounds,'grading':['Every introduced marker and its numeric value must be present in active or archived facts after every compaction.','Only F01A=101 may be superseded, at round 10, and its replacement F01A=9999 must exist.','JSON validity alone is not a pass; record missing markers, unexpected supersessions and all failures.','One dependent chain; no independent success-rate or latency-baseline claim.','Keep source messages and the fact ledger snapshot for reproducible manual audit; all content is authored test data.']}
    if args.preflight:
        args.root.mkdir(parents=True,exist_ok=True)
        p=args.root/'protocol.json'
        if p.exists():raise ValueError('Protocol exists')
        p.write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n')
        print('Frozen 10 dependent rounds, 29 unique markers and one correction; no provider calls.')
        return
    assert json.loads((args.root/'protocol.json').read_text())==protocol
    output=args.root/'attempts.json'
    if output.exists():raise ValueError('Attempts exist')
    load_dotenv(repo/'.env')
    for suffix in ('BASE_URL','API_KEY','MODEL'):os.environ['MAIN_AGENT_'+suffix]=os.environ['RESUME_ANALYSIS_AGENT_'+suffix]
    os.environ['CONVERSATION_SUMMARY_AGENT_DISABLE_THINKING']='true'
    config=ConversationSummaryAgentConfig.from_env(main_config=OpenAICompatibleAgentConfig.from_env(prefix='MAIN_AGENT'))
    assert config.provider.model==protocol['model'] and config.max_output_tokens==1200 and config.disable_thinking
    worker=OpenAIConversationSummaryWorker(config.provider,max_output_tokens=1200,disable_thinking=True)
    client=worker._client
    usage=[]
    from types import SimpleNamespace
    def observed(**kwargs):
        response=client.chat.completions.create(**kwargs)
        usage.append({'finish_reason':response.choices[0].finish_reason if response.choices else None,
            'completion_tokens':getattr(getattr(response,'usage',None),'completion_tokens',None)})
        return response
    worker._client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=observed)))
    store=CareerContextStore(args.root/'context.sqlite3')
    seed=ContextManager(store)
    manager=ContextManager(store,summary_worker=worker,summary_batch_size=2)
    expected={};rows=[]
    for index, round_ in enumerate(rounds,1):
        context=seed.load_for_turn(user_id='marker-fixture',conversation_id='chain',user_message=round_['content'])
        seed.commit_turn(context=context,task=ConversationTaskState(),assistant_message='已收到这些项目事实。')
        expected.update((fact['marker'],fact['value']) for fact in round_['facts'])
        start=perf_counter();before=store.get_conversation_summary(user_id='marker-fixture',conversation_id='chain')
        manager._compact_one_batch(user_id='marker-fixture',conversation_id='chain',trigger='seam',phase='commit',require_occupancy=False,
            measure=lambda boundary:manager._recent_pressure(user_id='marker-fixture',conversation_id='chain',after_sequence=boundary,user_message=''))
        summary=store.get_conversation_summary(user_id='marker-fixture',conversation_id='chain')
        active=store.list_summary_facts(user_id='marker-fixture',conversation_id='chain',statuses=('active',))
        archived=store.list_summary_facts(user_id='marker-fixture',conversation_id='chain',statuses=('archived',))
        retired=store.list_summary_facts(user_id='marker-fixture',conversation_id='chain',statuses=('superseded',))
        missing=[marker for marker,value in expected.items() if not any(marker in fact.fact and str(value) in fact.fact for fact in (*active,*archived))]
        unexpected=[f.model_dump() for f in retired if index<10 or 'F01A' not in f.fact or any(marker in f.fact for marker in expected if marker!='F01A')]
        row={'round':index,'wall_ms':round((perf_counter()-start)*1000),'advanced':summary is not None and summary.through_sequence==(index*2),
            'active_count':len(active),'archived_count':len(archived),'superseded_count':len(retired),'missing_markers':missing,'unexpected_supersessions':unexpected,
            'ledger':{name:[fact.model_dump() for fact in facts] for name,facts in [('active',active),('archived',archived),('superseded',retired)]}}
        if usage:row['provider']=usage.pop(0)
        row['passed']=row['advanced'] and not missing and not unexpected
        if index==10:
            row['correction_retired']=any('F01A' in f.fact and '101' in f.fact for f in retired)
            row['passed']=row['passed'] and row['correction_retired']
        rows.append(row);output.write_text(json.dumps({'protocol':protocol,'attempts':rows},ensure_ascii=False,indent=2)+'\n')
        print(json.dumps({k:v for k,v in row.items() if k!='ledger'},ensure_ascii=False),flush=True)
    manager.close();seed.close()
if __name__=='__main__':main()
