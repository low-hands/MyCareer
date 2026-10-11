"""Isolated E2E: model-generated query -> production episode tools -> final answer.

This measures episode answering, not the entire MainAgentRuntime. Request tapes
match schemas, projected memory, messages and model options exactly. Offline
replay constructs no provider clients. Test split has no recording entrypoint:
release it only after the frozen development gates pass.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import sqlite3
import sys
import tempfile

HERE = Path(__file__).resolve().parent
SYSTEM = ('你是求职事件记忆助手。当前时间 2026-10-08T12:00:00+08:00。'
          '根据用户问题查找已保存的求职事件并回答。只能使用提供的记录；'
          '记录没有支持用户问到的事件时如实说明没有找到，不猜测结果。')
# Load beside this runner so frozen lexical --source-root does not change the grader.
_grader_spec = importlib.util.spec_from_file_location('episode_deterministic_grader', HERE/'deterministic_grading.py')
_grader = importlib.util.module_from_spec(_grader_spec)
_grader_spec.loader.exec_module(_grader)
grade_answer = _grader.grade_answer



class StaleRequest(ValueError):
    pass


class RequestTape:
    def __init__(self, client=None, recording=None):
        self.client = client
        self.rows = [] if recording is None else recording
        self.position = 0

    def complete(self, **request):
        request = json.loads(json.dumps(request, ensure_ascii=False))
        if self.client is None:
            if self.position >= len(self.rows) or self.rows[self.position]['request'] != request:
                raise StaleRequest('actual model request differs from recording')
            row = self.rows[self.position]
            self.position += 1
        else:
            row = {'request': request}
            self.rows.append(row)
            try:
                response = self.client.chat.completions.create(**request)
                row['response'] = response.model_dump(mode='json')
            except Exception as error:
                row['error_type'] = type(error).__name__
        if 'error_type' in row:
            raise RuntimeError('recorded provider failure: '+row['error_type'])
        return row['response']['choices'][0]['message']

    def verify(self):
        if self.client is None and self.position != len(self.rows):
            raise StaleRequest('unused model responses remain')


def run_one(case, number, data, args, model, options, client, embedding_client):
    target = args.root/f"{case['id']}-{number:02}.json"
    if args.record and target.exists():
        raise ValueError('refusing to overwrite any attempt')
    recorded = None if args.record else json.loads(target.read_text())
    if recorded is not None and recorded['fixture_sha256'] != args.fixture_digest:
        return {'id':case['id'],'sample':number,'status':'stale','passed':False}
    from career_agent.domain.episodes import CareerEpisodeDraft
    from career_agent.storage.episodes import SQLiteCareerEpisodeStore
    from career_agent.agent.capabilities.registry import MainAgentToolRegistry
    requests = None if recorded is None else recorded['requests']
    historical_judge = []
    if recorded is not None and 'grading_version' not in recorded:
        # Legacy judge is a suffix, identified by its request shape, never by score.
        if (requests and requests[-1]['request'].get('response_format') == {'type':'json_object'}
                and requests[-1]['request'].get('messages',[{}])[0].get('content','').startswith(
                    '判断最终回答是否正确回应用户询问的事件和事实。')):
            historical_judge = requests[-1:]
            requests = requests[:-1]
    model_tape = RequestTape(client=client, recording=requests)
    embedding = None
    index = None
    result = {'id':case['id'],'category':case['category'],'sample':number,'passed':False,'status':'valid',
              'grading_version':_grader.GRADING_VERSION,'grading_digest':_grader.GRADING_DIGEST}
    with tempfile.TemporaryDirectory() as tmp:
        store = SQLiteCareerEpisodeStore(Path(tmp)/'context.sqlite3')
        for item in data['corpus']:
            episode = store.upsert(CareerEpisodeDraft(user_id='eval_user',source_run_id=item['id'],
                kind=item['kind'],title=item['title'],summary=item['summary'],
                occurred_at=datetime.fromisoformat(item['occurred_at'])))
            stable = 'career_episode_'+hashlib.md5(item['id'].encode()).hexdigest()
            with sqlite3.connect(store.path) as conn:
                conn.execute('UPDATE career_episodes SET id=? WHERE id=?',(stable,episode.id))
                conn.execute('UPDATE career_episodes_fts SET episode_id=? WHERE episode_id=?',(stable,episode.id))
        if args.arm=='hybrid':
            from career_agent.agent.context.episode_semantic import SQLiteCareerEpisodeSemanticIndex
            embedding = EmbeddingTape(client=embedding_client) if args.record else EmbeddingTape(recording=recorded['embedding'])
            index = SQLiteCareerEpisodeSemanticIndex(episode_path=store.path,
                cache_path=Path(tmp)/'career_embeddings.sqlite3',client=embedding)
            store.semantic_index = index
        registry = MainAgentToolRegistry(episode_store=store)
        schemas = [schema for schema in registry.schemas() if schema['function']['name']=='search_career_episodes']
        try:
            projection = store.project_relevant(user_id='eval_user',query=case['user_message'])
            # Model sees a real projection and generates every query itself.
            context = [{key:value for key,value in item.model_dump(mode='json').items()
                        if key in ('kind','title','summary','occurred_at','matched_by')}
                       | {'detail_ref':'episode:'+item.id} for item in projection]
            messages = [{'role':'system','content':SYSTEM+'\n过往事件目录：'+json.dumps(context,ensure_ascii=False)},
                        {'role':'user','content':case['user_message']}]
            for step in range(5):
                message = model_tape.complete(model=model,messages=messages,tools=schemas,
                    max_tokens=1600,temperature=0,
                    tool_choice={"type":"function","function":{"name":"search_career_episodes"}} if step==0 else "auto",
                    **options)
                calls = message.get('tool_calls') or []
                if not calls:
                    result['answer'] = message.get('content') or ''
                    break
                messages.append({key:value for key,value in message.items()
                                 if key in ('role','content','tool_calls') and value is not None})
                for call in calls:
                    if call['function']['name'] != 'search_career_episodes':
                        raise ValueError('unexpected tool')
                    arguments = json.loads(call['function']['arguments'])
                    observation = registry.invoke_atomic_tool('search_career_episodes',
                        {**arguments,'user_id':'eval_user'})
                    messages.append({'role':'tool','tool_call_id':call['id'],
                                     'content':json.dumps(observation.payload,ensure_ascii=False)})
            else:
                raise ValueError('model did not finish within frozen five-step budget')
            result['judgement'] = grade_answer(case, data, result['answer'])
            result['passed'] = result['judgement']['passed']
            result['grading_version'] = _grader.GRADING_VERSION
            result['grading_digest'] = _grader.GRADING_DIGEST
            if historical_judge:
                result['historical_judge_preserved'] = True
        except StaleRequest as error:
            result.update(status='stale',error=str(error))
        except Exception as error:
            result.update(status='infrastructure_error',error_type=type(error).__name__)
        finally:
            if index is not None:
                index.close()
            if embedding is not None:
                try:
                    embedding.verify()
                except Exception:
                    result.update(status='missing_embedding_recording',passed=False, embedding_recording_status='missing')
            try:
                model_tape.verify()
            except StaleRequest:
                result.update(status='stale',passed=False, request_recording_status='stale')
    if args.record:
        payload = {**result,'fixture_sha256':args.fixture_digest,'requests':model_tape.rows,
                   'embedding':embedding.snapshot() if embedding is not None else None}
        target.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--record',action='store_true')
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--source-root',type=Path,default=HERE.parents[1]/'src')
    parser.add_argument('--arm',choices=['lexical','hybrid'],required=True)
    parser.add_argument('--model',required=True)
    parser.add_argument('--env-prefix',default='MAIN_AGENT')
    parser.add_argument('--vendor',choices=['qwen','deepseek'],required=True)
    parser.add_argument('--jobs',type=int,default=8)
    parser.add_argument('--all-dev',action='store_true')
    args=parser.parse_args()
    sys.path.insert(0,str(args.source_root.resolve()))
    # Recording/replay helper has no production dependencies, including baseline source.
    spec=importlib.util.spec_from_file_location('episode_embedding_tape',HERE.parents[1]/'src/career_agent/evaluation/embedding_tape.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    global EmbeddingTape
    EmbeddingTape=module.EmbeddingTape
    fixture=HERE/'frozen_v2/scenarios.json'
    data=json.loads(fixture.read_text())
    protocol=json.loads(fixture.with_name('protocol.json').read_text())
    args.fixture_digest=hashlib.sha256(fixture.read_bytes()).hexdigest()
    assert args.fixture_digest == _grader.PROTOCOL['fixture_sha256'], 'grader fixture changed'
    assert args.fixture_digest==protocol['scenarios_sha256']
    cases=[case for case in data['queries'] if case['split']=='dev' and
           (args.all_dev or case['id'] in protocol['e2e_experiment_ids'])]
    options={'extra_body':{'enable_thinking':False}} if args.vendor=='qwen' else {'extra_body':{'thinking':{'type':'disabled'}}}
    client=embedding_client=None
    if args.record:
        from dotenv import load_dotenv
        load_dotenv()
        from openai import OpenAI
        from career_agent.agent.providers.openai_client import OpenAICompatibleAgentConfig
        config=OpenAICompatibleAgentConfig.from_env(prefix=args.env_prefix)
        assert config.model==args.model
        endpoint=config.endpoint.removesuffix('/chat/completions')
        client=OpenAI(base_url=endpoint,api_key=config.api_key,timeout=45,max_retries=0)
        if args.arm=='hybrid':
            from career_agent.agent.context.semantic_retrieval import CareerEmbeddingConfig,OpenAICompatibleEmbeddingClient
            embedding_client=OpenAICompatibleEmbeddingClient(CareerEmbeddingConfig.optional_from_env())
    args.root.mkdir(parents=True,exist_ok=True)
    # A missing recording never falls through to a live provider.
    tasks=[(case,n) for case in cases for n in range(protocol['samples_per_scenario'])]
    results=[]
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures={pool.submit(run_one,case,n,data,args,args.model,options,client,embedding_client):(case,n) for case,n in tasks}
        for future in as_completed(futures):
            case,n=futures[future]
            try:
                result=future.result()
            except FileNotFoundError:
                result={'id':case['id'],'category':case['category'],'sample':n,'status':'unrecorded','passed':False}
            results.append(result)
            if len(results)%20==0:
                print(f"completed {len(results)}/{len(tasks)}",flush=True)
    from collections import Counter
    report={'measurement':'isolated_episode_query_and_final_answer','arm':args.arm,'model':args.model,
        'fixture_sha256':args.fixture_digest,'grading_version':_grader.GRADING_VERSION,
        'grading_digest':_grader.GRADING_DIGEST,'sample_count':len(tasks),
        'samples_passed':sum(row['passed'] and row['status']=='valid' for row in results),
        'grading_status_counts':dict(Counter(row.get('judgement',{}).get('status','not_graded')
            for row in results if row['status']=='valid')),
        'valid_samples':sum(row['status']=='valid' for row in results),
        'pending_human_review':[{'id':row['id'],'sample':row['sample'],'answer':row.get('answer'),
            'judgement':row.get('judgement')} for row in results if row['status']=='valid' and
            row.get('judgement',{}).get('status')=='human_review'],
        'status_counts':dict(Counter(row['status'] for row in results)),
        'results':sorted(results,key=lambda row:(row['id'],row['sample']))}
    (args.root/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({key:value for key,value in report.items() if key!='results'},ensure_ascii=False))
    return int(any(row['status']!='valid' or row.get('judgement',{}).get('status')=='human_review' for row in results))


if __name__=='__main__':
    raise SystemExit(main())
