import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('episode_e2e', ROOT/'evals/episode_retrieval/run_e2e.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_request_replay_matches_exact_content_and_consumes_all():
    rows = [{'request':{'model':'fixed','messages':[{'role':'user','content':'原查询'}]},
             'response':{'choices':[{'message':{'role':'assistant','content':'没找到'}}]}}]
    tape = runner.RequestTape(recording=rows)
    with pytest.raises(runner.StaleRequest):
        tape.complete(model='fixed',messages=[{'role':'user','content':'改变查询'}])
    with pytest.raises(runner.StaleRequest):
        tape.verify()
    assert tape.complete(**rows[0]['request'])['content']=='没找到'
    tape.verify()


def test_recording_snapshots_request_before_messages_mutate():
    response = SimpleNamespace(model_dump=lambda **kw: {'choices':[{'message':{'role':'assistant','content':'答案'}}]})
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw:response)))
    tape=runner.RequestTape(client=client)
    messages=[{'role':'user','content':'查询'}]
    tape.complete(model='fixed',messages=messages)
    messages.append({'role':'assistant','content':'后续'})
    assert len(tape.rows[0]['request']['messages'])==1


def test_frozen_fixture_counts_and_hash():
    import json, hashlib
    from collections import Counter
    fixture=ROOT/'evals/episode_retrieval/frozen_v2/scenarios.json'
    protocol=json.loads(fixture.with_name('protocol.json').read_text())
    data=json.loads(fixture.read_text())
    assert hashlib.sha256(fixture.read_bytes()).hexdigest()==protocol['scenarios_sha256']
    assert min(Counter(row['category'] for row in data['queries']).values())>=12
    assert Counter(row['split'] for row in data['queries'])=={'dev':68,'test':32}
    assert all(row['split']=='dev' for row in data['queries'] if row['id'] in protocol['e2e_experiment_ids'])


def test_retrieval_record_and_replay_are_identical_even_when_all_vectors_tie(tmp_path, monkeypatch):
    import json
    import sys
    spec = importlib.util.spec_from_file_location('episode_hybrid_eval', ROOT/'evals/episode_retrieval/run_hybrid.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class FixedEmbedding:
        model_id = 'scripted-tie-fixture'

        def embed(self, texts):
            return [[1.,0.] for _ in texts]

    monkeypatch.setattr(module.CareerEmbeddingConfig,'optional_from_env',lambda:object())
    monkeypatch.setattr(module,'OpenAICompatibleEmbeddingClient',lambda _:FixedEmbedding())
    monkeypatch.setattr(sys,'argv',['run_hybrid.py','--record','--root',str(tmp_path)])
    module.main()
    expected=(tmp_path/'hybrid-dev-report.json').read_bytes()

    def forbidden(*args, **kwargs):
        raise AssertionError('offline evaluation constructed a service client')

    monkeypatch.setattr(module,'OpenAICompatibleEmbeddingClient',forbidden)
    monkeypatch.setattr(sys,'argv',['run_hybrid.py','--root',str(tmp_path)])
    module.main()
    assert (tmp_path/'hybrid-dev-report.json').read_bytes()==expected
    report=json.loads(expected)
    assert len(report['rows'])==68*3
    assert report['embedding_failures']==0


def test_deterministic_grading_requires_identity_time_round_outcome():
    case={'id':'Q14','relevant':{'E03':2},'user_message':'哪次面试被压力面了'}
    data={'corpus':[{'id':'E03','title':'字节跳动 · 推荐算法工程师 · 第 2 轮面试'},
                    {'id':'other','title':'腾讯 · 后台开发工程师'}]}
    answer='2026年4月9日，字节跳动二面未通过，面试官连续追问。'
    assert runner.grade_answer(case,data,answer)['status']=='passed'
    for part in ('4月9日','字节跳动','二面','未通过','连续追问'):
        assert runner.grade_answer(case,data,answer.replace(part,''))['status']=='failed'
    assert runner.grade_answer(case,data,answer.replace('2026','2025'))['status']=='failed'
    assert runner.grade_answer(case,data,answer.replace('二面','一面'))['status']=='failed'
    assert runner.grade_answer(case,data,answer+' 腾讯一面也聊了项目。')['status']=='human_review'


def test_negative_grading_rejects_other_companies_and_negated_absence():
    case={'id':'none','relevant':{},'user_message':'华为的面试记录'}
    data={'corpus':[{'id':'one','title':'腾讯 · 面试'}, {'id':'two','title':'字节跳动 · 面试'}]}
    assert runner.grade_answer(case,data,'没有找到华为的面试记录。')['status']=='passed'
    assert runner.grade_answer(case,data,'没有找到华为的面试记录，不过腾讯完成一面。')['status']=='failed'
    assert runner.grade_answer(case,data,'未找到华为记录；Tencent 有面试。')['status']=='failed'
    assert runner.grade_answer(case,data,'华为未通过面试。')['status']=='failed'
    assert runner.grade_answer(case,data,'不是没有这条记录。')['status']=='human_review'
    assert runner.grade_answer(case,data,'用户问：没有这条记录吗？')['status']=='human_review'
    assert runner.grade_answer(case,data,'我没找到华为的记录，但有其他记录。')['status']=='human_review'


def test_last_year_cannot_be_answered_with_current_year_and_unruled_needs_review():
    data={'corpus':[{'id':'E03','title':'字节跳动 · 面试'}]}
    absent={'id':'Q42','relevant':{},'user_message':'去年投的字节'}
    assert runner.grade_answer(absent,data,'没有找到2025年的字节跳动投递记录。')['status']=='passed'
    assert runner.grade_answer(absent,data,'没有找到去年记录，2026年投了字节跳动。')['status']=='failed'
    assert runner.grade_answer({'id':'unconfigured','relevant':{'E03':2}},data,'字节跳动的面试')['status']=='human_review'


def test_e2e_records_no_judge_and_regrades_legacy_judge_separately(tmp_path, monkeypatch):
    import json
    from career_agent.evaluation.embedding_tape import EmbeddingTape
    monkeypatch.setattr(runner,'EmbeddingTape',EmbeddingTape,raising=False)
    calls=[]
    response=SimpleNamespace(model_dump=lambda **kw:{'choices':[{'message':{
        'role':'assistant','content':'没有找到华为的面试记录。'}}]})
    def complete(**kw):
        calls.append(kw)
        return response
    client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=complete)))
    args=SimpleNamespace(root=tmp_path, record=True, fixture_digest='test-fixture',arm='lexical')
    data={'corpus':[]}
    case={'id':'missing','category':'unanswerable','relevant':{},'user_message':'华为的面试记录'}
    result=runner.run_one(case,0,data,args,'fixture',{},client,None)
    assert result['passed'] and len(calls)==1
    assert all('response_format' not in call for call in calls)
    path=tmp_path/'missing-00.json'
    recording=json.loads(path.read_text())
    recording.pop('grading_version')
    recording['requests'].append({'request':{'response_format':{'type':'json_object'},'messages':[
        {'role':'system','content':'判断最终回答是否正确回应用户询问的事件和事实。历史裁判'}]},
        'response':{'choices':[{'message':{'content':'{"passed":false}'}}]}})
    path.write_text(json.dumps(recording,ensure_ascii=False))
    original=path.read_bytes()
    args.record=False
    replay=runner.run_one(case,0,data,args,'fixture',{},None,None)
    assert replay['status']=='valid' and replay['passed']
    assert replay['historical_judge_preserved']
    assert len(calls)==1 and path.read_bytes()==original


def test_known_practice_rubrics_and_literal_salary_formats():
    examples=[('Q17','E07','3月5日模拟面试练了秒杀系统，限流和库存扣减清楚，缓存击穿需加强。'),
              ('Q18','E09','9月25日HR面模拟，期望35k*15，被追问现薪时回答被动。'),
              ('Q22','E26','4月1日推荐算法模拟面试，双塔召回讲清楚，冷启动回答空泛。')]
    for id,event,answer in examples:
        data={'corpus':[{'id':event,'title':'模拟面试'}]}
        case={'id':id,'relevant':{event:2}}
        assert runner.grade_answer(case,data,answer)['status']=='passed'
        assert runner.grade_answer(case,data,answer+'2027年。')['status']=='failed'
    case={'id':'Q14','relevant':{'E03':2}}
    data={'corpus':[{'id':'E03','title':'字节跳动 · 面试'}]}
    assert runner.grade_answer(case,data,'4月9日字节跳动二面未通过面试，连续追问。')['status']=='passed'
    assert runner.grade_answer(case,data,'4月9日字节跳动二面未通过，但是后来通过面试，连续追问。')['status']=='failed'


def test_absence_words_cannot_hide_invented_target_outcomes():
    case={'id':'none','relevant':{},'user_message':'华为的面试记录'}
    data={'corpus':[]}
    assert runner.grade_answer(case,data,'华为未通过面试。')['status']=='failed'
    assert runner.grade_answer(case,data,'未通过华为面试，没收到反馈。')['status']=='failed'
    assert runner.grade_answer(case,data,'没找到华为面试记录，华为未通过。')['status']=='human_review'
    assert runner.grade_answer(case,data,'没有这条记录，华为一面是9月1日。')['status']=='human_review'


def test_fact_matches_do_not_accept_longer_numbers_or_product_identifiers():
    case={'id':'Q18','relevant':{'E09':2}}
    data={'corpus':[{'id':'E09','title':'模拟面试'}]}
    answer='9月25日HR面，期望35k*150，回答被动。'
    assert runner.grade_answer(case,data,answer)['status']=='failed'
    assert runner.grade_answer(case,data,answer.replace('35k*150','35k*15').replace('9月25日','9/250'))['status']=='failed'
    case={'id':'Q14','relevant':{'E03':2}}
    data={'corpus':[{'id':'E03','title':'字节跳动 · 面试'}]}
    assert runner.grade_answer(case,data,'4月9日ByteDanceCloud二面未通过，连续追问。')['status']=='failed'


def test_quoting_a_company_or_key_fact_is_not_itself_ambiguous():
    absent={'id':'none','relevant':{},'user_message':'华为的面试记录'}
    assert runner.grade_answer(absent,{'corpus':[]},'没有找到“华为”的面试记录。')['status']=='passed'
    assert runner.grade_answer(absent,{'corpus':[]},'用户只说了“没有华为的面试记录”。')['status']=='human_review'
    case={'id':'Q14','relevant':{'E03':2}}
    data={'corpus':[{'id':'E03','title':'字节跳动 · 面试'}]}
    assert runner.grade_answer(case,data,'4月9日“字节跳动”二面未通过，面试官“连续追问”。')['status']=='passed'
    assert runner.grade_answer(case,data,'可能是4月9日字节跳动二面未通过，连续追问。')['status']=='human_review'
