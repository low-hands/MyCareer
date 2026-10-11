from career_agent.storage.aliases import expand_organization_aliases
from career_agent.storage.episodes import SQLiteCareerEpisodeStore
from career_agent.storage.career_history import CareerHistoryStore


def test_expansion_is_conservative_idempotent_and_keeps_surface_terms():
    assert expand_organization_aliases('ByteDance 一面') == 'ByteDance 一面 字节跳动'
    assert expand_organization_aliases('Pinduoduo') == 'Pinduoduo 拼多多'
    assert expand_organization_aliases('腾讯 Tencent') == '腾讯 Tencent'
    assert expand_organization_aliases('PDDish myByteDance_id') == 'PDDish myByteDance_id'
    query = expand_organization_aliases('ByteDance 二面')
    assert expand_organization_aliases(query) == query


def test_episode_and_fact_tokenizers_share_canonical_aliases(tmp_path):
    store = SQLiteCareerEpisodeStore(tmp_path/'context.sqlite3')
    history = CareerHistoryStore(tmp_path/'resume.sqlite3')
    assert '拼多多' in store.query_terms(user_id='u1',query='Pinduoduo')
    assert '拼多多' in history.current_evidence_query_terms(user_id='u1',query='Pinduoduo').match_tokens
    assert '字节跳' in history.current_evidence_query_terms(user_id='u1',query='ByteDance').match_tokens


def test_job_library_search_expands_the_same_aliases(tmp_path):
    from datetime import datetime, timezone
    from career_agent.domain.job_discovery import JobDetail, Provenance
    from career_agent.storage.jobs import SQLiteJobPostingRepository
    now = datetime(2026,9,1,tzinfo=timezone.utc)
    repository = SQLiteJobPostingRepository(tmp_path/'jobs.sqlite3')
    detail = JobDetail(source_name='fixture',source_job_id='pdd-1',source_url='https://example.test/1',
        title='算法工程师',company_name='拼多多',description='推荐算法研发',city='上海',
        captured_at=now,provenance=Provenance(source_name='fixture',source_job_id='pdd-1',
            source_url='https://example.test/1',captured_at=now,operation='detail',adapter_version='fixture'))
    saved = repository.save_captured_detail(user_id='u1',detail=detail)
    for query in ('Pinduoduo','拼多多'):
        hits=repository.search_saved_jobs(user_id='u1',query=query,include_dismissed=False)
        assert [hit.job_posting_id for hit in hits] == [saved.posting.id]
    assert repository.search_saved_jobs(user_id='other',query='Pinduoduo',include_dismissed=False) == ()


def test_technical_and_product_words_do_not_expand():
    for query in ('JVM 字节码 面试', '字节数怎么计算', '头条新闻 岗位', '字节', '头条', 'PDD', '拼夕夕', '鹅厂'):
        assert expand_organization_aliases(query) == query


def test_next_alias_dataset_is_separate_and_contains_unknown_aliases():
    import json
    from pathlib import Path
    from career_agent.storage.aliases import ORGANIZATION_ALIASES
    root = Path(__file__).resolve().parents[2] / 'evals/episode_retrieval/alias_v3'
    cases = json.loads((root / 'cases.json').read_text(encoding='utf-8'))
    known = {a.casefold() for aliases in ORGANIZATION_ALIASES.values() for a in aliases}
    for category in ('known_identity', 'unseen_alias', 'ambiguous_negative'):
        selected = [c for c in cases if c['category'] == category]
        assert len(selected) >= 12
        assert {c['split'] for c in selected} == {'dev', 'test'}
    for case in cases:
        if case['category'] == 'unseen_alias':
            assert case['surface'].casefold() not in known
            if case['split'] == 'dev':
                assert expand_organization_aliases(case['query']) == case['query']
