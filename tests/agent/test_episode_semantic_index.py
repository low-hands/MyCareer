from datetime import datetime, timezone
import sqlite3

import pytest

from career_agent.agent.context.episode_semantic import SQLiteCareerEpisodeSemanticIndex
from career_agent.agent.capabilities.registry import MainAgentToolRegistry
from career_agent.domain.episodes import CareerEpisodeDraft
from career_agent.storage.episodes import SQLiteCareerEpisodeStore
from career_agent.storage.context import CareerContextStore
from career_agent.evaluation.embedding_tape import EmbeddingTape, MissingEmbeddingRecording


class Client:
    model_id = 'fixed-test-v1'

    def __init__(self):
        self.calls = []
        self.fail = False

    def embed(self, texts):
        self.calls.append(tuple(texts))
        if self.fail:
            raise ConnectionError('synthetic failure')
        return [(1., 0.) if ('限流' in text or '高并发' in text) else (0., 1.) for text in texts]


def draft(run='one', *, user='u1', summary='练习限流和库存扣减', kind='mock_interview', title='系统设计'):
    return CareerEpisodeDraft(user_id=user, source_run_id=run, kind=kind,
        title=title, summary=summary, occurred_at=datetime(2026, 9, 1, tzinfo=timezone.utc))


@pytest.fixture
def setup_index(tmp_path):
    store = SQLiteCareerEpisodeStore(tmp_path/'context.sqlite3')
    client = Client()
    index = SQLiteCareerEpisodeSemanticIndex(episode_path=store.path,
        cache_path=tmp_path/'career_embeddings.sqlite3', client=client)
    yield store, client, index
    index.close()


def test_lazy_vectors_digest_and_model_namespace(setup_index):
    store, client, index = setup_index
    episode = store.upsert(draft())
    store.semantic_index = index
    assert store.search(user_id='u1', query='高并发')[0].matched_by == 'semantic'
    assert client.calls[0] == ('系统设计\n练习限流和库存扣减',)
    store.search(user_id='u1', query='高并发')
    assert len(client.calls) == 3  # episode once, query twice
    # Summary-only change invalidates the digest without changing episode ID.
    store.semantic_index = None
    store.upsert(draft(summary='新的摘要'))
    store.semantic_index = index
    store.search(user_id='u1', query='高并发')
    assert client.calls[-2] == ('系统设计\n新的摘要',)
    client.model_id = 'fixed-test-v2'
    store.search(user_id='u1', query='高并发')
    with sqlite3.connect(index._cache_path) as conn:
        rows = conn.execute('SELECT episode_id,model_id,dimensions FROM career_episode_embeddings').fetchall()
    assert rows == [(episode.id, 'fixed-test-v1', 2), (episode.id, 'fixed-test-v2', 2)]


def test_filter_before_embedding_and_both_route(setup_index):
    store, client, index = setup_index
    selected = store.upsert(draft())
    store.upsert(draft('private', user='other', summary='私人数据'))
    store.upsert(draft('wrong-kind', kind='application', summary='不应向量化'))
    store.semantic_index = index
    result = store.search(user_id='u1', query='限流', kinds=('mock_interview',))
    assert [item.id for item in result] == [selected.id]
    assert result[0].matched_by == 'both'
    assert client.calls[0] == ('系统设计\n练习限流和库存扣减',)
    assert store.search(user_id='u1', query='高并发', start_datetime=datetime(2026,10,1,tzinfo=timezone.utc)) == ()
    assert len(client.calls) == 2
    with pytest.raises(ValueError):
        store.search(user_id='u1',query='高并发',kinds=('invalid',))
    assert len(client.calls) == 2


def test_failure_falls_back_and_traces(setup_index, monkeypatch):
    store, client, index = setup_index
    expected = store.upsert(draft())
    store.semantic_index = index
    client.fail = True
    events = []
    monkeypatch.setattr('career_agent.agent.context.episode_semantic.record_active_trace',
                        lambda *args, **kwargs: events.append((args, kwargs)))
    assert store.search(user_id='u1',query='限流')[0].id == expected.id
    assert events[0][0][0] == 'episode_semantic_fallback'
    assert '限流' not in str(events)


def test_delete_sync_all_models_and_orphan_cleanup(setup_index):
    store, client, index = setup_index
    episode = store.upsert(draft(),memory_scope_keys=('scope',))
    store.semantic_index = index
    store.search(user_id='u1',query='高并发')
    with store._connect() as conn:
        assert store.delete_for_scope_on(conn,user_id='u1',scope_key='scope',semantic_index=index) == 1
    with sqlite3.connect(index._cache_path) as conn:
        assert conn.execute('SELECT count(*) FROM career_episode_embeddings').fetchone()[0] == 0
    # Simulate separate-store deletion / process failure before cleanup.
    store.semantic_index = None
    other = store.upsert(draft('two'))
    store.semantic_index = index
    store.search(user_id='u1',query='高并发')
    with store._connect() as conn:
        conn.execute('DELETE FROM career_episodes WHERE id=?',(other.id,))
    assert store.search(user_id='u1',query='高并发') == ()
    index.prune_orphans()  # explicit maintenance, not search
    with sqlite3.connect(index._cache_path) as conn:
        assert conn.execute('SELECT count(*) FROM career_episode_embeddings').fetchone()[0] == 0


def test_background_writes_and_deleted_queue_cannot_revive(setup_index):
    store, client, index = setup_index
    store.semantic_index = index
    episode = store.upsert(draft())
    index.close()
    assert client.calls == [('系统设计\n练习限流和库存扣减',)]
    with store._connect() as conn:
        conn.execute('DELETE FROM career_episodes WHERE id=?',(episode.id,))
    index._warm((episode.id,))
    with sqlite3.connect(index._cache_path) as conn:
        assert conn.execute('SELECT count(*) FROM career_episode_embeddings').fetchone()[0] == 0


def test_context_transaction_warms_only_after_commit(setup_index):
    store, client, index = setup_index
    context = CareerContextStore(store.path, episode_semantic_index=index)
    from career_agent.agent.contracts.task_state import ConversationTaskState
    context.upsert_task(user_id='u1',conversation_id='c1',task=ConversationTaskState(),episode_drafts=(draft(),))
    index.close()
    assert client.calls == [('系统设计\n练习限流和库存扣减',)]


def test_tool_and_projection_report_matching_route(setup_index):
    store, client, index = setup_index
    store.upsert(draft())
    store.semantic_index = index
    observation = MainAgentToolRegistry(episode_store=store).invoke_atomic_tool(
        'search_career_episodes',{'user_id':'u1','query':'高并发'})
    assert observation.payload['items'][0]['matched_by'] == 'semantic'
    assert store.project_relevant(user_id='u1',query='高并发')[0].matched_by == 'semantic'


def test_embedding_tape_replays_both_texts_and_detects_swallowed_missing(setup_index):
    store, client, index = setup_index
    recording = EmbeddingTape(client)
    assert recording.embed(('标题\n摘要','查询')) == [[0.,1.],[0.,1.]]
    tape = EmbeddingTape(recording=recording.snapshot())
    assert tape.embed(('标题\n摘要','查询')) == [[0.,1.],[0.,1.]]
    assert len(client.calls) == 1
    index._client = tape
    store.upsert(draft())
    store.semantic_index = index
    store.search(user_id='u1',query='未录制查询')
    with pytest.raises(MissingEmbeddingRecording):
        tape.verify()


def test_bad_dimensions_fallback_and_limit_validation(setup_index):
    store, client, index = setup_index
    episode = store.upsert(draft())
    store.semantic_index = index
    with pytest.raises(ValueError):
        store.search(user_id='u1',query='限流',limit=0)
    assert not client.calls
    store.search(user_id='u1',query='限流')
    with sqlite3.connect(index._cache_path) as conn:
        conn.execute("UPDATE career_episode_embeddings SET vector_json='[1,0,0]', dimensions=3")
    assert store.search(user_id='u1',query='限流')[0].id == episode.id
    assert store.search(user_id='u1',query='限流')[0].matched_by == 'lexical'


def test_provider_race_cannot_return_deleted_episode(setup_index):
    store, client, index = setup_index
    episode = store.upsert(draft())
    store.semantic_index = index
    original = client.embed

    def deleting_embed(texts):
        vectors = original(texts)
        if texts == ('高并发',):
            with store._connect() as conn:
                conn.execute('DELETE FROM career_episodes WHERE id=?',(episode.id,))
        return vectors

    client.embed = deleting_embed
    assert store.search(user_id='u1',query='高并发') == ()
    store.search(user_id='u1',query='高并发')
    index.prune_orphans()
    with sqlite3.connect(index._cache_path) as conn:
        assert conn.execute('SELECT count(*) FROM career_episode_embeddings').fetchone()[0] == 0


def test_projection_applies_time_kind_and_conversation_scope_before_embeddings(setup_index):
    store, client, index = setup_index
    selected = store.upsert(draft())
    store.upsert(draft('wrong-kind', kind='application'))
    store.upsert(draft('current').model_copy(update={'conversation_id':'current'}))
    store.semantic_index = index
    hits = store.project_relevant(user_id='u1',query='高并发',kinds=('mock_interview',),
        exclude_conversation_id='current',start_datetime=datetime(2026,9,1,tzinfo=timezone.utc))
    assert [item.id for item in hits] == [selected.id]
    assert len(client.calls[0]) == 1
    assert store.project_relevant(user_id='u1',query='高并发',
        end_datetime=datetime(2026,8,31,tzinfo=timezone.utc)) == ()


def test_turn_query_cache_expires_and_keeps_model_namespaces(setup_index):
    from career_agent.agent.context.episode_semantic import episode_query_vector_scope
    store, client, index = setup_index
    store.upsert(draft())
    store.semantic_index = index
    with episode_query_vector_scope():
        store.project_relevant(user_id='u1', query='高并发')
        store.project_relevant(user_id='u1', query='高并发')
        store.search(user_id='u1', query='高并发')
        assert client.calls.count(('高并发',)) == 1
        client.model_id = 'another-model'
        store.search(user_id='u1', query='高并发')
        assert client.calls.count(('高并发',)) == 2
    with episode_query_vector_scope():
        store.search(user_id='u1', query='高并发')
    assert client.calls.count(('高并发',)) == 3


def test_query_failures_are_not_retried_in_same_turn(setup_index):
    from career_agent.agent.context.episode_semantic import episode_query_vector_scope
    store, client, index = setup_index
    item = store.upsert(draft())
    index._warm((item.id,))
    store.semantic_index = index
    client.fail = True
    with episode_query_vector_scope():
        assert store.search(user_id='u1', query='限流')[0].matched_by == 'lexical'
        assert store.search(user_id='u1', query='限流')[0].matched_by == 'lexical'
    assert client.calls.count(('限流',)) == 1


def test_background_network_does_not_block_search_or_deletion(setup_index):
    from threading import Event
    from concurrent.futures import ThreadPoolExecutor
    store, client, index = setup_index
    item = store.upsert(draft(), memory_scope_keys=('scope',))
    entered, release = Event(), Event()
    original = client.embed
    def blocked(texts):
        entered.set()
        assert release.wait(5)
        return original(texts)
    client.embed = blocked
    index.schedule_episode_ids((item.id,))
    assert entered.wait(2)
    store.semantic_index = index
    def search_and_delete():
        assert store.search(user_id='u1', query='限流')[0].matched_by == 'lexical'
        with store._connect() as conn:
            assert store.delete_for_scope_on(conn, user_id='u1', scope_key='scope', semantic_index=index) == 1
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(search_and_delete)
            try:
                future.result(timeout=2)
            finally:
                release.set()
    finally:
        release.set()
    index.close()
    with sqlite3.connect(index._cache_path) as conn:
        assert conn.execute('SELECT count(*) FROM career_episode_embeddings').fetchone()[0] == 0


def test_semantic_top_k_and_hot_search_do_not_prune(setup_index, monkeypatch):
    store, client, index = setup_index
    for n in range(25):
        store.upsert(draft(str(n), summary='无关事件'))
    store.semantic_index = index
    monkeypatch.setattr(index, 'prune_orphans', lambda: pytest.fail('search must not prune'))
    assert len(store.search(user_id='u1', query='完全没有这个名字', limit=100)) == 20
    assert len(store.search(user_id='u1', query='完全没有这个名字', limit=100)) == 20


def test_embedding_cannot_persist_changed_text(setup_index):
    store, client, index = setup_index
    item = store.upsert(draft())
    original = client.embed
    def editing(texts):
        vectors = original(texts)
        store.upsert(draft(summary='已改写'))
        return vectors
    client.embed = editing
    assert index.rank_episode_ids(query='高并发', candidates=(item,)) == ()
    with sqlite3.connect(index._cache_path) as conn:
        assert conn.execute('SELECT count(*) FROM career_episode_embeddings').fetchone()[0] == 0


def test_runtime_turn_establishes_and_releases_query_scope():
    from career_agent.agent.context.episode_semantic import _TURN_QUERIES
    from career_agent.agent.runtime.main_agent_runtime import MainAgentRuntime
    from types import SimpleNamespace
    scopes = []
    def run(**kwargs):
        assert _TURN_QUERIES.get() is not None
        scopes.append(_TURN_QUERIES.get())
        return 'done'
    runtime = object.__new__(MainAgentRuntime)
    runtime._turn_coordinator = SimpleNamespace(run=run)
    for user in ('one', 'two'):
        assert runtime.run_turn(user_id=user, conversation_id='c', user_message='q') == 'done'
        assert _TURN_QUERIES.get() is None
    assert scopes[0] is not scopes[1]


def test_startup_backfills_batches_without_foreground_backlog(setup_index):
    from threading import Event
    store, client, index = setup_index
    index._batch_size = 2
    items=[store.upsert(draft(str(i),summary='不同事件'+str(i))) for i in range(5)]
    entered,release=Event(),Event()
    original=client.embed
    def blocked(texts):
        if not entered.is_set():
            entered.set()
            assert release.wait(5)
        return original(texts)
    client.embed=blocked
    index.schedule_backfill()
    assert entered.wait(2)
    store.semantic_index=index
    try:
        assert store.search(user_id='u1',query='不同事件',limit=10)
        assert client.calls==[]  # no query embed and no foreground missing batches
    finally:
        release.set()
    index.close()
    assert [len(batch) for batch in client.calls]==[2,2,1]
    assert not index._backfilling.is_set()
    with sqlite3.connect(index._cache_path) as conn:
        assert conn.execute('SELECT count(*) FROM career_episode_embeddings').fetchone()[0]==5


def test_backfill_reuses_current_vectors_and_updates_stale_digests(setup_index):
    store, client, index = setup_index
    one=store.upsert(draft('one'))
    two=store.upsert(draft('two'))
    index._warm((one.id,two.id))
    client.calls.clear()
    store.upsert(draft('two',summary='新摘要'))
    index.schedule_backfill()
    index.close()
    assert client.calls==[('系统设计\n新摘要',)]


def test_backfill_failure_clears_flag_and_does_not_fail_foreground(setup_index):
    store, client, index = setup_index
    store.upsert(draft())
    client.fail=True
    index.schedule_backfill()
    index.close()
    assert not index._backfilling.is_set()
    store.semantic_index=index
    assert store.search(user_id='u1',query='限流')[0].matched_by=='lexical'


def test_configured_factory_starts_existing_episode_backfill(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from career_agent.agent.context import episode_semantic as module
    store=SQLiteCareerEpisodeStore(tmp_path/'context.sqlite3')
    store.upsert(draft())
    client=Client()
    monkeypatch.setattr(module.CareerEmbeddingConfig,'optional_from_env',
                        lambda **kwargs:SimpleNamespace(batch_size=2))
    monkeypatch.setattr(module,'OpenAICompatibleEmbeddingClient',lambda config:client)
    index=module.optional_episode_semantic_index(episode_path=store.path,
        cache_path=tmp_path/'career_embeddings.sqlite3',environ={})
    index.close()
    assert client.calls==[('系统设计\n练习限流和库存扣减',)]
