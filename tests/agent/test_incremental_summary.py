from __future__ import annotations
from threading import Event
import pytest
from career_agent.agent.context.manager import ContextManager
from career_agent.agent.context.summary_facts import fact_id
from career_agent.agent.contracts.memory import ConversationSummaryDelta, SummaryKeyFact
from career_agent.agent.contracts.task_state import ConversationTaskState
from career_agent.storage.context import CareerContextStore

class DeltaWorker:
    def __init__(self):
        self.correct = None
    def extract_delta(self, *, previous, messages, known_facts):
        # Each new source contains one uniquely identifiable fact.
        return ConversationSummaryDelta(new_facts=tuple(SummaryKeyFact(fact=m.content, source_sequence=m.sequence) for m in messages),
            superseded_fact_ids=(self.correct,) if self.correct else ())


def test_fact_overflow_is_archived_and_explicit_correction_only_retires_its_id(tmp_path):
    store=CareerContextStore(tmp_path/'memory.sqlite3')
    worker=DeltaWorker()
    manager=ContextManager(store,summary_worker=worker,recent_message_limit=2,summary_batch_size=2)
    originals=[]
    for i in range(10):
        context=manager.load_for_turn(user_id='u',conversation_id='c',user_message=f'fact_{i}: count={i*100}')
        manager.commit_turn(context=context,task=ConversationTaskState(),assistant_message=f'confirmed_{i}')
        originals += [f'fact_{i}: count={i*100}',f'confirmed_{i}']
    facts=store.list_summary_facts(user_id='u',conversation_id='c')
    assert {f.fact for f in facts}==set(originals[:18])
    summary=store.get_conversation_summary(user_id='u',conversation_id='c')
    assert len(summary.content.key_facts)==12
    assert len(store.list_summary_facts(user_id='u',conversation_id='c',statuses=('archived',)))==6
    archived=facts[0]
    worker.correct=fact_id(archived)
    context=manager.load_for_turn(user_id='u',conversation_id='c',user_message='correction next')
    manager.commit_turn(context=context,task=ConversationTaskState(),assistant_message='replacement confirmed')
    assert archived not in store.list_summary_facts(user_id='u',conversation_id='c')
    assert archived in store.list_summary_facts(user_id='u',conversation_id='c',statuses=('superseded',))
    assert len(store.list_summary_facts(user_id='u',conversation_id='c'))==19
    assert store.list_summary_facts(user_id='other',conversation_id='c')==()
    reopened=CareerContextStore(store.path)
    assert reopened.list_summary_facts(user_id='u',conversation_id='c')==store.list_summary_facts(user_id='u',conversation_id='c')


def seed(store):
    manager=ContextManager(store)
    for i in range(5):
        context=manager.load_for_turn(user_id='u',conversation_id='c',user_message=f'fact_{i}')
        manager.commit_turn(context=context,task=ConversationTaskState(),assistant_message=f'answer_{i}')

class BlockingWorker(DeltaWorker):
    def __init__(self):
        super().__init__()
        self.entered=Event();self.release=Event();self.calls=0
    def extract_delta(self, **kwargs):
        self.calls+=1;self.entered.set()
        assert self.release.wait(3)
        return super().extract_delta(**kwargs)

@pytest.mark.parametrize('delete',[False,True])
def test_foreground_does_not_wait_and_background_cannot_resurrect_deleted_conversation(tmp_path,delete):
    store=CareerContextStore(tmp_path/'memory.sqlite3');seed(store)
    worker=BlockingWorker()
    manager=ContextManager(store,summary_worker=worker,background_summary=True,recent_message_limit=2)
    try:
        # Worker is held until after load AND commit have both returned.
        context=manager.load_for_turn(user_id='u',conversation_id='c',user_message='next')
        assert worker.entered.wait(1)
        assert not worker.release.is_set()
        assert context.through_sequence==0
        manager.commit_turn(context=context,task=ConversationTaskState(),assistant_message='done')
        assert worker.calls==1
        assert store.get_conversation_summary(user_id='u',conversation_id='c') is None
        if delete:
            assert store.delete_conversation(user_id='u',conversation_id='c')
    finally:
        worker.release.set();manager.close()
    summary=store.get_conversation_summary(user_id='u',conversation_id='c')
    if delete:
        assert summary is None
        assert store.list_summary_facts(user_id='u',conversation_id='c')==()
    else:
        assert summary.through_sequence==8
        assert len(store.list_summary_facts(user_id='u',conversation_id='c'))==8


def test_background_failures_preserve_sources_and_open_existing_circuit(tmp_path):
    from career_agent.agent.providers.openai_client import AgentWorkerError
    class FailingWorker:
        calls=0
        def summarize(self, **kwargs):
            self.calls+=1
            raise AgentWorkerError("CONVERSATION_SUMMARY_TIMEOUT", "Timeout")
    store=CareerContextStore(tmp_path/'memory.sqlite3');seed(store)
    worker=FailingWorker()
    manager=ContextManager(store,summary_worker=worker,background_summary=True,recent_message_limit=2)
    try:
        for index in range(4):
            manager.load_for_turn(user_id='u',conversation_id='c',user_message='next')
            manager._summary_pool.submit(lambda: None).result(timeout=2)
        assert worker.calls==3
        assert store.get_conversation_summary(user_id='u',conversation_id='c') is None
        assert len(store.list_messages_after(user_id='u',conversation_id='c',after_sequence=0,limit=20))==10
        assert store.list_summary_facts(user_id='u',conversation_id='c')==()
    finally:
        manager.close()


def test_older_visible_facts_are_backfilled_on_schema_upgrade(tmp_path):
    import sqlite3
    from career_agent.agent.contracts.memory import ConversationSummaryContent
    store=CareerContextStore(tmp_path/'memory.sqlite3')
    fact=SummaryKeyFact(fact='salary 300k',source_sequence=1)
    store.compact_conversation_summary(user_id='u',conversation_id='c',expected_previous_through_sequence=0,
        through_sequence=1,content=ConversationSummaryContent(key_facts=(fact,)))
    with sqlite3.connect(store.path) as connection:
        connection.execute('DROP TABLE conversation_fact_archive')
        connection.execute("UPDATE schema_versions SET version=17 WHERE component='agent_context'")
    reopened=CareerContextStore(store.path)
    assert reopened.list_summary_facts(user_id='u',conversation_id='c')==(fact,)
