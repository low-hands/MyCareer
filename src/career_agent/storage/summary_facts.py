"""SQLite ledger behind the bounded conversation fact projection."""
from __future__ import annotations
import json
import sqlite3
from collections.abc import Sequence
from career_agent.agent.contracts.memory import SummaryKeyFact
from career_agent.agent.context.summary_facts import fact_id


def ensure_schema(connection: sqlite3.Connection) -> None:
    connection.execute('''CREATE TABLE IF NOT EXISTS conversation_fact_archive (
        user_id TEXT NOT NULL, conversation_id TEXT NOT NULL, fact_id TEXT NOT NULL,
        fact TEXT NOT NULL, source_sequence INTEGER NOT NULL,
        status TEXT NOT NULL CHECK(status IN ('active','archived','superseded')),
        PRIMARY KEY(user_id, conversation_id, fact_id))''')
    # Backfill pre-delta visible facts. Idempotent; never undo a supersession.
    for user, conversation, payload in connection.execute('SELECT user_id, conversation_id, content_json FROM conversation_summaries').fetchall():
        for raw in json.loads(payload).get('key_facts', []):
            fact=SummaryKeyFact.model_validate(raw)
            connection.execute('INSERT OR IGNORE INTO conversation_fact_archive VALUES (?,?,?,?,?,?)',
                (user, conversation, fact_id(fact),fact.fact,fact.source_sequence,'active'))


def reconcile(connection: sqlite3.Connection, *, user_id: str, conversation_id: str, facts: Sequence[SummaryKeyFact], visible: Sequence[SummaryKeyFact], superseded: Sequence[str]) -> None:
    visible_ids = {fact_id(fact) for fact in visible}
    for fact in facts:
        connection.execute('''INSERT INTO conversation_fact_archive VALUES (?,?,?,?,?,?)
            ON CONFLICT(user_id,conversation_id,fact_id) DO UPDATE SET status=excluded.status
            WHERE conversation_fact_archive.status != 'superseded' ''',
            (user_id,conversation_id,fact_id(fact),fact.fact,fact.source_sequence,
             'active' if fact_id(fact) in visible_ids else 'archived'))
    for identity in superseded:
        connection.execute('UPDATE conversation_fact_archive SET status=? WHERE user_id=? AND conversation_id=? AND fact_id=?',
            ('superseded',user_id,conversation_id,identity))
