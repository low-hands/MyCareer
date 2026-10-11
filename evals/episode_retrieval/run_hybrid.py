"""Record/replay development retrieval vectors; never consume test queries."""
import argparse
from collections import defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import tempfile
import sqlite3

from career_agent.domain.episodes import CareerEpisodeDraft
from career_agent.storage.episodes import SQLiteCareerEpisodeStore
from career_agent.agent.context.episode_semantic import SQLiteCareerEpisodeSemanticIndex
from career_agent.agent.context.semantic_retrieval import CareerEmbeddingConfig, OpenAICompatibleEmbeddingClient
from career_agent.evaluation.embedding_tape import EmbeddingTape


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--record', action='store_true')
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    fixture = Path(__file__).parent/'frozen_v2/scenarios.json'
    digest = hashlib.sha256(fixture.read_bytes()).hexdigest()
    data = json.loads(fixture.read_text(encoding="utf-8"))
    protocol = json.loads(fixture.with_name('protocol.json').read_text(encoding="utf-8"))
    assert digest == protocol['scenarios_sha256'], 'frozen scenarios changed'
    tape_path = args.root/'embeddings.json'
    args.root.mkdir(parents=True, exist_ok=True)
    if args.record:
        if tape_path.exists():
            parser.error('recording already exists; use a new experiment directory')
        from dotenv import load_dotenv
        load_dotenv()
        config = CareerEmbeddingConfig.optional_from_env()
        if config is None:
            parser.error('embedding configuration missing')
        tape = EmbeddingTape(OpenAICompatibleEmbeddingClient(config))
    else:
        tape = EmbeddingTape(recording=json.loads(tape_path.read_text(encoding="utf-8"))['embedding'])
        assert json.loads(tape_path.read_text(encoding="utf-8"))['fixture_sha256'] == digest
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp)
        store = SQLiteCareerEpisodeStore(path/'context.sqlite3')
        ids = {}
        for item in data['corpus']:
            episode = store.upsert(CareerEpisodeDraft(user_id='eval_user', source_run_id=item['id'],
                kind=item['kind'],title=item['title'],summary=item['summary'],
                occurred_at=datetime.fromisoformat(item['occurred_at'])))
            stable = 'career_episode_'+hashlib.md5(item['id'].encode()).hexdigest()
            with sqlite3.connect(store.path) as conn:
                conn.execute('UPDATE career_episodes SET id=? WHERE id=?',(stable,episode.id))
                conn.execute('UPDATE career_episodes_fts SET episode_id=? WHERE episode_id=?',(stable,episode.id))
            ids[stable] = item['id']
        index = SQLiteCareerEpisodeSemanticIndex(episode_path=store.path,
            cache_path=path/'career_embeddings.sqlite3',client=tape)
        # Attach only after seeding: deterministic replay has no background provider calls.
        store.semantic_index = index
        try:
            for query in data['queries']:
                if query['split'] != 'dev':
                    continue
                for mode in ('projection','tool_keywords','tool_sentence'):
                    filters = query['filters']
                    text = query['tool_query'] if mode == 'tool_keywords' else query['user_message']
                    if mode == 'projection':
                        hits = store.project_relevant(user_id='eval_user',query=text)
                    else:
                        hits = store.search(user_id='eval_user',query=text,limit=8,
                            start_datetime=datetime.fromisoformat(filters['start']) if 'start' in filters else None,
                            end_datetime=datetime.fromisoformat(filters['end']) if 'end' in filters else None,
                            kinds=tuple(filters.get('kinds',())))
                    ranked = [ids[hit.id] for hit in hits]
                    relevant = query['relevant']
                    targets = {doc for doc, grade in relevant.items() if grade == 2} or set(relevant)
                    import math
                    k = 8 if query['id']=='Q38' else 5
                    dcg = sum(relevant.get(doc,0)/math.log2(i+2) for i,doc in enumerate(ranked[:k]))
                    ideal = sum(grade/math.log2(i+2) for i,grade in enumerate(sorted(relevant.values(),reverse=True)[:k]))
                    rows.append(dict(id=query['id'],category=query['category'],mode=mode,ranked=ranked,
                        matched_by=[hit.matched_by for hit in hits],recall=len(targets & set(ranked[:k]))/len(targets) if targets else None,
                        ndcg=dcg/ideal if ideal else None))
            tape.verify()
        finally:
            index.close()
            if args.record:
                tape_path.write_text(json.dumps({'fixture_sha256':digest,'embedding':tape.snapshot()},ensure_ascii=False,indent=2)+'\n', encoding='utf-8')
    grouped = defaultdict(list)
    for row in rows:
        if row['recall'] is not None:
            grouped[row['mode']+'/'+row['category']].append(row)
    summary = {key:{metric:sum(row[metric] for row in group)/len(group) for metric in ('recall','ndcg')}
               for key,group in grouped.items()}
    report = {'fixture_sha256':digest,'status':'valid','rows':rows,'summary':summary,
        'embedding_failures':sum('error_type' in row for row in tape.records.values())}
    (args.root/'hybrid-dev-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n', encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
