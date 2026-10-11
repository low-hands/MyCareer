"""Offline development-only calibration after observed E2E abstention regression.

All candidates and the gate are fixed in calibration protocol before running.
Does not install a threshold in production or consume any test queries.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import sys
from collections import defaultdict

HERE = Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--embedding-root',type=Path,required=True)
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args()
    protocol=json.loads((args.root/'protocol.json').read_text())
    spec=importlib.util.spec_from_file_location('episode_hybrid_evaluator',HERE/'run_hybrid.py')
    runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
    base_class=runner.SQLiteCareerEpisodeSemanticIndex
    baseline=defaultdict(list)
    for row in json.loads(args.baseline.read_text()):
        if 'recall' in row:baseline[row['mode']+'/'+row['category']].append(row)
    baselines={key:{m:sum(row[m] for row in group)/len(group) for m in ('recall','ndcg')}
               for key,group in baseline.items()}
    results=[]
    for threshold in protocol['thresholds']:
        class CalibratedIndex(base_class):
            def rank_episode_ids(self, *, query, candidates):
                ranked=super().rank_episode_ids(query=query,candidates=candidates)
                if not ranked:return ()
                query_vector=self._client.embed((query,))[0]
                vectors=self._cached_vectors(self._texts(candidates))
                return tuple(id for id in ranked if self._cosine(query_vector,vectors[id])>=threshold)
        runner.SQLiteCareerEpisodeSemanticIndex=CalibratedIndex
        root=args.root/f'threshold-{threshold:.2f}'
        root.mkdir(exist_ok=True)
        shutil.copyfile(args.embedding_root/'embeddings.json',root/'embeddings.json')
        sys.argv=['run_hybrid.py','--root',str(root)]
        runner.main()
        report=json.loads((root/'hybrid-dev-report.json').read_text())
        failures=[]
        for key,metrics in report['summary'].items():
            baseline_metrics=baselines[key]
            if key.endswith('/paraphrase'):
                if metrics['recall']-baseline_metrics['recall']<0.10-1e-9:
                    failures.append(key+': insufficient recall gain')
            elif any(metrics[m]<baseline_metrics[m]-1e-9 for m in ('recall','ndcg')):
                failures.append(key+': regression')
        results.append({'threshold':threshold,'retrieval_gate_passed':not failures,'failures':failures,
                        'summary':report['summary']})
    summary={'status':'candidate_available' if any(r['retrieval_gate_passed'] for r in results) else 'no_feasible_threshold',
             'measurement':'offline_retrieval_only_not_E2E_quality','results':results,
             'production_threshold_installed':False}
    (args.root/'calibration-report.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({key:value for key,value in summary.items() if key!='results'}))


if __name__=='__main__':
    main()
