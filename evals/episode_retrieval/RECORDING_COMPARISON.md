# Episode search recording check (2026-10-08)

The episode retrieval query set still measures recall at 0.78 for automatic
projection, 0.80 for keyword tool queries, and 0.78 for sentence tool queries.
The original in-repository baseline was 0.42 / 0.78 / 0.17.

The Main Agent recordings were kept in separate model directories:

| Model | Cassette directory | Active scenarios | Stale | Missing |
| --- | --- | ---: | ---: | ---: |
| `qwen3.7-plus` | `evals/main_agent_search` | 47 | 0 | 0 |
| `deepseek-flash` | `evals/main_agent_deepseek_search` | 47 | 0 | 0 |

The comparison below scores recorded final decisions with the same current
scenario assertions before and after the tool-description change. It uses only
scenarios present in both recordings. It deliberately skips prompt/schema
fingerprints, since the older cassettes predate other tool-selection changes.

| Model | Comparable scenarios | Behavior failures before → after | Quality failures before → after |
| --- | ---: | ---: | ---: |
| Qwen | 45 | 10 → 9 | 1 → 1 |
| DeepSeek | 44 | 9 → 13 | 1 → 1 |

Per-scenario changes are in the local reports produced by
`compare_recordings.py`. Most changed scenarios did not offer
`search_career_episodes`. Other prompt and selection changes landed between
the old and new recordings, and each model was sampled anew. These batches do
not establish that the episode-search description caused the overall
difference. The DeepSeek behavior regression needs separate investigation.

The misplaced DeepSeek recordings previously written into the Qwen directory
were backed up at `evals/episode_retrieval/results/misplaced_deepseek_20261008/`
before the Qwen directory was restored from HEAD. The `results/` directory is
ignored by Git.
