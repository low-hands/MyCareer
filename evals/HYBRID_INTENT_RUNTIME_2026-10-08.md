# Hybrid intent ranking, 2026-10-08

The runtime now warms the capability index in batches of at most 20 and ranks
turn intent with lexical and semantic scores for both Qwen and DeepSeek.
Explicit `search_capabilities` keeps its existing hybrid behavior.

| Model | Prior 9 × 3 | Hybrid 9 × 3 | Decision |
| --- | ---: | ---: | --- |
| Qwen `qwen3.7-plus` | 25/27 | 27/27 | Enable hybrid intent |
| DeepSeek `deepseek-flash` | 21/27 | 24/27 | Enable hybrid intent; retain the `planning_to_apply_does_not_create_an_application` 3/3 to 2/3 result as a monitoring point |

The DeepSeek failure called forbidden `analyze_job`. That tool was offered in
both prior and hybrid recordings, so the result may reflect model variation.
The user clarified that this single-scenario result should not cause rollback.

Offline selection with the fixed Qwen embedding score file measured DEV
coverage 54/62, HOLDOUT coverage 25/29, unsolicited write offers 25/13,
unreachable offers 0, and waiting reoffers 0. The score file covers all nine
re-recorded scenarios. One additional message introduced after the snapshot
is absent; two other search scenarios using it currently fall back to lexical
ranking. Automatic approval review rejected a full 119-message embedding
refresh as excessive external text transfer. A narrower, explicitly approved
refresh can fill that gap later.

Raw paired recordings are in `output/hybrid_qwen_20261008/` and
`output/hybrid_deepseek_20261008/`. The comparison script is
`output/compare_hybrid_recordings.py`; it grades terminal business decisions
without requiring old schema fingerprints to match today's code.
