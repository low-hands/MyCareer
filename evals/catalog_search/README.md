# search_catalog 开发集评估

当前启用英文复数归一及独立语义候选。语义相似度 ≥0.55 可直接参与 RRF，不依赖词法门槛；纯词法回退继续采用所有工具共享的有效词门槛：有效词不超过 2 个时命中 1 个，超过 2 个时命中 2 个。有效词仅从名称、别名、摘要统计，排除高频词。动作词表已撤回。

fixture v4 按用户要求改为**只检查前五召回**；WRITE 与 `propose_*` 只记录出现次数，不影响通过与否。查询、目标和 dev/holdout 分组与 v3 完全一致；v3 原文件已保存，版本差异由测试检查。

最新结果见 [混合检索报告](HYBRID_RETRIEVAL_2026-10-08.md)：模型风格纯词法 top-1/3/5 为 7/14、10/14、12/14，混合为 6/14、12/14、14/14，达到第 1 步 top-5 至少 12/14 的要求。用户整句混合为 21/43、29/43、34/43。top-1 并未全面改善，runtime 覆盖从 54/62 降为 53/62，限制已逐条报告。此前 [统一门槛报告](UNIFORM_GATE_2026-10-08.md) 保留为历史阶段结果。

15 条 dev 中 14 条有召回目标，1 条没有目标、只观测（passed=null）；11 条 holdout 全部有目标，继续未评分。报告中的 WRITE、propose_* 和去重并集按查询×工具计数；还提供非写请求的对应子计数，以及写请求误带非目标 WRITE 的观测值。

```sh
.venv/bin/python -m career_agent.evaluation.catalog_search --output /tmp/catalog-dev-v4.json
.venv/bin/python -m career_agent.evaluation.catalog_search --check --semantic-scores evals/catalog_search/model_query_semantic_scores.json --output /tmp/catalog-hybrid-dev-check.json
```

输出路径必须不存在。无语义参数时 `--check` 因 2 条纯词法召回缺失返回非零；提供 `--semantic-scores` 时检查混合列，当前返回 0。动作工具出现不会导致失败。

最终规则确定且工作区干净后，才可人工执行一次 `--final-holdout --check --semantic-scores evals/catalog_search/model_query_semantic_scores.json --output evals/catalog_search/final_holdout.json`。命令在评分和创建文件前检查包含未跟踪文件的 git status。测试仅检查 holdout 结构与政策变更前后的一致性，不执行其查询。

历史第 0/1 步及已撤回第 2 步的 JSON/报告保留作审查记录，不代表当前评分口径。dev_probes.json 属于已撤回动作词表方案的审查记录，不被当前 evaluator 或测试加载。历史 README 见 README_v3_historical.md。

26 条查询相似度已由独立 CAREER_EMBEDDING 接口录制；这包含 holdout 的相似度记录，但没有执行 holdout 检索排名/召回评分。刷新录制可运行 `python -m career_agent.evaluation.catalog_search_embeddings --output <new-path>`。验证时会检查 catalogue digest、fixture SHA-256 和 26 条查询完整性。
