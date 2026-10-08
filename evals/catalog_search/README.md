# search_catalog 开发集评估

当前采用所有工具共享的有效词门槛：有效词不超过 2 个时命中 1 个，超过 2 个时命中 2 个。有效词仅从名称、别名、摘要统计，排除高频词。动作词表已撤回。

fixture v4 按用户要求改为**只检查前五召回**；WRITE 与 `propose_*` 只记录出现次数，不影响通过与否。查询、目标和 dev/holdout 分组与 v3 完全一致；v3 原文件已保存，版本差异由测试检查。

最新结果与局限见 [统一门槛报告](UNIFORM_GATE_2026-10-08.md)：模型风格 top-1/3/5 为 7/14、10/14、11/14，低于第 1 步的 8/14、12/14、12/14；用户整句为 17/43、27/43、34/43。runtime 覆盖保持 54/62，unrequested_write_offer_count 从 24 增至 27。**召回不降的预期未满足。**

15 条 dev 中 14 条有召回目标，1 条没有目标、只观测（passed=null）；11 条 holdout 全部有目标，继续未评分。报告中的 WRITE、propose_* 和去重并集按查询×工具计数；还提供非写请求的对应子计数，以及写请求误带非目标 WRITE 的观测值。

```sh
.venv/bin/python -m career_agent.evaluation.catalog_search --output /tmp/catalog-dev-v4.json
.venv/bin/python -m career_agent.evaluation.catalog_search --check --output /tmp/catalog-dev-v4-check.json
```

输出路径必须不存在。当前 `--check` 因 3 条召回缺失返回非零，动作工具出现不会导致失败。

最终规则确定且工作区干净后，才可人工执行一次 `--final-holdout --check --output evals/catalog_search/final_holdout.json`。命令在评分和创建文件前检查包含未跟踪文件的 git status。测试仅检查 holdout 结构与政策变更前后的一致性，不执行其查询。

历史第 0/1 步及已撤回第 2 步的 JSON/报告保留作审查记录，不代表当前评分口径。dev_probes.json 属于已撤回动作词表方案的审查记录，不被当前 evaluator 或测试加载。历史 README 见 README_v3_historical.md。
