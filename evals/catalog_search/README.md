# search_catalog 检索评估

## 当前规则

生产检索采用业界基线（`cbe1765`）：

- **词法**：字段加权 BM25F（名称、别名 ×5，摘要 ×2，参数名、例句 ×1）；例句并入工具文档（doc2query），不再单独打分或设证据门槛；中文按双字切词，英文做简单复数归一。
- **语义**：相似度 ≥ 0.55 的工具取前 10，作为独立候选。
- **融合**：RRF（k=60）；查询与工具名称或完整别名完全一致时排在最前。
- **意图预加载**按当前消息检索时过滤停用词（英文停用词表 + 中文虚词），`search_capabilities` 的显式查询不过滤。

## 数据与评分

- `model_queries.json`（fixture v4）：dev 查询及期望工具，**只检查期望工具是否进入前五**；WRITE 与 `propose_*` 只记录出现次数，不影响通过。
- `model_query_semantic_scores.json`：由 CAREER_EMBEDDING 接口录制的查询相似度，验证时检查 catalogue digest、fixture SHA-256 和查询完整性。刷新：`python -m career_agent.evaluation.catalog_search_embeddings --output <new-path>`。
- **holdout 只评一次。** 第一份 holdout 已在 `final_holdout.json` 中最终评分；第二份 `holdout_v2_queries.json`（16 条）已在 `holdout_v2_result.json` 中评分。两份都**不再重跑，也不依据其结果调整规则**。下次修改检索规则需要新的 holdout。

第二份 holdout 上，业界基线（即当前生产规则，评分时尚未加入停用词过滤）前五召回为 15/16，唯一缺失是 `check my inbox for recruiter replies` 找不到 `sync_application_emails`。

## 运行

输出路径必须不存在。

```sh
# dev 评估，纯词法
.venv/bin/python -m career_agent.evaluation.catalog_search --output /tmp/catalog-dev.json
# dev 评估，词法 + 录制的语义分数
.venv/bin/python -m career_agent.evaluation.catalog_search --check \
  --semantic-scores evals/catalog_search/model_query_semantic_scores.json \
  --output /tmp/catalog-hybrid-dev.json
```

`compare_simplified.py` 是改用业界基线前，旧规则与基线在 dev 上的对比脚本；其说明见文件头。

这一目录下带日期的 JSON 是各阶段的评估结果，保留作审查。
