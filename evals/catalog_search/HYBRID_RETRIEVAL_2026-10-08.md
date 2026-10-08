# 英文复数归一与独立语义候选（2026-10-08）

**验收：混合检索模型风格 dev top-5 为 14/14，高于第 1 步的 12/14，全部 14 条有目标的开发查询通过。** 纯词法仍有两条召回缺失；混合 top-1 没有改善，runtime 覆盖还出现一条回退，见下文。

## 实现与英文归一单独验收

查询和目录共用 `_tokens`：普通英文复数去掉尾部 s，sses/shes/ches/xes/zes 去掉 es；保留 status、analysis、process 这类单数尾缀。resumes 归为 resume，matches 归为 match。没有引入分词或词干依赖，也不处理全部不规则复数。

先只应用英文归一，记录 `plural_lexical_dev_2026-10-08.json`：模型风格 top-1/3/5 从 7/10/11（分母 14）变为 7/10/12，resume 的 list_resumes 进入第 5 名。对模型风格及既有用户整句的每条 dev，在 top-1/3/5 分别检查原已命中的目标，均没有丢失。用户整句维持 17/27/34（分母 43），该阶段 runtime 仍为 54/62。

然后修改语义通道：相似度 ≥0.55 的目录工具直接进入语义候选，不再经过词法有效词门槛；保留语义前 10、RRF 的 1/(60+rank)、精确名称/别名优先和最终 top-k。词法门槛本身继续统一用于所有工具。

## 分数来源与 holdout 边界

单独加载项目 .env 中 CAREER_EMBEDDING_*，在允许联网的执行环境成功调用 qwen3.7-text-embedding，返回 1024 维向量。没有依赖 DeepSeek 生成接口。26 条查询的分数保存在 `model_query_semantic_scores.json`，包含模型、时间、catalogue digest、fixture SHA-256 和采样范围。按运行时 SemanticCapabilityIndex 保存达到 0.55 的前 10 个相似度，分数保留 6 位小数；这是供当前阈值重放的候选分数，不是全部工具的原始向量。

fixture v4 未修改，SHA-256 仍为 `dca9a3dae15fdcdefa1df50d1d4e38052ab0072ff1c02cfebf0b00d60f85a621`。按用户要求为 dev 和 holdout 共 26 条记录 embedding/相似度，但**holdout 未调用 search_catalog、未计算 top-k 召回或通过率**。开发报告及契约只运行 15 条 dev。已有用户整句使用原录制的同型号分数，未重录它们；目录 embedding 文本未变，原 catalogue digest 仍匹配。

## 纯词法与混合检索

两列使用相同的英文归一、词法门槛、fixture 和前五评分口径，只在右列注入录制分数。WRITE / propose_* 仅观测，不影响通过。

| 指标 | 纯词法 | 词法 + 语义 |
| --- | --- | --- |
| 模型风格 top-1 | 7/14 | 6/14 |
| 模型风格 top-3 | 10/14 | 12/14 |
| 模型风格 top-5 | 12/14 | 14/14 |
| 模型风格 WRITE 次数 | 37 | 36 |
| 模型风格 propose_* 次数 | 2 | 2 |
| 模型风格 动作工具去重次数 | 39 | 37 |
| 模型风格 写请求非目标 WRITE 次数 | 9 | 9 |
| 用户整句 top-1 | 17/43 | 21/43 |
| 用户整句 top-3 | 27/43 | 29/43 |
| 用户整句 top-5 | 34/43 | 34/43 |
| 用户整句 WRITE 次数 | 120 | 116 |
| 用户整句 propose_* 次数 | 6 | 6 |
| 用户整句 动作工具去重次数 | 126 | 122 |
| 用户整句 写请求非目标 WRITE 次数 | 43 | 39 |

第 1 步模型风格 top-1/3/5 为 8/14、12/14、12/14；混合结果为 6/14、12/14、14/14。**top-5 验收满足，但不能宣称所有排名指标都改善。** 用户整句第 1 步为 17/43、28/43、34/43；混合为 21/43、29/43、34/43。

## 重点查询目标位次

以下位次用未截到前五的完整 RRF 次序诊断；不改变评分的前五要求。

| 查询 → 目标 | 纯词法 | 词法 + 语义 |
| --- | --- | --- |
| resume → list_resumes | 5 | 2 |
| 帮我看看我的经历 → search_career_memory | 6 | 2 |
| 查看我的简历 → list_resumes | 7 | 4 |
| 面试 → list_interviews | 2 | 2 |
| calendar → list_calendar_links | 5 | 5 |

经历和查看简历回到前五；面试和 calendar 在这次融合中没有进一步提前。纯词法的两个缺失目标仍通过 strict xfail 保留，不因为混合通过而删掉纯词法回归断言。

## runtime 的额外影响

| 指标 | 第 1 步 | 英文归一后、旧语义门槛 | 独立语义候选后 |
| --- | --- | --- | --- |
| 覆盖需求步骤 | 54/62 | 54/62 | 53/62 |
| unrequested_write_offer_count | 24 | 27 | 27 |

独立语义候选新增的一条未覆盖发生在 `selection_dev_cross_application_then_email` 的第 1 步（从 0 计数）：resolve_email_event 掉出意图预加载前五，未进入 offered；该变化由同一进程对照原语义门槛定位，英文归一在两边相同。逐步变化见 `hybrid_runtime_changes_2026-10-08.json`。本轮没有调整状态规则、RRF 权重或给工具加特殊优先级来掩盖这条回退。原 54/62 验收断言保留为 strict xfail，明确不满足。

## 验证与复现

582 passed、3 holdout 测试未执行、4 strict xfailed。四个预期失败为两个 dev 纯词法召回缺失、独立的纯词法经历召回测试和 runtime 54/62 覆盖断言。新增混合 dev 每条目标契约均普通通过，并有 top-5 ≥12/14 的验收断言。另覆盖 s/es 归一、无词法证据的 READ/WRITE/propose 语义准入、阈值边界、未知工具排除、录制来源过期检查及只记录 26 条相似度而不评 holdout。

最终开发评估的 --check 返回 0，验证的是混合列；没有语义分数参数时仍检查纯词法列。

```sh
.venv/bin/python -m career_agent.evaluation.catalog_search --check \
  --semantic-scores evals/catalog_search/model_query_semantic_scores.json \
  --output /tmp/catalog-hybrid-dev.json
```

需要重新录分数时使用单独的 recording 命令（输出路径必须不存在，需网络和 CAREER_EMBEDDING_*）：

```sh
.venv/bin/python -m career_agent.evaluation.catalog_search_embeddings \
  --output /tmp/model-query-scores.json
```

工具 schema 本身未修改；逐步选择的工具集合变化，使 trajectory 指纹变为 `4e6e8011210fa93410e09387382ab21c420b57ff8fec8b1ebbc4b71c3a6b93f2`。更新对应契约期望，但未改写任何既有模型决策录制中的新鲜度散列，不能把本次离线检索验收当成模型录制质量验收。
