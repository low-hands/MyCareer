# 统一检索门槛与召回口径（2026-10-08）

所有工具共享有效查询词 ≤2 命中 1 个、>2 命中 2 个的门槛，不按 effect 或 `propose_` 名称分支；精确工具名、namespace、完整别名路径、BM25 排序和 top-k 截断保留。有效词仍只从名称、别名、摘要计数并排除高频词。没有动作词表。

**预期检查：召回不降未满足。** 模型风格 top-5 从 12/14 降为 11/14；用户整句 top-5 保持 34/43，但 top-3 从 28/43 降为 27/43。放宽门槛让更多写工具进入候选并挤占读工具名次。本轮按指定规则实施，没有额外调整排序。

## 固定数据与通过条件

fixture v4 仅按用户授权撤掉禁止 effect/动作的评分字段及更新评分说明；26 条查询、期望工具、前五要求和分组逐项与 v3 相同。v3 原文件保留为 `model_queries_v3_recall_and_effects.json`。v4 SHA-256：`dca9a3dae15fdcdefa1df50d1d4e38052ab0072ff1c02cfebf0b00d60f85a621`。holdout 11 条继续未评分。

只按期望工具是否进入前五判定通过。WRITE、`propose_*`、二者去重并集均只记录暴露次数，不参与失败数或 `--check`。没有期望工具的用例 `passed=null`，仅观测，不计入召回分母。明确写请求误带其他 WRITE 的原指标继续保留为观测。

## 开发集对比

基线使用第 1 步元数据词表方案的已保存 top-5；不重跑或改写基线。曝光指标由基线逐行 offered 重算，同口径比较。均按查询 × 工具计数。

| 指标 | 第 1 步 dev | 统一门槛 dev |
| --- | --- | --- |
| 模型风格 top-1 | 8/14 | 7/14 |
| 模型风格 top-3 | 12/14 | 10/14 |
| 模型风格 top-5 | 12/14 | 11/14 |
| 模型风格 WRITE 总次数 | 19 | 38 |
| 模型风格 propose_* 总次数 | 1 | 2 |
| 模型风格 动作工具去重总次数 | 20 | 40 |
| 模型风格 非写请求 WRITE 次数 | 7 | 24 |
| 模型风格 非写请求 propose_* 次数 | 1 | 2 |
| 模型风格 非写请求动作并集次数 | 8 | 26 |
| 模型风格 写请求非目标 WRITE 次数 | 7 | 9 |
| 用户整句 top-1 | 17/43 | 17/43 |
| 用户整句 top-3 | 28/43 | 27/43 |
| 用户整句 top-5 | 34/43 | 34/43 |
| 用户整句 WRITE 总次数 | 106 | 120 |
| 用户整句 propose_* 总次数 | 4 | 6 |
| 用户整句 动作工具去重总次数 | 110 | 126 |
| 用户整句 非写请求 WRITE 次数 | 56 | 63 |
| 用户整句 非写请求 propose_* 次数 | 2 | 2 |
| 用户整句 非写请求动作并集次数 | 58 | 65 |
| 用户整句 写请求非目标 WRITE 次数 | 36 | 43 |
| runtime 覆盖需求步骤 | 54/62 | 54/62 |
| runtime unrequested_write_offer_count | 24 | 27 |

runtime 共 64 步，使用既有录制语义分数；直接查询只使用词法检索，两者口径不同。此次同时修复新会话回读工具的选择一致性，故 schema token 总量的变化不能单独归因于检索门槛。`pre_uniform_dev_2026-10-08.json` 在该修复完成、统一门槛尚未应用时记录，召回/覆盖/写暴露与第 1 步相同，供隔离验证。

## 具体回退与未解决项

- `帮我看看我的经历`：`search_career_memory` 从第 3 名降为第 6 名；前五为 confirm_career_fact、propose_memory_tombstone、confirm_memory_amendment、propose_memory_amendment、resolve_claim_source。这是新增召回回退。
- `面试`：list_interviews 从第 1 降为第 2；`calendar`：list_calendar_links 从第 3 降为第 5。
- 用户整句 `刚才提到的待办都列给我看看。`：list_action_items 从第 3 降为第 4。
- 原有 `resume` 单复数不匹配和 `查看我的简历` 前五排序缺失仍未解决。

模型 dev 召回失败为 3 条；旧报告失败 6 条包含动作禁止，与现在不可直接比较，不能把减少的失败数算成召回提升。`--check` 当前仍应返回非零，原因仅为这 3 条缺失。

## 验证与边界

相关检索、评估、selection、回读、runtime、context manager 和模型适配器测试：545 passed、3 deselected（holdout）、4 strict xfailed。四个 xfail 是三个 dev 召回失败和单独的经历召回回归测试；新回退显式标记，没有删除召回断言。

检索准入仅决定工具发现与提供；执行仍保留 offered-set、授权、审批与 registry 守卫。此处没有修改执行权限。录制失效与指纹追溯见 `../HISTORY_SELECTION_AND_FINGERPRINT_2026-10-08.md`。
