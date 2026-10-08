> 历史说明：当前规则和评分口径已由 v4 替代，见 README.md。

# 模型风格 search_catalog 基线（2026-10-08，v3）

第 0 步补评估时未调整检索门槛、排序、catalogue 元数据或语义分数。第 1 步现已完成普通读工具的动态证据门槛，第 1 步随后收窄有效词计数到名称/别名/摘要；第 1 步结果、IDF 对比与限制见 [有效词词表报告](READ_VOCABULARY_2026-10-08.md)，原第 1 步结果见 [历史报告](STEP1_2026-10-08.md)。第 2 步动作词表及配套门槛、测试已按用户要求撤回；[第 2 步报告](STEP2_2026-10-08.md)、step2 JSON 与 dev_probes.json 仅保留为历史审查记录，不再被当前 evaluator 或测试加载。holdout 仍未评分。第 0 步基线采用 `baseline_v3_2026-10-08.json`；v1、v2 报告保留为历史记录，其用例与召回分母已被本次修订替代；v1 还缺少当时的工作区状态，不能仅凭 HEAD 认定可完整复现。

## 审阅后重新固定数据

`model_queries.json` v3 在任何规则调优和 holdout 评分之前固定：15 条 dev，11 条 holdout。不移动原用例分组；保留原样别名用于覆盖直接发现路径，分别新增三条非别名的明确写请求：

| split | query | 前五必须出现 |
| --- | --- | --- |
| dev | 把这个岗位加入投递 | create_application |
| dev | 帮我生成一份定制简历 | draft_resume_tailoring |
| dev | 把简历导出成 PDF | export_resume_artifact |
| holdout | 帮我把当前职位加到投递清单里 | create_application |
| holdout | 请按这个岗位起草一版定制简历 | draft_resume_tailoring |
| holdout | 请将这份简历导出为 PDF 文件 | export_resume_artifact |

测试通过元数据断言这些新增查询均不是工具名、namespace 或完整 alias，不执行 holdout 检索。

每条用例记录 expected_tools、required_k=5 和 forbidden_effects。读查询通过 `forbid_action_tools=true` 调用共用 `needs_action_evidence` 禁止动作工具，fixture 不再保存 `propose_` 前缀定义。明确写请求允许动作工具。所有目标工具必须进入前 k 名。

四条 `64aea67` 对照中，`看看我的面试安排` 必须召回 `list_interviews`，`我的投递记录有哪些` 必须召回 `list_applications`。dev 的 `帮我看看我的经历` 与 holdout 的 `查看我保存的经历` 统一要求召回 `search_career_memory`，同时检查动作暴露。仅 `这个岗位不错` 保留为**只检查动作暴露**的观测对照：它没有明确的必需读操作，因此不计入召回分母。

dev 有 14 条召回需求（9 条读、5 条写）和 1 条纯暴露对照；holdout 的 11 条全部有召回目标（6 条读、5 条写）。空结果无法通过任何召回需求。

holdout **已固定、未评分**，在最终规则确定后只评一次。新测试仅检查其数据结构、非别名属性、召回用例数量和固定文件散列，不执行其查询。原有 holdout 不重新评分。

## 共用动作分类

`search.py` 定义共用 `needs_action_evidence(descriptor)`，统一判断：

```python
descriptor.effect == "WRITE" or descriptor.name.startswith("propose_")
```

五个 effect 为 READ 的 `propose_*` 同样属于动作工具。evaluator 和语义压力测试都使用这一共用函数，另有测试覆盖全部五个 READ proposal。

第 1 步使用此分类将 WRITE 和 READ proposal 排除在短查询放宽范围之外。当前没有逐工具动作词表；这些工具恢复为原有的至少两个非高频命中词门槛（工具名、namespace 或完整别名精确匹配仍可直接通过）。

## 第 0 步历史基线

以下数值来自第 0 步 v3 报告。当前代码恢复至词表收窄后的 `step1_metadata_vocab_dev_2026-10-08.json` 状态：禁止动作工具次数 8、非目标 WRITE 7、整句 control WRITE 6、runtime `unrequested_write_offer_count` 24。这次写工具指标下降来自读工具挤占前五，写工具门槛未改变，不能计入第 2 步效果。归因与比较口径见 [有效词词表报告](READ_VOCABULARY_2026-10-08.md)。

直接模型查询调用 `search_catalog(query=..., limit=5)`，不注入语义分数，不经过上下文和可达性筛选。现有用户整句开发集单独评分；运行时 selection 使用现有录制语义分数，口径不同。

| 直接 search_catalog | top-1 | top-3 | top-5 |
| --- | --- | --- | --- |
| 模型风格 dev v3（14 条需求） | 5/14 (35.71%) | 7/14 (50.00%) | 7/14 (50.00%) |
| 既有用户整句 dev（43 条需求） | 17/43 (39.53%) | 28/43 (65.12%) | 34/43 (79.07%) |

模型风格 dev 共 10/15 条违反契约，禁止动作工具共出现 10 次（按查询和工具计数）。新增的三条非别名写查询均在前五命中，具体位次为 2、3、1。五个短词查询仍返回空；`查看我的简历` 仍缺少目标读工具且暴露写工具；四条历史对照仍均违反动作暴露约束。

新增只观测、不影响 `passed`、`failed_case_count` 或 `--check` 退出码的指标 `non_target_write_offer_count`：仅在 expected_tools 中存在 WRITE 目标的查询上，统计前五中 effect 为 WRITE 且不属于 expected_tools 的工具，按查询和工具计数。逐条名单见 `non_target_write_offered`，分母见 `write_query_count`。这里的 WRITE 按字面 effect 统计，不把 READ proposal 混入这个指标；READ proposal 的禁止准入仍由共用动作分类处理。

当前模型风格 dev 的 5 条写查询共误带 **8 次**非目标 WRITE：两条简历起草各 3 次，投递创建 2 次，两条导出均为 0。既有用户整句 dev 对应为 18 条写需求、38 次非目标 WRITE；这也是独立观测指标，不改变既有评分。

既有 runtime selection dev：64 个步骤，54/62 个需求步骤覆盖；`unrequested_write_offer_count=25`，`unreachable_offer_count=0`，`waiting_reoffer_count=0`，`schema_tokens_proxy_total=122371`。既有整句 dev control WRITE 暴露为 8 次。保留原 evaluator 对 namespace 的判定语义。

详细前五结果及失败原因见 v3 JSON。报告除 HEAD、fixture SHA-256 外，还记录评分开始前的 `git_status_porcelain`、`worktree_clean`，以及源代码、用例和录制语义分数文件的内容散列。v3 基线明确标记为未提交工作区；不读取无关未跟踪文件的内容（例如 `.env`）。

## 复现与最终验收

从仓库根目录执行；输出文件必须不存在：

```sh
.venv/bin/python -m career_agent.evaluation.catalog_search --output /tmp/catalog-dev-v3.json
.venv/bin/python -m career_agent.evaluation.catalog_search --check --output /tmp/catalog-dev-v3-check.json
.venv/bin/python -m pytest tests/agent/test_catalog_search_evaluation.py tests/agent/test_tool_selection_evaluation.py tests/agent/test_capability_search.py -k 'not holdout' -q
```

`--check` 对缺失目标、禁止 effect 或共用分类判定的禁止动作返回非零。撤回后恢复 6 条 dev 契约和 4 条合成语义压力用例的 strict xfail；修复后 XPASS，必须移除对应标记。fixture 散列防止调参期间无意修改分组或期望。

最终规则确定、工作区干净后，人工只执行一次以下命令并保留结果：

```sh
.venv/bin/python -m career_agent.evaluation.catalog_search --final-holdout --check --output evals/catalog_search/final_holdout.json
```

`--final-holdout` 在评分及创建输出文件之前检查 `git status --porcelain --untracked-files=all`，包括未跟踪文件，非空直接拒绝。干净指开始评分时的状态；随后新生成的报告会成为未跟踪文件。独占创建防止覆盖原路径，不限制更换路径重复运行，因此仍需遵守只评一次的流程。

## 动作词表撤回验证

2026-10-08 按用户要求撤回第 2 步动作词表、动作与主题门槛及配套测试，保留第 0、1 步。`rollback_step2_dev_2026-10-08.json` 的模型查询和既有开发集结果逐项与 `step1_metadata_vocab_dev_2026-10-08.json` 相同；holdout 未评分。相关测试 117 passed、10 xfailed、7 deselected；排除 holdout 测试及已在原 HEAD 复现的四条既有失败（schema fingerprint 一条、available_now 投影三条）。
