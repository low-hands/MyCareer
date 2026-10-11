# 情景记忆混合检索评估

`frozen_v2/scenarios.json` 在生产改动和新方案结果之前按意图扩充并冻结：100 条，8 类，每类至少 12 条，开发集 68、测试集 32。历史 v1 和原基线保留。`protocol.json` 固定场景摘要值、模型、每场景 10 次和验收标准。测试集在开发门禁通过、实现固定后最后跑一次；当前门禁未通过，测试集未运行，开发工具没有测试集采样入口。

验收要求：三个检索模式的开发集换说法 Recall@5 都至少增加 10 个百分点；其他类别 Recall 和 nDCG 均不得下降；未找到类的最终回答正确率不得下降。Q38 延续原 Recall@8 例外。不根据测试集调参数。

端到端实验固定 12 个开发场景（4 个换说法、全部 8 个未找到），Qwen `qwen3.7-plus` / DeepSeek `deepseek-flash`，纯词法 / 混合两路，每场景 10 次，共 480 个独立样本。模型自行生成每次查询和过滤参数，随后整理最终回答。使用生产事件存储、投影和工具；这是隔离的事件查询与回答链路，**不冒充完整 MainAgentRuntime 评估**。旧 runtime 和轨迹另行离线重放。其余开发类别由完整检索集检验，不能声称这些类别已有端到端模型质量证据。

历史 v2 使用被测模型自评分，结果仅作历史参考，保留原始录制及审计，不覆盖。

下一轮采用独立冻结的 `grading_v3/protocol.json` 确定性规则，不再调用模型裁判。找到类必须出现身份和日期、轮次、结果或练习关键事实；未找到类须明确说无记录，提到其他语料公司的事件即失败。规则未覆盖、含歧义断言或难以绑定的补充事件单列 `human_review`，导出独立人工审核队列，审核前不计通过。新增规则/规则变化是新评估版本，不可冒充历史可比成绩。独立审核者使用最终回答、原始语料和规则，先不看旧模型自评分；审核决定另存且保留自动理由。pending 队列未处理完时不宣称验收完成。

## 录制与严格离线回放

```bash
# 开发集真实向量录制：合成数据，不读取用户记忆
PYTHONPATH=src .venv/bin/python evals/episode_retrieval/run_hybrid.py --record --root output/独立实验/retrieval
# 只读取已录制的查询/事件向量，不创建 provider client
PYTHONPATH=src .venv/bin/python evals/episode_retrieval/run_hybrid.py --root output/独立实验/retrieval
# 端到端录制，固定每场景 10 次，已有 attempt 拒绝覆盖
.venv/bin/python evals/episode_retrieval/run_e2e.py --record --root output/独立实验/qwen-hybrid --arm hybrid --model qwen3.7-plus --vendor qwen --env-prefix RESUME_ANALYSIS_AGENT
# 端到端离线回放，无服务调用
.venv/bin/python evals/episode_retrieval/run_e2e.py --root output/独立实验/qwen-hybrid --arm hybrid --model qwen3.7-plus --vendor qwen
```

DeepSeek 使用 `--vendor deepseek --model deepseek-flash --env-prefix MAIN_AGENT`。纯词法基线使用 `--arm lexical --source-root <冻结提交的src目录>`，基线提交见协议。可用 `git archive <提交> src | tar -x -C <隔离目录>` 重新建立只读基线。`--all-dev` 是另一批更大实验，使用独立目录；没有自动扩大采样。开发过程中不运行测试集。

向量 tape 保存准确文本、模型 namespace、向量及失败类型，涵盖事件标题＋摘要和模型查询。缺少记录即 `missing_embedding_recording`；即使生产代码捕获异常回退，评估仍不能把缺失录制算作有效。模型 tape 逐请求匹配 schema、目录、查询观察、模型和生成参数；被测助手响应必须全部消费，差异为 stale。历史模型裁判末尾请求按已知形状隔离保留，其答案不作为判分；离线重判写入新目录，标记 v3 版本及规则摘要值。缓存数据库和事件 ID 每样本独立、ID 确定，避免 RRF 同分时随机 UUID 使重放排名不一致。

模型最终回答按意图与事件事实判分，工具命中不等于答对。无相关事件时须如实说明没有找到；相近公司、错误轮次、错误年份不能替代目标。录制的失败与基础设施异常全部保留，不补采成功样本。

## 录制影响

`search_career_episodes` 的工具参数 schema 未改。返回条目和目录增加 `matched_by`，排名可能不同。仅实际模型请求变化的录制判 stale：包括消费事件工具结果、注入事件目录，以及受共享别名展开影响的事实/岗位查询。源码变化本身不判 stale，不改历史指纹。本轮已有 47 条轨迹和 19 个 runtime 场景不包含实际变化后的事件 payload/目录，离线回放没有 stale；这不代表新行为已有质量证据。

## 阈值条件

初版无语义阈值，RRF k=60、投影 0.65/0.35 均不调。只有开发端到端出现未找到正确率下降后，才运行 `run_threshold_calibration.py`。校准仅回放开发集向量，候选阈值、选择标准写入独立 protocol，再跑曲线。离线检索门禁通过的候选仍须单独冻结并重新录制受影响的端到端场景，不能拿旧模型请求证明新阈值质量。没有可行候选时不向生产写入阈值，不打开测试集。

本轮完整录制、审计、基础设施失败和报告保存在本地 `output/episode-hybrid-v2/`；实现及数据外发限制见本地 `docs/iteration/116-episode-hybrid-retrieval.md`。
