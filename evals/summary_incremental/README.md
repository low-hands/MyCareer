# 增量事实与后台摘要

## 事实账本

摘要 worker 输出 `new_facts` 与 `superseded_fact_ids`，不重写旧事实表。新增事实只能引用本次压缩的消息；更正必须使用已披露的精确 fact ID 并给出替代事实。其余事实由代码合并保留：摘要最多显示最新 12 条，超出条数或字符预算的旧事实进入 `conversation_fact_archive`，不会因模型省略而消失。fact ID 由来源序号和文本确定；被更正的旧条目标记为 superseded。

`agent_context` schema v18 建立账本，并在迁移时一次性回填旧摘要中的可见事实。账本按 user/conversation 隔离；`list_summary_facts(statuses=("archived",))` 可读取归档，原文仍可用 `read_conversation_span` 取回。遗忘记忆使摘要失效、或删除整个对话时，同时清除对应账本。

限制：代码只保证**已提取且未被明确更正**的事实不丢失，不保证模型提取了所有事实，也不保证更正语义正确。worker 只收到当前活动事实（最多 12 条）和与本批消息有本地词项交集的归档事实。归档按匹配词项数排序，最多 24 条、2048 token，并逐条核对包含系统提示、schema、旧摘要和本批消息的完整请求是否满足 worker 输入预算，首次请求预算检查失败就停止，可能略过后面更短的事实，以限制完整请求计数次数；整条事实保留，不截断事实或源消息。未选中事实仍保留在账本，更正 ID 只能来自本次披露的清单。词项匹配不保证召回所有语义相关的旧事实；基础请求超预算时，先从批次末尾移除整条消息直到前缀可装入，再针对该前缀召回归档。只推进实际压缩前缀的水位，其余消息保留给后续批次；单条消息与活动事实仍超预算时会报告超限，不截断原文。

主 agent 的摘要投影另含 `archived_fact_count`，按当前会话账本统计未被更正的归档事实，旧摘要加载时也会重算。它由 harness 管理，不交给摘要 worker 生成；零计数也不保证摘要提取了所有原始事实。

## 触发与调度

生产 CLI/API 启用后台摘要；直接构造 `ContextManager` 默认同步，供测试和离线任务使用。

- 只按 token 触发。分别计算 `request_occupancy = request_tokens / max_input_tokens` 与 `recent_occupancy = recent_tokens / recent_context_tokens`，任一个越线即触发。近期消息从摘要水位起按 token 装填，条数只保留 2048（可配 128–8192）的保险上限。
- **软阈值**（默认 0.60，`CONTEXT_COMPACT_SOFT_THRESHOLD`）：提交阶段按下一轮的预测占用率判断，后台排队，不等待、不提示。
- **硬阈值**（默认 0.75）：加载阶段先测量；低于硬阈值直接加载，不等在途任务。达到硬阈值时，等待该会话的后台任务（上限为摘要超时 + 5 秒），或在没有任务时同步摘要，两种情况都提示"正在压缩较早的对话记录，稍后开始判断……"。等待后重新测量，仍超硬阈值才同步兜底。
- workflow 与主对话隔离，退出时只写回结论和资源 ID，不在边界上强制摘要。
- 一个后台 worker，默认最多 8 个运行/排队会话（`summary_job_limit`），每个会话最多一个任务；队列满时记录 `context_compaction_skipped`。
- 后台结果用摘要水位 CAS 提交，并在事务内核对源消息仍存在、未被抑制；失败不推进水位。连续 3 次失败后暂停 10 分钟（进程内）。重启后从持久化水位恢复。

后台摘要的 provider 计时带 `background=true`，可能晚于 `turn_completed`，不计入该轮等待。

## 当前结论

高压力 fixture（18 轮，含三份长 JD），软/硬阈值 0.60/0.75：

| 轮间间隔 | 返回轮数 | 后台摘要 | 加载等待 | 同步摘要 |
|---|---:|---:|---:|---:|
| 0 秒（最坏情况） | 18/18 | 6 | 4 | 0 |
| 20 秒 | 18/18 | 6 | 0 | 0 |

20 秒是假设的阅读间隔，未从自然会话测得。fixture 是合成的依赖链，不代表任务质量成功率。数据见 `dual_budget_fact_pressure_2026-10-10/`。

## 复现

固定标记探针（10 轮依赖链，真实调用摘要模型，经过实际 `ContextManager` 和 SQLite 账本，逐轮核对每个事实标记是否保留，第 10 轮更正一条旧事实）。`--preflight` 只写出冻结协议，不调用模型；上次结果见 `qwen_markers/`：

```bash
.venv/bin/python evals/summary_incremental/probe_incremental_summary.py --root output/summary-markers --preflight
.venv/bin/python evals/summary_incremental/probe_incremental_summary.py --root output/summary-markers
```

高压力采样的只读汇总（把 gap20 换成 gap0 得到另一组）：

```bash
.venv/bin/python evals/summary_incremental/dual_budget_fact_pressure_2026-10-10/report_compaction.py \
  output/summary-dual-fact-pressure-gap20-2026-10-10/run-events.sqlite3 \
  --manifest evals/summary_incremental/dual_budget_fact_pressure_2026-10-10/capture_gap20.json \
  --protocol evals/summary_incremental/dual_budget_fact_pressure_2026-10-10/protocol_gap20.json \
  --output /tmp/summary-dual-budget-report.json
```

原始 trace 在被忽略提交的 `output/` 下。
