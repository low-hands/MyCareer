# 按阶段的延迟基线

`career_agent.evaluation.latency` 只读 `run-events.sqlite3`，输出 JSON 和 Markdown。仅依赖 Python 标准库；不初始化业务服务、不调用模型、不迁移数据库、不读取对话正文。以 run 为单位选择样本，不在时间边界截断一轮的事件。

```bash
.venv/bin/python -m career_agent.evaluation.latency ~/.career-agent/run-events.sqlite3 \
  --output output/latency.json --markdown output/latency.md

# 从有工具埋点、且上下文较长的轮次中建立可比较的子集。
.venv/bin/python -m career_agent.evaluation.latency ~/.career-agent/run-events.sqlite3 \
  --since 2026-10-09T00:00:00-07:00 --min-context-chars 8000 --require-tool-trace \
  --output output/latency-long.json --markdown output/latency-long.md
```

`--since` 包含起点，`--until` 不包含终点，要求 ISO 8601 带时区。时间筛选依据每轮最早的 trace 事件；上下文筛选依据首个决策的 `context_chars`，不是包含系统提示和工具 schema 的完整请求 token 数。

| 阶段 | 统计事件 | 口径 |
|---|---|---|
| 决策 | `model_attempt`、`model_succeeded`、`model_failed`，category=`orchestrator_decision` | 逻辑决策次数与耗时，包括内部重试等待；`attempt_count` 另计真实请求次数。 |
| 专项 worker | 其他有 category 的模型事件，按 stage 细分 | worker 模型调用耗时，包含失败。 |
| embedding | `provider_request`，stage=`embedding` | 一次 `embed()` 操作，含批处理；turn 外的启动预热不计入。 |
| 工具执行 | `capability_executed`，按 `details.tool_name` 细分 | 包含 worker、embedding、业务 I/O 的整体工具耗时。 |
| 摘要压缩 | `provider_request`，stage=`conversation_summary` | `summarize()` 调用含校验；后台调用带 `background=true`，单独汇总。 |

各阶段输出单次总量/均值/p50/p95/max、成功与失败耗时、每轮次数和每轮阶段耗时分布。p50/p95 用 nearest rank；真实的 0ms 保留，缺失数据不填 0。工具与 worker/embedding 嵌套，不能相加为端到端延迟；trace 首尾跨度也不等于用户看到回复的延迟。

## 真实采样

```bash
# 使用已有 .env 连接，真实调用模型；目录必须不存在。
.venv/bin/python -m career_agent.evaluation.latency_capture \
  --work-root output/latency-new --conversations 3
.venv/bin/python -m career_agent.evaluation.latency output/latency-new/run-events.sqlite3 \
  --output output/latency-new/summary.json --markdown output/latency-new/summary.md
```

采样器使用 CLI 的生产 composition root、实际 SQLite 业务存储、真实决策模型、worker、embedding 和摘要服务；store 路径重定向到独立目录。失败或等待用户输入时停止该会话，保留已尝试的轮次，不补采成功样本。会话串行运行，共享岗位和简历存储，因此包含首次计算与缓存读取。`--inter-turn-delay-seconds` 可在两轮之间加入固定的阅读间隔，间隔不计入 wall time。

`scenarios.json` 是编写的代表性资料（完整简历、JD、九轮职业讨论，随后查询记录、分析 JD、请求匹配和追问），不能代替自然使用数据。`capture.json` 保存源码和 fixture 指纹、模型配置、逐轮 wall time 和失败状态；trace 数据库存放在被忽略提交的 `output/` 下。

工具后的 `final` 与 `ask_user` 分开统计：

```bash
.venv/bin/python evals/latency/report_post_tool_response.py \
  output/<capture-dir>/run-events.sqlite3 \
  --capture-manifest output/<capture-dir>/capture.json \
  --output output/<capture-dir>/post_tool.json \
  --markdown output/<capture-dir>/post_tool.md
```

摘要服务的固定输入探针（输出路径必须不存在）：

```bash
CONVERSATION_SUMMARY_AGENT_DISABLE_THINKING=true .venv/bin/python evals/latency/probe_summary.py \
  --samples 10 --diagnostic --output output/summary-probe.json
```

## 当前结论

- 摘要服务需要关闭思考（`CONVERSATION_SUMMARY_AGENT_DISABLE_THINKING=true`），否则在 30 秒超时内无法完成。provider 的全局默认仍为 False，其他环境需显式设置。
- 摘要已改为后台执行，触发规则和高压力采样见 [增量事实与后台摘要](../summary_incremental/README.md)。
- 含 `text_rejected`（模型输出纯文本被拒后重试）的决策占串行基线整批 wall time 的 47%，其中包含有效生成，不能全部视为浪费。首请求 `tool_choice=required` 的定向实验未通过行为护栏，见 [required_tool_choice](../required_tool_choice/README.md)。

这一目录下的 JSON 是各次采样的结构化结果与冻结协议。
