# Qwen 首请求 `tool_choice=required` 定向实验

目的：主 agent 输出纯文本时会被拒绝并重试（`text_rejected`），这类决策耗时很长。实验检查首个请求就使用 `required` 能否在不破坏行为的前提下去掉这次重试。

`run_probe.py` 只在探针子类中覆盖请求参数；生产请求、Qwen vendor 标记、主 agent 提示和 schema 不变。四个已有场景、每个 10 次，判分规则与次数在调用前冻结，串行采样，不补采成功样本，不调用 embedding。

## 结论

`required` 去掉了纯文本拒绝，但未通过行为护栏，**不在生产启用**：

| 场景 | 本批 required |
|---|---:|
| 卡片报告一句话概括 | 6/10 |
| 五项缺失事实用问卷 | 10/10 |
| 按已披露报告 ID 读取 | 10/10 |
| 缺报告 ID、不能错用其他报告 | 0/10 |

缺 ID 场景中，被强制调用工具的模型多次产生无效参数或使用未披露的 reference。对照是此前的 3 样本录制，不是同步匹配的 A/B。

## 文件

- `qwen_required_live/`：正式 4×10 批次的协议、全部尝试与汇总；
- `qwen_provider_compatibility/`：单次服务兼容性检查；
- `qwen_initial_required/`：探针配置错误导致在调用前被本地拒绝的尝试，没有发出 provider 请求，保留作审计。
