# 评估恢复基线

Qwen（`qwen3.7-plus`）与 DeepSeek（`deepseek-flash`）两套原 47 条决策轨迹的全量重录结果，以及 [runtime 质量评估](../runtime_quality/README.md) 的首批录制报告，作为后续改动的对照基线。录制只发送合成评估场景，使用各自已有的连接配置并关闭思考。

| 文件 | 内容 |
|---|---|
| `provider_probe.json` | 两个服务的连通性检查 |
| `*_trajectory_before.json` | 录制前的过期审计 |
| `*_trajectory_report.json` | 47 条轨迹重录后的报告（每场景 3 个样本） |
| `qwen_trajectory_attempt_report.json` | 一次终止失败样本的原始尝试报告 |
| `*_runtime_report.json` | runtime 质量场景的 pass^3 报告 |
| `*_runtime_initial_marker_policy.json` | 初版判分规则下的报告，仅供审计，不属于当前分母 |

模型质量仍有失败，报告如实保留，不应解读为全部通过。离线回放命令见 [runtime 质量评估](../runtime_quality/README.md)。
