# 新会话回读一致性与 schema 指纹追溯（2026-10-08）

新会话（through_sequence=0 且 recent_from_sequence 为 None/1）在 prepare_capability_selection 前排除 read_conversation_span。offered_names、schemas 和 available_now 使用同一选择快照，删除投影阶段的单独过滤。预加载、显式加载和意图命中都不能绕过此排除。

MainAgentContext.has_compressed_history 调用唯一的 compressed_history_available 水位判断；selection、参数投影和 registry 守卫共用这一判断。runtime 投影携带内部历史可用性标记，registry 仍校验水位作为兜底；直接调用未提供水位时保持原有行为，伪造 True 标记也不能覆盖新会话水位。显式回读 prelude 和模型请求未提供工具时的解析也复用该属性，新会话不会隐式加载该工具。

原三个 available_now 一致性测试恢复通过。新增 None/1 冷会话、只有遗漏前缀、只有摘要水位等边界用例；新增 runtime 强行调用测试，断言 not_offered 且没有执行工具。既有 registry 不可用回读测试保留并覆盖缺少/矛盾投影标记。

## 过期指纹来自哪些提交

使用 git log -p 检查 schema 与测试历史，并将 23019e1 的父提交和该提交的 src 导出到独立临时目录运行同一指纹脚本；没有改动当前 checkout 来探测历史。

- 743e81b 最后更新此测试的 prompt/trajectory 期望；schema 全量散列仍为 4e3d8460…，不含 search 为 7492097e…。
- a964b93 有意启用 hybrid intent ranking 和 recorded evaluation strategy，改变逐步提供的 schemas；没有同步 trajectory 指纹期望。23019e1 的父提交实测 trajectory 已为 58bca189…，不同于测试中的 07956099…。
- **23019e1（feat(memory): ground compressed history readback）是 schema 散列过期的直接原因。** from_sequence/through_sequence 从必填整数改为默认 None 的可选整数，允许 runtime 从水位确定完整回读范围，并修订 read_conversation_span 描述和 query 提示。该提交的配套投影、registry、场景和测试明确使用省略边界的行为；紧随其后的 6f801b5 专门刷新模型录制，确认这是有意改动。

| 指纹 | 23019e1 父提交 | 23019e1 |
| --- | --- | --- |
| 全量 schema | 4e3d8460ad7985cc7c2c915065c0e666347666fc8194cdb723a359e430db9ae0 | 9e8d3f50d3fc0f690d60ef631fe0352aa53b0a6c9f9fea56c153e9ca3bf0441e |
| 去掉 search schema | 7492097e01607e7d57c5e2ad29c6a0fe65cb99edc2d64b946cc530dbb2a9dcb3 | 5441650204630b35a8ac5201e4c34289b1d1885436711643cd441e91fed8e14d |
| prompt | c4fb185d167566a51b3d333c8d367d9487fc06d4882bb1b01718f11562b66da8 | 05c2e750c8665eac45975a069093627c733d01c217a91be256ec9bc3b657a719 |
| trajectory | 58bca18992cabfe701018be786f7f5b4f15840a4e0bd74d5ac18f6a2c96d5804 | 916c64318e657349869a62e420be3149b6fd0b55e5720aebc0662211508866d8 |

工作区第 1 步读召回改动进一步使 trajectory 为 55604b2c…；本次冷会话 schemas 修复后为 44f1caab…；随后按用户要求统一读写门槛后为 **5efade0d1a7d0c79264c16b302e66e8fa1e8fb01a86b6fc6d9d9371c463cfcb7**。后三项不改变工具 schema 本身，只改变逐步提供的工具集合。测试在追溯完成后才更新全部四个期望值，并加注释说明来源。

## 录制依赖与失效范围

record_search_catalogue 写入 trajectory_prompt_fingerprint，search_cassette_staleness 对该值严格比较，replay_search_sample 还校验后续决策的 schema/context 快照。因此不能仅改录制中的散列来宣称录制有效。

使用正式 SEARCH_SCENARIOS，对 evals/main_agent_search（qwen3.7-plus）和 evals/main_agent_deepseek_search（deepseek-flash）各 47 条当前场景录制只做元数据审计。修复前两套各有 40 条元数据新鲜、7 条已过期；修复后以及统一门槛后均为 3 条新鲜、44 条过期，其中每套 37 条是由新会话工具选择变化新增的失效。逐项存储/当前指纹见 HISTORY_SELECTION_FINGERPRINT_AUDIT_2026-10-08.json；新鲜仅指元数据，不等于模型质量验收通过。未将孤立旧场景或 zip 归档算入。

已尝试使用当前配置的 deepseek-flash 重录。受限环境 DNS 解析失败；允许联网后，实际 HTTP 客户端及不使用环境代理的诊断仍返回 ConnectError，无法连接配置的模型服务。停止无效重试，未修改任何录制文件、未伪造新鲜散列。Qwen 当前没有对应 MAIN_AGENT 配置。**两套失效录制仍需在服务可用、对应模型配置齐备后重录，不能用于当前代码的模型质量结论。**

## 验证与提交说明

原四个失败测试均通过；最终相关测试 545 passed、3 holdout 测试未执行、4 strict xfailed（召回问题，见 catalog_search/UNIFORM_GATE_2026-10-08.md）。没有评 catalog-search holdout。

提交说明应明确：修复冷会话 read_conversation_span 的 offered/schema/projection 一致性；schema 散列过期由 23019e1 的有意可选边界/说明改动引起，trajectory 还包含 a964b93 的意图预加载及本次统一门槛/冷会话选择变化；录制必须重新生成，目前因服务连接失败未完成。
