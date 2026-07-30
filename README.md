# Career Agent

> [English](README.en.md) | 中文

智能求职助手 —— 基于 LLM + MCP 的多 Agent 架构，帮助你高效搜索岗位、分析 JD、准备面试。

## 特性

- **MCP 工具集成** — 通过 [boss-agent-cli](https://github.com/can4hou6joeng4/boss-agent-cli) 接入 Boss 直聘 50+ 工具（搜索、详情、AI 分析等）
- **Agent Loop 运行时** — 自研 observe-decide-act 循环，支持多步工具调用
- **防幻觉 Prompt** — 严格接地规则，禁止编造岗位信息
- **权限控制** — 工具按 READ / WRITE / EXTERNAL 分级，模型只能调用被授权的工具
- **Trace 可视化** — `--trace` 参数实时显示每轮工具调用轨迹
- **Schema 自动修复** — 兼容非标准 MCP 工具定义（如 `int` → `integer`）
- **岗位简历池** — 按岗位方向管理多个不可变简历版本，SQLite 按租户持久化
- **可插拔架构** — Runtime / Context / TraceSink 均为 Protocol，方便替换

## 架构

```
用户请求
  ↓
AgentLoopRuntime（observe → decide → act 循环）
  ↓
ToolRegistry（权限过滤 + 超时控制）
  ↓
MCP Server（boss-agent-cli 50+ 工具）
  ↓
格式化输出（表格 / 分析 / 建议）
```

## 快速开始

### 环境要求

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) 包管理器
- Boss 直聘账号（用于 MCP 工具认证）

### 安装

```bash
git clone git@github.com:low-hands/MyCareer.git
cd MyCareer
uv sync
```

### 配置

创建 `.env` 文件（或 export 环境变量）：

```bash
ORCHESTRATOR_MODEL=gpt-4o
ORCHESTRATOR_API_KEY=sk-your-key-here
ORCHESTRATOR_BASE_URL=https://api.openai.com/v1

RESUME_MODEL=gpt-4o
RESUME_API_KEY=sk-your-key-here
RESUME_BASE_URL=https://api.openai.com/v1

CAREER_AGENT_TENANT_ID=local
CAREER_AGENT_DATA_DIR=~/.career-agent
```

### Boss 直聘登录

MCP 工具需要 Boss 直聘登录态（只需一次）：

```bash
uv run boss login
uv run boss status   # 验证登录状态
```

### 运行

```bash
uv run career-agent            # 启动（带 MCP 工具）
uv run career-agent --trace    # 启动并显示工具调用轨迹
uv run career-agent --no-mcp   # 纯对话模式（不启动 MCP）
```

启动时也可以直接将简历加入指定岗位池：

```bash
uv run career-agent \
  --resume "/path/to/resume.pdf" \
  --resume-role "Agent开发"
```

会话中的简历池命令：

```text
/resume add "Agent开发" "/path/to/resume.pdf" "通用版"
/resume list
/resume list "Agent开发"
/resume show <version-id>
/resume use <version-id>
```

数据默认保存在 `~/.career-agent/`。当前本地模式使用单个 SQLite
数据库，并通过 `tenant_id` 对所有简历查询进行隔离；后续可将同一
Repository 接口迁移到 PostgreSQL。

### 测试

```bash
uv run pytest
uv run ruff check src/
```

## 项目结构

```
career-agent/
├── src/career_agent/
│   ├── cli.py                          # CLI 入口
│   ├── contracts.py                    # AgentRequest / AgentResult / ToolPermission
│   ├── prompts.py                      # Prompt 模板（策略层，工具动态注入）
│   ├── context.py                      # ContextManager（facts 注入）
│   ├── tools.py                        # ToolRegistry（权限 + 超时）
│   ├── tracing.py                      # TraceSink Protocol + InMemoryTraceSink
│   ├── models.py                       # 模型客户端工厂
│   ├── mcp_client.py                   # MCP 客户端封装
│   ├── jobs/
│   │   ├── models.py                   # JobPosting 岗位事实模型
│   │   ├── sqlite_repo.py              # 岗位 SQLite Repository
│   │   ├── search_models.py            # SearchCriteria / SearchRun / 解析结果
│   │   ├── search_repository.py        # 不可变搜索快照接口
│   │   ├── search_sqlite_repo.py       # SearchRun SQLite 实现
│   │   ├── search_service.py           # 发布结果与跨轮引用解析
│   │   ├── providers/                   # 外部岗位来源 Adapter（Boss MCP）
│   │   ├── discovery_service.py         # 搜索、持久化、详情获取业务链
│   │   ├── agent_tools.py               # 按任务生成的 function-calling 工具
│   │   ├── grounding_guard.py           # 岗位搜索/详情接地策略
│   │   └── wiring.py                    # MCP → 业务服务 → Agent 工具接线
│   ├── resumes/
│   │   ├── models.py                   # 简历领域模型（ResumeData / ResumeVersion）
│   │   ├── parser.py                   # PDF/文本提取与 LLM 结构化
│   │   ├── repository.py               # 简历版本持久化接口
│   │   ├── sqlite_repo.py              # SQLite Repository 实现
│   │   └── service.py                  # 岗位简历池业务服务
│   ├── state.py                        # 对话状态持久化
│   ├── evaluation.py                   # 评估框架
│   └── runtime/
│       ├── base.py                     # AgentRuntime Protocol
│       ├── agent_loop_runtime.py       # 默认 Agent Loop
│       ├── response_guard.py            # 终态答案审查与工具证据契约
│       └── langgraph_runtime.py        # LangGraph 适配器
└── tests/
```

## 设计原则

1. **Runtime 框架优先** — 领域代码依赖 `AgentRuntime` Protocol，不依赖具体框架
2. **工具权限声明** — 工具必须声明权限级别才能暴露给模型
3. **Prompt 分层** — 静态策略层 + 动态注入层（工具 schema / state / facts）
4. **可追踪** — 每次运行有稳定的 thread_id 和 trace_id
5. **安全边界** — 模型输出不是权威，外部操作需要显式授权

## 演进路线

- [x] 单 Agent + MCP 工具集成
- [x] Prompt 工程（防幻觉 + 中文输出）
- [x] Schema 兼容 + Trace 可视化
- [ ] LangGraph 混合架构（外层编排 + 内层 Agent Loop）
- [x] 简历解析（PDF/文本 + 结构化事实抽取）
- [x] 多租户岗位简历池（SQLite持久化 + 多版本）
- [ ] 岗位匹配
- [ ] 多 Specialist 节点（JD 分析、面试准备等）
- [ ] 评估体系 + Golden Cases

## License

MIT
