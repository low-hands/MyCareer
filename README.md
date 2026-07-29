# Career Agent

> [English](#english) | 中文

智能求职助手 —— 基于 LLM + MCP 的多 Agent 架构，帮助你高效搜索岗位、分析 JD、准备面试。

## 特性

- **MCP 工具集成** — 通过 [boss-agent-cli](https://github.com/can4hou6joeng4/boss-agent-cli) 接入 Boss 直聘 50+ 工具（搜索、详情、AI 分析等）
- **Agent Loop 运行时** — 自研 observe-decide-act 循环，支持多步工具调用
- **防幻觉 Prompt** — 严格接地规则，禁止编造岗位信息
- **权限控制** — 工具按 READ / WRITE / EXTERNAL 分级，模型只能调用被授权的工具
- **Trace 可视化** — `--trace` 参数实时显示每轮工具调用轨迹
- **Schema 自动修复** — 兼容非标准 MCP 工具定义（如 `int` → `integer`）
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
│   ├── state.py                        # 对话状态持久化
│   ├── evaluation.py                   # 评估框架
│   └── runtime/
│       ├── base.py                     # AgentRuntime Protocol
│       ├── agent_loop_runtime.py       # 默认 Agent Loop
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
- [ ] 简历解析 + 岗位匹配
- [ ] 多 Specialist 节点（JD 分析、面试准备等）
- [ ] 评估体系 + Golden Cases

## License

MIT

---

<a id="english"></a>

# Career Agent (English)

An intelligent job-seeking assistant powered by LLM + MCP multi-agent architecture. Search jobs, analyze JDs, and prepare for interviews efficiently.

## Features

- **MCP Tool Integration** — 50+ tools via [boss-agent-cli](https://github.com/can4hou6joeng4/boss-agent-cli) (search, details, AI analysis, etc.)
- **Agent Loop Runtime** — Custom observe-decide-act loop with multi-step tool calling
- **Anti-Hallucination Prompt** — Strict grounding rules, no fabricated job listings
- **Permission Control** — Tools classified by READ / WRITE / EXTERNAL levels
- **Trace Visualization** — `--trace` flag shows tool call trajectory per turn
- **Auto Schema Fix** — Handles non-standard MCP tool schemas (e.g. `int` → `integer`)
- **Pluggable Architecture** — Runtime / Context / TraceSink are all Protocols

## Architecture

```
User Request
  ↓
AgentLoopRuntime (observe → decide → act loop)
  ↓
ToolRegistry (permission filter + timeout)
  ↓
MCP Server (boss-agent-cli, 50+ tools)
  ↓
Formatted Output (tables / analysis / suggestions)
```

## Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- Boss Zhipin account (for MCP tool authentication)

### Install

```bash
git clone git@github.com:low-hands/MyCareer.git
cd MyCareer
uv sync
```

### Configure

Create a `.env` file (or export environment variables):

```bash
ORCHESTRATOR_MODEL=gpt-4o
ORCHESTRATOR_API_KEY=sk-your-key-here
ORCHESTRATOR_BASE_URL=https://api.openai.com/v1
```

### Boss Zhipin Login

MCP tools require a one-time Boss Zhipin login:

```bash
uv run boss login
uv run boss status   # verify login status
```

### Run

```bash
uv run career-agent            # start with MCP tools
uv run career-agent --trace    # start with tool call tracing
uv run career-agent --no-mcp   # bare chat mode (no MCP)
```

### Test

```bash
uv run pytest
uv run ruff check src/
```

## Project Structure

```
career-agent/
├── src/career_agent/
│   ├── cli.py                          # CLI entry point
│   ├── contracts.py                    # AgentRequest / AgentResult / ToolPermission
│   ├── prompts.py                      # Prompt templates (strategy layer, tools injected dynamically)
│   ├── context.py                      # ContextManager (facts injection)
│   ├── tools.py                        # ToolRegistry (permissions + timeouts)
│   ├── tracing.py                      # TraceSink Protocol + InMemoryTraceSink
│   ├── models.py                       # Model client factory
│   ├── mcp_client.py                   # MCP client wrapper
│   ├── state.py                        # Conversation state persistence
│   ├── evaluation.py                   # Evaluation framework
│   └── runtime/
│       ├── base.py                     # AgentRuntime Protocol
│       ├── agent_loop_runtime.py       # Default Agent Loop
│       └── langgraph_runtime.py        # LangGraph adapter
└── tests/
```

## Design Principles

1. **Runtime-First** — Domain code depends on `AgentRuntime` Protocol, not specific frameworks
2. **Permission Declaration** — Tools must declare permission level before exposure to the model
3. **Layered Prompts** — Static strategy layer + dynamic injection layer (tool schemas / state / facts)
4. **Traceable** — Every run has stable thread_id and trace_id
5. **Safety Boundary** — Model output is never authority; external operations require explicit approval

## Roadmap

- [x] Single Agent + MCP tool integration
- [x] Prompt engineering (anti-hallucination + Chinese output)
- [x] Schema compatibility + Trace visualization
- [ ] LangGraph hybrid architecture (outer orchestration + inner Agent Loop)
- [ ] Resume parsing + job matching
- [ ] Multiple Specialist nodes (JD analysis, interview prep, etc.)
- [ ] Evaluation system + Golden Cases

## License

MIT
