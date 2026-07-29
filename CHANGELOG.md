# 更新日志

> [English](CHANGELOG.en.md) | 中文

本项目所有重要变更记录。

## [Unreleased]

### 新增

- **简历解析模块** (`src/career_agent/resume.py`)
  - PDF 文本提取（pypdf）+ 纯文本支持
  - LLM 结构化解析（`with_structured_output(ResumeData)`）
  - `ResumeData` 模型，Experience 分类：
    - `WorkExperience` — 工作/实习
    - `ProjectExperience` — 项目经历
    - `Publication` — 论文/发表
    - `Award` — 竞赛/奖项
  - `skills`（软技能）与 `tech_stack`（技术栈）分离
  - `ResumeLabel` 枚举（16 个计算机方向：backend, ml, llm, cv, agent 等）
  - `to_facts()` 结构化输出（公司:岗位, 项目:概要, 论文:级别）
  - 多简历支持，标签管理

- **简历 CLI 集成**
  - `--resume <path>` 启动时加载简历
  - `/resume [label] <path>` 对话中加载
  - `/resume list` 查看所有已加载简历
  - `/resume use <label>` 切换活跃简历
  - 简历 facts 自动注入 system prompt

- **Trace 可视化**（`--trace` 参数）
  - 显示每轮工具调用轨迹（工具名 + step 编号）

- **Schema 自动修复**（`_sanitize_tools`）
  - 递归修复 MCP 工具的非标准 JSON Schema 类型
  - `int` → `integer`, `float` → `number`, `bool` → `boolean`

### 变更

- **Prompt 工程** — 重写 system prompt：
  - 防幻觉接地规则（禁止编造岗位信息）
  - 中文输出格式化（表格展示岗位列表）
  - 安全边界（不自动投递、不替用户做决定）

- **MCP 集成** — 从 `mcp-jobs` 切换到 `boss-agent-cli`（50 个工具）

### 依赖

- 新增 `pypdf`（PDF 简历解析）

## [0.1.0] — 2025-07-29

### 初始版本

- `AgentLoopRuntime` — 自研 observe-decide-act 函数调用循环
- `LangGraphRuntime` — LangGraph 适配器（带 checkpoint）
- `ToolRegistry` — 权限感知的工具注册（READ/WRITE/EXTERNAL）
- `TraceSink` Protocol + `InMemoryTraceSink` — 可插拔运行追踪
- `ContextManager` Protocol + `BasicContextManager` — system prompt + facts 注入
- `ConversationStore` Protocol + `InMemoryConversationStore` — 线程级对话状态
- MCP 客户端集成（`langchain-mcp-adapters`）
- CLI 入口，支持 `--no-mcp` 纯对话模式
- 角色化环境变量（`ORCHESTRATOR_*`）
- 评估框架骨架
- 完整测试套件（ruff + pytest）
