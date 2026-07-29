# 更新日志

> [English](CHANGELOG.en.md) | 中文

本项目所有重要变更记录。

## [Unreleased]

### 新增

- **简历解析模块** (`src/career_agent/resume.py`)
  - PDF 文本提取（PyMuPDF）+ 纯文本支持
  - LLM 结构化解析（`with_structured_output(ResumeData)`）
  - `ResumeData` 模型，Experience 分类：
    - `WorkExperience` — 工作/实习
    - `ProjectExperience` — 项目经历
    - `Publication` — 论文/发表
    - `Award` — 竞赛/奖项
  - 工作与项目经历按 `highlights` 逐条保留
  - `skills` 与 `technologies` 保持轻量 `list[str]`
  - 简历明示求职目标使用 `stated_target_*` 字段，与模型推荐岗位分离
  - `to_facts()` 完整保留技能、技术、教育与经历内容
  - 多简历 CLI 标签由用户或文件名提供，不再由解析模型推断

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

- **简历事实边界**
  - 移除解析 Schema 中的 `ResumeLabel`
  - 禁止根据经历推断目标岗位、目标城市、期望薪资和个人简介
  - 所有缺失字段允许为空，避免“必填”要求诱发编造
  - `description` 改为 `highlights`，删除未使用的论文作者和奖项年份/等级字段
  - `skills` 仅提取明确技能栏目，禁止从项目描述重新归纳
  - `technologies` 仅容纳语言、框架、库、模型、开发/部署工具和软件平台，排除 API、数据集与评测基准
  - 增加微信号和教育 GPA 的事实提取

- **PDF 解析器**
  - 从 `pypdf` 切换到 PyMuPDF（`fitz`），改善中文简历的可读文本与版面内容提取
  - 真实中文技术简历回归中成功提取 2,324 字符，并保留教育、工作和项目段落
  - 明确扫描件仍需未来增加 OCR fallback

- **Prompt 工程** — 重写 system prompt：
  - 防幻觉接地规则（禁止编造岗位信息）
  - 中文输出格式化（表格展示岗位列表）
  - 安全边界（不自动投递、不替用户做决定）

- **MCP 集成** — 从 `mcp-jobs` 切换到 `boss-agent-cli`（50 个工具）

### 依赖

- 新增 `pymupdf`（中文 PDF 简历文本提取）
- 移除未使用的 `pypdf`

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
