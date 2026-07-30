# 更新日志

> [English](CHANGELOG.en.md) | 中文

本项目所有重要变更记录。

## [Unreleased]

### 新增

- **岗位事实存储** (`src/career_agent/jobs/`)
  - `JobPosting` 保存来源可追溯的岗位摘要或完整 JD，不混入匹配结果
  - SQLite Repository 强制按 `tenant_id` 查询和更新
  - 依次使用来源岗位 ID、规范化 URL、内容指纹去重
  - 搜索摘要刷新不会清除已保存的完整 JD 或其他非空详情
  - `SearchCriteria` 保存用户原始搜索意图，`SearchRun` 保存不可变的展示结果快照
  - SearchRun SQLite 表只保存有序 `job_id` 引用，不复制完整 JD
  - 支持按位置或公司/岗位/城市/薪资解析跨轮引用，并区分唯一命中、歧义和未找到
  - Boss MCP Provider 使用 `boss_export` 只读内联模式获取完整筛选结果，并通过 `boss_detail` 补充完整 JD
  - 上游 JSON 信封统一转换为 `JobPosting`，保留详情定位符但不保存凭据
  - 每轮生成受 tenant/thread 上下文约束的搜索和详情工具，LLM 只生成 `SearchCriteria`/`SearchResultSelector`
  - CLI 已接通搜索业务链；原始 Boss MCP 工具不再直接暴露给 LLM
  - 会话存储跨轮复用，业务工具按轮捕获可信的 tenant/thread/user_query 上下文
  - Agent loop 新增终态 Response Guard；明确的岗位搜索/详情请求必须有对应的成功工具调用证据
  - 无证据答案会在有限预算内反馈重试，耗尽后返回安全提示；工具错误不计为有效证据

- **简历领域模型与解析模块** (`src/career_agent/resumes/models.py`, `src/career_agent/resumes/parser.py`)
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
  - 简历岗位池元数据与解析事实分离，不再由解析模型推断

- **岗位简历池** (`src/career_agent/resumes/`)
  - 按自由岗位方向 `role_type` 对简历版本分池
  - 每个池内版本号独立递增，版本绑定不可变原始文件及解析快照
  - 单个 SQLite 数据库通过强制 `tenant_id` 查询实现租户隔离
  - SHA-256 防止同一租户在同一岗位池重复上传相同文件
  - 解析失败不写数据库、不遗留文件
  - 当前选中版本持久化，程序重启后可恢复

- **简历 CLI 集成**
  - `--resume <path> --resume-role <role>` 启动时加入岗位池
  - `/resume add <role> <path> [note]` 添加版本
  - `/resume list [role]` 按岗位池查看版本
  - `/resume show <version-id>` 查看版本详情
  - `/resume use <version-id>` 选择活跃版本
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
