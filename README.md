# My_Career

My_Career 是一个面向完整求职流程的本地 Career Agent。它把对话式 Agent、岗位与简历资料库、长期记忆、投递与面试状态，以及需要用户确认的外部操作放在同一个工作区中。

## 当前状态

**已经可以作为本地个人版使用。** 当前代码已覆盖从岗位采集、简历分析到投递和面试准备的主要流程，数据会持久化到本机 SQLite；刷新页面、重启服务或切换对话后，可以重新读取已经保存的业务状态和报告。

它目前仍是本地 MVP，不应直接作为公网、多用户或多实例服务部署：

- 后端只支持单进程、单 worker；同一套数据库不能同时由多个 API 实例打开。
- 对话、分析、调研、定制和模拟面试依赖配置的 OpenAI-compatible 模型服务；供应商超时或 `429/502/503/504` 会影响对应功能。
- BOSS 扩展依赖当前页面 DOM，BOSS 改版后可能需要更新选择器；它不会绕过登录或风控。
- Gmail、Google Calendar 和 QQ 邮箱需要单独完成账号授权。
- 当前没有后台调度器，邮件同步、日报刷新和日历操作仍由用户主动触发。

当前版本最近一次完整自动化验证为 **1695 passed，3 xfailed**。3 个 xfail 记录的是模型第一步偶尔选错工具的行为评测；涉及未确认偏好、公司报告绑定的实际执行路径已有 Runtime 兜底，明确历史范围场景最坏会多问一次，不会用附近无关内容冒充答案。

## 主要功能

| 模块 | 已实现能力 |
| --- | --- |
| 对话式 Agent | 流式对话、多步工具调用、结构化 observation、Human-in-the-loop 确认、失败恢复与幂等回放 |
| 岗位库 | 通过 Chrome 扩展显式保存 BOSS JD、完整 JD 快照、本地检索、岗位比较、关闭与追求状态管理 |
| 简历库 | 导入 PDF/TXT/Markdown、不可变版本管理、查看与下载原文件、在对话中精确绑定某个版本 |
| 简历分析 | 提取履历事实、让用户确认或修正、基于岗位做匹配分析，未经确认的推断不能直接成为候选人事实 |
| 简历定制 | 基于指定简历版本和完整 JD 生成修改建议、逐项审阅、Reviewer 校验、生成并下载定稿 |
| 公司调研 | 针对已保存岗位检索公开资料，保存来源与报告，并校验报告和目标公司的实体绑定 |
| 投递管理 | 记录投递、维护状态和事件流、生成行动项与每日 brief |
| 面试 | 记录面试轮次、生成岗位相关准备材料、进行可暂停和恢复的模拟面试并生成复盘报告 |
| 邮件与日历 | Gmail 只读同步、QQ IMAP、招聘邮件识别、Google Calendar 预览与确认后写入 |
| 长期记忆 | 管理已确认履历事实和偏好、隔离 Agent 未确认推断、BM25 检索及可选向量检索、修订和删除传播 |
| 本地可靠性 | API key 权限隔离、单实例锁、并发背压、外部写入确认、执行状态持久化、备份/校验/恢复 |

## Agent 架构

- **Main Agent：** 基于 LangGraph 实现 `hydrate → decide → authorize → act → observe → present/interrupt` 的 ReAct 控制循环，负责理解当前请求并决定下一步；每次只执行一个工具，工具结果以结构化 observation 回灌模型。
- **专家工作流：** 岗位调研和简历定制使用 Deep Agents 执行受约束的领域任务；模拟面试使用带 SQLite checkpoint 的 LangGraph 子图，支持暂停、恢复和复盘。
- **渐进式披露：** 岗位调研和简历定制 Worker 通过 Deep Agents 的 Skills 系统按需读取领域说明；模拟面试只为当前 `plan/ask/evaluate/report` 操作装配相关参考资料，避免把所有专家流程长期塞入提示词。
- **确定性 Runtime：** Runtime 负责参数校验、资源归属、公司实体绑定、权限预算、幂等、持久化和副作用；模型负责判断和生成内容，但不能绕过这些执行约束。
- **上下文工程：** 当前轮只装配与任务有关的会话摘要、状态、已确认资料和必要文档摘录；完整 JD、简历和报告保存在业务库中，在明确选中后按需读取，避免让无限增长的历史直接占满模型窗口。
- **可治理记忆：** 记忆区分已确认事实、待确认偏好和 Agent working notes；未确认推断被隔离，不能直接影响推荐或匹配，同时保留来源、版本、修订和派生内容删除语义。
- **可回放评测：** SQLite Runtime trace 记录模型决策、工具调用、结果状态、token 与缓存指标；trajectory cassette 可以回放固定场景，并将模型选错第一步和 Runtime 是否成功兜底分开评估。

## 快速启动

### 1. 环境要求

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- Node.js `^20.19.0` 或 `>=22.12.0`
- 一个支持 Chat Completions 的 HTTPS OpenAI-compatible 模型接口
- 如需采集 BOSS 岗位：Chrome、Edge、Brave 或 Arc 等 Chromium 浏览器

### 2. 安装依赖

在仓库根目录执行：

```bash
uv sync --extra test
cd web
npm ci
cd ..
```

### 3. 配置模型

复制环境变量示例：

```bash
cp .env.example .env
```

至少填写下面两组配置。两组可以先使用同一个模型服务：

```dotenv
# Main Agent：负责对话、任务路由和工具选择
MAIN_AGENT_BASE_URL=https://你的服务/v1
MAIN_AGENT_API_KEY=你的密钥
MAIN_AGENT_MODEL=你的模型名
MAIN_AGENT_TIMEOUT_SECONDS=120

# 专家模型：负责简历分析/匹配/定制、调研、邮件处理和模拟面试
RESUME_ANALYSIS_AGENT_BASE_URL=https://你的服务/v1
RESUME_ANALYSIS_AGENT_API_KEY=你的密钥
RESUME_ANALYSIS_AGENT_MODEL=你的模型名
```

`BASE_URL` 可以填写 `/v1` 基地址，也可以直接填写完整的 `/chat/completions` 地址。`.env` 已被 Git 忽略，不要提交任何真实密钥。

`MAIN_AGENT_TIMEOUT_SECONDS=120` 是当前推荐值。Main Agent 对连接错误以及 `429/502/503/504` 最多做 3 次有限指数退避；持续不可用时会明确失败，不会无限重试或把不完整回答交给用户。

### 4. 创建本地 API key

在启动 API 之前创建两把 key。两条命令必须使用相同的 `--user-id`，这样 Web 和扩展保存的数据属于同一个本地用户：

```bash
uv run career-agent api-keys issue \
  --user-id local-user \
  --name local-web \
  --scope workspace:read \
  --scope workspace:write \
  --scope chat:write \
  --scope settings:write \
  --no-expiry

uv run career-agent api-keys issue \
  --user-id local-user \
  --name boss-extension \
  --scope capture:write \
  --no-expiry
```

每条命令只会显示一次 `secret`。在 `web/.env` 中分别填写：

```dotenv
CAREER_AGENT_WEB_API_KEY=第一条命令返回的_secret
VITE_CAPTURE_API_KEY=第二条命令返回的_secret
```

Web key 由 Vite 本地代理注入请求；权限更窄的 capture key 只交给 BOSS 扩展。`web/.env` 同样已被 Git 忽略。

### 5. 启动后端

在仓库根目录开启第一个终端：

```bash
uv run career-agent-api
```

服务默认监听 `http://127.0.0.1:8000`。可检查：

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/ready
```

`health` 表示 HTTP 服务已启动；`ready` 表示模型配置和 Runtime 也已成功初始化。如果 `ready` 返回 503，按响应中的配置错误检查 `.env`。

### 6. 启动前端

在第二个终端执行：

```bash
cd web
npm run dev
```

然后用浏览器打开 [http://127.0.0.1:5173/](http://127.0.0.1:5173/)。修改 `web/.env` 后需要重启 Vite。

### 7. 安装 BOSS 岗位收藏扩展（可选）

1. 在 Chrome 打开 `chrome://extensions` 并启用“开发者模式”。
2. 点击“加载已解压的扩展程序”，选择仓库中的 `browser-extension/`。
3. 用**同一个 Chrome profile** 打开 Career Agent Web 页面，并在该 profile 中登录 BOSS。
4. 正常打开 BOSS 岗位详情；扩展解析完整 JD 后会在右下角显示保存卡片，只有用户点击后才写入本地岗位库。

扩展不会自动搜索、滚动、投递、发送消息，也不会读取 Cookie 或绕过平台验证。完整说明见 [browser-extension/README.md](browser-extension/README.md)。

## 推荐的首次使用顺序

1. 在“简历管理”导入 PDF、TXT 或 Markdown 简历；也可以在对话框用“+”或拖拽文件导入，二者写入同一个简历库。
2. 打开该简历的原文件确认版本无误，再让 Agent 分析并确认提取出的履历事实。
3. 使用扩展保存一个完整 BOSS JD，或先浏览已经保存的岗位库。
4. 在对话中让 Agent 比较岗位、分析“指定简历版本 × 指定岗位”，或生成公司调研。
5. 发起简历定制，逐项确认修改后生成和下载定稿。
6. 记录投递和面试轮次，再生成面试准备材料或开始模拟面试。
7. 如有需要，连接 Gmail/QQ 邮箱与 Google Calendar；所有外部写入仍需显式确认。

对话附件只发送 `resume_version` 的内部引用。Runtime 会校验该版本是否属于当前用户，并只向模型装配任务所需的元数据和有界文本；对话 transcript 不保存一份重复的简历原文。删除简历后，旧对话仍保留名称等安全快照，但不能继续读取已删除文件。

## 运行与数据边界

### 单进程限制

后端是本地 SQLite、单进程部署，**只支持单 worker**：

```bash
uv run career-agent-api                              # 固定 workers=1
uv run uvicorn career_agent.api.app:app --workers 1 # 也可以
```

不要使用 `--workers 2+`、`WEB_CONCURRENCY>1`，也不要让多个 API 实例共用默认的 `~/.career-agent/*.sqlite3`。启动时进程会取得 `~/.career-agent/api-server.lock`；第二个实例会立即失败，进程退出后锁由内核释放。

会写数据库的 CLI 命令也使用同一把锁，因此 API 运行时会拒绝执行 `chat`、`actions settle`、`memory apply`、`settings set`、`target-role create`、`resume import`、`email/calendar add-account` 和 `api-keys issue/revoke`。只读命令不取锁。

同一进程默认最多同时运行 3 个 turn，可用 `CAREER_AGENT_MAX_CONCURRENT_TURNS` 修改。达到上限时 API 返回 `503 TURN_CAPACITY_EXHAUSTED`；同一对话已有 turn 在运行时返回 `409 CONVERSATION_TURN_IN_PROGRESS`。关闭服务会先等待正在执行的 turn，默认排空时间为 30 秒，可用 `CAREER_AGENT_SHUTDOWN_DRAIN_SECONDS` 调整。

### 数据与隐私

- 业务数据默认保存在 `~/.career-agent/` 下的多个 SQLite 文件和 `working-notes/`；API key 默认保存在仓库的 `data/api_keys.sqlite3`。
- 岗位、简历、投递、对话和报告默认不会上传到本项目自建服务器，因为项目没有中心服务器。
- 这不是完全离线应用：执行模型任务时，当前任务所需的提示、资料摘录或正文会发送到你配置的模型服务；使用 Google、QQ 或 BOSS 时，也受相应外部服务的数据政策约束。
- Gmail/Calendar/QQ 的连接凭据保存在系统 keyring，页面和 API 不回传 secret；`.env` 中的模型和 OAuth 密钥仍由本机用户负责保护。

## 备份与恢复

工作区包括 `~/.career-agent/*.sqlite3`、`~/.career-agent/working-notes/` 和 `$CAREER_AGENT_DATA_DIR/api_keys.sqlite3`。建议在开始保存正式简历和求职记录前建立备份习惯：

```bash
uv run career-agent backup create
uv run career-agent backup create --dest /Volumes/usb/career-2026-09-14
uv run career-agent backup verify --source ~/.career-agent-backups/20260914T120000Z
uv run career-agent backup restore --source ~/.career-agent-backups/20260914T120000Z --yes
```

- `create` 使用 SQLite 在线备份 API 逐库复制，运行 `integrity_check` 并生成包含文件大小和 SHA-256 的 `manifest.json`。
- 创建和恢复一致性备份前必须停止 API；命令取不到 `api-server.lock` 时会拒绝运行。`--allow-running-api` 会将备份明确标记为不一致快照。
- `verify` 检查文件、摘要和 SQLite 完整性；`restore` 只有在校验通过后才覆盖，并默认先创建 safety copy。
- 如果使用非默认数据库路径，给 `backup` 传入与 `chat` 相同的 `--*-store` 参数。

## Gmail OAuth 配置

Gmail 连接需要先创建 Google OAuth Client。普通用户授权 Gmail 时不会看到或填写 Client Secret。

1. 打开 [Google Cloud Console](https://console.cloud.google.com/) 并创建或选择项目。
2. 在 API Library 启用 **Gmail API**。
3. 在 Google Auth Platform 配置 OAuth consent screen；测试模式下把自己的 Google 邮箱加入 Test users。
4. 在 Clients/Credentials 中创建 **Web application** 类型的 OAuth Client。
5. 添加 Authorized redirect URI：`http://127.0.0.1:8000/v1/connections/google/callback`。
6. 在仓库根目录的 `.env` 中添加：

   ```dotenv
   GOOGLE_OAUTH_CLIENT_ID=你的_Client_ID
   GOOGLE_OAUTH_CLIENT_SECRET=你的_Client_Secret
   GOOGLE_OAUTH_CALLBACK_URL=http://127.0.0.1:8000/v1/connections/google/callback
   CAREER_AGENT_WEB_URL=http://127.0.0.1:5173
   ```

7. 重启后端，在“邮件追踪”页面点击“连接 Gmail”。该入口只申请 Gmail 只读权限。

不要把 Client Secret 提交到仓库或放入 `web/.env`。

## Google Calendar（可选）

项目内置面试月历不依赖 Google Calendar。不连接 Google 时，项目中的面试仍会显示在前端月历。

如需同步到 Google Calendar，在同一个 Google Cloud 项目启用 **Google Calendar API**，复用上面的 OAuth Client 和回调地址，再从“面试日历”页面连接。Calendar 入口只申请日历事件权限；外部写入经过 proposal → 用户确认 → 执行 → 审计流程。

## QQ 邮箱连接

QQ 邮箱使用 IMAP 授权码，不使用登录密码：

1. 登录 [QQ 邮箱网页版](https://mail.qq.com/)。
2. 打开“设置 → 账号与安全 → 安全设置”。
3. 开启 IMAP/SMTP 第三方客户端服务并生成授权码。
4. 在应用“邮件追踪”页面点击“连接 QQ 邮箱”。
5. 填写完整 QQ 邮箱地址和授权码。

授权码保存在系统 keyring；页面和 API 不会返回授权码或邮件正文。

## 开发验证

```bash
uv run pytest
cd web
npm run test
npm run typecheck
npm run build
```

后端 API 契约变更后，按仓库现有 contract 生成流程更新 `web/src/contracts/api-contract.ts`，再运行前端测试和构建。
