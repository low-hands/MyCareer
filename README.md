# My_Career

Career Agent：面向求职流程的本地 Agent 与管理工作区。

## 运行后端

后端是本地 SQLite、单进程部署，**只支持单 worker**：

```bash
career-agent-api                                  # 等价于 uvicorn --workers 1
uvicorn career_agent.api.app:app --workers 1      # 也可以
```

不要用 `--workers 2+`、`WEB_CONCURRENCY>1` 启动，也不要让多个 API 实例共用同一套业务数据库（默认 `~/.career-agent/*.sqlite3`）。同一会话的 turn 互斥（`ConversationRunGate`）只在单进程内生效；多进程共用数据库会让同一会话并发执行两个 turn。

启动时进程会对会话库所在目录的 `api-server.lock`（默认 `~/.career-agent/api-server.lock`）取独占锁并持有到退出。锁跟着业务数据库走，与只存 API key 的 `CAREER_AGENT_DATA_DIR` 无关。第二个进程取不到锁会立即启动失败并给出提示；`WEB_CONCURRENCY` 等变量大于 1 也会在启动时被拒绝。锁由内核随进程退出自动释放，崩溃后无需手工清理。

同一进程内同时运行的 turn 数有上限（默认 3，`CAREER_AGENT_MAX_CONCURRENT_TURNS` 可改）。每个 turn 在模型调用期间独占一个工作线程，不设上限时多个对话同时发送会把线程占满、等模型超时才恢复。超出上限的请求立即返回 `503 TURN_CAPACITY_EXHAUSTED`（带 `Retry-After`），提示"当前任务较多，请稍后再试"；没有排队。同一对话重复发送仍然是 `409 CONVERSATION_TURN_IN_PROGRESS`。

## 备份与恢复

工作区 = `~/.career-agent/*.sqlite3` 全部业务库 + `~/.career-agent/working-notes/` + `$CAREER_AGENT_DATA_DIR/api_keys.sqlite3`。请在开始保存正式简历和求职记录前养成备份习惯：

```bash
career-agent backup create                       # 复制到 ~/.career-agent-backups/<UTC 时间戳>/
career-agent backup create --dest /Volumes/usb/career-2026-09-14
career-agent backup verify --source ~/.career-agent-backups/20260914T120000Z
career-agent backup restore --source ~/.career-agent-backups/20260914T120000Z --yes
```

- `create` 用 SQLite 在线备份 API 逐库复制（WAL 里已提交的页也带上），逐库跑 `integrity_check`，并写 `manifest.json`（每个文件的大小 + SHA-256）。目标目录必须不存在或为空。可以在 API 运行时执行。
- `verify` 核对 manifest 里每个文件存在、大小和 SHA-256 一致、数据库通过 `integrity_check`；有任何问题退出码为 5。
- `restore` 先 `verify`，不通过则一个文件都不动；必须停掉 API（restore 会尝试取 `api-server.lock`，取不到即拒绝）；覆盖前先把当前工作区备份到 `~/.career-agent-backups/pre-restore-<时间戳>/`（`--no-safety-copy` 可跳过）；按文件名匹配当前配置的库路径，所以换机器、换用户名也能恢复；恢复时会清掉目标库旁的 `-wal/-shm/-journal` 残留，恢复完再跑一次 `integrity_check`。
- 非默认库路径的用户给 `backup` 传和 `chat` 一样的 `--*-store` 参数。

## Gmail OAuth 配置

Gmail 连接需要应用管理员先创建 Google OAuth Client。普通用户授权 Gmail 时不会看到或填写 Client Secret。

1. 打开 [Google Cloud Console](https://console.cloud.google.com/) 并创建或选择项目。
2. 在 API Library 启用 **Gmail API**。
3. 在 Google Auth Platform 配置 OAuth consent screen；测试模式下需要把自己的 Google 邮箱加入 Test users。
4. 在 Clients/Credentials 中创建 **Web application** 类型的 OAuth Client。
5. 添加 Authorized redirect URI：

   ```text
   http://127.0.0.1:8000/v1/connections/google/callback
   ```

6. 在仓库根目录创建 `.env`：

   ```dotenv
   GOOGLE_OAUTH_CLIENT_ID=你的_Client_ID
   GOOGLE_OAUTH_CLIENT_SECRET=你的_Client_Secret
   GOOGLE_OAUTH_CALLBACK_URL=http://127.0.0.1:8000/v1/connections/google/callback
   CAREER_AGENT_WEB_URL=http://127.0.0.1:5173
   ```

7. 重启后端，在“邮件追踪”页面点击“连接 Gmail”。该入口只申请 Gmail 只读权限。

`.env` 已被 Git 忽略。不要把 Client Secret 提交到仓库或放入 `web/.env`。

## Google Calendar（可选）

项目内置面试月历不依赖 Google Calendar。不连接 Google 时，项目面试仍会显示在前端月历。

如需把确认后的面试同步到 Google Calendar：

1. 在同一个 Google Cloud 项目中启用 **Google Calendar API**。
2. 复用上面的 OAuth Client 和回调地址。
3. 在“面试日历”页面点击“连接 Google Calendar”。

Calendar 入口只申请日历事件权限。外部日历写入仍经过 Agent 的提案与确认流程。

## QQ 邮箱连接

QQ 邮箱使用 IMAP 授权码，不使用 Google OAuth，也不要填写 QQ 登录密码。

1. 登录 [QQ 邮箱网页版](https://mail.qq.com/)。
2. 打开“设置 → 账号与安全 → 安全设置”。
3. 开启 IMAP/SMTP 第三方客户端服务，并生成授权码。
4. 在应用“邮件追踪”页面点击“连接 QQ 邮箱”。
5. 填写完整 QQ 邮箱地址和刚生成的授权码。

授权码会保存到系统钥匙串；页面和 API 不会返回授权码或邮件正文。
