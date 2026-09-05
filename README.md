# My_Career

Career Agent：面向求职流程的本地 Agent 与管理工作区。

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
