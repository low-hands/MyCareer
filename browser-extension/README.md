# Career Agent 岗位收藏扩展

这是一个被动的 Chrome Manifest V3 扩展。它只解析用户已经打开的 BOSS 直聘岗位详情，并在用户点击“保存到 Career Agent”后，把结构化 JD 发送到本机 `127.0.0.1:8000`。

它不会自动搜索、滚动、点击岗位、调用 BOSS 内部接口、投递简历或发送消息，也不会上传整页文本、Cookie 或登录凭证。

## 浏览器前提

**Web 页面和 BOSS 必须在同一个装了本扩展的 Chromium 浏览器里，与系统默认浏览器无关。**

这条前提是隐含在整条链路里的，不满足时不会报错，只会静默失联：

- Main Agent 的 `open_job_search` 只把搜索 URL 交给前端，由前端 `window.open`。网页无法指定用哪个应用打开，所以搜索页**永远开在你打开 Web 页面的那个浏览器里**，不会切到系统默认浏览器，也不会拉起 Chrome。
- 扩展依赖 `chrome.runtime` / `chrome.storage` 和 `service_worker` 形式的 background，因此只在 Chrome、Edge、Brave、Arc 等 Chromium 浏览器中可用。Firefox 的 MV3 不支持这种 background 形式，Safari 需要经 Xcode 转换打包，当前这份代码在两者中都不生效。
- 如果在 Safari 里使用 Web 页面：搜索页照常打开，浏览也一切正常，但右下角**永远不会出现保存卡片**，岗位无法进入本地库。

因此 macOS 默认浏览器保持 Safari 完全没有问题，只需用 Chrome 打开 Career Agent 的 Web 页面即可。

另外，BOSS 的登录状态属于具体浏览器 profile：**必须在装了扩展的那个浏览器里登录 BOSS**，在其他浏览器登录过不算。

## 本地安装

1. 启动 Career Agent 的 FastAPI 和 Web 页面。
2. 在 Chrome 打开 `chrome://extensions`，开启“开发者模式”。
3. 点击“加载已解压的扩展程序”，选择本目录。
4. 在**同一个 Chrome** 里先打开一次 `http://127.0.0.1:5173/`，让扩展同步单独签发的 `capture:write` API key。这一步不可跳过：扩展通过 app bridge 从 Web 页面的 `localStorage` 读取 capture key 并写入 `chrome.storage`，缺少它时保存请求会失败。
5. 在同一个 Chrome 里登录并正常浏览 BOSS 直聘。详情加载完整后，右下角会出现预览；只有点击保存才会写入本地岗位库。

如果 BOSS 出现登录墙或安全验证，请在页面中手动完成。扩展不会尝试绕过验证。

## 已知限制

- API 与 Web 的地址目前写死为 `127.0.0.1:8000` 和 `127.0.0.1:5173`（见 `manifest.json` 与 `background.js`）。改动服务端口时需要同步修改扩展。
- 详情页选择器尚未在真实 BOSS 页面上校准。页面改版导致字段不完整时，卡片只显示未就绪，不会用整页原文兜底入库。
