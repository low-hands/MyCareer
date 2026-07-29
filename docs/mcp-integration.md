# MCP 集成设计与核心概念

本文档沉淀本项目接入 MCP（Model Context Protocol）过程中涉及的核心概念、架构选择与部署演进路径。适合作为新人入门、日后回顾或对外解释设计决策的参考资料。

## 1. MCP 是什么

MCP 是一个**工具/数据暴露协议**，让 LLM 应用能够以统一方式调用外部能力。它定义了三件事：

- **协议格式**：JSON-RPC 2.0（请求/响应的 JSON 信封规范）
- **能力类型**：tools（工具调用）、resources（数据读取）、prompts（提示模板）
- **传输方式**：stdio（子进程管道）、SSE、streamable HTTP

在本项目中，我们只用到 **tools** 和 **stdio 传输**。

### 1.1 JSON-RPC 2.0

MCP 的消息格式就是 JSON-RPC 2.0。一次请求长这样：

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {"name": "search_jobs", "arguments": {"query": "后端"}}
}
```

一次响应长这样：

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {"content": [{"type": "text", "text": "...岗位列表..."}]}
}
```

`id` 字段用来匹配请求和响应——同一时间可以有多个并发请求，靠 `id` 区分谁是谁的回包。

### 1.2 传输方式

| 传输 | 场景 | 进程关系 |
|---|---|---|
| **stdio** | CLI / 单机应用 | 父进程 fork 子进程，通过 stdin/stdout 通信 |
| **SSE / HTTP** | 生产服务 | 独立部署的 server，多个 client 可共享 |

MCP 协议层**不关心传输**——同一段 `tools/call` JSON 可以在 stdio 上跑，也可以在 HTTP 上跑。这是解耦的关键。

## 2. 子进程 + stdio 的物理形态

stdio 模式下，MCP server 是一个被主程序 fork 出来的**子进程**。

```
career-agent (父进程)              boss-mcp (子进程)
       │                               │
       │ ── fork 子进程 ──→            │
       │                               │
       │ ── stdin ──→                  │
       │    {"method":"initialize"}    │
       │                               │
       │ ←── stdout ──                 │
       │    {"result":{...}}           │
       │                               │
       │ ── stdin ──→                  │
       │    {"method":"tools/call"}    │
       │                               │
       │ ←── stdout ──                 │
       │    {"result":{...}}           │
```

**三个关键角色**：

- **子进程** = MCP server 的运行形态（跑在另一个进程里）
- **stdio** = 与它通信的管道（stdin/stdout）
- **JSON-RPC** = 管道里流淌的消息格式

三者缺一不可：没子进程没地方跑、没 stdio 没管道、没 JSON-RPC 双方鸡同鸭讲。

### 2.1 server 是被 client 启动的

一个常见误解："server 应该一直在那儿等"。实际上 stdio 模式下：

- **启动方向**：client 主动 fork 出 server 子进程
- **运行方向**：server 被动等请求、处理、响应

类比：你要和会计谈话，不是会计在你家门口蹲着，而是你打电话让他上线，然后开始一问一答。

## 3. 工具发现与调用的生命周期（`langchain-mcp-adapters` 0.1.0+）

当前版本的 `MultiServerMCPClient` **不再是 context manager**——没有 `async with`，没有长连接的子进程。

### 3.1 工具发现：`await client.get_tools()`

```python
client = MultiServerMCPClient({
    "jobs": {"command": "uv", "args": ["run", "boss-mcp"], "transport": "stdio"},
})
tools = await client.get_tools()   # ← 这一步发现工具
```

`get_tools()` 内部做了五步：

1. **fork 子进程**：`os.exec("uv", "boss-mcp")`
2. **建立 session**：通过 stdio 发 `initialize` 握手
3. **发 `tools/list` 请求**：server 返回它暴露的所有工具（名称、描述、JSON Schema 参数定义）
4. **转换为 LangChain 工具**：每个 MCP tool 被包成 `StructuredTool`，附带一个闭包（记住"怎么连回 server"）
5. **关闭 session、终止子进程**：发现完成后，临时子进程被销毁

### 3.2 工具调用：每次调用 fork 一个新子进程

每个 `StructuredTool` 内部保存了连接配置（`connection`），而不是保存活的 session。所以当模型决定调用某个工具时：

```
模型输出 tool_call: search_jobs(query="后端")
   │
   ▼
StructuredTool.coroutine(**args)       ← 官方包内部
   │
   ├─ create_session(connection)       ← fork 新的 boss-mcp 子进程
   ├─ await session.initialize()       ← 握手
   ├─ await session.call_tool(...)     ← 发 tools/call，等响应
   ├─ 关闭 session、终止子进程          ← 清理
   └─ 返回结果给模型
```

**每次工具调用 = 一次完整的 fork → 握手 → 调用 → 清理**。

### 3.3 这个设计的取舍

| 维度 | 长连接（旧版 `async with`） | 短连接（当前 0.1.0+） |
|---|---|---|
| 性能 | 快（子进程一直活着） | 慢（每次调用 fork） |
| 简单性 | 需要管生命周期 | 不管，用完即毁 |
| 资源泄漏风险 | 有（忘了 `__aexit__`） | 无 |
| 适合场景 | 高频调用 | CLI / 低频调用 |

对 CLI 场景（用户一问一答，间隔几秒），fork 开销可以忽略。生产高频场景应走 HTTP（见第 7 节）。

## 4. client / server 的关系模型

### 4.1 一个 client 可连多个 server

```python
client = MultiServerMCPClient({
    "jobs":    {"command": "uv", "args": ["run", "boss-mcp"]},
    "resume":  {"command": "uv", "args": ["resume-mcp"]},
    "remote":  {"url": "http://...", "transport": "http"},
})
```

- **client 实例 ↔ server 进程**：1 对 N
- **每条 session ↔ 对应 server**：1 对 1（每条 session 是一条独占通道）
- **server 之间互不知道**：jobs-server 不知道 resume-server 的存在

```
┌─ career-agent 进程 ─────────────────────────────┐
│                                                  │
│  MultiServerMCPClient                            │
│     ├── session A ──→ jobs-server (stdio, 独占)  │
│     ├── session B ──→ resume-server (stdio)      │
│     └── session C ──→ remote-server (HTTP)       │
└──────────────────────────────────────────────────┘
```

### 4.2 server 能被多个 client 共享吗？

| Transport | 多个 client 共享一个 server | 原因 |
|---|---|---|
| stdio | ❌ 不行 | stdin/stdout 是点对点管道 |
| HTTP/SSE | ✅ 可以 | 像 web server 一样接多个客户端 |

### 4.3 本项目当前状态

只有一个 server（boss-mcp），dict 里就一项。**只有真需要第二个 server 时再加**，不要预先做多 server 架构。

## 5. 技术选型：为什么用 `langchain-mcp-adapters`

### 5.1 演进历程

1. **自写 226 行适配器**：用官方 `mcp` SDK 直接拼 `stdio_client` + `ClientSession` + JSON Schema → Pydantic 转换 + 错误处理
2. **调研主流项目**：DeerFlow、Open Deep Research、LangAlpha 都在用 `langchain-mcp-adapters`
3. **切换为官方适配器**：226 行 → 57 行，再删 pass-through wrapper → 当前 ~40 行

### 5.2 官方包已经做了的事

- JSON Schema → Pydantic 自动转换
- 工具调用错误转成 `ToolMessage(status="error")`，模型能自纠而不是 run 崩溃
- 多 server 支持（dict 配置）
- stdio + SSE + streamable HTTP 全支持

### 5.3 项目内只保留薄包装

[mcp_client.py](../src/career_agent/mcp_client.py) 当前只有：

- `MCPServerConfig`（类型声明，IDE 补全用）
- `MultiServerMCPClient` 的 re-export（隔离上游包路径变更）
- `load_tools(client)` 一行转发（未来加 trace / 权限预检的入口）

**不允许**再为官方已有的能力加 pass-through wrapper。

## 6. 与项目的集成点

### 6.1 数据流

```
用户输入
   │
   ▼
AgentLoopRuntime.run()
   │
   ├─→ 模型决定调 search_jobs
   │
   ├─→ ToolRegistry.execute("search_jobs", args, permissions={READ})
   │      └─ 权限检查 → 超时控制 → tool.ainvoke(args)
   │
   ├─→ MCPClientTool._arun(**args)  ← 官方包的内部适配器
   │      └─ client.session.call_tool("search_jobs", args)
   │
   ├─→ stdio → boss-mcp 子进程 → 真实招聘 API
   │
   └─→ 工具结果作为 observation 回模型 → 最终输出
```

### 6.2 权限策略

MCP 工具注册时统一标 `ToolPermission.READ`（boss-mcp 只读招聘数据）。未来若有"投递岗位"等写操作，那个工具要标 `EXTERNAL`，并在 `AgentRequest.allowed_permissions` 里显式打开。

## 7. 部署演进：从 CLI 到生产

### 7.1 CLI 模式（当前）

```
uv run career-agent
└─ backend 进程
    └─ fork boss-mcp 子进程（stdio）
```

特点：
- 生命周期耦合（backend 死 = server 死）
- 只能单实例使用
- 部署时顺便装 `uv`

### 7.2 生产模式（未来）

```
前端 React ──HTTP──→ 后端 FastAPI ──HTTP──→ boss-mcp 独立服务 (port 8765)
```

**项目代码只需改一行配置**：

```python
# CLI 模式
{"jobs": {"command": "uv", "args": ["run", "boss-mcp"], "transport": "stdio"}}

# 生产模式
{"jobs": {"url": "http://boss-mcp:8765/mcp", "transport": "http"}}
```

其他代码（`load_tools`、`ToolRegistry`、runtime、评测）**全部不动**。这是 MCP 协议解耦的价值。

### 7.3 如果 boss-mcp 不支持 HTTP

boss-mcp 当前只提供 stdio，生产部署需要二选一：

- **写个小 wrapper**：FastAPI 包 stdio，对外暴露 HTTP。约 100 行。
- **换 server**：用支持 HTTP 的替代（如 `jobspy-mcp-server`）或自己用 FastMCP 写一个。

## 8. 关键反模式与禁令

1. **禁止再为官方包已有的能力写包装**：例如 `connect()` 这种 pass-through async context manager 已被删除
2. **禁止用 `async with MultiServerMCPClient(...)`**：0.1.0+ 已移除 context manager 支持，会直接抛 `NotImplementedError`
3. **禁止预先做多 server 架构**：等真实需求出现再加，YAGNI
4. **禁止在生产部署中让 backend fork server 子进程**：应该走 HTTP 让 server 独立部署

## 9. 相关文件

- 模块实现：[`src/career_agent/mcp_client.py`](../src/career_agent/mcp_client.py)
- CLI 集成：[`src/career_agent/cli.py`](../src/career_agent/cli.py)
- 工具注册：[`src/career_agent/tools.py`](../src/career_agent/tools.py)
- 上游官方文档：<https://github.com/langchain-ai/langchain-mcp-adapters>
- MCP 协议规范：<https://modelcontextprotocol.io>
