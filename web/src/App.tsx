import { FormEvent, useEffect, useMemo, useReducer, useRef, useState } from "react";

import { streamChat } from "./api/sse";
import { chatReducer, initialChatState } from "./chat/reducer";
import { InteractionCard } from "./components/InteractionCard";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "/api";

function localId(key: string, prefix: string): string {
  const existing = window.localStorage.getItem(key);
  if (existing) return existing;
  const created = `${prefix}-${crypto.randomUUID()}`;
  window.localStorage.setItem(key, created);
  return created;
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function isAllowedJobSearchUrl(value: string): boolean {
  try {
    const url = new URL(value);
    return url.protocol === "https:" &&
      (url.hostname === "www.zhipin.com" || url.hostname.endsWith(".zhipin.com"));
  } catch {
    return false;
  }
}

export default function App() {
  const [state, dispatch] = useReducer(chatReducer, initialChatState);
  const [draft, setDraft] = useState("");
  const [userId] = useState(() => localId("career-agent:user-id", "user"));
  const [conversationId, setConversationId] = useState(() =>
    localId("career-agent:conversation-id", "conversation"),
  );
  const controller = useRef<AbortController | null>(null);
  const transcript = useRef<HTMLDivElement | null>(null);
  const busy = state.phase === "running";
  const canSubmit = draft.trim().length > 0 && !busy;

  useEffect(() => () => controller.current?.abort(), []);
  useEffect(() => {
    transcript.current?.scrollTo({ top: transcript.current.scrollHeight, behavior: "smooth" });
  }, [state.messages, state.progress, state.interaction]);

  const statusLabel = useMemo(() => {
    if (state.phase === "running") return "处理中";
    if (state.phase === "awaiting_input") return "等待你的回复";
    if (state.phase === "failed") return "本轮失败";
    return "可以开始";
  }, [state.phase]);

  async function sendMessage(rawMessage: string): Promise<void> {
    const message = rawMessage.trim();
    if (!message || busy) return;
    setDraft("");
    const nextController = new AbortController();
    controller.current = nextController;
    dispatch({
      type: "submit",
      messageId: crypto.randomUUID(),
      assistantMessageId: crypto.randomUUID(),
      content: message,
    });
    try {
      for await (const event of streamChat(
        { user_id: userId, conversation_id: conversationId, message },
        { apiBaseUrl: API_BASE_URL, signal: nextController.signal },
      )) {
        if (
          event.type === "client_action" &&
          event.action === "open_url" &&
          isAllowedJobSearchUrl(event.url)
        ) {
          window.open(event.url, "_blank", "noopener,noreferrer");
        }
        dispatch({ type: "stream_event", event });
      }
    } catch (error) {
      if (nextController.signal.aborted) return;
      dispatch({
        type: "transport_failed",
        message: error instanceof Error ? error.message : "连接失败，请稍后重试。",
      });
    } finally {
      if (controller.current === nextController) controller.current = null;
    }
  }

  function submit(event: FormEvent<HTMLFormElement>): void {
    event.preventDefault();
    void sendMessage(draft);
  }

  function newConversation(): void {
    if (busy) return;
    const next = `conversation-${crypto.randomUUID()}`;
    window.localStorage.setItem("career-agent:conversation-id", next);
    setConversationId(next);
    dispatch({ type: "reset" });
    setDraft("");
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark" aria-hidden="true">C</span>
          <div>
            <strong>Career Agent</strong>
            <span>你的职业行动工作台</span>
          </div>
        </div>
        <button className="new-chat" type="button" onClick={newConversation} disabled={busy}>
          新对话
        </button>
      </header>

      <section className="workspace">
        <aside className="context-panel">
          <p className="eyebrow">CURRENT FOCUS</p>
          <h1>把复杂求职任务，变成下一步行动。</h1>
          <p className="context-copy">
            查岗位、看匹配、改简历、跟进投递，或者开始一场模拟面试。
          </p>
          <div className="status-card">
            <span className={`status-dot ${busy ? "is-active" : ""}`} />
            <div>
              <small>Agent 状态</small>
              <strong>{statusLabel}</strong>
            </div>
          </div>
          <div className="suggestions">
            <span>可以这样问</span>
            {["帮我找适合的 AI 产品岗位", "分析我和这份 JD 的匹配度", "开始一场技术模拟面试"].map(
              (suggestion) => (
                <button
                  type="button"
                  key={suggestion}
                  disabled={busy}
                  onClick={() => void sendMessage(suggestion)}
                >
                  {suggestion}
                </button>
              ),
            )}
          </div>
        </aside>

        <section className="chat-panel" aria-label="Career Agent 对话">
          <div className="transcript" ref={transcript} aria-live="polite">
            {state.messages.length === 0 ? (
              <div className="welcome">
                <span className="welcome-index">01</span>
                <h2>今天想推进哪件事？</h2>
                <p>我会在执行过程中告诉你正在做什么，需要选择时会停下来问你。</p>
              </div>
            ) : null}

            {state.messages.map((message) => (
              <article className={`message message-${message.role}`} key={message.id}>
                <span className="message-role">{message.role === "user" ? "你" : "Career Agent"}</span>
                <div className="message-content">
                  {message.content || (message.role === "assistant" && busy ? <span className="typing">● ● ●</span> : null)}
                </div>
              </article>
            ))}

            {state.progress ? (
              <div className="progress-row" role="status">
                <span className="spinner" aria-hidden="true" />
                {state.progress}
              </div>
            ) : null}

            {state.interaction ? (
              <InteractionCard
                interaction={state.interaction}
                disabled={busy}
                onReply={(reply) => void sendMessage(reply)}
              />
            ) : null}

            {state.artifacts.map((artifact) => (
              <div className="artifact-card" key={artifact.artifact_id}>
                <span className="artifact-icon" aria-hidden="true">↗</span>
                <div>
                  <strong>{artifact.filename}</strong>
                  <small>{artifact.media_type} · {formatBytes(artifact.byte_size)}</small>
                </div>
                <span className="artifact-pending">已生成</span>
              </div>
            ))}

            {state.clientActions.map((action) => (
              <a
                className="client-action-card"
                key={`${action.action}:${action.url}`}
                href={action.url}
                target="_blank"
                rel="noreferrer"
              >
                <span aria-hidden="true">↗</span>
                <span>
                  <strong>{action.label}</strong>
                  <small>如果浏览器没有自动打开，请点击这里</small>
                </span>
              </a>
            ))}

            {state.error ? <div className="error-banner" role="alert">{state.error}</div> : null}
          </div>

          <form className="composer" onSubmit={submit}>
            <label htmlFor="message">给 Career Agent 发消息</label>
            <div className="composer-box">
              <textarea
                id="message"
                rows={2}
                value={draft}
                disabled={busy}
                placeholder={state.phase === "awaiting_input" ? "输入你的选择或回答…" : "描述你现在想完成的事情…"}
                onChange={(event) => setDraft(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter" && !event.shiftKey && canSubmit) {
                    event.preventDefault();
                    event.currentTarget.form?.requestSubmit();
                  }
                }}
              />
              <button type="submit" disabled={!canSubmit} aria-label="发送消息">↑</button>
            </div>
            <small>Enter 发送 · Shift + Enter 换行</small>
          </form>
        </section>
      </section>
    </main>
  );
}
