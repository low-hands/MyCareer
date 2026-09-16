import { CSSProperties, DragEvent, FormEvent, PointerEvent as ReactPointerEvent, useEffect, useMemo, useReducer, useRef, useState } from "react";

import {
  type ConversationTranscript,
  type ConversationView,
  type JobCapturedEventView,
  acknowledgeJobCapture,
  deleteConversation,
  fetchConversationMessages,
  fetchConversations,
  fetchPendingJobCaptures,
} from "./api/client";
import { seedCaptureApiKey } from "./api/auth";
import { streamChat, type InteractionResponse, type TurnInputResource } from "./api/sse";
import {
  DEFAULT_ATTACHMENT_PROMPT,
  attachmentFromImport,
  droppedResumeFile,
  formatBytes,
  toInputResources,
  toMessageResources,
  withAttachment,
  type ResumeAttachment,
} from "./chat/attachments";
import {
  CAPTURE_POLL_INTERVAL_MS,
  JOB_CAPTURED_MESSAGE,
  continuePendingJobCapture,
  nextCaptureEvents,
  openJobSearchViaBridge,
} from "./chat/jobCapture";
import { chatReducer, initialChatState } from "./chat/reducer";
import {
  RECOVERY_ATTEMPTS,
  RECOVERY_INTERVAL_MS,
  hydrationFrom,
  sleep,
  turnIsStored,
} from "./chat/recovery";
import { InteractionCard } from "./components/InteractionCard";
import { MessageResourceCard } from "./components/MessageResourceCard";
import { ResumeImporter } from "./components/ResumeImporter";
import { MarkdownContent } from "./components/MarkdownContent";
import { AppIcon, type AppIconName } from "./components/AppIcon";
import { DailyBriefPanel } from "./pages/DailyBrief";
import { consumeGoogleOAuthCallback } from "./oauthCallback";
import {
  ApplicationsPanel,
  CalendarPanel,
  DashboardPanel,
  EmailPanel,
  JobsPanel,
  ResearchPanel,
  ResumesPanel,
} from "./pages/WorkspaceViews";

type View =
  | "dashboard"
  | "chat"
  | "jobs"
  | "applications"
  | "email"
  | "brief"
  | "calendar"
  | "resumes"
  | "research";

const VIEW_GROUPS: {
  label: string;
  views: { id: View; label: string; description: string; icon: AppIconName }[];
}[] = [
  {
    label: "工作台",
    views: [
      { id: "dashboard", label: "Dashboard", description: "求职进度总览", icon: "dashboard" },
      { id: "chat", label: "对话", description: "让 Agent 执行任务", icon: "chat" },
      { id: "brief", label: "日报", description: "今天的行动安排", icon: "brief" },
    ],
  },
  {
    label: "求职管理",
    views: [
      { id: "jobs", label: "岗位库", description: "JD 与结构化分析", icon: "search" },
      { id: "applications", label: "投递记录", description: "岗位申请与状态", icon: "applications" },
      { id: "email", label: "邮件追踪", description: "招聘邮件与待确认事件", icon: "mail" },
      { id: "calendar", label: "面试日历", description: "面试安排与同步", icon: "calendar" },
      { id: "resumes", label: "简历管理", description: "简历家族与版本", icon: "document" },
      { id: "research", label: "公司研究", description: "业务和产品资料", icon: "building" },
    ],
  },
];

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "/api";

interface PendingRequest {
  conversationId: string;
  message: string;
  interactionResponse?: InteractionResponse;
  inputResources: TurnInputResource[];
  idempotencyKey: string;
}
const GOOGLE_OAUTH_CALLBACK = consumeGoogleOAuthCallback();

function localId(key: string, prefix: string): string {
  const existing = window.localStorage.getItem(key);
  if (existing) return existing;
  const created = `${prefix}-${crypto.randomUUID()}`;
  window.localStorage.setItem(key, created);
  return created;
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

// Provision the separately scoped extension credential before its bridge's
// first synchronization attempt. The bridge retries, but a deterministic
// initial state avoids making startup timing part of authentication.
seedCaptureApiKey();

export default function App() {
  const [state, dispatch] = useReducer(chatReducer, initialChatState);
  const [draft, setDraft] = useState("");
  const [conversationId, setConversationId] = useState(() =>
    localId("career-agent:conversation-id", "conversation"),
  );
  const [view, setView] = useState<View>(GOOGLE_OAUTH_CALLBACK?.view ?? "chat");
  const [integrationNotice, setIntegrationNotice] = useState(GOOGLE_OAUTH_CALLBACK);
  // Bumped when a turn ends so panels refetch: acting in chat has to show up on
  // the board without the user reloading the page.
  const [completedTurns, setCompletedTurns] = useState(0);
  const [conversations, setConversations] = useState<ConversationView[]>([]);
  const [conversationListError, setConversationListError] = useState<string | null>(null);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [deletingConversationId, setDeletingConversationId] = useState<string | null>(null);
  // Bumped to re-read the current conversation's transcript without changing
  // conversations: the "重新读取" fallback after a recovery that found nothing.
  const [transcriptReloads, setTranscriptReloads] = useState(0);
  const [conversationPanelWidth, setConversationPanelWidth] = useState(() => {
    const saved = Number(window.localStorage.getItem("career-agent:conversation-panel-width"));
    return Number.isFinite(saved) && saved >= 230 && saved <= 460 ? saved : 310;
  });
  // Resume versions picked for the next message. They are already in the
  // library; sending attaches their ids, never their contents.
  const [attachments, setAttachments] = useState<ResumeAttachment[]>([]);
  // A file dropped or picked in chat, waiting for the import form to name it.
  const [pendingUpload, setPendingUpload] = useState<File | null>(null);
  const [importerOpen, setImporterOpen] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const controller = useRef<AbortController | null>(null);
  // The last request as sent, key included. Re-sending it under the same key
  // lets the server answer from its receipt when the turn did run, and run it
  // when it did not; the client never has to know which.
  const lastRequest = useRef<PendingRequest | null>(null);
  const transcript = useRef<HTMLDivElement | null>(null);
  // Jobs saved from the agent's own BOSS search that have not yet re-entered
  // their conversation. They live on the server until acknowledged, so a
  // page that was closed during the save still picks them up here.
  const [captureEvents, setCaptureEvents] = useState<JobCapturedEventView[]>([]);
  const [captureRefreshes, setCaptureRefreshes] = useState(0);
  const captureInFlight = useRef(new Set<string>());
  const captureAttempted = useRef(new Set<string>());
  const captureCommitted = useRef(new Set<string>());
  const captureController = useRef<AbortController | null>(null);
  const activeConversationId = useRef(conversationId);
  const activeChatState = useRef(state);
  activeConversationId.current = conversationId;
  activeChatState.current = state;
  const busy = state.phase === "running" || state.phase === "recovering";
  const canSubmit = (draft.trim().length > 0 || attachments.length > 0) && !busy && !historyLoading;

  useEffect(
    () => () => {
      controller.current?.abort();
      captureController.current?.abort();
    },
    [],
  );
  useEffect(() => {
    const request = new AbortController();
    void fetchConversations({ apiBaseUrl: API_BASE_URL, signal: request.signal })
      .then((items) => {
        setConversations(items);
        setConversationListError(null);
      })
      .catch((cause: unknown) => {
        if (!request.signal.aborted) {
          setConversationListError(cause instanceof Error ? cause.message : "读取历史会话失败。");
        }
      });
    return () => request.abort();
  }, [completedTurns]);
  useEffect(() => {
    const request = new AbortController();
    setHistoryLoading(true);
    void fetchConversationMessages(conversationId, {
      apiBaseUrl: API_BASE_URL,
      signal: request.signal,
    })
      .then((transcript) => {
        dispatch({ type: "hydrate", ...hydrationFrom(conversationId, transcript) });
      })
      .catch((cause: unknown) => {
        if (!request.signal.aborted) {
          dispatch({
            type: "transport_failed",
            message: cause instanceof Error ? cause.message : "恢复历史会话失败。",
          });
        }
      })
      .finally(() => {
        if (!request.signal.aborted) setHistoryLoading(false);
      });
    return () => request.abort();
  }, [conversationId, transcriptReloads]);
  useEffect(() => {
    transcript.current?.scrollTo({ top: transcript.current.scrollHeight, behavior: "smooth" });
  }, [state.messages, state.progress, state.interaction]);
  useEffect(() => {
    const request = new AbortController();
    void fetchPendingJobCaptures({ apiBaseUrl: API_BASE_URL, signal: request.signal })
      .then((events) => {
        if (request.signal.aborted) return;
        captureAttempted.current.clear();
        for (const eventId of captureCommitted.current) {
          if (!events.some((event) => event.id === eventId)) {
            captureCommitted.current.delete(eventId);
          }
        }
        setCaptureEvents(events);
      })
      .catch(() => undefined);
    const timer = window.setInterval(
      () => setCaptureRefreshes((count) => count + 1),
      CAPTURE_POLL_INTERVAL_MS,
    );
    return () => {
      request.abort();
      window.clearInterval(timer);
    };
  }, [captureRefreshes]);
  useEffect(() => {
    // The extension nudges the page right after a save; the event itself is
    // still read from the backend, never taken from the message.
    function onMessage(event: MessageEvent): void {
      if (event.source !== window || event.origin !== window.location.origin) return;
      const data = event.data as { type?: unknown } | null;
      if (data?.type === JOB_CAPTURED_MESSAGE) setCaptureRefreshes((count) => count + 1);
    }
    window.addEventListener("message", onMessage);
    return () => window.removeEventListener("message", onMessage);
  }, []);
  useEffect(() => {
    // One captured job at a time. The continuation checks the target
    // conversation before posting and stays detached from the visible chat.
    if (busy || historyLoading) return;
    const [next] = nextCaptureEvents(
      captureEvents, captureInFlight.current, captureAttempted.current,
    );
    if (!next) return;
    captureInFlight.current.add(next.id);
    captureAttempted.current.add(next.id);
    void continueFromCapture(next);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [captureEvents, conversationId, busy, historyLoading]);

  const statusLabel = useMemo(() => {
    if (state.phase === "running") return "处理中";
    if (state.phase === "recovering") return "正在恢复";
    if (state.phase === "awaiting_input") return "等待你的回复";
    if (state.phase === "failed") return "本轮失败";
    return "可以开始";
  }, [state.phase]);

  async function sendMessage(
    rawMessage: string,
    interactionResponse?: InteractionResponse,
    resources: ResumeAttachment[] = attachments,
  ): Promise<void> {
    // An interaction reply answers the agent's question; it does not carry
    // attachments, and it must not consume the ones queued for the next message.
    const attached = interactionResponse ? [] : resources;
    const message = rawMessage.trim() || (attached.length ? DEFAULT_ATTACHMENT_PROMPT : "");
    if (!message || busy || historyLoading) return;
    setDraft("");
    if (attached.length) setAttachments([]);
    const request: PendingRequest = {
      conversationId,
      message,
      interactionResponse,
      inputResources: toInputResources(attached),
      idempotencyKey: crypto.randomUUID(),
    };
    lastRequest.current = request;
    dispatch({
      type: "submit",
      messageId: crypto.randomUUID(),
      assistantMessageId: crypto.randomUUID(),
      content: message,
      resources: toMessageResources(attached),
    });
    await streamTurn(request);
  }

  async function resendLastRequest(): Promise<void> {
    const request = lastRequest.current;
    if (!request || request.conversationId !== conversationId || busy || historyLoading) return;
    if (state.activeAssistantMessageId) {
      dispatch({ type: "resubmit" });
    } else {
      dispatch({
        type: "submit",
        messageId: crypto.randomUUID(),
        assistantMessageId: crypto.randomUUID(),
        content: request.message,
      });
    }
    await streamTurn(request);
  }

  async function continueFromCapture(event: JobCapturedEventView): Promise<void> {
    const request = new AbortController();
    const chatStateAtStart = activeChatState.current;
    captureController.current = request;
    try {
      if (captureCommitted.current.has(event.id)) {
        await acknowledgeJobCapture(event.id, {
          apiBaseUrl: API_BASE_URL,
          signal: request.signal,
        });
        setCaptureEvents((current) => current.filter((item) => item.id !== event.id));
        captureCommitted.current.delete(event.id);
        return;
      }
      const result = await continuePendingJobCapture(event, {
        apiBaseUrl: API_BASE_URL,
        signal: request.signal,
      });
      if (result.status !== "committed") return;
      captureCommitted.current.add(event.id);
      setCompletedTurns((count) => count + 1);
      if (
        activeConversationId.current !== event.conversation_id
        || activeChatState.current !== chatStateAtStart
      ) return;
      for (const streamEvent of result.events) {
        if (
          streamEvent.type === "client_action"
          && streamEvent.action === "open_url"
          && isAllowedJobSearchUrl(streamEvent.url)
        ) {
          void openJobSearch(
            streamEvent.url,
            streamEvent.capture_intent_id ?? null,
            streamEvent.capture_intent_expires_at ?? null,
          );
        }
      }
      const transcript = await fetchConversationMessages(event.conversation_id, {
        apiBaseUrl: API_BASE_URL,
        signal: request.signal,
      });
      if (
        activeConversationId.current === event.conversation_id
        && activeChatState.current === chatStateAtStart
      ) {
        dispatch({
          type: "hydrate",
          ...hydrationFrom(event.conversation_id, transcript),
        });
      }
    } catch {
      // The durable event remains pending and will be reconsidered after polling.
    } finally {
      if (captureController.current === request) captureController.current = null;
      captureInFlight.current.delete(event.id);
    }
  }

  async function streamTurn(request: PendingRequest): Promise<void> {
    const { message, interactionResponse } = request;
    const nextController = new AbortController();
    controller.current = nextController;
    let turnStarted = false;
    try {
      const body = {
        conversation_id: request.conversationId,
        message,
        interaction_response: interactionResponse,
        ...(request.inputResources.length ? { input_resources: request.inputResources } : {}),
      };
      const options = {
        apiBaseUrl: API_BASE_URL,
        signal: nextController.signal,
        idempotencyKey: request.idempotencyKey,
      };
      for await (const event of streamChat(body, options)) {
        turnStarted = true;
        if (
          event.type === "client_action" &&
          event.action === "open_url" &&
          isAllowedJobSearchUrl(event.url)
        ) {
          void openJobSearch(event.url, event.capture_intent_id ?? null, event.capture_intent_expires_at ?? null);
        }
        if (
          event.type === "turn_completed" ||
          event.type === "turn_suspended" ||
          event.type === "turn_failed"
        ) {
          setCompletedTurns((count) => count + 1);
        }
        dispatch({ type: "stream_event", event });
      }
    } catch (error) {
      if (nextController.signal.aborted) return;
      const reason = error instanceof Error ? error.message : "连接失败，请稍后重试。";
      if (!turnStarted) {
        // Rejected or unreachable before any event: nothing ran server-side.
        dispatch({ type: "transport_failed", message: reason });
        return;
      }
      dispatch({ type: "transport_lost" });
      await recoverTurn(message, reason, nextController.signal);
    } finally {
      if (controller.current === nextController) controller.current = null;
    }
  }

  /**
   * The stream died after the turn began. The server finishes the turn on its
   * own and stores the reply, so read the transcript back until the reply for
   * `sentMessage` shows up, then show that instead of a failure. A turn can
   * legitimately outlast the polling window (long tool runs), so giving up
   * says "unconfirmed", not "failed", and leaves a manual re-read.
   */
  async function recoverTurn(sentMessage: string, reason: string, signal: AbortSignal): Promise<void> {
    for (let attempt = 0; attempt < RECOVERY_ATTEMPTS; attempt += 1) {
      if (attempt > 0) await sleep(RECOVERY_INTERVAL_MS, signal);
      if (signal.aborted) return;
      let transcript: ConversationTranscript;
      try {
        transcript = await fetchConversationMessages(conversationId, {
          apiBaseUrl: API_BASE_URL,
          signal,
        });
      } catch {
        continue;
      }
      if (!turnIsStored(transcript, sentMessage)) continue;
      dispatch({ type: "hydrate", ...hydrationFrom(conversationId, transcript) });
      setCompletedTurns((count) => count + 1);
      return;
    }
    if (signal.aborted) return;
    dispatch({
      type: "transport_failed",
      message: `${reason} 连接中断后没有读到这一轮的回复；服务器可能仍在处理，稍后可点“重新读取”，或“重新发送”同一请求。`,
    });
  }

  /**
   * Prefer the extension so the capture intent is bound to the BOSS tab; the
   * URL is the same either way and never carries the intent. Without the
   * extension the page opens BOSS itself and a save becomes a plain library save.
   */
  async function openJobSearch(
    url: string,
    captureIntentId: string | null,
    captureIntentExpiresAt: string | null,
  ): Promise<void> {
    const opened = await openJobSearchViaBridge({ url, captureIntentId, captureIntentExpiresAt });
    if (!opened) window.open(url, "_blank", "noopener,noreferrer");
  }

  function submit(event: FormEvent<HTMLFormElement>): void {
    event.preventDefault();
    void sendMessage(draft);
  }

  function attachResume(attachment: ResumeAttachment): void {
    setAttachments((current) => withAttachment(current, attachment));
  }

  function openImporter(file: File | null): void {
    if (busy) return;
    setPendingUpload(file);
    setImporterOpen(true);
  }

  function closeImporter(): void {
    setImporterOpen(false);
    setPendingUpload(null);
  }

  function handleDrop(event: DragEvent<HTMLElement>): void {
    event.preventDefault();
    setDragActive(false);
    const file = droppedResumeFile(event.dataTransfer.files);
    if (file) openImporter(file);
  }

  function handleDragOver(event: DragEvent<HTMLElement>): void {
    if (!Array.from(event.dataTransfer.types).includes("Files")) return;
    event.preventDefault();
    if (!busy) setDragActive(true);
  }

  function newConversation(): void {
    if (busy) return;
    const next = `conversation-${crypto.randomUUID()}`;
    window.localStorage.setItem("career-agent:conversation-id", next);
    setConversationId(next);
    dispatch({ type: "reset" });
    setDraft("");
    setAttachments([]);
    setView("chat");
  }

  function startAgentTask(prompt: string, resource?: ResumeAttachment): void {
    setView("chat");
    const resources = resource ? withAttachment(attachments, resource) : attachments;
    if (busy || historyLoading) {
      setDraft(prompt);
      if (resource) setAttachments(resources);
      return;
    }
    void sendMessage(prompt, undefined, resources);
  }

  function openConversation(nextConversationId: string): void {
    if (busy) return;
    setView("chat");
    if (nextConversationId === conversationId) return;
    window.localStorage.setItem("career-agent:conversation-id", nextConversationId);
    setConversationId(nextConversationId);
  }

  async function removeConversation(item: ConversationView): Promise<void> {
    if (busy || deletingConversationId) return;
    if (!window.confirm(`确定删除会话“${item.title}”的聊天记录吗？执行审计记录会继续保留，此操作无法撤销。`)) return;
    setDeletingConversationId(item.id);
    try {
      await deleteConversation(item.id, { apiBaseUrl: API_BASE_URL });
      setConversations((current) => current.filter((entry) => entry.id !== item.id));
      setConversationListError(null);
      if (item.id === conversationId) {
        const next = `conversation-${crypto.randomUUID()}`;
        window.localStorage.setItem("career-agent:conversation-id", next);
        setConversationId(next);
        dispatch({ type: "reset" });
        setDraft("");
        setAttachments([]);
      }
    } catch (cause) {
      setConversationListError(cause instanceof Error ? cause.message : "删除会话失败。");
    } finally {
      setDeletingConversationId(null);
    }
  }

  function beginConversationPanelResize(event: ReactPointerEvent<HTMLButtonElement>): void {
    if (event.button !== 0) return;
    event.preventDefault();
    const startX = event.clientX;
    const startWidth = conversationPanelWidth;
    let latestWidth = startWidth;
    document.body.classList.add("is-resizing-panel");
    const move = (nextEvent: PointerEvent) => {
      latestWidth = Math.min(460, Math.max(230, startWidth + nextEvent.clientX - startX));
      setConversationPanelWidth(latestWidth);
    };
    const stop = () => {
      document.body.classList.remove("is-resizing-panel");
      window.localStorage.setItem("career-agent:conversation-panel-width", String(latestWidth));
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", stop);
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", stop, { once: true });
  }

  function resizeConversationPanelBy(delta: number): void {
    setConversationPanelWidth((current) => {
      const next = Math.min(460, Math.max(230, current + delta));
      window.localStorage.setItem("career-agent:conversation-panel-width", String(next));
      return next;
    });
  }

  return (
    <main className="app-shell">
      {integrationNotice ? (
        <div
          className={`oauth-callback-banner is-${integrationNotice.status}`}
          role={integrationNotice.status === "connected" ? "status" : "alert"}
        >
          <span>{integrationNotice.message}</span>
          <button type="button" onClick={() => setIntegrationNotice(null)} aria-label="关闭提示">×</button>
        </div>
      ) : null}
      <aside className="app-sidebar">
        <div className="brand">
          <span className="brand-mark"><AppIcon name="sparkles" size={23} /></span>
          <div>
            <strong>Career Agent</strong>
            <span>你的职业行动工作台</span>
          </div>
        </div>
        <nav className="view-nav" aria-label="视图">
          {VIEW_GROUPS.map((group) => (
            <div className="nav-group" key={group.label}>
              <span className="nav-group-label">{group.label}</span>
              {group.views.map((entry) => (
                <button
                  type="button"
                  key={entry.id}
                  className={view === entry.id ? "is-active" : ""}
                  aria-current={view === entry.id ? "page" : undefined}
                  onClick={() => setView(entry.id)}
                >
                  <span className="view-icon"><AppIcon name={entry.icon} size={20} /></span>
                  <span>
                    <strong>{entry.label}</strong>
                    <small>{entry.description}</small>
                  </span>
                </button>
              ))}
            </div>
          ))}
          <div className="conversation-history mobile-conversation-history">
            <span className="nav-group-label">最近对话</span>
            {conversationListError ? <small className="history-error">暂时无法读取</small> : null}
            {conversations.length > 0 ? conversations.map((item) => (
              <div className="mobile-conversation-item-row" key={item.id}>
                <button
                  type="button"
                  className={`conversation-item ${item.id === conversationId ? "is-current" : ""}`}
                  disabled={busy || deletingConversationId === item.id}
                  onClick={() => openConversation(item.id)}
                  title={item.title}
                >
                  <span className="conversation-dot" />
                  <span>
                    <strong>{item.title}</strong>
                    <small>{new Date(item.last_active_at).toLocaleDateString("zh-CN", { month: "numeric", day: "numeric" })} · {Math.ceil(item.message_count / 2)} 轮</small>
                  </span>
                </button>
                <button
                  type="button"
                  className="mobile-conversation-delete"
                  disabled={busy || deletingConversationId !== null}
                  onClick={() => void removeConversation(item)}
                  aria-label={`删除会话：${item.title}`}
                >
                  <AppIcon name="trash" size={14} />
                </button>
              </div>
            )) : <small className="history-empty">完成第一轮对话后会出现在这里</small>}
          </div>
        </nav>
        <div className="sidebar-footer">
          <div className="system-state">
            <span className={`status-dot ${busy ? "is-active" : ""}`} />
            <span><strong>{statusLabel}</strong><small>Agent 状态</small></span>
          </div>
          <button className="new-chat" type="button" onClick={newConversation} disabled={busy}>
            <AppIcon name="plus" size={18} />
            新对话
          </button>
          <small>所有实际改动仍会在对话中确认</small>
        </div>
      </aside>

      <div className="app-content">
        <DashboardPanel
          apiBaseUrl={API_BASE_URL}
          refreshToken={completedTurns}
          hidden={view !== "dashboard"}
          onAskAgent={startAgentTask}
        />
        <ApplicationsPanel
          apiBaseUrl={API_BASE_URL}
          refreshToken={completedTurns}
          hidden={view !== "applications"}
          onAskAgent={startAgentTask}
          onOpenConversation={openConversation}
        />
        <JobsPanel
          apiBaseUrl={API_BASE_URL}
          refreshToken={completedTurns}
          hidden={view !== "jobs"}
          onAskAgent={startAgentTask}
        />
        <DailyBriefPanel
          apiBaseUrl={API_BASE_URL}
          refreshToken={completedTurns}
          hidden={view !== "brief"}
        />
        <CalendarPanel
          apiBaseUrl={API_BASE_URL}
          refreshToken={completedTurns}
          hidden={view !== "calendar"}
          onAskAgent={startAgentTask}
        />
        <EmailPanel
          apiBaseUrl={API_BASE_URL}
          refreshToken={completedTurns}
          hidden={view !== "email"}
          onAskAgent={startAgentTask}
        />
        <ResumesPanel
          apiBaseUrl={API_BASE_URL}
          refreshToken={completedTurns}
          hidden={view !== "resumes"}
          onAskAgent={startAgentTask}
        />
        <ResearchPanel
          apiBaseUrl={API_BASE_URL}
          refreshToken={completedTurns}
          hidden={view !== "research"}
          onAskAgent={startAgentTask}
        />

        <section
          className="workspace"
          hidden={view !== "chat"}
          style={{ "--conversation-panel-width": `${conversationPanelWidth}px` } as CSSProperties}
        >
          <aside className="context-panel conversation-panel" aria-label="历史对话">
            <header className="conversation-panel-header">
              <div>
                <p className="eyebrow">CONVERSATIONS</p>
                <h1>历史对话</h1>
                <p>选择一段对话继续推进</p>
              </div>
              <button type="button" onClick={newConversation} disabled={busy} aria-label="新建对话">
                <AppIcon name="plus" size={18} />
              </button>
            </header>
            <div className="conversation-panel-list">
              {!conversations.some((item) => item.id === conversationId) ? (
                <button type="button" className="conversation-panel-item is-current" disabled>
                  <span className="conversation-avatar"><AppIcon name="chat" size={17} /></span>
                  <span><strong>新对话</strong><small>尚未发送第一条消息</small></span>
                  <span className="conversation-current-mark" />
                </button>
              ) : null}
              {conversationListError ? <div className="conversation-panel-error">暂时无法读取历史对话</div> : null}
              {conversations.map((item) => (
                <div className="conversation-panel-row" key={item.id}>
                  <button
                    type="button"
                    className={`conversation-panel-item ${item.id === conversationId ? "is-current" : ""}`}
                    disabled={busy || deletingConversationId === item.id}
                    onClick={() => openConversation(item.id)}
                  >
                    <span className="conversation-avatar"><AppIcon name="chat" size={17} /></span>
                    <span>
                      <strong>{item.title}</strong>
                      <small>{item.last_message_preview}</small>
                      <time>{new Date(item.last_active_at).toLocaleDateString("zh-CN", { month: "numeric", day: "numeric" })} · {Math.ceil(item.message_count / 2)} 轮</time>
                    </span>
                    {item.id === conversationId ? <span className="conversation-current-mark" /> : null}
                  </button>
                  <button
                    type="button"
                    className="conversation-delete"
                    disabled={busy || deletingConversationId !== null}
                    onClick={() => void removeConversation(item)}
                    aria-label={`删除会话：${item.title}`}
                    title="删除会话"
                  >
                    <AppIcon name="trash" size={15} />
                  </button>
                </div>
              ))}
            </div>
            <div className="conversation-panel-status">
              <span className={`status-dot ${busy ? "is-active" : ""}`} />
              <span><strong>{statusLabel}</strong><small>Agent 状态</small></span>
            </div>
            <button
              type="button"
              className="conversation-resize-handle"
              aria-label="拖动调整会话栏宽度"
              onPointerDown={beginConversationPanelResize}
              onKeyDown={(event) => {
                if (event.key === "ArrowLeft") resizeConversationPanelBy(-20);
                if (event.key === "ArrowRight") resizeConversationPanelBy(20);
              }}
            ><span /></button>
          </aside>

          <section className="chat-panel" aria-label="Career Agent 对话">
          <div className="transcript" ref={transcript} aria-live="polite">
            {state.messages.length === 0 && !historyLoading ? (
              <div className="welcome">
                <div className="welcome-visual">
                  <span className="orbit orbit-one" />
                  <span className="orbit orbit-two" />
                  <span className="welcome-core"><AppIcon name="sparkles" size={34} /></span>
                </div>
                <h2>今天想推进哪件事？</h2>
                <p>描述你的目标，我会拆解任务、执行工具，并在需要你决定时停下来。</p>
                <div className="capability-chips">
                  <span><AppIcon name="document" size={14} /> 简历</span>
                  <span><AppIcon name="search" size={14} /> 岗位</span>
                  <span><AppIcon name="calendar" size={14} /> 面试</span>
                </div>
              </div>
            ) : null}

            {historyLoading ? <div className="history-loading"><span className="spinner" /> 正在恢复对话…</div> : null}

            {state.messages.filter((message) =>
              Boolean(
                message.content
                || message.resources?.length
                || (message.role === "assistant" && busy),
              )
            ).map((message) => (
              <article className={`message message-${message.role}`} key={message.id}>
                <span className="message-role">{message.role === "user" ? "你" : "Career Agent"}</span>
                <div className="message-content">
                  {message.content ? (
                    message.role === "assistant" ? (
                      <MarkdownContent content={message.content} className="markdown-content" />
                    ) : (
                      message.content
                    )
                  ) : message.role === "assistant" && busy ? (
                    <span className="typing">● ● ●</span>
                  ) : null}
                </div>
                {message.resources?.map((resource) => (
                  <MessageResourceCard
                    key={`${resource.kind}-${resource.resourceId}`}
                    resource={resource}
                    apiBaseUrl={API_BASE_URL}
                  />
                ))}
              </article>
            ))}

            {state.progress ? (
              <div className="progress-row" role="status">
                <span className="spinner" aria-hidden="true" />
                {state.progress}
              </div>
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

            {state.error ? (
              <div className="error-banner" role="alert">
                {state.error}
                {state.phase === "failed" ? (
                  <button
                    type="button"
                    className="error-banner-action"
                    onClick={() => setTranscriptReloads((count) => count + 1)}
                    disabled={historyLoading}
                  >
                    重新读取
                  </button>
                ) : null}
                {state.phase === "failed" && lastRequest.current?.conversationId === conversationId ? (
                  <button
                    type="button"
                    className="error-banner-action"
                    onClick={() => void resendLastRequest()}
                    disabled={historyLoading}
                  >
                    重新发送
                  </button>
                ) : null}
              </div>
            ) : null}
          </div>

          <div
            className={`composer ${state.interaction ? "has-interaction" : ""} ${dragActive ? "is-drop-target" : ""}`}
            onDragOver={handleDragOver}
            onDragLeave={() => setDragActive(false)}
            onDrop={handleDrop}
          >
            {importerOpen ? (
              <div className="composer-importer" role="dialog" aria-label="导入简历并附到消息">
                <div className="composer-importer-header">
                  <strong>导入简历并附到消息</strong>
                  <small>与简历库使用同一导入服务；原文件只保存一份在简历库。</small>
                </div>
                <ResumeImporter
                  key={pendingUpload ? `${pendingUpload.name}-${pendingUpload.size}-${pendingUpload.lastModified}` : "picker"}
                  apiBaseUrl={API_BASE_URL}
                  initialFile={pendingUpload}
                  submitLabel="导入并附到消息"
                  onCancel={closeImporter}
                  onImported={(result) => {
                    attachResume(attachmentFromImport(result));
                    setCompletedTurns((count) => count + 1);
                    closeImporter();
                  }}
                />
              </div>
            ) : null}
            {attachments.length > 0 ? (
              <div className="composer-attachments" aria-label="将随消息发送的简历版本">
                {attachments.map((item) => (
                  <span className="attachment-chip" key={item.resumeVersionId}>
                    <AppIcon name="document" size={14} />
                    <span>
                      <strong>{item.name} v{item.versionNumber}</strong>
                      <small>{item.documentFormat.toUpperCase()} · {formatBytes(item.byteSize)}</small>
                    </span>
                    <button
                      type="button"
                      aria-label={`移除附件：${item.name} v${item.versionNumber}`}
                      disabled={busy}
                      onClick={() => setAttachments((current) => current.filter((entry) => entry.resumeVersionId !== item.resumeVersionId))}
                    >
                      ×
                    </button>
                  </span>
                ))}
                <small className="attachment-privacy">发送后，提取出的简历内容会发给配置的模型供应商。</small>
              </div>
            ) : null}
            {state.interaction ? (
              <InteractionCard
                interaction={state.interaction}
                disabled={busy}
                apiBaseUrl={API_BASE_URL}
                onReply={(reply) =>
                  void sendMessage(reply.message, reply.interactionResponse)
                }
              />
            ) : null}
            <form className="composer-entry" onSubmit={submit}>
              <label htmlFor="message">给 Career Agent 发消息</label>
              <div className="composer-box">
                <button
                  type="button"
                  className="composer-attach"
                  aria-label="附上简历"
                  title="选择或拖入简历文件（PDF、TXT、Markdown）"
                  disabled={busy}
                  onClick={() => openImporter(null)}
                >
                  <AppIcon name="plus" size={17} />
                </button>
                <textarea
                  id="message"
                  rows={2}
                  value={draft}
                  disabled={busy}
                  placeholder={
                    state.phase === "awaiting_input"
                      ? "输入你的选择或回答…"
                      : attachments.length
                        ? `不填则发送“${DEFAULT_ATTACHMENT_PROMPT}”`
                        : "描述你现在想完成的事情，或拖入一份简历…"
                  }
                  onChange={(event) => setDraft(event.target.value)}
                  onKeyDown={(event) => {
                    if (event.key === "Enter" && !event.shiftKey && canSubmit) {
                      event.preventDefault();
                      event.currentTarget.form?.requestSubmit();
                    }
                  }}
                />
                <button type="submit" disabled={!canSubmit} aria-label="发送消息"><AppIcon name="arrow-up" size={19} /></button>
              </div>
              <small>Enter 发送 · Shift + Enter 换行 · 拖入简历可直接导入并附到消息</small>
            </form>
          </div>
          </section>
        </section>
      </div>
    </main>
  );
}
