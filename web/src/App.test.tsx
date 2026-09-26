// @vitest-environment jsdom
import { act, StrictMode } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import App from "./App";
import {
  deleteConversation, fetchConversationMessages, fetchConversations, fetchPendingJobCaptures,
  type ConversationTranscript, type ResumeImportResult,
} from "./api/client";
import { ChatStreamHttpError, followLiveTurn, streamChat } from "./api/sse";
import type { ChatAttachment } from "./chat/attachments";
import type { PublicStreamEvent } from "./chat/events";

vi.mock("./api/client", async (importOriginal) => ({
  ...await importOriginal<typeof import("./api/client")>(),
  deleteConversation: vi.fn(),
  fetchConversationMessages: vi.fn(),
  fetchConversations: vi.fn(),
  fetchPendingJobCaptures: vi.fn(),
}));
vi.mock("./api/sse", async (importOriginal) => ({
  ...await importOriginal<typeof import("./api/sse")>(),
  followLiveTurn: vi.fn(),
  streamChat: vi.fn(),
}));
vi.mock("./pages/DailyBrief", () => ({ DailyBriefPanel: () => null }));
vi.mock("./pages/WorkspaceViews", () => ({
  DashboardPanel: () => null,
  ApplicationsPanel: () => null,
  CalendarPanel: () => null,
  EmailPanel: () => null,
  JobsPanel: () => null,
  ResearchPanel: (props: { onAskAgent: (prompt: string) => void }) => (
    <button data-testid="ask-research" onClick={() => props.onAskAgent("让 Agent 研究这家公司")}>
      让 Agent 研究公司
    </button>
  ),
  ResumesPanel: (props: {
    onStartStandaloneTask: (prompt: string, resource: ChatAttachment) => void;
  }) => (
    <div>
      <button data-testid="analyze" onClick={() => props.onStartStandaloneTask(
        "分析简历 Library v1",
        {
          resumeId: "library", resumeVersionId: "library-v1", name: "Library",
          versionNumber: 1, documentFormat: "pdf", byteSize: 10, uploadedAt: null,
        },
      )}>分析所选版本</button>
      <button data-testid="analyze-other" onClick={() => props.onStartStandaloneTask(
        "分析简历 Library v2",
        {
          resumeId: "library", resumeVersionId: "library-v2", name: "Library",
          versionNumber: 2, documentFormat: "pdf", byteSize: 10, uploadedAt: null,
        },
      )}>分析另一个版本</button>
    </div>
  ),
}));
vi.mock("./components/ResumeImporter", () => ({
  ResumeImporter: (props: { onImported: (result: ResumeImportResult) => void }) => (
    <button data-testid="import" onClick={() => props.onImported({
      resume_id: "queued", resume_version_id: "queued-v2", name: "Queued",
      version_number: 2, document_format: "pdf", byte_size: 20,
    })}>附上待发送简历</button>
  ),
}));

function transcript(old = false): ConversationTranscript {
  return {
    messages: old ? [{
      role: "assistant", content: "旧岗位上下文与面试问题", resources: [],
      created_at: "2026-09-01T00:00:00Z",
    }] : [],
    active_workflow: old ? "mock_interview" : null,
    phase: old ? "answering" : null,
    pending_interaction: null,
    pending_interaction_body: null,
  };
}

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((complete) => { resolve = complete; });
  return { promise, resolve };
}

let root: Root;
let container: HTMLDivElement;

function element<T extends Element>(selector: string): T {
  const found = container.querySelector<T>(selector);
  if (!found) throw new Error(`Missing element: ${selector}`);
  return found;
}

async function click(selector: string): Promise<void> {
  await act(async () => element<HTMLButtonElement>(selector).click());
}

async function typeDraft(value: string): Promise<void> {
  await act(async () => {
    const input = element<HTMLTextAreaElement>("#message");
    Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value")?.set?.call(input, value);
    input.dispatchEvent(new Event("input", { bubbles: true }));
  });
}

async function mount(): Promise<void> {
  await act(async () => root.render(<StrictMode><App /></StrictMode>));
}

beforeEach(() => {
  vi.clearAllMocks();
  vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
  HTMLElement.prototype.scrollTo = vi.fn();
  window.localStorage.clear();
  window.localStorage.setItem("career-agent:conversation-id", "old");
  vi.mocked(fetchPendingJobCaptures).mockResolvedValue([]);
  vi.mocked(fetchConversations).mockResolvedValue([{
    id: "old", status: "active", title: "旧对话", last_message_preview: "旧岗位",
    message_count: 2, active_workflow: "mock_interview", phase: "answering",
    created_at: "2026-09-01T00:00:00Z", last_active_at: "2026-09-01T00:00:00Z",
  }]);
  vi.mocked(fetchConversationMessages).mockImplementation(async (id) => transcript(id === "old"));
  vi.mocked(streamChat).mockImplementation(async function* (): AsyncGenerator<PublicStreamEvent> {
    yield { type: "turn_started", turn_id: "test-turn" };
    yield { type: "turn_completed", turn_id: "test-turn" };
  });
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
});

afterEach(async () => {
  await act(async () => root.unmount());
  container.remove();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe("resume-library conversation isolation", () => {
  it("restores the old draft, queued version and interview when returning from analysis", async () => {
    await mount();
    await typeDraft("尚未发送的旧草稿");
    await click('[aria-label="附上简历"]');
    await click('[data-testid="import"]');
    await click('[data-testid="analyze"]');

    expect(streamChat).toHaveBeenCalledTimes(1);
    const [request] = vi.mocked(streamChat).mock.calls[0];
    expect(request.conversation_id).not.toBe("old");
    expect(request.input_resources).toEqual([{ kind: "resume_version", id: "library-v1" }]);
    expect(request.message).toContain("Library v1");
    expect(container.textContent).not.toContain("旧岗位上下文与面试问题");
    expect(element<HTMLTextAreaElement>("#message").value).toBe("");
    expect(container.querySelector(".composer-attachments")).toBeNull();

    await click(".conversation-panel-row .conversation-panel-item");
    expect(element<HTMLTextAreaElement>("#message").value).toBe("尚未发送的旧草稿");
    expect(element(".composer-attachments").textContent).toContain("Queued v2");
    expect(container.textContent).toContain("旧岗位上下文与面试问题");
    expect(streamChat).toHaveBeenCalledTimes(1);
  });

  it("creates one task for batched double clicks and waits for the new transcript", async () => {
    const hydration = deferred<ConversationTranscript>();
    vi.mocked(fetchConversationMessages).mockImplementation(async (id) =>
      id === "old" ? transcript(true) : hydration.promise);
    await mount();
    const ids = vi.spyOn(crypto, "randomUUID");
    await act(async () => {
      element<HTMLButtonElement>('[data-testid="analyze"]').click();
      element<HTMLButtonElement>('[data-testid="analyze"]').click();
    });
    expect(ids).toHaveBeenCalledTimes(1);
    expect(streamChat).not.toHaveBeenCalled();
    await act(async () => hydration.resolve(transcript()));
    await act(async () => root.render(<StrictMode><App /></StrictMode>));
    expect(streamChat).toHaveBeenCalledTimes(1);
  });

  it("starts library analysis at once and leaves the running turn to finish in the background", async () => {
    const completion = deferred<void>();
    vi.mocked(streamChat).mockImplementationOnce(async function* (): AsyncGenerator<PublicStreamEvent> {
      yield { type: "turn_started", turn_id: "old-turn" };
      await completion.promise;
      yield { type: "turn_completed", turn_id: "old-turn" };
    });
    await mount();
    await typeDraft("回答当前问题");
    await click('[aria-label="发送消息"]');
    await click('[data-testid="analyze"]');

    // Only this page's stream of the old turn is let go; the server runs it on.
    expect(vi.mocked(streamChat).mock.calls[0][1]?.signal?.aborted).toBe(true);
    expect(streamChat).toHaveBeenCalledTimes(2);
    expect(vi.mocked(streamChat).mock.calls[0][0].conversation_id).toBe("old");
    expect(vi.mocked(streamChat).mock.calls[1][0].conversation_id).not.toBe("old");
    expect(vi.mocked(streamChat).mock.calls[1][0].input_resources).toEqual([
      { kind: "resume_version", id: "library-v1" },
    ]);
    await act(async () => completion.resolve());
  });

  it("does not send an analysis cancelled before it started, and can start it again", async () => {
    const history = deferred<ConversationTranscript>();
    vi.mocked(fetchConversationMessages).mockImplementation(async (id) =>
      id === "old" ? transcript(true) : history.promise);
    await mount();
    await click('[data-testid="analyze"]');
    // The new conversation is still loading, so the task waits and can be cancelled.
    await click(".standalone-task-notice button");
    expect(container.querySelector(".standalone-task-notice")).toBeNull();
    await act(async () => history.resolve(transcript()));
    expect(streamChat).not.toHaveBeenCalled();

    await click('[data-testid="analyze"]');
    expect(streamChat).toHaveBeenCalledTimes(1);
    expect(vi.mocked(streamChat).mock.calls[0][0].input_resources).toEqual([
      { kind: "resume_version", id: "library-v1" },
    ]);
  });

  it("ignores a repeated click after the same analysis has started streaming", async () => {
    const completion = deferred<void>();
    vi.mocked(streamChat).mockImplementationOnce(async function* (): AsyncGenerator<PublicStreamEvent> {
      yield { type: "turn_started", turn_id: "analysis-turn" };
      await completion.promise;
      yield { type: "turn_completed", turn_id: "analysis-turn" };
    });
    await mount();
    await click('[data-testid="analyze"]');
    const ids = vi.spyOn(crypto, "randomUUID");
    await click('[data-testid="analyze"]');
    expect(ids).not.toHaveBeenCalled();
    await act(async () => completion.resolve());
    expect(streamChat).toHaveBeenCalledTimes(1);
  });

  it("starts another version at once while the first analysis keeps running", async () => {
    const completion = deferred<void>();
    vi.mocked(streamChat).mockImplementationOnce(async function* (): AsyncGenerator<PublicStreamEvent> {
      yield { type: "turn_started", turn_id: "first-analysis" };
      await completion.promise;
      yield { type: "turn_completed", turn_id: "first-analysis" };
    });
    await mount();
    await click('[data-testid="analyze"]');
    const firstConversation = window.localStorage.getItem("career-agent:conversation-id");
    await click('[data-testid="analyze-other"]');
    expect(streamChat).toHaveBeenCalledTimes(2);
    const [next] = vi.mocked(streamChat).mock.calls[1];
    expect(next.conversation_id).not.toBe(firstConversation);
    expect(next.input_resources).toEqual([{ kind: "resume_version", id: "library-v2" }]);
    await act(async () => completion.resolve());
  });

  it("lets the reader open another conversation while a turn runs, and marks the running one", async () => {
    vi.mocked(fetchConversations).mockResolvedValue([
      {
        id: "old", status: "active", title: "旧对话", last_message_preview: "旧岗位",
        message_count: 2, active_workflow: null, phase: null,
        created_at: "2026-09-01T00:00:00Z", last_active_at: "2026-09-02T00:00:00Z",
      },
      {
        id: "other", status: "active", title: "另一个对话", last_message_preview: "…",
        message_count: 2, active_workflow: null, phase: null,
        created_at: "2026-09-01T00:00:00Z", last_active_at: "2026-09-01T00:00:00Z",
      },
    ]);
    const completion = deferred<void>();
    vi.mocked(streamChat).mockImplementationOnce(async function* (): AsyncGenerator<PublicStreamEvent> {
      yield { type: "turn_started", turn_id: "old-turn" };
      await completion.promise;
      yield { type: "turn_completed", turn_id: "old-turn" };
    });
    await mount();
    await typeDraft("帮我研究公司");
    await click('[aria-label="发送消息"]');

    const other = [...container.querySelectorAll<HTMLButtonElement>(".conversation-panel-item")]
      .find((item) => item.textContent?.includes("另一个对话"))!;
    expect(other.disabled).toBe(false);
    await act(async () => other.click());

    expect(window.localStorage.getItem("career-agent:conversation-id")).toBe("other");
    expect(vi.mocked(streamChat).mock.calls[0][1]?.signal?.aborted).toBe(true);
    const oldRow = [...container.querySelectorAll(".conversation-panel-row")]
      .find((row) => row.textContent?.includes("旧对话"))!;
    expect(oldRow.textContent).toContain("运行中");
    expect(oldRow.querySelector<HTMLButtonElement>(".conversation-delete")!.disabled).toBe(true);
    const otherRow = [...container.querySelectorAll(".conversation-panel-row")]
      .find((row) => row.textContent?.includes("另一个对话"))!;
    expect(otherRow.querySelector<HTMLButtonElement>(".conversation-delete")!.disabled).toBe(false);
    await act(async () => completion.resolve());
  });

  it("asks in the row before deleting a conversation, and Escape keeps it", async () => {
    vi.mocked(deleteConversation).mockResolvedValue({ conversation_id: "old" } as never);
    await mount();
    const row = () => [...container.querySelectorAll(".conversation-panel-row")]
      .find((item) => item.textContent?.includes("旧对话") || item.querySelector(".inline-delete-confirm"));
    await act(async () => row()!.querySelector<HTMLButtonElement>(".conversation-delete")!.click());
    const confirm = container.querySelector(".conversation-panel-row .inline-delete-confirm")!;
    expect(confirm.textContent).toContain("删除这段聊天？");
    expect(confirm.textContent).toContain("记忆都会保留");
    expect(document.activeElement?.textContent).toBe("取消");
    expect(deleteConversation).not.toHaveBeenCalled();

    await act(async () => confirm.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape", bubbles: true })));
    expect(container.querySelector(".conversation-panel-row .inline-delete-confirm")).toBeNull();
    expect(deleteConversation).not.toHaveBeenCalled();

    await act(async () => row()!.querySelector<HTMLButtonElement>(".conversation-delete")!.click());
    const remove = [...container.querySelectorAll<HTMLButtonElement>(".conversation-panel-row .inline-delete-actions button")]
      .find((button) => button.textContent === "删除")!;
    await act(async () => remove.click());
    expect(deleteConversation).toHaveBeenCalledWith("old", expect.anything());
    expect([...container.querySelectorAll(".conversation-panel-row")].some((item) => item.textContent?.includes("旧对话"))).toBe(false);
  });

  it("shows a running turn it can follow as it unfolds: the message, what is done, then the rest", async () => {
    const more = deferred<void>();
    vi.mocked(fetchConversationMessages).mockImplementation(async () => ({
      ...transcript(true),
      active_workflow: null,
      turn_running: true,
      running_turn_message: "帮我研究字节",
    }));
    vi.mocked(followLiveTurn).mockImplementationOnce(async function* (): AsyncGenerator<PublicStreamEvent> {
      yield { type: "turn_started", turn_id: "live" };
      yield { type: "content_delta", delta: "已经查到" };
      yield { type: "client_action", action: "open_url", url: "https://www.zhipin.com/web/geek/job?query=x" } as PublicStreamEvent;
      await more.promise;
      yield { type: "content_delta", delta: "三条业务线" };
      yield { type: "turn_completed", turn_id: "live" };
    });
    const opened = vi.spyOn(window, "open").mockReturnValue(null);
    await mount();

    expect(followLiveTurn).toHaveBeenCalledWith("old", expect.anything());
    expect(container.textContent).toContain("帮我研究字节");
    expect(container.textContent).toContain("已经查到");
    expect(container.textContent).not.toContain("这一轮还在后台运行");
    expect(element<HTMLButtonElement>('[aria-label="发送消息"]').disabled).toBe(true);
    await act(async () => more.resolve());
    expect(container.textContent).toContain("已经查到三条业务线");
    // A replayed action already happened when the turn ran.
    expect(opened).not.toHaveBeenCalled();
    opened.mockRestore();
  });

  it("reads the stored reply when the followed turn settled before it attached", async () => {
    let running = true;
    vi.mocked(fetchConversationMessages).mockImplementation(async () => ({
      ...transcript(true),
      active_workflow: null,
      messages: running ? [] : [
        { role: "user", content: "帮我研究字节", resources: [], created_at: "2026-09-01T00:00:00Z" },
        { role: "assistant", content: "研究完成", resources: [], created_at: "2026-09-01T00:00:01Z" },
      ],
      turn_running: running,
      running_turn_message: running ? "帮我研究字节" : null,
    }));
    vi.mocked(followLiveTurn).mockImplementationOnce(async function* (): AsyncGenerator<PublicStreamEvent> {
      running = false;
      throw new ChatStreamHttpError(404, "NO_LIVE_TURN", "没有");
    });
    await mount();
    await act(async () => undefined);

    expect(container.textContent).toContain("研究完成");
    expect(followLiveTurn).toHaveBeenCalledTimes(1);
  });

  it("shows a turn still running on the server and reads its reply once stored", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    try {
      let running = true;
      vi.mocked(fetchConversationMessages).mockImplementation(async () => ({
        ...transcript(true),
        active_workflow: null,
        messages: running ? [] : [
          { role: "user", content: "帮我研究公司", resources: [], created_at: "2026-09-01T00:00:00Z" },
          { role: "assistant", content: "研究完成", resources: [], created_at: "2026-09-01T00:00:01Z" },
        ],
        turn_running: running,
      }));
      await mount();
      expect(container.textContent).toContain("这一轮还在后台运行");
      await typeDraft("再问一句");
      expect(element<HTMLButtonElement>('[aria-label="发送消息"]').disabled).toBe(true);

      running = false;
      await act(async () => { await vi.advanceTimersByTimeAsync(4100); });
      expect(container.textContent).toContain("研究完成");
      expect(container.textContent).not.toContain("这一轮还在后台运行");
    } finally {
      vi.useRealTimers();
    }
  });

  it("runs a request made from a workspace page in its own new conversation", async () => {
    vi.mocked(fetchConversationMessages).mockResolvedValue(transcript());
    await mount();
    // A resume queued in the open conversation's composer stays there.
    await click('[aria-label="附上简历"]');
    await click('[data-testid="import"]');

    await click('[data-testid="ask-research"]');

    expect(streamChat).toHaveBeenCalledTimes(1);
    const [request] = vi.mocked(streamChat).mock.calls[0];
    expect(request.conversation_id).not.toBe("old");
    expect(request.message).toBe("让 Agent 研究这家公司");
    expect(request.input_resources ?? []).toEqual([]);
  });

  it("keeps a normal in-conversation analysis on its selected resume version", async () => {
    vi.mocked(fetchConversationMessages).mockResolvedValue(transcript());
    await mount();
    await click('[aria-label="附上简历"]');
    await click('[data-testid="import"]');
    await typeDraft("结合刚才这个岗位分析简历");
    await click('[aria-label="发送消息"]');
    expect(vi.mocked(streamChat).mock.calls[0][0]).toMatchObject({
      conversation_id: "old",
      message: "结合刚才这个岗位分析简历",
      input_resources: [{ kind: "resume_version", id: "queued-v2" }],
    });
  });

  it("waits for the old history to load before switching to the library task", async () => {
    const history = deferred<ConversationTranscript>();
    vi.mocked(fetchConversationMessages).mockImplementation(async (id) =>
      id === "old" ? history.promise : transcript());
    await mount();
    await click('[data-testid="analyze"]');
    expect(window.localStorage.getItem("career-agent:conversation-id")).toBe("old");
    expect(streamChat).not.toHaveBeenCalled();

    await act(async () => history.resolve(transcript(true)));
    expect(streamChat).toHaveBeenCalledTimes(1);
    expect(vi.mocked(streamChat).mock.calls[0][0].conversation_id).not.toBe("old");
  });

  it("ignores a late transcript from a conversation that was switched away from", async () => {
    const history = deferred<ConversationTranscript>();
    vi.mocked(fetchConversationMessages).mockImplementation(async (id) =>
      id === "old" ? history.promise : transcript());
    await mount();
    await click('[aria-label="新建对话"]');
    await typeDraft("新对话草稿");
    await act(async () => history.resolve(transcript(true)));
    expect(container.textContent).not.toContain("旧岗位上下文与面试问题");
    expect(element<HTMLTextAreaElement>("#message").value).toBe("新对话草稿");
    expect(streamChat).not.toHaveBeenCalled();
  });

  it("explains why Enter cannot send while conversation history is loading", async () => {
    const history = deferred<ConversationTranscript>();
    vi.mocked(fetchConversationMessages).mockReturnValue(history.promise);
    await mount();
    await typeDraft("不要重复这条消息");

    await act(async () => {
      element<HTMLTextAreaElement>("#message").dispatchEvent(new KeyboardEvent("keydown", {
        key: "Enter", bubbles: true, cancelable: true,
      }));
    });

    expect(container.textContent).toContain("正在恢复这段对话，恢复完成后即可发送。");
    expect(element<HTMLTextAreaElement>("#message").value).toBe("不要重复这条消息");
    expect(streamChat).not.toHaveBeenCalled();
    await act(async () => history.resolve(transcript(true)));
  });

  it("does not send when Enter only confirms pinyin in an input method", async () => {
    await mount();
    await typeDraft("nihao");
    const box = element<HTMLTextAreaElement>("#message");

    // Chrome marks the committing Enter as composing; Safari sends keyCode 229.
    await act(async () => {
      box.dispatchEvent(new KeyboardEvent("keydown", {
        key: "Enter", isComposing: true, bubbles: true, cancelable: true,
      }));
      box.dispatchEvent(new KeyboardEvent("keydown", {
        key: "Enter", keyCode: 229, bubbles: true, cancelable: true,
      }));
    });
    expect(streamChat).not.toHaveBeenCalled();
    expect(box.value).toBe("nihao");

    await act(async () => {
      box.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter", bubbles: true, cancelable: true }));
    });
    expect(streamChat).toHaveBeenCalledTimes(1);
  });

  it("shows repeated tailoring stages as a second review round", async () => {
    const completion = deferred<void>();
    vi.mocked(streamChat).mockImplementation(async function* (): AsyncGenerator<PublicStreamEvent> {
      yield { type: "turn_started", turn_id: "turn-progress" };
      for (const [step_key, step_label] of [
        ["resume_job_match", "正在比对简历与岗位要求"],
        ["resume_tailoring", "正在起草定制简历"],
        ["resume_draft_review", "正在审校简历草稿"],
        ["resume_tailoring", "正在起草定制简历"],
        ["resume_draft_review", "正在审校简历草稿"],
      ]) {
        yield {
          type: "progress", stage: "running_capability",
          message: `${step_label}……`, step_key, step_label,
        };
      }
      await completion.promise;
      yield { type: "turn_completed", turn_id: "turn-progress" };
    });
    await mount();
    await typeDraft("定制简历");
    await click('[aria-label="发送消息"]');

    expect(container.textContent).toContain("正在起草定制简历（第 2 轮）");
    expect(container.textContent).toContain("正在审校简历草稿（第 2 轮）");
    await act(async () => completion.resolve());
  });
});
