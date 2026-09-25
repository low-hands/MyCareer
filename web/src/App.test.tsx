// @vitest-environment jsdom
import { act, StrictMode } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import App from "./App";
import {
  fetchConversationMessages, fetchConversations, fetchPendingJobCaptures,
  type ConversationTranscript, type ResumeImportResult,
} from "./api/client";
import { streamChat } from "./api/sse";
import type { ChatAttachment } from "./chat/attachments";
import type { PublicStreamEvent } from "./chat/events";

vi.mock("./api/client", async (importOriginal) => ({
  ...await importOriginal<typeof import("./api/client")>(),
  fetchConversationMessages: vi.fn(),
  fetchConversations: vi.fn(),
  fetchPendingJobCaptures: vi.fn(),
}));
vi.mock("./api/sse", () => ({ streamChat: vi.fn() }));
vi.mock("./pages/DailyBrief", () => ({ DailyBriefPanel: () => null }));
vi.mock("./pages/WorkspaceViews", () => ({
  DashboardPanel: () => null,
  ApplicationsPanel: () => null,
  CalendarPanel: () => null,
  EmailPanel: () => null,
  JobsPanel: () => null,
  ResearchPanel: () => null,
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

  it("queues library analysis behind a running turn without interrupting it", async () => {
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
    expect(streamChat).toHaveBeenCalledTimes(1);
    expect(window.localStorage.getItem("career-agent:conversation-id")).toBe("old");
    expect(vi.mocked(streamChat).mock.calls[0][1]?.signal?.aborted).toBe(false);

    await act(async () => completion.resolve());
    expect(streamChat).toHaveBeenCalledTimes(2);
    expect(vi.mocked(streamChat).mock.calls[0][0].conversation_id).toBe("old");
    expect(vi.mocked(streamChat).mock.calls[1][0].conversation_id).not.toBe("old");
    expect(vi.mocked(streamChat).mock.calls[1][0].input_resources).toEqual([
      { kind: "resume_version", id: "library-v1" },
    ]);
  });

  it("can retry a cancelled queued analysis without sending the cancelled task", async () => {
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
    await click(".standalone-task-notice button");
    expect(container.querySelector(".standalone-task-notice")).toBeNull();
    expect(vi.mocked(streamChat).mock.calls[0][1]?.signal?.aborted).toBe(false);

    await act(async () => completion.resolve());
    expect(streamChat).toHaveBeenCalledTimes(1);
    expect(window.localStorage.getItem("career-agent:conversation-id")).toBe("old");

    await click('[data-testid="analyze"]');
    expect(streamChat).toHaveBeenCalledTimes(2);
    expect(vi.mocked(streamChat).mock.calls[1][0].conversation_id).not.toBe("old");
    expect(vi.mocked(streamChat).mock.calls[1][0].input_resources).toEqual([
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

  it("can queue another version while the first standalone analysis is streaming", async () => {
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
    expect(streamChat).toHaveBeenCalledTimes(1);
    expect(window.localStorage.getItem("career-agent:conversation-id")).toBe(firstConversation);
    await act(async () => completion.resolve());
    expect(streamChat).toHaveBeenCalledTimes(2);
    const [next] = vi.mocked(streamChat).mock.calls[1];
    expect(next.conversation_id).not.toBe(firstConversation);
    expect(next.input_resources).toEqual([{ kind: "resume_version", id: "library-v2" }]);
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
