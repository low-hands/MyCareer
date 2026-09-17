// @vitest-environment jsdom
import { act, StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { expect, it, vi } from "vitest";

import App from "./App";
import { fetchConversationMessages, fetchConversations, fetchPendingJobCaptures } from "./api/client";
import { streamChat } from "./api/sse";
import type { ChatAttachment } from "./chat/attachments";
import type { PublicStreamEvent } from "./chat/events";

vi.mock("./api/client", async (importOriginal) => ({
  ...await importOriginal<typeof import("./api/client")>(),
  fetchConversationMessages: vi.fn(), fetchConversations: vi.fn(), fetchPendingJobCaptures: vi.fn(),
}));
vi.mock("./api/sse", () => ({ streamChat: vi.fn() }));
vi.mock("./pages/DailyBrief", () => ({ DailyBriefPanel: () => null }));
vi.mock("./pages/WorkspaceViews", () => ({
  DashboardPanel: () => null, ApplicationsPanel: () => null, CalendarPanel: () => null,
  EmailPanel: () => null, ResearchPanel: () => null, ResumesPanel: () => null,
  JobsPanel: (props: {
    onStartStandaloneTask: (prompt: string, resource: ChatAttachment, additionalResources: ChatAttachment[]) => void;
  }) => (
    <button data-testid="match" onClick={() => props.onStartStandaloneTask(
      "匹配简历 v1 和 JD v1",
      { resumeId: "resume", resumeVersionId: "resume-v1", name: "Resume", versionNumber: 1, documentFormat: "text", byteSize: 20, uploadedAt: null },
      [{ kind: "jd_snapshot", jdSnapshotId: "jd-v1", jobPostingId: "job", title: "Engineer", companyName: "Example", jdVersion: 1 }],
    )}>开始匹配</button>
  ),
}));

it("sends both immutable references once in a new hydrated conversation", async () => {
  vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
  HTMLElement.prototype.scrollTo = vi.fn();
  window.localStorage.clear();
  window.localStorage.setItem("career-agent:conversation-id", "old");
  vi.mocked(fetchPendingJobCaptures).mockResolvedValue([]);
  vi.mocked(fetchConversations).mockResolvedValue([]);
  vi.mocked(fetchConversationMessages).mockResolvedValue({
    messages: [], active_workflow: null, phase: null, pending_interaction: null, pending_interaction_body: null,
  });
  vi.mocked(streamChat).mockImplementation(async function* (): AsyncGenerator<PublicStreamEvent> {
    yield { type: "turn_started", turn_id: "test" };
    yield { type: "turn_completed", turn_id: "test" };
  });
  const container = document.createElement("div");
  document.body.append(container);
  const root = createRoot(container);
  try {
    await act(async () => root.render(<StrictMode><App /></StrictMode>));
    const button = container.querySelector<HTMLButtonElement>('[data-testid="match"]')!;
    await act(async () => { button.click(); button.click(); });
    expect(streamChat).toHaveBeenCalledTimes(1);
    const [request] = vi.mocked(streamChat).mock.calls[0];
    expect(request.conversation_id).not.toBe("old");
    expect(request.input_resources).toEqual([
      { kind: "resume_version", id: "resume-v1" },
      { kind: "jd_snapshot", id: "jd-v1" },
    ]);
    expect(fetchConversationMessages).toHaveBeenCalledWith(request.conversation_id, expect.anything());
  } finally {
    await act(async () => root.unmount());
    container.remove();
    vi.unstubAllGlobals();
  }
});
