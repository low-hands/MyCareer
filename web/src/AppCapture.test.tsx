// @vitest-environment jsdom
import { act, StrictMode } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, expect, it, vi } from "vitest";

import App from "./App";
import {
  acknowledgeJobCapture, fetchConversationMessages, fetchConversations,
  fetchPendingJobCaptures, retryJobCapture,
  type ConversationTranscript, type JobCapturedEventView,
} from "./api/client";
import { streamChat } from "./api/sse";

vi.mock("./api/client", async (importOriginal) => ({
  ...await importOriginal<typeof import("./api/client")>(),
  fetchConversationMessages: vi.fn(), fetchConversations: vi.fn(),
  fetchPendingJobCaptures: vi.fn(), acknowledgeJobCapture: vi.fn(), retryJobCapture: vi.fn(),
}));
vi.mock("./api/sse", () => ({ streamChat: vi.fn() }));
vi.mock("./pages/DailyBrief", () => ({ DailyBriefPanel: () => null }));
vi.mock("./pages/WorkspaceViews", () => ({
  DashboardPanel: () => null, ApplicationsPanel: () => null,
  CalendarPanel: () => null, EmailPanel: () => null, JobsPanel: () => null,
  ResearchPanel: () => null, ResumesPanel: () => null,
}));

let root: Root;
let container: HTMLDivElement;
const capture: JobCapturedEventView = {
  id: "jobcap-one", conversation_id: "original", job_posting_id: "job",
  jd_snapshot_id: "snapshot-v1", title: "AI", company_name: "Example",
  created_at: "2026-09-01T00:00:00Z", continuation_status: "pending",
  continuation_turn_id: null,
};

function transcript(content: string): ConversationTranscript {
  return {
    messages: [{ role: "assistant", content, resources: [], created_at: capture.created_at }],
    active_workflow: null, phase: null, pending_interaction: null, pending_interaction_body: null,
  };
}

async function mount() {
  await act(async () => root.render(<StrictMode><App /></StrictMode>));
}

beforeEach(() => {
  vi.resetAllMocks();
  vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
  HTMLElement.prototype.scrollTo = vi.fn();
  window.localStorage.clear();
  window.localStorage.setItem("career-agent:conversation-id", "original");
  vi.mocked(fetchPendingJobCaptures).mockResolvedValue([capture]);
  vi.mocked(fetchConversations).mockResolvedValue([]);
  vi.mocked(fetchConversationMessages).mockResolvedValue(transcript("原有会话"));
  vi.mocked(acknowledgeJobCapture).mockResolvedValue(true);
  vi.mocked(retryJobCapture).mockResolvedValue(true);
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

it("does not start or acknowledge a pending backend continuation", async () => {
  await mount();
  expect(container.textContent).toContain("原有会话");
  expect(streamChat).not.toHaveBeenCalled();
  expect(acknowledgeJobCapture).not.toHaveBeenCalled();
});

it("refreshes a completed continuation and acknowledges without opening a turn", async () => {
  vi.mocked(fetchPendingJobCaptures).mockResolvedValue([{ ...capture, continuation_status: "completed" }]);
  vi.mocked(fetchConversationMessages).mockResolvedValue(transcript("后端已保存岗位"));
  await mount();
  expect(container.textContent).toContain("后端已保存岗位");
  expect(acknowledgeJobCapture).toHaveBeenCalledTimes(1);
  expect(streamChat).not.toHaveBeenCalled();
});

it("keeps the visible conversation when another conversation receives a capture", async () => {
  window.localStorage.setItem("career-agent:conversation-id", "other");
  vi.mocked(fetchPendingJobCaptures).mockResolvedValue([{ ...capture, continuation_status: "completed" }]);
  vi.mocked(fetchConversationMessages).mockImplementation(async (id) =>
    transcript(id === "other" ? "当前会话内容" : "原会话新岗位"));
  await mount();
  expect(container.textContent).toContain("当前会话内容");
  expect(container.textContent).not.toContain("原会话新岗位");
  expect(window.localStorage.getItem("career-agent:conversation-id")).toBe("other");
  expect(acknowledgeJobCapture).toHaveBeenCalledTimes(1);
  expect(streamChat).not.toHaveBeenCalled();
});

it("retries failed captures through the backend endpoint", async () => {
  vi.mocked(fetchPendingJobCaptures).mockResolvedValue([{ ...capture, continuation_status: "failed" }]);
  await mount();
  const retry = [...container.querySelectorAll("button")].find((button) => button.textContent === "重试续接");
  expect(retry).toBeDefined();
  await act(async () => retry?.click());
  expect(retryJobCapture).toHaveBeenCalledWith(capture.id, expect.any(Object));
  expect(streamChat).not.toHaveBeenCalled();
  expect(acknowledgeJobCapture).not.toHaveBeenCalled();
});
