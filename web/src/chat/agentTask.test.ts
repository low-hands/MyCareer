import { describe, expect, it } from "vitest";

import { newStandaloneAgentTask, standaloneAgentTaskStep } from "./agentTask";
import type { ResumeAttachment } from "./attachments";

const resource: ResumeAttachment = {
  resumeId: "resume-1",
  resumeVersionId: "resume_version-v1",
  name: "AI Resume",
  versionNumber: 1,
  documentFormat: "pdf",
  byteSize: 1024,
  uploadedAt: null,
};

describe("standalone agent task", () => {
  it("gets its own conversation and keeps the exact resume version", () => {
    const task = newStandaloneAgentTask("帮我分析简历“AI Resume”的 v1", resource);
    expect(task.conversationId).toMatch(/^conversation-/);
    expect(task.resource.resumeVersionId).toBe("resume_version-v1");
    expect(newStandaloneAgentTask("x", resource).conversationId).not.toBe(task.conversationId);
  });

  it("waits for a running turn instead of interrupting it", () => {
    const task = newStandaloneAgentTask("p", resource);
    expect(
      standaloneAgentTaskStep(task, {
        conversationId: "conversation-old",
        hydratedConversationId: "conversation-old",
        busy: true,
        historyLoading: false,
      }),
    ).toBe("wait");
  });

  it("switches conversations only when the old chat is idle", () => {
    const task = newStandaloneAgentTask("p", resource);
    expect(
      standaloneAgentTaskStep(task, {
        conversationId: "conversation-old",
        hydratedConversationId: "conversation-old",
        busy: false,
        historyLoading: false,
      }),
    ).toBe("switch");
  });

  it("does not send until the new conversation has been hydrated", () => {
    const task = newStandaloneAgentTask("p", resource);
    expect(
      standaloneAgentTaskStep(task, {
        conversationId: task.conversationId,
        hydratedConversationId: "conversation-old",
        busy: false,
        historyLoading: true,
      }),
    ).toBe("wait");
    expect(
      standaloneAgentTaskStep(task, {
        conversationId: task.conversationId,
        hydratedConversationId: "conversation-old",
        busy: false,
        historyLoading: false,
      }),
    ).toBe("wait");
    expect(
      standaloneAgentTaskStep(task, {
        conversationId: task.conversationId,
        hydratedConversationId: task.conversationId,
        busy: false,
        historyLoading: false,
      }),
    ).toBe("send");
  });
});
