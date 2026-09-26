import { describe, expect, it } from "vitest";

import { newStandaloneAgentTask, standaloneAgentTaskStep } from "./agentTask";
import { isJobAttachment, type JobAttachment, type ResumeAttachment } from "./attachments";

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
    expect(isJobAttachment(task.resource)).toBe(false);
    expect(task.resource).toMatchObject({ resumeVersionId: "resume_version-v1" });
    expect(newStandaloneAgentTask("x", resource).conversationId).not.toBe(task.conversationId);
  });

  it("carries a JD snapshot on its own, with no resume beside it", () => {
    const job: JobAttachment = {
      kind: "jd_snapshot",
      jdSnapshotId: "snapshot-3",
      jobPostingId: "job-1",
      title: "AI 产品经理",
      companyName: "示例科技",
      jdVersion: 3,
    };
    const task = newStandaloneAgentTask("分析 JD", job);
    expect(isJobAttachment(task.resource)).toBe(true);
    expect(task.resource).toMatchObject({ jdSnapshotId: "snapshot-3" });
  });

  it("switches away from a running turn, which finishes on the server", () => {
    const task = newStandaloneAgentTask("p", resource);
    expect(
      standaloneAgentTaskStep(task, {
        conversationId: "conversation-old",
        hydratedConversationId: "conversation-old",
        busy: true,
        historyLoading: false,
      }),
    ).toBe("switch");
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
