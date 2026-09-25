import { describe, expect, it } from "vitest";

import type { ConversationTranscript } from "../api/client";
import { hydrationFrom, turnIsStored } from "./recovery";

function transcript(
  messages: { role: "user" | "assistant"; content: string }[],
  extra: Partial<ConversationTranscript> = {},
): ConversationTranscript {
  return {
    messages: messages.map((message) => ({
      ...message,
      created_at: "2026-09-13T00:00:00Z",
      resources: [],
    })),
    active_workflow: null,
    phase: null,
    pending_interaction: null,
    pending_interaction_body: null,
    ...extra,
  };
}

describe("turnIsStored", () => {
  it("is false while our message is still the last thing in the transcript", () => {
    expect(turnIsStored(transcript([{ role: "user", content: "分析岗位" }]), "分析岗位")).toBe(false);
    expect(turnIsStored(transcript([]), "分析岗位")).toBe(false);
  });

  it("is true once a reply follows our message", () => {
    const stored = transcript([
      { role: "user", content: "分析岗位" },
      { role: "assistant", content: "核心要求是……" },
    ]);
    expect(turnIsStored(stored, "分析岗位")).toBe(true);
  });

  it("accepts a clipped copy of a long paste as our message", () => {
    const stored = transcript([
      { role: "user", content: "岗位描述：负责后端" },
      { role: "assistant", content: "已收到" },
    ]);
    expect(turnIsStored(stored, "岗位描述：负责后端服务的设计与实现……")).toBe(true);
  });

  it("does not mistake an older exchange for ours", () => {
    const stored = transcript([
      { role: "user", content: "你好" },
      { role: "assistant", content: "你好，需要什么帮助？" },
    ]);
    expect(turnIsStored(stored, "分析岗位")).toBe(false);
  });

  it("counts a pending interaction as the turn having landed", () => {
    const suspended = transcript([{ role: "user", content: "开始" }], {
      pending_interaction: {
        type: "interaction_required",
        interaction_id: "interaction_0123456789abcdef0123",
        kind: "confirmation",
        scope: "capability_confirmation",
        prompt: "确认执行？",
        options: [],
        allow_free_text: false,
      },
    });
    expect(turnIsStored(suspended, "开始")).toBe(true);
  });
});

describe("hydrationFrom", () => {
  it("appends the pending interaction body as an assistant message", () => {
    const hydration = hydrationFrom(
      "c1",
      transcript([{ role: "user", content: "开始" }], {
        active_workflow: "mock_interview",
        pending_interaction_body: "第一题：请介绍你自己。",
      }),
    );
    expect(hydration.messages.map((message) => message.content)).toEqual([
      "开始",
      "第一题：请介绍你自己。",
    ]);
    expect(hydration.awaitingInput).toBe(true);
  });

  it("does not wait for input once a workflow has left the conversation", () => {
    // The server reports the empty slot as the string "none", not null.
    const hydration = hydrationFrom(
      "c1",
      transcript([{ role: "user", content: "开始" }], { active_workflow: "none" }),
    );
    expect(hydration.awaitingInput).toBe(false);
  });

  it("restores a resume attachment's snapshot, including that it is no longer reachable", () => {
    const stored = transcript([{ role: "user", content: "帮我分析这份简历" }]);
    stored.messages[0].resources = [
      {
        kind: "resume_version",
        resource_id: "version-2",
        status_at_delivery: null,
        anchored_by_other_job: null,
        title: "主简历 v2",
        description: "pdf · 2048 bytes",
        available: false,
        resume_id: null,
      },
    ];
    const [message] = hydrationFrom("c1", stored).messages;
    expect(message.resources).toEqual([
      {
        kind: "resume_version",
        resourceId: "version-2",
        statusAtDelivery: null,
        anchoredByOtherJob: null,
        title: "主简历 v2",
        description: "pdf · 2048 bytes",
        available: false,
        resumeId: null,
      },
    ]);
  });
});
