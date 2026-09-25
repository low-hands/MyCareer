import { describe, expect, it } from "vitest";

import { chatReducer, initialChatState, visibleMessageResources } from "./reducer";

describe("visibleMessageResources", () => {
  const savedJob = { kind: "saved_job" as const, resourceId: "jd-1" };

  it("hides an identical resource repeated on the adjacent assistant reply", () => {
    const messages = [
      { id: "u", role: "user" as const, content: "保存了", resources: [savedJob] },
      { id: "a", role: "assistant" as const, content: "已记录", resources: [savedJob] },
    ];
    expect(visibleMessageResources(messages, 0)).toEqual([savedJob]);
    expect(visibleMessageResources(messages, 1)).toEqual([]);
  });

  it("keeps a resource on a later independent assistant message", () => {
    const messages = [
      { id: "u", role: "user" as const, content: "保存了", resources: [savedJob] },
      { id: "a1", role: "assistant" as const, content: "已记录" },
      { id: "a2", role: "assistant" as const, content: "再看一下", resources: [savedJob] },
    ];
    expect(visibleMessageResources(messages, 2)).toEqual([savedJob]);
  });
});

describe("chatReducer", () => {
  it("aggregates token deltas and reaches a committed terminal state", () => {
    let state = chatReducer(initialChatState, {
      type: "submit",
      messageId: "user-1",
      assistantMessageId: "assistant-1",
      content: "分析岗位",
    });
    state = chatReducer(state, {
      type: "stream_event",
      event: { type: "content_delta", delta: "核心" },
    });
    state = chatReducer(state, {
      type: "stream_event",
      event: { type: "content_delta", delta: "要求" },
    });
    state = chatReducer(state, {
      type: "stream_event",
      event: { type: "turn_completed", turn_id: "turn-1" },
    });

    expect(state.phase).toBe("completed");
    expect(state.messages.at(-1)?.content).toBe("核心要求");
  });

  it("keeps a structured interaction when the turn suspends", () => {
    let state = chatReducer(initialChatState, {
      type: "submit",
      messageId: "user-1",
      assistantMessageId: "assistant-1",
      content: "开始",
    });
    state = chatReducer(state, {
      type: "stream_event",
      event: {
        type: "interaction_required",
        interaction_id: "interaction_1234567890abcdef1234",
        kind: "single_selection",
        prompt: "选择岗位",
        options: [{ label: "岗位 A", selection_index: 1 }],
        allow_free_text: false,
      },
    });
    // The form is usable as soon as it appears. A delayed/lost terminal SSE
    // frame must not strand the user with a disabled interaction card.
    expect(state.phase).toBe("awaiting_input");
    state = chatReducer(state, {
      type: "stream_event",
      event: {
        type: "turn_suspended",
        turn_id: "turn-1",
        reason: "interaction_required",
        interaction_id: "interaction_1234567890abcdef1234",
      },
    });

    expect(state.phase).toBe("awaiting_input");
    expect(state.interaction?.options[0]?.selection_index).toBe(1);
  });

  it("keeps an answered question in the transcript instead of an empty bubble", () => {
    let state = chatReducer(initialChatState, {
      type: "submit",
      messageId: "user-1",
      assistantMessageId: "assistant-1",
      content: "开始模拟面试",
    });
    state = chatReducer(state, {
      type: "stream_event",
      event: {
        type: "interaction_required",
        interaction_id: "interaction_1234567890abcdef1234",
        kind: "free_text",
        prompt: "模拟面试题：讲一个你负责的项目。",
        options: [],
        allow_free_text: true,
      },
    });
    // A free-text question is a message right away, not a card.
    expect(state.messages.at(-1)?.content).toBe("模拟面试题：讲一个你负责的项目。");
    expect(state.interaction?.kind).toBe("free_text");

    state = chatReducer(state, {
      type: "submit",
      messageId: "user-2",
      assistantMessageId: "assistant-2",
      content: "我负责检索评测。",
    });

    expect(state.messages.map((message) => message.content)).toEqual([
      "开始模拟面试",
      "模拟面试题：讲一个你负责的项目。",
      "我负责检索评测。",
      "",
    ]);
  });

  it("keeps a browser action as an explicit clickable fallback", () => {
    const state = chatReducer(initialChatState, {
      type: "stream_event",
      event: {
        type: "client_action",
        action: "open_url",
        url: "https://www.zhipin.com/web/geek/job?query=AI",
        label: "在 BOSS 搜索 AI",
      },
    });

    expect(state.clientActions).toHaveLength(1);
    expect(state.clientActions[0]?.action).toBe("open_url");
  });

  it("attaches a report reference to the reply that produced it", () => {
    /*
     * On the message, not on the turn: the reply keeps a short summary and the
     * report is reachable only through this reference, so a conversation with
     * two reports has to keep each next to its own reply.
     */
    const submitted = chatReducer(initialChatState, {
      type: "submit",
      messageId: "u-1",
      assistantMessageId: "a-1",
      content: "帮我复盘刚才的模拟面试",
    });
    const streamed = chatReducer(submitted, {
      type: "stream_event",
      event: { type: "content_delta", delta: "项目深度可以，系统设计偏弱。" },
    });
    const first = chatReducer(streamed, {
      type: "stream_event",
      event: {
        type: "report_ready",
        kind: "mock_interview_report",
        resource_id: "rep-1",
      },
    });
    // A second report in the same turn must not replace the first: the server
    // stores both on the message (`resources`, plural) and so must we.
    const second = chatReducer(first, {
      type: "stream_event",
      event: {
        type: "report_ready",
        kind: "interview_retro_report",
        resource_id: "retro-1",
      },
    });
    // The same report delivered again updates its card rather than adding one.
    const state = chatReducer(second, {
      type: "stream_event",
      event: {
        type: "report_ready",
        kind: "mock_interview_report",
        resource_id: "rep-1",
        status_at_delivery: "outdated",
      },
    });

    const assistant = state.messages.find((item) => item.id === "a-1");
    expect(assistant?.resources).toEqual([
      { kind: "mock_interview_report", resourceId: "rep-1", statusAtDelivery: "outdated" },
      { kind: "interview_retro_report", resourceId: "retro-1" },
    ]);
    // The summary is what stays in the bubble; the report is behind the card.
    expect(assistant?.content).toBe("项目深度可以，系统设计偏弱。");
    expect(state.messages.find((item) => item.id === "u-1")?.resources).toBeUndefined();
  });

  it("keeps a JD card and a report card side by side on one assistant message", () => {
    const submitted = chatReducer(initialChatState, {
      type: "submit",
      messageId: "u-1",
      assistantMessageId: "a-1",
      content: "保存后帮我调研一下这家公司",
    });
    const streamed = chatReducer(submitted, {
      type: "stream_event",
      event: { type: "content_delta", delta: "已保存「量霸科技｜AI Agent 实习生」。" },
    });
    const withJob = chatReducer(streamed, {
      type: "stream_event",
      event: {
        type: "job_resource_ready",
        kind: "saved_job",
        resource_id: "jds-1",
        job_posting_id: "job-1",
        title: "量霸科技｜AI Agent 实习生",
        description: "JD 第 1 版 · BOSS直聘",
      },
    });
    const state = chatReducer(withJob, {
      type: "stream_event",
      event: { type: "report_ready", kind: "job_research_report", resource_id: "rep-1" },
    });

    const assistant = state.messages.find((item) => item.id === "a-1");
    // The JD text never enters the bubble; the card carries the snapshot id.
    expect(assistant?.content).toBe("已保存「量霸科技｜AI Agent 实习生」。");
    expect(assistant?.resources).toEqual([
      {
        kind: "saved_job",
        resourceId: "jds-1",
        jobPostingId: "job-1",
        title: "量霸科技｜AI Agent 实习生",
        description: "JD 第 1 版 · BOSS直聘",
        available: true,
      },
      { kind: "job_research_report", resourceId: "rep-1" },
    ]);
  });

  it("hydrates a persisted conversation without inventing a running turn", () => {
    const state = chatReducer(initialChatState, {
      type: "hydrate",
      messages: [
        { id: "history-1", role: "user", content: "分析这份岗位" },
        {
          id: "history-2",
          role: "assistant",
          content: "这是岗位分析。",
          resources: [{ kind: "job_research_report", resourceId: "report-1" }],
        },
      ],
    });

    expect(state.phase).toBe("completed");
    expect(state.messages.map((item) => item.content)).toEqual([
      "分析这份岗位",
      "这是岗位分析。",
    ]);
    expect(state.activeAssistantMessageId).toBeNull();
    // A reload has to leave the report reachable, or the transcript keeps a
    // summary of something the user can no longer open.
    expect(state.messages[1]?.resources?.[0]?.resourceId).toBe("report-1");
  });

  it("treats a dropped stream as recovering, not failed, until the transcript answers", () => {
    let state = chatReducer(initialChatState, {
      type: "submit",
      messageId: "user-1",
      assistantMessageId: "assistant-1",
      content: "分析岗位",
    });
    state = chatReducer(state, {
      type: "stream_event",
      event: { type: "content_delta", delta: "核心" },
    });
    state = chatReducer(state, { type: "transport_lost" });

    expect(state.phase).toBe("recovering");
    expect(state.error).toBeNull();
    expect(state.progress).toContain("连接中断");
    // The partial reply stays on screen while we check.
    expect(state.messages.at(-1)?.content).toBe("核心");

    const recovered = chatReducer(state, {
      type: "hydrate",
      messages: [
        { id: "h-0", role: "user", content: "分析岗位" },
        { id: "h-1", role: "assistant", content: "核心要求是……" },
      ],
    });
    expect(recovered.phase).toBe("completed");
    expect(recovered.messages.at(-1)?.content).toBe("核心要求是……");
  });

  it("resubmits a failed exchange in place and overwrites only its reply", () => {
    let state = chatReducer(initialChatState, {
      type: "submit",
      messageId: "user-1",
      assistantMessageId: "assistant-1",
      content: "分析岗位",
    });
    state = chatReducer(state, {
      type: "stream_event",
      event: { type: "content_delta", delta: "核心" },
    });
    state = chatReducer(state, { type: "transport_failed", message: "断了" });

    state = chatReducer(state, { type: "resubmit" });

    expect(state.phase).toBe("running");
    expect(state.error).toBeNull();
    expect(state.messages.map((message) => message.id)).toEqual(["user-1", "assistant-1"]);
    expect(state.messages.at(-1)?.content).toBe("");

    state = chatReducer(state, {
      type: "stream_event",
      event: { type: "content_delta", delta: "核心要求是……" },
    });
    state = chatReducer(state, {
      type: "stream_event",
      event: { type: "turn_completed", turn_id: "turn-1" },
    });
    expect(state.phase).toBe("completed");
    expect(state.messages.at(-1)?.content).toBe("核心要求是……");
  });

  it("shows the safe application error code for a failed turn", () => {
    const state = chatReducer(initialChatState, {
      type: "stream_event",
      event: {
        type: "turn_failed",
        turn_id: "turn-1",
        code: "MAIN_AGENT_REJECTED_400",
        message: "当前模型配置不支持该请求。",
      },
    });
    expect(state.phase).toBe("failed");
    expect(state.error).toBe(
      "当前模型配置不支持该请求。（错误码：MAIN_AGENT_REJECTED_400）",
    );
  });

  it("keeps structured capability stages and counts writer-review revisions", () => {
    let state = chatReducer(initialChatState, {
      type: "stream_event",
      event: {
        type: "progress", stage: "running_capability", message: "正在比对简历与岗位要求……",
        step_key: "resume_job_match", step_label: "正在比对简历与岗位要求",
      },
    });
    state = chatReducer(state, {
      type: "stream_event",
      event: {
        type: "progress", stage: "running_capability", message: "正在比对简历与岗位要求（已等待 30 秒）……",
        step_key: "resume_job_match", step_label: "正在比对简历与岗位要求",
      },
    });
    state = chatReducer(state, {
      type: "stream_event",
      event: {
        type: "progress", stage: "running_capability", message: "正在起草定制简历……",
        step_key: "resume_tailoring", step_label: "正在起草定制简历",
      },
    });
    state = chatReducer(state, {
      type: "stream_event",
      event: {
        type: "progress", stage: "running_capability", message: "正在审校简历草稿……",
        step_key: "resume_draft_review", step_label: "正在审校简历草稿",
      },
    });
    state = chatReducer(state, {
      type: "stream_event",
      event: {
        type: "progress", stage: "running_capability", message: "正在起草定制简历……",
        step_key: "resume_tailoring", step_label: "正在起草定制简历",
      },
    });
    state = chatReducer(state, {
      type: "stream_event",
      event: {
        type: "progress", stage: "running_capability", message: "正在审校简历草稿……",
        step_key: "resume_draft_review", step_label: "正在审校简历草稿",
      },
    });

    expect(state.progressSteps).toEqual([
      { key: "resume_job_match", label: "正在比对简历与岗位要求", occurrence: 1, completed: true },
      { key: "resume_tailoring", label: "正在起草定制简历", occurrence: 1, completed: true },
      { key: "resume_draft_review", label: "正在审校简历草稿", occurrence: 1, completed: true },
      { key: "resume_tailoring", label: "正在起草定制简历", occurrence: 2, completed: true },
      { key: "resume_draft_review", label: "正在审校简历草稿", occurrence: 2, completed: false },
    ]);
    expect(new Set(state.progressSteps.map(
      (step) => `${step.key}-${step.occurrence}`,
    )).size).toBe(state.progressSteps.length);
  });
});
