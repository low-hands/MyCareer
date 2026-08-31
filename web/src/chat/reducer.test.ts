import { describe, expect, it } from "vitest";

import { chatReducer, initialChatState } from "./reducer";

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

  it("hydrates a persisted conversation without inventing a running turn", () => {
    const state = chatReducer(initialChatState, {
      type: "hydrate",
      messages: [
        { id: "history-1", role: "user", content: "分析这份岗位" },
        { id: "history-2", role: "assistant", content: "这是岗位分析。" },
      ],
    });

    expect(state.phase).toBe("completed");
    expect(state.messages.map((item) => item.content)).toEqual([
      "分析这份岗位",
      "这是岗位分析。",
    ]);
    expect(state.activeAssistantMessageId).toBeNull();
  });
});
