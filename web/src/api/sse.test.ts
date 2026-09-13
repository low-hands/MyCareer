import { afterEach, describe, expect, it, vi } from "vitest";

import { SseParser, streamChat } from "./sse";

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("SseParser", () => {
  it("handles chunk boundaries, CRLF, comments and multi-line data", () => {
    const parser = new SseParser();
    expect(parser.push(": keep-alive\r")).toEqual([]);
    expect(parser.push("\nevent: content_delta\r\ndata: {\"type\":\r\n")).toEqual([]);
    expect(parser.push("data: \"content_delta\",\"delta\":\"你好\"}\r\n\r\n")).toEqual([
      {
        event: "content_delta",
        data: '{"type":\n"content_delta","delta":"你好"}',
      },
    ]);
  });

  it("does not dispatch an unterminated event at end of stream", () => {
    const parser = new SseParser();
    expect(parser.push('event: progress\ndata: {"type":"progress"}')).toEqual([]);
    expect(parser.finish()).toEqual([]);
  });
});

describe("streamChat", () => {
  it("sends a bound interaction response in the POST body", async () => {
    const fetchMock = vi.fn(async (_input: RequestInfo | URL, init?: RequestInit) =>
      new Response(
        'event: turn_completed\ndata: {"type":"turn_completed","turn_id":"turn-1"}\n\n',
        { status: 200 },
      ),
    );
    vi.stubGlobal("fetch", fetchMock);

    for await (const _event of streamChat({
      conversation_id: "c1",
      message: "确认并导入",
      interaction_response: {
        interaction_id: "interaction_0123456789abcdef0123",
        scope: "resume_analysis_confirmation",
        action: "confirm",
      },
    })) {
      // Consume the response so the request completes.
    }

    const request = fetchMock.mock.calls[0][1];
    expect(JSON.parse(String(request?.body))).toEqual({
      conversation_id: "c1",
      message: "确认并导入",
      interaction_response: {
        interaction_id: "interaction_0123456789abcdef0123",
        scope: "resume_analysis_confirmation",
        action: "confirm",
      },
    });
  });

  it("sends the idempotency key as a header when given one", async () => {
    const fetchMock = vi.fn(async (_input: RequestInfo | URL, init?: RequestInit) =>
      new Response(
        'event: turn_completed\ndata: {"type":"turn_completed","turn_id":"turn-1"}\n\n',
        { status: 200 },
      ),
    );
    vi.stubGlobal("fetch", fetchMock);

    for await (const _event of streamChat(
      { conversation_id: "c1", message: "你好" },
      { idempotencyKey: "req-1" },
    )) {
      // Consume the response so the request completes.
    }

    const headers = fetchMock.mock.calls[0][1]?.headers as Record<string, string>;
    expect(headers["Idempotency-Key"]).toBe("req-1");
  });

  it("decodes typed events across response chunks", async () => {
    const encoder = new TextEncoder();
    const body = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(encoder.encode("event: content_delta\ndata: {\"type\":\"content_"));
        controller.enqueue(encoder.encode("delta\",\"delta\":\"你好\"}\n\n"));
        controller.enqueue(
          encoder.encode(
            "event: turn_completed\ndata: {\"type\":\"turn_completed\",\"turn_id\":\"turn-1\"}\n\n",
          ),
        );
        controller.close();
      },
    });
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => new Response(body, { status: 200 })),
    );

    const events = [];
    for await (const event of streamChat({
      conversation_id: "c1",
      message: "你好",
    })) {
      events.push(event);
    }

    expect(events).toEqual([
      { type: "content_delta", delta: "你好" },
      { type: "turn_completed", turn_id: "turn-1" },
    ]);
  });

  it("turns a conversation conflict into a safe client error", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () =>
        new Response(
          JSON.stringify({
            detail: {
              code: "CONVERSATION_TURN_IN_PROGRESS",
              message: "Internal English message",
            },
          }),
          { status: 409, headers: { "Content-Type": "application/json" } },
        ),
      ),
    );

    const consume = async () => {
      for await (const _event of streamChat({
        conversation_id: "c1",
        message: "再次发送",
      })) {
        // The response fails before yielding an event.
      }
    };

    await expect(consume()).rejects.toMatchObject({
      status: 409,
      code: "CONVERSATION_TURN_IN_PROGRESS",
      message: "这个对话还有一轮正在处理中，请等待它结束后再发送。",
    });
  });

  it("replaces the browser's Failed to fetch with an actionable message", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => {
        throw new TypeError("Failed to fetch");
      }),
    );

    const consume = async () => {
      for await (const _event of streamChat({
        conversation_id: "c1",
        message: "你好",
      })) {
        // Network failure occurs before an event can be yielded.
      }
    };

    await expect(consume()).rejects.toMatchObject({
      status: 0,
      code: "API_UNREACHABLE",
      message: "无法连接 Career Agent 服务，请确认 FastAPI 已启动。",
    });
  });
});
