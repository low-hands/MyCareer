import {
  parsePublicStreamEvent,
  type InteractionScope,
  type PublicStreamEvent,
} from "../chat/events";

export interface ChatStreamRequest {
  conversation_id: string;
  message: string;
  interaction_response?: InteractionResponse;
}

export interface InteractionResponse {
  interaction_id: string;
  scope: InteractionScope;
  action: "confirm" | "cancel";
}

export interface SseFrame {
  event: string;
  data: string;
}

export class SseParser {
  private line = "";
  private eventName = "message";
  private dataLines: string[] = [];
  private pendingCarriageReturn = false;

  push(chunk: string): SseFrame[] {
    const frames: SseFrame[] = [];
    for (const character of chunk) {
      if (this.pendingCarriageReturn) {
        this.finishLine(frames);
        this.pendingCarriageReturn = false;
        if (character === "\n") continue;
      }
      if (character === "\r") {
        this.pendingCarriageReturn = true;
      } else if (character === "\n") {
        this.finishLine(frames);
      } else {
        this.line += character;
      }
    }
    return frames;
  }

  finish(): SseFrame[] {
    const frames: SseFrame[] = [];
    if (this.pendingCarriageReturn) {
      this.finishLine(frames);
      this.pendingCarriageReturn = false;
    }
    return frames;
  }

  private finishLine(frames: SseFrame[]): void {
    const line = this.line;
    this.line = "";
    if (line === "") {
      if (this.dataLines.length > 0) {
        frames.push({ event: this.eventName, data: this.dataLines.join("\n") });
      }
      this.eventName = "message";
      this.dataLines = [];
      return;
    }
    if (line.startsWith(":")) return;

    const separator = line.indexOf(":");
    const field = separator === -1 ? line : line.slice(0, separator);
    let value = separator === -1 ? "" : line.slice(separator + 1);
    if (value.startsWith(" ")) value = value.slice(1);
    if (field === "event") this.eventName = value;
    if (field === "data") this.dataLines.push(value);
  }
}

export class ChatStreamHttpError extends Error {
  constructor(
    readonly status: number,
    readonly code: string,
    message: string,
  ) {
    super(message);
  }
}

async function errorFromResponse(response: Response): Promise<ChatStreamHttpError> {
  let code = `HTTP_${response.status}`;
  let message = "连接 Career Agent 失败，请稍后重试。";
  try {
    const body = (await response.json()) as {
      detail?: { code?: unknown; message?: unknown };
    };
    if (typeof body.detail?.code === "string") code = body.detail.code;
    if (typeof body.detail?.message === "string") message = body.detail.message;
  } catch {
    // The server may return a proxy-generated non-JSON error page.
  }
  if (code === "CONVERSATION_TURN_IN_PROGRESS") {
    message = "这个对话还有一轮正在处理中，请等待它结束后再发送。";
  }
  if (code === "TURN_CAPACITY_EXHAUSTED") {
    message = "当前任务较多，请稍后再试。";
  }
  return new ChatStreamHttpError(response.status, code, message);
}

export interface ChatStreamOptions {
  apiBaseUrl?: string;
  signal?: AbortSignal;
  /**
   * Sent as `Idempotency-Key`. The server anchors this turn's external writes
   * (calendar, applications) to it, so a later request carrying the same key
   * for the same conversation is recognised as the same attempt instead of
   * writing twice. It does not make the whole turn idempotent: the model still
   * runs and a new reply is still stored.
   */
  idempotencyKey?: string;
}

export async function* streamChat(
  request: ChatStreamRequest,
  options: ChatStreamOptions = {},
): AsyncGenerator<PublicStreamEvent> {
  const apiBaseUrl = (options.apiBaseUrl ?? "/api").replace(/\/$/, "");
  const headers: Record<string, string> = {
    Accept: "text/event-stream",
    "Content-Type": "application/json",
  };
  if (options.idempotencyKey) headers["Idempotency-Key"] = options.idempotencyKey;
  let response: Response;
  try {
    response = await fetch(`${apiBaseUrl}/v1/chat/stream`, {
      method: "POST",
      headers,
      body: JSON.stringify(request),
      signal: options.signal,
    });
  } catch (error) {
    if (options.signal?.aborted) throw error;
    throw new ChatStreamHttpError(
      0,
      "API_UNREACHABLE",
      "无法连接 Career Agent 服务，请确认 FastAPI 已启动。",
    );
  }
  if (!response.ok) throw await errorFromResponse(response);
  if (!response.body) {
    throw new ChatStreamHttpError(502, "STREAM_BODY_MISSING", "服务没有返回流式响应。");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  const parser = new SseParser();
  try {
    while (true) {
      const { done, value } = await reader.read();
      const frames = parser.push(decoder.decode(value, { stream: !done }));
      for (const frame of frames) {
        const event = parsePublicStreamEvent(JSON.parse(frame.data));
        if (frame.event !== "message" && frame.event !== event.type) {
          throw new Error("SSE_EVENT_TYPE_MISMATCH");
        }
        yield event;
      }
      if (done) break;
    }
    for (const frame of parser.finish()) {
      const event = parsePublicStreamEvent(JSON.parse(frame.data));
      yield event;
    }
  } finally {
    reader.releaseLock();
  }
}
