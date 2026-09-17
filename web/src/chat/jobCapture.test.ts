import { afterEach, describe, expect, it, vi } from "vitest";

import type { ConversationTranscript, JobCapturedEventView } from "../api/client";
import type { PublicStreamEvent } from "./events";
import {
  captureConversationAcceptsInput,
  captureFollowUpMessage,
  captureInputResources,
  continuePendingJobCapture,
  nextCaptureEvents,
  streamCaptureTurn,
} from "./jobCapture";

const capture: JobCapturedEventView = {
  id: "jobcap-1",
  conversation_id: "c1",
  job_posting_id: "job-1",
  jd_snapshot_id: "snapshot-1",
  title: "AI 产品经理",
  company_name: "示例科技",
  created_at: "2026-09-12T12:00:00Z",
};
const request = {
  conversation_id: capture.conversation_id,
  message: captureFollowUpMessage(capture),
  input_resources: captureInputResources(capture),
};
const started: PublicStreamEvent = { type: "turn_started", turn_id: "turn-1" };
const completed: PublicStreamEvent = { type: "turn_completed", turn_id: "turn-1" };
const suspended: PublicStreamEvent = {
  type: "turn_suspended",
  turn_id: "turn-1",
  reason: "interaction_required",
  interaction_id: "interaction-1",
};
const interaction: PublicStreamEvent = {
  type: "interaction_required",
  interaction_id: "interaction_0123456789abcdef0123",
  kind: "single_selection",
  prompt: "请回答当前面试问题",
  options: [],
  allow_free_text: true,
};

function response(events: PublicStreamEvent[]): Response {
  return new Response(events.map((event) => `data: ${JSON.stringify(event)}\n\n`).join(""));
}

function transcript(
  overrides: Partial<ConversationTranscript> = {},
): Response {
  return Response.json({
    messages: [],
    active_workflow: null,
    phase: null,
    pending_interaction: null,
    pending_interaction_body: null,
    ...overrides,
  });
}

async function consume(): Promise<PublicStreamEvent[]> {
  const events: PublicStreamEvent[] = [];
  for await (const event of streamCaptureTurn(request, capture.id)) events.push(event);
  return events;
}

afterEach(() => vi.unstubAllGlobals());

describe("capture follow-up", () => {
  it("records the save and offers analysis instead of asking for it", () => {
    const message = captureFollowUpMessage(capture);
    expect(message).toContain("示例科技");
    expect(message).toContain("让 Agent 分析");
    expect(message).not.toContain("继续分析");
    expect(captureInputResources(capture)).toEqual([{ kind: "jd_snapshot", id: "snapshot-1" }]);
  });
});

describe("capture acknowledgements", () => {
  it.each([completed, suspended])("acknowledges only after $type", async (terminal) => {
    const events: PublicStreamEvent[] = [
      started,
      { type: "progress", stage: "loading_context", message: "loading" },
      { type: "content_delta", delta: "reply before commit" },
      { type: "progress", stage: "saving", message: "saving" },
      terminal,
    ];
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(response(events))
      .mockResolvedValueOnce(Response.json({ acknowledged: true }));
    vi.stubGlobal("fetch", fetchMock);
    const signal = new AbortController().signal;
    const stream = streamCaptureTurn(request, capture.id, { apiBaseUrl: "/api/", signal });

    for (const event of events.slice(0, -1)) {
      expect((await stream.next()).value).toEqual(event);
      expect(fetchMock).toHaveBeenCalledTimes(1);
    }
    expect((await stream.next()).value).toEqual(terminal);
    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(fetchMock).toHaveBeenLastCalledWith(
      `/api/v1/job-captures/events/${capture.id}/ack`,
      expect.objectContaining({ method: "POST", signal }),
    );
    expect((await stream.next()).done).toBe(true);
    expect(JSON.parse(fetchMock.mock.calls[0][1].body)).toEqual(request);
    expect(fetchMock.mock.calls[0][1].headers["Idempotency-Key"]).toBe(capture.id);
  });

  it.each(["INPUT_RESOURCE_REJECTED", "TURN_COMMIT_FAILED"])(
    "keeps the capture pending after %s and retries the same request",
    async (code) => {
      const failed: PublicStreamEvent[] = [
        started,
        { type: "progress", stage: "loading_context", message: "loading" },
        ...(code === "TURN_COMMIT_FAILED"
          ? [{ type: "content_delta" as const, delta: "not yet stored" }]
          : []),
        { type: "turn_failed", turn_id: "turn-1", code, message: "failed" },
      ];
      const fetchMock = vi.fn()
        .mockResolvedValueOnce(response(failed))
        .mockResolvedValueOnce(response([started, completed]))
        .mockResolvedValueOnce(Response.json({ acknowledged: true }));
      vi.stubGlobal("fetch", fetchMock);

      expect(await consume()).toEqual(failed);
      expect(fetchMock).toHaveBeenCalledTimes(1);
      expect(await consume()).toEqual([started, completed]);
      expect(fetchMock).toHaveBeenCalledTimes(3);
      for (const [, init] of fetchMock.mock.calls.slice(0, 2)) {
        expect(init.headers["Idempotency-Key"]).toBe(capture.id);
        expect(JSON.parse(init.body)).toEqual(request);
      }
    },
  );

  it("leaves an interrupted stream unacknowledged", async () => {
    let reads = 0;
    const body = new ReadableStream<Uint8Array>({
      pull(controller) {
        if (reads++ === 0) {
          controller.enqueue(new TextEncoder().encode(`data: ${JSON.stringify(started)}\n\n`));
        } else {
          controller.error(new Error("connection lost"));
        }
      },
    });
    const fetchMock = vi.fn().mockResolvedValueOnce(new Response(body));
    vi.stubGlobal("fetch", fetchMock);

    await expect(consume()).rejects.toThrow("connection lost");
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("does not acknowledge a stream that ends without a committed terminal event", async () => {
    const fetchMock = vi.fn().mockResolvedValueOnce(response([started]));
    vi.stubGlobal("fetch", fetchMock);

    expect(await consume()).toEqual([started]);
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("retries a failed ACK through the committed request's replay", async () => {
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(response([started, completed]))
      .mockResolvedValueOnce(new Response(null, { status: 503 }))
      .mockResolvedValueOnce(response([started, completed]))
      .mockResolvedValueOnce(Response.json({ acknowledged: true }));
    vi.stubGlobal("fetch", fetchMock);

    expect(await consume()).toEqual([started, completed]);
    expect(await consume()).toEqual([started, completed]);
    expect(fetchMock).toHaveBeenCalledTimes(4);
    expect(fetchMock.mock.calls[0][1].headers["Idempotency-Key"]).toBe(capture.id);
    expect(fetchMock.mock.calls[2][1].headers["Idempotency-Key"]).toBe(capture.id);
  });
});

describe("capture retry scheduling", () => {
  it.each([
    [410, "JOB_CAPTURE_CONVERSATION_UNAVAILABLE"],
    [404, "JOB_CAPTURE_NOT_FOUND"],
  ])("discards a cached capture rejected with %s after deletion", async (status, code) => {
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(transcript())
      .mockResolvedValueOnce(Response.json(
        { detail: { code, message: "原会话不可用" } }, { status: Number(status) },
      ));
    vi.stubGlobal("fetch", fetchMock);

    expect(await continuePendingJobCapture(capture)).toEqual({
      status: "discarded", events: [],
    });
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it("does not discard captures on a temporary server failure", async () => {
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(transcript())
      .mockResolvedValueOnce(Response.json(
        { detail: { code: "TURN_CAPACITY_EXHAUSTED" } }, { status: 503 },
      ));
    vi.stubGlobal("fetch", fetchMock);

    await expect(continuePendingJobCapture(capture)).rejects.toMatchObject({ status: 503 });
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it("matches the runtime phases that release mock-interview input", () => {
    expect(captureConversationAcceptsInput({
      active_workflow: "mock_interview",
      phase: "mock_interview_answer_required",
      pending_interaction: null,
    })).toBe(false);
    expect(captureConversationAcceptsInput({
      active_workflow: "mock_interview",
      phase: "mock_interview_checkpoint_missing",
      pending_interaction: null,
    })).toBe(true);
    expect(captureConversationAcceptsInput({
      active_workflow: null,
      phase: null,
      pending_interaction: interaction,
    })).toBe(false);
  });

  it("polls occupied conversations without repeatedly posting the capture", async () => {
    const fetchMock = vi.fn()
      .mockImplementation(() => Promise.resolve(transcript({
        active_workflow: "mock_interview",
        phase: "mock_interview_answer_required",
      })));
    vi.stubGlobal("fetch", fetchMock);

    expect((await continuePendingJobCapture(capture)).status).toBe("waiting");
    expect((await continuePendingJobCapture(capture)).status).toBe("waiting");
    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(fetchMock.mock.calls.every(([url]) => String(url).includes("/messages"))).toBe(true);
  });

  it("waits after an ownership race and resumes with the same request id", async () => {
    const failed: PublicStreamEvent[] = [
      started,
      { type: "progress", stage: "loading_context", message: "loading" },
      {
        type: "turn_failed",
        turn_id: "turn-1",
        code: "INPUT_RESOURCE_REJECTED",
        message: "interview owns the conversation",
      },
    ];
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(transcript())
      .mockResolvedValueOnce(response(failed))
      .mockResolvedValueOnce(transcript({
        active_workflow: "mock_interview",
        phase: "mock_interview_answer_required",
      }))
      .mockResolvedValueOnce(transcript())
      .mockResolvedValueOnce(response([started, completed]))
      .mockResolvedValueOnce(Response.json({ acknowledged: true }));
    vi.stubGlobal("fetch", fetchMock);

    expect(await continuePendingJobCapture(capture)).toEqual({ status: "failed", events: failed });
    expect((await continuePendingJobCapture(capture)).status).toBe("waiting");
    expect((await continuePendingJobCapture(capture)).status).toBe("committed");

    const streamCalls = fetchMock.mock.calls.filter(
      ([url]) => String(url).endsWith("/v1/chat/stream"),
    );
    expect(streamCalls).toHaveLength(2);
    expect(streamCalls.every(([, init]) => init.headers["Idempotency-Key"] === capture.id)).toBe(true);
  });

  it("defers rejected captures until the next poll without blocking other captures", () => {
    const later = { ...capture, id: "jobcap-2", created_at: "2026-09-12T12:01:00Z" };
    const events = [later, capture];
    const attempted = new Set([capture.id]);
    const inFlight = new Set<string>();

    expect(nextCaptureEvents(events, inFlight, attempted)).toEqual([later]);
    attempted.clear();
    inFlight.add(capture.id);
    expect(nextCaptureEvents(events, inFlight, attempted)).toEqual([later]);
    inFlight.clear();
    expect(nextCaptureEvents(events, inFlight, attempted)).toEqual([capture, later]);
    expect(events).toEqual([later, capture]);
  });
});
