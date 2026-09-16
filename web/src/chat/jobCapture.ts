import {
  acknowledgeJobCapture,
  fetchConversationMessages,
  type ConversationTranscript,
  type JobCapturedEventView,
} from "../api/client";
import {
  streamChat,
  type ChatStreamOptions,
  type ChatStreamRequest,
  type TurnInputResource,
} from "../api/sse";
import type { PublicStreamEvent } from "./events";

/** Message names shared with the extension's app bridge (`browser-extension/app-bridge.js`). */
export const OPEN_JOB_SEARCH_MESSAGE = "career-agent:open-job-search";
export const OPEN_JOB_SEARCH_RESULT_MESSAGE = "career-agent:open-job-search-result";
export const JOB_CAPTURED_MESSAGE = "career-agent:job-captured";

/** How long to wait for the bridge before falling back to a plain `window.open`. */
export const BRIDGE_TIMEOUT_MS = 800;
/** Pending events are re-read on this cadence; the bridge nudge only makes it sooner. */
export const CAPTURE_POLL_INTERVAL_MS = 15_000;

export interface OpenJobSearchRequest {
  url: string;
  captureIntentId: string | null;
  captureIntentExpiresAt: string | null;
}

/**
 * Ask the extension to open the BOSS search so the capture intent rides with
 * the tab. Resolves `true` when the extension took the request, `false` when
 * no bridge answered in time (extension missing or disabled); the caller then
 * opens the URL itself and the save will land as a plain library save.
 */
export function openJobSearchViaBridge(
  request: OpenJobSearchRequest,
  target: Window = window,
  timeoutMs: number = BRIDGE_TIMEOUT_MS,
): Promise<boolean> {
  const requestId = crypto.randomUUID();
  return new Promise((resolve) => {
    const finish = (ok: boolean): void => {
      target.removeEventListener("message", onMessage);
      target.clearTimeout(timer);
      resolve(ok);
    };
    function onMessage(event: MessageEvent): void {
      if (event.source !== target || event.origin !== target.location.origin) return;
      const data = event.data as { type?: unknown; request_id?: unknown; ok?: unknown } | null;
      if (!data || data.type !== OPEN_JOB_SEARCH_RESULT_MESSAGE || data.request_id !== requestId) {
        return;
      }
      finish(data.ok === true);
    }
    const timer = target.setTimeout(() => finish(false), timeoutMs);
    target.addEventListener("message", onMessage);
    target.postMessage(
      {
        type: OPEN_JOB_SEARCH_MESSAGE,
        request_id: requestId,
        url: request.url,
        capture_intent_id: request.captureIntentId,
        capture_intent_expires_at: request.captureIntentExpiresAt,
      },
      target.location.origin,
    );
  });
}

/** The user-visible message that opens the follow-up turn for a captured job. */
export function captureFollowUpMessage(event: JobCapturedEventView): string {
  return (
    `我已经从 BOSS 保存了岗位「${event.title} · ${event.company_name}」。` +
    `请基于这份 JD 继续分析。`
  );
}

/** The exact JD version just saved, verified server-side against the caller before the turn runs. */
export function captureInputResources(event: JobCapturedEventView): TurnInputResource[] {
  return [{ kind: "jd_snapshot", id: event.jd_snapshot_id }];
}

export function captureConversationAcceptsInput(
  transcript: Pick<ConversationTranscript, "active_workflow" | "phase" | "pending_interaction">,
): boolean {
  if (transcript.pending_interaction) return false;
  if (transcript.active_workflow !== "mock_interview") return true;
  return transcript.phase === "mock_interview_checkpoint_missing"
    || transcript.phase === "mock_interview_graph_incompatible";
}

export async function* streamCaptureTurn(
  request: ChatStreamRequest,
  captureEventId: string,
  options: ChatStreamOptions = {},
): AsyncGenerator<PublicStreamEvent> {
  for await (const event of streamChat(request, { ...options, idempotencyKey: captureEventId })) {
    if (event.type === "turn_completed" || event.type === "turn_suspended") {
      try {
        await acknowledgeJobCapture(captureEventId, {
          apiBaseUrl: (options.apiBaseUrl ?? "/api").replace(/\/$/, ""),
          signal: options.signal,
        });
      } catch {
        // A pending event retries with the same key and replays the committed turn.
      }
    }
    yield event;
  }
}

export type CaptureContinuationResult =
  | { status: "waiting"; events: [] }
  | { status: "failed"; events: PublicStreamEvent[] }
  | { status: "committed"; events: PublicStreamEvent[] };

export async function continuePendingJobCapture(
  event: JobCapturedEventView,
  options: ChatStreamOptions = {},
): Promise<CaptureContinuationResult> {
  const apiBaseUrl = (options.apiBaseUrl ?? "/api").replace(/\/$/, "");
  const transcript = await fetchConversationMessages(event.conversation_id, {
    apiBaseUrl,
    signal: options.signal,
  });
  if (!captureConversationAcceptsInput(transcript)) return { status: "waiting", events: [] };

  const events: PublicStreamEvent[] = [];
  for await (const streamEvent of streamCaptureTurn(
    {
      conversation_id: event.conversation_id,
      message: captureFollowUpMessage(event),
      input_resources: captureInputResources(event),
    },
    event.id,
    { ...options, apiBaseUrl },
  )) {
    events.push(streamEvent);
  }
  return events.some(
    (streamEvent) =>
      streamEvent.type === "turn_completed" || streamEvent.type === "turn_suspended",
  )
    ? { status: "committed", events }
    : { status: "failed", events };
}

/** Events not yet taken up, oldest first, skipping ones already in flight. */
export function nextCaptureEvents(
  events: readonly JobCapturedEventView[],
  inFlight: ReadonlySet<string>,
  attempted?: ReadonlySet<string>,
): JobCapturedEventView[] {
  return events
    .filter((event) => !inFlight.has(event.id) && !attempted?.has(event.id))
    .sort((left, right) => left.created_at.localeCompare(right.created_at));
}
