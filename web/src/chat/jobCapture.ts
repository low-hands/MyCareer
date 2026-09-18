import type { JobCapturedEventView } from "../api/client";

/** Message names shared with the extension's app bridge (`browser-extension/app-bridge.js`). */
export const OPEN_JOB_SEARCH_MESSAGE = "career-agent:open-job-search";
export const OPEN_JOB_SEARCH_RESULT_MESSAGE = "career-agent:open-job-search-result";
export const JOB_CAPTURED_MESSAGE = "career-agent:job-captured";

/** How long to wait for the bridge before falling back to a plain `window.open`. */
export const BRIDGE_TIMEOUT_MS = 800;
/** Pending events are re-read on this cadence; the bridge nudge only makes it sooner. */
export const CAPTURE_POLL_INTERVAL_MS = 15_000;

export function captureDeliveryReady(event: JobCapturedEventView): boolean {
  return event.continuation_status === "completed" || event.continuation_status === "discarded";
}

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
