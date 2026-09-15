import type { ConversationTranscript } from "../api/client";
import type { ChatAction, ChatMessage } from "./reducer";

/** How many times, and how far apart, a dropped stream re-reads the transcript. */
export const RECOVERY_ATTEMPTS = 8;
export const RECOVERY_INTERVAL_MS = 4000;

export type Hydration = Omit<Extract<ChatAction, { type: "hydrate" }>, "type">;

/** The reducer's `hydrate` payload for a transcript, shared by the initial
 * load and by recovery so both restore exactly the same view. */
export function hydrationFrom(
  conversationId: string,
  transcript: ConversationTranscript,
): Hydration {
  const messages: ChatMessage[] = transcript.messages.map((message, index) => ({
    id: `history-${conversationId}-${index}`,
    role: message.role,
    content: message.content,
    resources: message.resources.map((resource) => ({
      kind: resource.kind,
      resourceId: resource.resource_id,
      statusAtDelivery: resource.status_at_delivery,
      anchoredByOtherJob: resource.anchored_by_other_job,
      title: resource.title,
      ...(resource.kind === "resume_version"
        ? {
            description: resource.description ?? null,
            available: resource.available ?? null,
            resumeId: resource.resume_id ?? null,
          }
        : {}),
    })),
  }));
  if (transcript.pending_interaction_body) {
    messages.push({
      id: `pending-${conversationId}`,
      role: "assistant",
      content: transcript.pending_interaction_body,
      resources: [],
    });
  }
  return {
    messages,
    interaction: transcript.pending_interaction,
    awaitingInput: Boolean(transcript.active_workflow),
  };
}

/**
 * Whether the server has finished the turn that `sentMessage` started.
 *
 * A turn stores the request and its reply in one write, so the reply is there
 * exactly when the last user message is ours and something follows it. The
 * stored request may be a clipped copy of a long paste, hence the prefix test.
 * A pending interaction also counts: the turn reached its question and is
 * waiting on us, which the hydrated view shows as such.
 *
 * Turns inside a running workflow (mock interview) never reach the transcript,
 * so this cannot confirm those; the caller then reports "unconfirmed".
 */
export function turnIsStored(
  transcript: ConversationTranscript,
  sentMessage: string,
): boolean {
  if (transcript.pending_interaction) return true;
  const messages = transcript.messages;
  let last = messages.length - 1;
  while (last >= 0 && messages[last].role !== "user") last -= 1;
  if (last === -1 || last === messages.length - 1) return false;
  const stored = messages[last].content;
  return sentMessage.startsWith(stored) || stored.startsWith(sentMessage);
}

export function sleep(ms: number, signal: AbortSignal): Promise<void> {
  return new Promise((resolve) => {
    if (signal.aborted) {
      resolve();
      return;
    }
    const timer = setTimeout(done, ms);
    function done(): void {
      clearTimeout(timer);
      signal.removeEventListener("abort", done);
      resolve();
    }
    signal.addEventListener("abort", done, { once: true });
  });
}
