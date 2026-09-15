import type { MessageResourceKind } from "../api/client";
import type { ReportDeliveryStatus } from "./events";
import type {
  ArtifactReadyEvent,
  ClientActionEvent,
  InteractionRequiredEvent,
  PublicStreamEvent,
} from "./types";

/**
 * `recovering`: the stream dropped after the turn had started. The server keeps
 * running the turn and stores the reply regardless, so the client is re-reading
 * the transcript rather than reporting a failure it cannot yet confirm.
 */
export type ChatPhase =
  | "idle"
  | "running"
  | "recovering"
  | "awaiting_input"
  | "completed"
  | "failed";

export interface MessageResource {
  kind: MessageResourceKind;
  resourceId: string;
  statusAtDelivery?: ReportDeliveryStatus | null;
  anchoredByOtherJob?: boolean | null;
  /** The card's heading before its body is fetched, when the kind alone does not say. */
  title?: string | null;
  /** Snapshot text kept with the message (format, size, upload time of a resume). */
  description?: string | null;
  /** False once the asset behind a `resume_version` was deleted; undefined when untracked. */
  available?: boolean | null;
  /** Set for a `resume_version` whose resume still exists, for the document link. */
  resumeId?: string | null;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  /**
   * The stored reports this message only summarizes.
   *
   * Attached to the message rather than kept in a turn-level list so a
   * restored transcript and a live turn agree: after a reload the references
   * come back on the message they belonged to, and several reports across a
   * conversation each stay next to the reply that produced them. Plural
   * because one turn can store two reports (a research report and a match);
   * the transcript endpoint returns them as `resources` for the same reason.
   */
  resources?: MessageResource[];
}

/** Appends `resource`, or replaces the entry with the same identity so a
 * re-delivered report updates its card instead of adding a second one. */
function withResource(
  resources: MessageResource[],
  resource: MessageResource,
): MessageResource[] {
  const index = resources.findIndex(
    (item) => item.kind === resource.kind && item.resourceId === resource.resourceId,
  );
  if (index === -1) return [...resources, resource];
  return resources.map((item, position) => (position === index ? resource : item));
}

export interface ChatState {
  phase: ChatPhase;
  messages: ChatMessage[];
  activeAssistantMessageId: string | null;
  progress: string | null;
  interaction: InteractionRequiredEvent | null;
  artifacts: ArtifactReadyEvent[];
  clientActions: ClientActionEvent[];
  error: string | null;
}

export const initialChatState: ChatState = {
  phase: "idle",
  messages: [],
  activeAssistantMessageId: null,
  progress: null,
  interaction: null,
  artifacts: [],
  clientActions: [],
  error: null,
};

export type ChatAction =
  | {
      type: "submit";
      messageId: string;
      assistantMessageId: string;
      content: string;
      /** Resume versions attached to the user message, shown as chips on it. */
      resources?: MessageResource[];
    }
  | {
      type: "hydrate";
      messages: ChatMessage[];
      interaction?: InteractionRequiredEvent | null;
      awaitingInput?: boolean;
    }
  | { type: "stream_event"; event: PublicStreamEvent }
  | { type: "resubmit" }
  | { type: "transport_failed"; message: string }
  | { type: "transport_lost" }
  | { type: "reset" };

function updateActiveMessage(state: ChatState, delta: string): ChatMessage[] {
  return state.messages.map((message) =>
    message.id === state.activeAssistantMessageId
      ? { ...message, content: message.content + delta }
      : message,
  );
}

export function chatReducer(state: ChatState, action: ChatAction): ChatState {
  if (action.type === "reset") return initialChatState;
  if (action.type === "hydrate") {
    return {
      ...initialChatState,
      phase: action.interaction || action.awaitingInput
        ? "awaiting_input"
        : action.messages.length
          ? "completed"
          : "idle",
      messages: action.messages,
      interaction: action.interaction ?? null,
    };
  }
  if (action.type === "submit") {
    return {
      ...state,
      phase: "running",
      messages: [
        ...state.messages,
        {
          id: action.messageId,
          role: "user",
          content: action.content,
          ...(action.resources?.length ? { resources: action.resources } : {}),
        },
        { id: action.assistantMessageId, role: "assistant", content: "" },
      ],
      activeAssistantMessageId: action.assistantMessageId,
      progress: "正在连接 Career Agent……",
      interaction: null,
      artifacts: [],
      clientActions: [],
      error: null,
    };
  }
  if (action.type === "resubmit") {
    // The same request goes out again under the same key, so the failed
    // exchange stays where it is and only its reply is written over.
    return {
      ...state,
      phase: "running",
      messages: state.messages.map((message) =>
        message.id === state.activeAssistantMessageId ? { ...message, content: "" } : message,
      ),
      progress: "正在重新连接 Career Agent……",
      interaction: null,
      artifacts: [],
      clientActions: [],
      error: null,
    };
  }
  if (action.type === "transport_failed") {
    return {
      ...state,
      phase: "failed",
      progress: null,
      error: action.message,
    };
  }
  if (action.type === "transport_lost") {
    return {
      ...state,
      phase: "recovering",
      progress: "连接中断，正在检查回复是否已保存……",
      error: null,
    };
  }

  const event = action.event;
  switch (event.type) {
    case "turn_started":
      return { ...state, phase: "running" };
    case "progress":
    case "capability_started":
    case "capability_completed":
      return { ...state, progress: event.message };
    case "content_delta":
      return { ...state, messages: updateActiveMessage(state, event.delta) };
    case "interaction_required":
      return { ...state, interaction: event, progress: null };
    case "artifact_ready":
      return { ...state, artifacts: [...state.artifacts, event] };
    case "report_ready": {
      const resource: MessageResource = {
        kind: event.kind,
        resourceId: event.resource_id,
        statusAtDelivery: event.status_at_delivery,
        anchoredByOtherJob: event.anchored_by_other_job,
      };
      return {
        ...state,
        messages: state.messages.map((message) =>
          message.id === state.activeAssistantMessageId
            ? { ...message, resources: withResource(message.resources ?? [], resource) }
            : message,
        ),
      };
    }
    case "client_action":
      return { ...state, clientActions: [...state.clientActions, event] };
    case "turn_suspended":
      return { ...state, phase: "awaiting_input", progress: null };
    case "turn_completed":
      return { ...state, phase: "completed", progress: null };
    case "turn_failed":
      return { ...state, phase: "failed", progress: null, error: event.message };
  }
}
