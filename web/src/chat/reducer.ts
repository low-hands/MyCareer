import type { ReportKind } from "./events";
import type {
  ArtifactReadyEvent,
  ClientActionEvent,
  InteractionRequiredEvent,
  PublicStreamEvent,
} from "./types";

export type ChatPhase = "idle" | "running" | "awaiting_input" | "completed" | "failed";

export interface MessageResource {
  kind: ReportKind;
  resourceId: string;
  statusAtDelivery?: "current" | "outdated" | "superseded" | null;
  anchoredByOtherJob?: boolean | null;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  /**
   * The stored report this message only summarizes, when it has one.
   *
   * Attached to the message rather than kept in a turn-level list so a
   * restored transcript and a live turn agree: after a reload the reference
   * comes back on the message it belonged to, and several reports across a
   * conversation each stay next to the reply that produced them.
   */
  resource?: MessageResource;
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
  | { type: "submit"; messageId: string; assistantMessageId: string; content: string }
  | {
      type: "hydrate";
      messages: ChatMessage[];
      interaction?: InteractionRequiredEvent | null;
      awaitingInput?: boolean;
    }
  | { type: "stream_event"; event: PublicStreamEvent }
  | { type: "transport_failed"; message: string }
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
        { id: action.messageId, role: "user", content: action.content },
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
  if (action.type === "transport_failed") {
    return {
      ...state,
      phase: "failed",
      progress: null,
      error: action.message,
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
    case "report_ready":
      return {
        ...state,
        messages: state.messages.map((message) =>
          message.id === state.activeAssistantMessageId
            ? {
                ...message,
                resource: {
                  kind: event.kind,
                  resourceId: event.resource_id,
                  statusAtDelivery: event.status_at_delivery,
                  anchoredByOtherJob: event.anchored_by_other_job,
                },
              }
            : message,
        ),
      };
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
