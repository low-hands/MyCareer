import type {
  ArtifactReadyEvent,
  ClientActionEvent,
  InteractionRequiredEvent,
  PublicStreamEvent,
} from "./types";

export type ChatPhase = "idle" | "running" | "awaiting_input" | "completed" | "failed";

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
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
  | { type: "hydrate"; messages: ChatMessage[]; awaitingInput?: boolean }
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
      phase: action.awaitingInput ? "awaiting_input" : action.messages.length ? "completed" : "idle",
      messages: action.messages,
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
