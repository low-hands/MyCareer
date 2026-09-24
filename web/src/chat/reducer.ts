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
  /** The posting a `saved_job` card's pinned JD version belongs to. */
  jobPostingId?: string | null;
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

/**
 * Avoid showing an attachment twice when an older stored turn pinned the exact
 * same resource to both the user message and its immediately following reply.
 * A later, independent mention remains visible because only a user/assistant
 * pair is considered.
 */
export function visibleMessageResources(
  messages: ChatMessage[],
  index: number,
): MessageResource[] {
  const message = messages[index];
  const resources = message?.resources ?? [];
  if (!message || message.role !== "assistant" || index === 0) return resources;

  const previous = messages[index - 1];
  if (previous?.role !== "user" || !previous.resources?.length) return resources;
  const previousKeys = new Set(
    previous.resources.map((resource) => `${resource.kind}:${resource.resourceId}`),
  );
  return resources.filter(
    (resource) => !previousKeys.has(`${resource.kind}:${resource.resourceId}`),
  );
}

export interface ProgressStep {
  key: string;
  label: string;
  occurrence: number;
  completed: boolean;
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
  progressSteps: ProgressStep[];
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
  progressSteps: [],
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
      progressSteps: [],
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
      progressSteps: [],
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
      progressSteps: [],
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
    case "progress": {
      if (!event.step_key || !event.step_label) {
        return { ...state, progress: event.message };
      }
      const stepKey = event.step_key;
      const stepLabel = event.step_label;
      const previous = state.progressSteps.at(-1);
      if (previous?.key === stepKey) {
        return {
          ...state,
          progress: event.message,
          progressSteps: state.progressSteps.map((step, index) =>
            index === state.progressSteps.length - 1
              ? { ...step, label: stepLabel }
              : step,
          ),
        };
      }
      const occurrence = state.progressSteps.reduce(
        (highest, step) => step.key === stepKey
          ? Math.max(highest, step.occurrence)
          : highest,
        0,
      ) + 1;
      return {
        ...state,
        progress: event.message,
        progressSteps: [
          ...state.progressSteps.map((step) => ({ ...step, completed: true })),
          {
            key: stepKey,
            label: stepLabel,
            occurrence,
            completed: false,
          },
        ].slice(-6),
      };
    }
    case "capability_started":
      return { ...state, progress: event.message };
    case "capability_completed":
      return {
        ...state,
        progress: event.message,
        progressSteps: state.progressSteps.map((step) => ({ ...step, completed: true })),
      };
    case "content_delta":
      return { ...state, messages: updateActiveMessage(state, event.delta) };
    case "interaction_required":
      // An interaction is already durable when the server publishes it. The
      // following turn_suspended event is useful confirmation, but the form
      // must not remain disabled if that final SSE frame is delayed or lost.
      return { ...state, phase: "awaiting_input", interaction: event, progress: null };
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
    case "job_resource_ready": {
      const resource: MessageResource = {
        kind: event.kind,
        resourceId: event.resource_id,
        jobPostingId: event.job_posting_id,
        title: event.title,
        description: event.description,
        available: true,
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
      return {
        ...state,
        phase: "failed",
        progress: null,
        error: `${event.message}（错误码：${event.code}）`,
      };
  }
}
