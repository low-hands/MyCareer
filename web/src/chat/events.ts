export type ProgressStage =
  | "loading_context"
  | "deciding"
  | "running_capability"
  | "presenting"
  | "saving";

export type Capability =
  | "job_search"
  | "job_research"
  | "resume"
  | "application_tracking"
  | "interview"
  | "calendar"
  | "action_center"
  | "career_task";

export interface InteractionOption {
  label: string;
  description?: string;
  selection_index?: number;
  value?: string;
}

export type PublicStreamEvent =
  | { type: "turn_started"; turn_id: string }
  | { type: "progress"; stage: ProgressStage; message: string }
  | {
      type: "capability_started";
      capability: Capability;
      message: string;
    }
  | {
      type: "capability_completed";
      capability: Capability;
      state: string;
      message: string;
    }
  | {
      type: "interaction_required";
      interaction_id: string;
      kind:
        | "single_selection"
        | "multiple_selection"
        | "confirmation"
        | "free_text"
        | "approval"
        | "file_upload";
      prompt: string;
      options: InteractionOption[];
      allow_free_text: boolean;
    }
  | { type: "content_delta"; delta: string }
  | {
      type: "artifact_ready";
      artifact_id: string;
      filename: string;
      media_type: string;
      byte_size: number;
    }
  | {
      type: "client_action";
      action: "open_url";
      url: string;
      label: string;
    }
  | {
      type: "turn_suspended";
      turn_id: string;
      reason: "interaction_required";
      interaction_id: string;
    }
  | { type: "turn_completed"; turn_id: string }
  | {
      type: "turn_failed";
      turn_id: string;
      code: string;
      message: string;
    };

const EVENT_TYPES = new Set<PublicStreamEvent["type"]>([
  "turn_started",
  "progress",
  "capability_started",
  "capability_completed",
  "interaction_required",
  "content_delta",
  "artifact_ready",
  "client_action",
  "turn_suspended",
  "turn_completed",
  "turn_failed",
]);

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

export function parsePublicStreamEvent(value: unknown): PublicStreamEvent {
  if (!isRecord(value) || typeof value.type !== "string") {
    throw new Error("SSE_EVENT_INVALID");
  }
  if (!EVENT_TYPES.has(value.type as PublicStreamEvent["type"])) {
    throw new Error("SSE_EVENT_UNSUPPORTED");
  }

  switch (value.type) {
    case "turn_started":
    case "turn_completed":
      if (typeof value.turn_id !== "string") throw new Error("SSE_EVENT_INVALID");
      break;
    case "progress":
      if (typeof value.stage !== "string" || typeof value.message !== "string") {
        throw new Error("SSE_EVENT_INVALID");
      }
      break;
    case "capability_started":
    case "capability_completed":
      if (typeof value.capability !== "string" || typeof value.message !== "string") {
        throw new Error("SSE_EVENT_INVALID");
      }
      break;
    case "interaction_required":
      if (
        typeof value.interaction_id !== "string" ||
        typeof value.kind !== "string" ||
        typeof value.prompt !== "string" ||
        !Array.isArray(value.options) ||
        typeof value.allow_free_text !== "boolean"
      ) {
        throw new Error("SSE_EVENT_INVALID");
      }
      break;
    case "content_delta":
      if (typeof value.delta !== "string") throw new Error("SSE_EVENT_INVALID");
      break;
    case "artifact_ready":
      if (
        typeof value.artifact_id !== "string" ||
        typeof value.filename !== "string" ||
        typeof value.media_type !== "string" ||
        typeof value.byte_size !== "number"
      ) {
        throw new Error("SSE_EVENT_INVALID");
      }
      break;
    case "client_action":
      if (
        value.action !== "open_url" ||
        typeof value.url !== "string" ||
        typeof value.label !== "string"
      ) {
        throw new Error("SSE_EVENT_INVALID");
      }
      break;
    case "turn_suspended":
      if (typeof value.turn_id !== "string" || typeof value.interaction_id !== "string") {
        throw new Error("SSE_EVENT_INVALID");
      }
      break;
    case "turn_failed":
      if (
        typeof value.turn_id !== "string" ||
        typeof value.code !== "string" ||
        typeof value.message !== "string"
      ) {
        throw new Error("SSE_EVENT_INVALID");
      }
      break;
  }

  return value as PublicStreamEvent;
}
