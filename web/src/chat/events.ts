/*
 * The closed sets below are runtime arrays rather than bare type unions so the
 * contract test (src/contracts/contract.test.ts) can compare them with the
 * enums the backend exports. A union alone is erased at compile time and could
 * silently lag a value the server had already started sending.
 */

export const PROGRESS_STAGES = [
  "loading_context",
  "deciding",
  "running_capability",
  "presenting",
  "saving",
] as const;
export type ProgressStage = (typeof PROGRESS_STAGES)[number];

export const CAPABILITIES = [
  "job_search",
  "job_research",
  "resume",
  "application_tracking",
  "interview",
  "calendar",
  "action_center",
  "career_task",
] as const;
export type Capability = (typeof CAPABILITIES)[number];

export const REPORT_KIND_LIST = [
  "job_research_report",
  "mock_interview_report",
  "interview_preparation",
  "interview_retro_report",
  "resume_job_match",
  "resume_tailoring_draft",
  "job_analysis",
] as const;
export type ReportKind = (typeof REPORT_KIND_LIST)[number];
export type ReportResourceKind = ReportKind | "delivered_body";
export const REPORT_KINDS = new Set<ReportKind>(REPORT_KIND_LIST);

export const REPORT_DELIVERY_STATUSES = ["current", "outdated", "superseded"] as const;
export type ReportDeliveryStatus = (typeof REPORT_DELIVERY_STATUSES)[number];

export const INTERACTION_KINDS = [
  "single_selection",
  "multiple_selection",
  "confirmation",
  "free_text",
  "approval",
  "file_upload",
  "questionnaire",
] as const;
export type InteractionKind = (typeof INTERACTION_KINDS)[number];

/**
 * Scopes whose buttons send a bound `InteractionResponse` back rather than
 * prose. The server routes the reply by this value, so an unknown scope must
 * fail at the parse boundary instead of producing a button that posts a
 * response the server will reject.
 */
export const INTERACTION_SCOPES = [
  "capability_confirmation",
  "questionnaire",
] as const;
export type InteractionScope = (typeof INTERACTION_SCOPES)[number];

export interface InteractionOption {
  label: string;
  description?: string | null;
  selection_index?: number | null;
  value?: string | null;
}

export interface UserQuestion {
  question_id: string;
  prompt: string;
  kind: "single" | "multiple" | "free_text";
  options: { value: string; label: string; meaning: "choice" | "none" | "other" }[];
  allow_free_text: boolean;
  allow_skip: boolean;
}

export type PublicStreamEvent =
  | { type: "turn_started"; turn_id: string }
  | {
      type: "progress";
      stage: ProgressStage;
      message: string;
      step_key?: string | null;
      step_label?: string | null;
    }
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
      kind: InteractionKind;
      prompt: string;
      options: InteractionOption[];
      questions?: UserQuestion[];
      allow_free_text: boolean;
      scope?: InteractionScope | null;
      /** The selection can also be answered by uploading this kind of file. */
      accepts_upload?: "resume" | null;
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
      type: "report_ready";
      kind: ReportKind;
      resource_id: string;
      status_at_delivery?: ReportDeliveryStatus | null;
      anchored_by_other_job?: boolean | null;
    }
  | {
      type: "job_resource_ready";
      kind: "saved_job";
      /** The immutable `jd_snapshot_id` the card opens, not the posting. */
      resource_id: string;
      job_posting_id: string;
      title?: string | null;
      description?: string | null;
    }
  | {
      type: "client_action";
      action: "open_url";
      url: string;
      label: string;
      /** Opaque capture intent to bind to the opened tab; never part of `url`. */
      capture_intent_id?: string | null;
      capture_intent_expires_at?: string | null;
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

export const EVENT_TYPES = [
  "turn_started",
  "progress",
  "capability_started",
  "capability_completed",
  "interaction_required",
  "content_delta",
  "artifact_ready",
  "report_ready",
  "job_resource_ready",
  "client_action",
  "turn_suspended",
  "turn_completed",
  "turn_failed",
] as const satisfies readonly PublicStreamEvent["type"][];

// Compile-time exhaustiveness: adding a variant to PublicStreamEvent without
// listing it here fails to type-check.
type MissingEventType = Exclude<PublicStreamEvent["type"], (typeof EVENT_TYPES)[number]>;
const _everyEventTypeListed: MissingEventType extends never ? true : never = true;
void _everyEventTypeListed;

const EVENT_TYPE_SET = new Set<string>(EVENT_TYPES);
const INTERACTION_KIND_SET = new Set<string>(INTERACTION_KINDS);
const INTERACTION_SCOPE_SET = new Set<string>(INTERACTION_SCOPES);
const REPORT_DELIVERY_STATUS_SET = new Set<string>(REPORT_DELIVERY_STATUSES);

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

export function parsePublicStreamEvent(value: unknown): PublicStreamEvent {
  if (!isRecord(value) || typeof value.type !== "string") {
    throw new Error("SSE_EVENT_INVALID");
  }
  if (!EVENT_TYPE_SET.has(value.type)) {
    throw new Error("SSE_EVENT_UNSUPPORTED");
  }

  switch (value.type) {
    case "turn_started":
    case "turn_completed":
      if (typeof value.turn_id !== "string") throw new Error("SSE_EVENT_INVALID");
      break;
    case "progress":
      // Stage and capability only choose a status line, never a code path, so
      // an unknown value is not worth failing the turn over: it is accepted
      // here and the contract test reports the set mismatch by name.
      if (typeof value.stage !== "string" || typeof value.message !== "string") {
        throw new Error("SSE_EVENT_INVALID");
      }
      if ((value.step_key == null) !== (value.step_label == null)) {
        throw new Error("SSE_EVENT_INVALID");
      }
      if (value.step_key != null && (
        typeof value.step_key !== "string" || typeof value.step_label !== "string"
      )) {
        throw new Error("SSE_EVENT_INVALID");
      }
      break;
    case "capability_started":
    case "capability_completed":
      if (typeof value.capability !== "string" || typeof value.message !== "string") {
        throw new Error("SSE_EVENT_INVALID");
      }
      if (value.type === "capability_completed" && typeof value.state !== "string") {
        throw new Error("SSE_EVENT_INVALID");
      }
      break;
    case "interaction_required":
      // The kind picks the widget and the scope decides whether a button posts
      // a bound response; both are closed sets the server routes on.
      if (
        typeof value.interaction_id !== "string" ||
        typeof value.kind !== "string" ||
        !INTERACTION_KIND_SET.has(value.kind) ||
        typeof value.prompt !== "string" ||
        !Array.isArray(value.options) ||
        (value.kind === "questionnaire" &&
          (value.scope !== "questionnaire" || !Array.isArray(value.questions)
            || value.questions.length < 2 || value.questions.length > 8
            || !value.questions.every((question: unknown, index: number) =>
              isRecord(question)
              && question.question_id === `q${index + 1}`
              && typeof question.prompt === "string"
              && ["single", "multiple", "free_text"].includes(String(question.kind))
              && Array.isArray(question.options)
              && question.options.every((option: unknown) => isRecord(option)
                && typeof option.value === "string" && typeof option.label === "string"
                && ["choice", "none", "other"].includes(String(option.meaning)))
              && typeof question.allow_free_text === "boolean"
              && typeof question.allow_skip === "boolean"))) ||
        typeof value.allow_free_text !== "boolean" ||
        (value.accepts_upload != null && value.accepts_upload !== "resume") ||
        (value.scope != null &&
          (typeof value.scope !== "string" || !INTERACTION_SCOPE_SET.has(value.scope)))
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
    case "report_ready":
      // The kind is checked against the closed set rather than accepted as any
      // string: it selects which read endpoint the card fetches, so an unknown
      // one would produce a card that can never resolve.
      if (
        typeof value.resource_id !== "string" ||
        typeof value.kind !== "string" ||
        !REPORT_KINDS.has(value.kind as ReportKind) ||
        (value.status_at_delivery != null &&
          (typeof value.status_at_delivery !== "string" ||
            !REPORT_DELIVERY_STATUS_SET.has(value.status_at_delivery))) ||
        (value.anchored_by_other_job != null &&
          typeof value.anchored_by_other_job !== "boolean")
      ) {
        throw new Error("SSE_EVENT_INVALID");
      }
      break;
    case "job_resource_ready":
      if (
        value.kind !== "saved_job" ||
        typeof value.resource_id !== "string" ||
        typeof value.job_posting_id !== "string" ||
        (value.title != null && typeof value.title !== "string") ||
        (value.description != null && typeof value.description !== "string")
      ) {
        throw new Error("SSE_EVENT_INVALID");
      }
      break;
    case "client_action":
      if (
        value.action !== "open_url" ||
        typeof value.url !== "string" ||
        typeof value.label !== "string" ||
        (value.capture_intent_id != null && typeof value.capture_intent_id !== "string") ||
        (value.capture_intent_expires_at != null &&
          typeof value.capture_intent_expires_at !== "string")
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
