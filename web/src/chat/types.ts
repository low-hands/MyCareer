import type { PublicStreamEvent } from "./events";

export type InteractionRequiredEvent = Extract<
  PublicStreamEvent,
  { type: "interaction_required" }
>;

export type ArtifactReadyEvent = Extract<
  PublicStreamEvent,
  { type: "artifact_ready" }
>;

export type ClientActionEvent = Extract<
  PublicStreamEvent,
  { type: "client_action" }
>;

export type { PublicStreamEvent } from "./events";
