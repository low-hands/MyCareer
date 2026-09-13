/*
 * The client's half of the API contract.
 *
 * `api-contract.ts` is generated from the backend's pydantic models and pinned
 * there by tests/api/test_public_contract.py. This file holds it against what
 * the client actually parses and types, so a shape change on either side goes
 * red here before it reaches a browser. Two kinds of check:
 *
 *  - runtime: every example event parses; every closed set the parser owns
 *    equals the enum the schema declares (both directions, so a value added
 *    on one side only is reported by name);
 *  - compile time: every response example `satisfies` the TypeScript interface
 *    the client reads it through. A field the client requires but the server
 *    no longer sends, or a `null` the client type does not admit, fails
 *    `tsc -b`. This is the check that would have caught the transcript
 *    `resource` → `resources` rename. It only sees a `null` that some example
 *    carries, so the backend test requires every nullable field to be null in
 *    at least one example.
 */

import { describe, expect, it } from "vitest";

import type {
  ApplicationMockInterviews,
  ApplicationView,
  CalendarWorkspace,
  CompanyResearchView,
  ConversationTranscript,
  ConversationView,
  DailyBrief,
  Dashboard,
  EmailWorkspace,
  ReportView,
  ResumeImportResult,
  ResumeView,
  SavedJobDetail,
  SavedJobView,
  TargetRoleView,
} from "../api/client";
import type { ChatStreamRequest } from "../api/sse";
import {
  CAPABILITIES,
  EVENT_TYPES,
  INTERACTION_KINDS,
  INTERACTION_SCOPES,
  PROGRESS_STAGES,
  REPORT_DELIVERY_STATUSES,
  REPORT_KIND_LIST,
  parsePublicStreamEvent,
} from "../chat/events";
import { API_CONTRACT } from "./api-contract";

/** `as const` makes arrays readonly tuples; the client interfaces use mutable
 * arrays. Casting through this keeps the narrow literals while dropping the
 * readonly-ness, which is not a property the wire format has. */
type Writable<T> = T extends readonly (infer U)[]
  ? Writable<U>[]
  : T extends object
    ? { -readonly [K in keyof T]: Writable<T[K]> }
    : T;

function writable<T>(value: T): Writable<T> {
  return value as Writable<T>;
}

function examples<K extends keyof typeof API_CONTRACT.reads>(
  name: K,
): Writable<(typeof API_CONTRACT.reads)[K]["examples"]> {
  return API_CONTRACT.reads[name].examples as Writable<
    (typeof API_CONTRACT.reads)[K]["examples"]
  >;
}

/* Compile-time checks. `satisfies` reports a missing required field or a
 * nullability mismatch; it does not complain about extra server fields, which
 * the client is free to ignore. `tsc -b` runs these; vitest only executes the
 * runtime assertions below. */
examples("DailyBriefResponse") satisfies DailyBrief[];
examples("DashboardResponse") satisfies Dashboard[];
examples("ApplicationView") satisfies ApplicationView[];
examples("ApplicationMockInterviewsResponse") satisfies ApplicationMockInterviews[];
examples("SavedJobView") satisfies SavedJobView[];
examples("SavedJobDetailView") satisfies SavedJobDetail[];
examples("ConversationView") satisfies ConversationView[];
examples("ConversationTranscriptResponse") satisfies ConversationTranscript[];
examples("ReportView") satisfies ReportView[];
examples("ResumeView") satisfies ResumeView[];
examples("ResumeImportResponse") satisfies ResumeImportResult[];
examples("TargetRoleView") satisfies TargetRoleView[];
examples("EmailWorkspaceResponse") satisfies EmailWorkspace[];
examples("CalendarWorkspaceResponse") satisfies CalendarWorkspace[];
examples("CompanyResearchView") satisfies CompanyResearchView[];

/* The request the client sends: the check runs the other way. The client's
 * type must be accepted by the server, so the server's example must be
 * expressible in the client's type — including the bound interaction scope. */
writable(API_CONTRACT.requests.ChatStreamRequest.examples) satisfies ChatStreamRequest[];

type SchemaProperty = {
  enum?: readonly string[];
  anyOf?: readonly object[];
};

/** The closed set a schema property accepts, whether declared directly or as
 * the non-null arm of `Literal[...] | None`. */
function schemaEnum(property: SchemaProperty): string[] {
  if (property.enum) return [...property.enum];
  for (const arm of property.anyOf ?? []) {
    if ("enum" in arm && Array.isArray(arm.enum)) return [...(arm.enum as string[])];
  }
  throw new Error("schema property declares no enum");
}

function sameSet(client: readonly string[], server: readonly string[]): void {
  const missingOnClient = server.filter((value) => !client.includes(value));
  const missingOnServer = client.filter((value) => !server.includes(value));
  expect({ missingOnClient, missingOnServer }).toEqual({
    missingOnClient: [],
    missingOnServer: [],
  });
}

const defs = API_CONTRACT.stream_events.schema.$defs;

describe("stream event contract", () => {
  it("parses every example the server can emit", () => {
    // One event per type and per behaviour-selecting literal. A parser that
    // rejects one of these is refusing something production sends.
    for (const raw of API_CONTRACT.stream_events.examples) {
      const parsed = parsePublicStreamEvent(raw);
      expect(parsed.type).toBe(raw.type);
    }
  });

  it("lists exactly the event types the schema discriminates on", () => {
    sameSet(EVENT_TYPES, Object.keys(API_CONTRACT.stream_events.schema.discriminator.mapping));
  });

  it("owns the same closed sets the schema declares", () => {
    sameSet(PROGRESS_STAGES, schemaEnum(defs.ProgressEvent.properties.stage));
    sameSet(CAPABILITIES, schemaEnum(defs.CapabilityStartedEvent.properties.capability));
    sameSet(CAPABILITIES, schemaEnum(defs.CapabilityCompletedEvent.properties.capability));
    sameSet(INTERACTION_KINDS, schemaEnum(defs.InteractionRequiredEvent.properties.kind));
    sameSet(INTERACTION_SCOPES, schemaEnum(defs.InteractionRequiredEvent.properties.scope));
    sameSet(REPORT_KIND_LIST, schemaEnum(defs.ReportReadyEvent.properties.kind));
    sameSet(
      REPORT_DELIVERY_STATUSES,
      schemaEnum(defs.ReportReadyEvent.properties.status_at_delivery),
    );
  });

  it("accepts both interaction scopes and binds a response to either", () => {
    /*
     * The capability-confirmation gate (owner rule → approval card) was added
     * server-side after the parser was written, and the parser rejected the
     * whole event. Pin the property directly rather than rely on the example
     * sweep above, because this is the one that broke.
     */
    const scoped = API_CONTRACT.stream_events.examples.filter(
      (event) => event.type === "interaction_required" && event.scope != null,
    );
    expect(scoped.map((event) => event.scope).sort()).toEqual([
      "capability_confirmation",
      "resume_analysis_confirmation",
    ]);
    for (const event of scoped) {
      expect(parsePublicStreamEvent(event).type).toBe("interaction_required");
    }
  });
});

describe("read response contract", () => {
  it("returns report references as a list on each transcript message", () => {
    /*
     * The runtime side of the `satisfies ConversationTranscript` check above.
     * A message can carry two reports; the client must render both rather
     * than read a singular field that no longer exists.
     */
    const withReports = examples("ConversationTranscriptResponse")
      .flatMap((transcript) => transcript.messages)
      .find((message) => message.resources.length > 0);
    expect(withReports?.resources.map((resource) => resource.kind)).toEqual([
      "job_research_report",
      "resume_job_match",
    ]);
    expect("resource" in (withReports ?? {})).toBe(false);
  });

  it("restores a pending interaction the stream parser accepts", () => {
    // The transcript reload path hands this to the same reducer as the live
    // stream does, so it has to pass the same parser.
    const pending = examples("ConversationTranscriptResponse")
      .map((transcript) => transcript.pending_interaction)
      .filter((interaction) => interaction != null);
    expect(pending).not.toHaveLength(0);
    for (const interaction of pending) {
      expect(parsePublicStreamEvent(interaction).type).toBe("interaction_required");
    }
  });

  it("describes every response the client fetches", () => {
    // The names the client's fetch functions read. A route the client starts
    // calling must be added here, and to the `satisfies` block above.
    const consumed = [
      "DailyBriefResponse",
      "DashboardResponse",
      "ApplicationView",
      "ApplicationMockInterviewsResponse",
      "SavedJobView",
      "SavedJobDetailView",
      "ConversationView",
      "ConversationTranscriptResponse",
      "ReportView",
      "ResumeView",
      "ResumeImportResponse",
      "TargetRoleView",
      "EmailWorkspaceResponse",
      "CalendarWorkspaceResponse",
      "CompanyResearchView",
    ];
    const described = Object.keys(API_CONTRACT.reads);
    expect(consumed.filter((name) => !described.includes(name))).toEqual([]);
  });
});
