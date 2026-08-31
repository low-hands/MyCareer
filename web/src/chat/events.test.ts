import { describe, expect, it } from "vitest";

import { parsePublicStreamEvent } from "./events";

describe("parsePublicStreamEvent", () => {
  it("accepts a report reference the card can resolve", () => {
    expect(
      parsePublicStreamEvent({
        type: "report_ready",
        kind: "interview_preparation",
        resource_id: "prep-1",
      }),
    ).toEqual({
      type: "report_ready",
      kind: "interview_preparation",
      resource_id: "prep-1",
    });
  });

  it("rejects a kind that names no read endpoint", () => {
    /*
     * The kind selects which endpoint the card fetches, so an unrecognised one
     * would render a card that can never open. Failing at the parse boundary
     * keeps that out of state entirely.
     */
    expect(() =>
      parsePublicStreamEvent({
        type: "report_ready",
        kind: "resume_analysis",
        resource_id: "rep-1",
      }),
    ).toThrow("SSE_EVENT_INVALID");
  });
});
