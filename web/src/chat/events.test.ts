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

  it("accepts a saved-job card pinned to an immutable JD snapshot", () => {
    expect(
      parsePublicStreamEvent({
        type: "job_resource_ready",
        kind: "saved_job",
        resource_id: "jds-1",
        job_posting_id: "job-1",
        title: "量霸科技｜AI Agent 实习生",
        description: "JD 第 1 版 · BOSS直聘",
      }),
    ).toMatchObject({ type: "job_resource_ready", resource_id: "jds-1", job_posting_id: "job-1" });
  });

  it("rejects a saved-job card that names no snapshot or posting", () => {
    expect(() =>
      parsePublicStreamEvent({ type: "job_resource_ready", kind: "saved_job", resource_id: "jds-1" }),
    ).toThrow("SSE_EVENT_INVALID");
    expect(() =>
      parsePublicStreamEvent({
        type: "job_resource_ready",
        kind: "job_posting",
        resource_id: "job-1",
        job_posting_id: "job-1",
      }),
    ).toThrow("SSE_EVENT_INVALID");
  });
});
