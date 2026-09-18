import { describe, expect, it } from "vitest";

import type { JobCapturedEventView } from "../api/client";
import { nextCaptureEvents } from "./jobCapture";

const capture: JobCapturedEventView = {
  id: "jobcap-1",
  conversation_id: "c1",
  job_posting_id: "job-1",
  jd_snapshot_id: "snapshot-1",
  title: "AI 产品经理",
  company_name: "示例科技",
  created_at: "2026-09-12T12:00:00Z",
  continuation_status: "completed",
  continuation_turn_id: "turn-1",
};

describe("capture delivery scheduling", () => {
  it("defers rejected captures until the next poll without blocking other captures", () => {
    const later = { ...capture, id: "jobcap-2", created_at: "2026-09-12T12:01:00Z" };
    const events = [later, capture];
    const attempted = new Set([capture.id]);
    const inFlight = new Set<string>();

    expect(nextCaptureEvents(events, inFlight, attempted)).toEqual([later]);
    attempted.clear();
    inFlight.add(capture.id);
    expect(nextCaptureEvents(events, inFlight, attempted)).toEqual([later]);
    inFlight.clear();
    expect(nextCaptureEvents(events, inFlight, attempted)).toEqual([capture, later]);
    expect(events).toEqual([later, capture]);
  });
});
