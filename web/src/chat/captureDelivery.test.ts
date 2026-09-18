import { describe, expect, it } from "vitest";

import type { JobCapturedEventView } from "../api/client";
import { captureDeliveryReady } from "./jobCapture";

const event: JobCapturedEventView = {
  id: "capture", conversation_id: "original", job_posting_id: "job",
  jd_snapshot_id: "v1", title: "AI", company_name: "Example", created_at: "",
  continuation_status: "pending", continuation_turn_id: null,
};

describe("backend capture delivery", () => {
  it.each(["pending", "failed"] as const)(
    "does not acknowledge or hydrate a %s continuation",
    (status) => expect(captureDeliveryReady({ ...event, continuation_status: status })).toBe(false),
  );

  it.each(["completed", "discarded"] as const)(
    "refreshes a %s delivery without starting a new turn",
    (status) => expect(captureDeliveryReady({ ...event, continuation_status: status })).toBe(true),
  );
});
