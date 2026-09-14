import { expect, it } from "vitest";

import { parsePublicStreamEvent } from "./events";
import { hydrationFrom } from "./recovery";

it("restores a UI-only body card from the transcript", () => {
  const hydrated = hydrationFrom("c1", {
    messages: [{
      role: "assistant",
      content: "已展示简历分析",
      created_at: "2026-09-14T00:00:00Z",
      resources: [{
        kind: "delivered_body",
        resource_id: "body-1",
        title: "简历分析",
        status_at_delivery: null,
        anchored_by_other_job: null,
      }],
    }],
    active_workflow: null,
    phase: null,
    pending_interaction: null,
    pending_interaction_body: null,
  });
  expect(hydrated.messages[0].resources).toEqual([{
    kind: "delivered_body",
    resourceId: "body-1",
    title: "简历分析",
    statusAtDelivery: null,
    anchoredByOtherJob: null,
  }]);
});

it("does not add the UI-only body kind to the streaming protocol", () => {
  expect(() => parsePublicStreamEvent({
    type: "report_ready",
    kind: "delivered_body",
    resource_id: "body-1",
  })).toThrow();
});
