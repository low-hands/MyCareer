import { describe, expect, it } from "vitest";

import { safeMeetingUrl } from "./WorkspaceViews";

describe("meeting links", () => {
  it("allows HTTPS links", () => {
    expect(safeMeetingUrl("https://meet.example/room")).toBe(
      "https://meet.example/room",
    );
  });

  it("rejects executable and insecure schemes", () => {
    expect(safeMeetingUrl("javascript:alert(1)")).toBeNull();
    expect(safeMeetingUrl("http://meet.example/room")).toBeNull();
    expect(safeMeetingUrl("not a url")).toBeNull();
  });
});
