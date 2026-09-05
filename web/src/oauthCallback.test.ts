import { describe, expect, it, vi } from "vitest";

import { consumeGoogleOAuthCallback } from "./oauthCallback";

describe("Google OAuth callback", () => {
  it("opens the matching panel and removes callback data from the URL", () => {
    const replaceState = vi.fn();

    const result = consumeGoogleOAuthCallback(
      {
        search: "?status=connected&kind=calendar",
        pathname: "/workspace",
        hash: "#today",
      },
      { replaceState },
    );

    expect(result).toMatchObject({ view: "calendar", status: "connected" });
    expect(replaceState).toHaveBeenCalledWith(null, "", "/workspace#today");
  });

  it("does not consume unrelated query parameters", () => {
    const replaceState = vi.fn();

    expect(consumeGoogleOAuthCallback(
      { search: "?page=2", pathname: "/", hash: "" },
      { replaceState },
    )).toBeNull();
    expect(replaceState).not.toHaveBeenCalled();
  });
});
