import { describe, expect, it } from "vitest";

import { requirementsByTier } from "../components/JobAnalysisDetails";
import { derivedStatusLabel, jobAnalysisPrompt, safeMeetingUrl } from "./WorkspaceViews";

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

describe("job library analysis state", () => {
  it("says the current JD version is unanalysed when the result is from an older snapshot", () => {
    expect(derivedStatusLabel("none")).toBe("待分析");
    expect(derivedStatusLabel("ready")).toBe("已完成");
    expect(derivedStatusLabel("stale")).toBe("当前版本待分析");
  });

  it("asks for a JD-only reading of the named job", () => {
    const prompt = jobAnalysisPrompt({ title: "AI 产品经理", company_name: "示例科技" });
    expect(prompt).toContain("示例科技 · AI 产品经理");
    expect(prompt).toContain("不要结合简历");
  });

  it("groups requirements S first and drops empty tiers", () => {
    const groups = requirementsByTier([
      { text: "b", tier: "B", kind: "inference", jd_quote: "q" },
      { text: "s1", tier: "S", kind: "fact", jd_quote: "q" },
      { text: "s2", tier: "S", kind: "fact", jd_quote: "q" },
    ]);
    expect(groups.map((group) => group.tier)).toEqual(["S", "B"]);
    expect(groups[0].items.map((item) => item.text)).toEqual(["s1", "s2"]);
  });
});
