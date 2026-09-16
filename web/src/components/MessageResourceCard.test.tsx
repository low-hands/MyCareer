import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import type { MessageResource } from "../chat/reducer";
import { MessageResourceCard } from "./MessageResourceCard";

function render(resource: MessageResource): string {
  return renderToStaticMarkup(<MessageResourceCard resource={resource} apiBaseUrl="/api" />);
}

describe("MessageResourceCard", () => {
  it("dispatches each kind to its own card", () => {
    expect(render({ kind: "resume_version", resourceId: "rv-1", title: "简历 v3" })).toContain(
      "resume-attachment",
    );
    const savedJob = render({
      kind: "saved_job",
      resourceId: "jds-1",
      title: "量霸科技｜AI Agent 实习生",
      description: "JD 第 1 版 · BOSS直聘",
    });
    expect(savedJob).toContain("saved-job-card");
    expect(savedJob).toContain("量霸科技｜AI Agent 实习生");
    expect(savedJob).toContain("查看完整 JD");
    const report = render({ kind: "job_research_report", resourceId: "rep-1" });
    expect(report).toContain("report-card");
    expect(report).not.toContain("saved-job-card");
    expect(report).toContain("公司调研报告");
  });

  it("keeps a deleted job's name while saying its JD is gone", () => {
    const html = render({
      kind: "saved_job",
      resourceId: "jds-1",
      title: "量霸科技｜AI Agent 实习生",
      description: "JD 第 1 版 · BOSS直聘",
      available: false,
    });
    expect(html).toContain("量霸科技｜AI Agent 实习生");
    expect(html).toContain("该岗位已删除或不可访问");
  });

  it("never renders an unknown kind as a report", () => {
    const html = render({
      kind: "future_thing" as MessageResource["kind"],
      resourceId: "x-1",
      title: "某资源",
    });
    expect(html).not.toContain("report-card");
    expect(html).toContain("future_thing");
  });
});
