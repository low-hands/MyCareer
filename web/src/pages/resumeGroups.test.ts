import { describe, expect, it } from "vitest";

import type { ResumeView, TargetRoleView } from "../api/client";
import { groupResumesByRole } from "./resumeGroups";

function resume(id: string, roleId: string, role: string): ResumeView {
  return {
    id, name: id, target_role: role, target_role_id: roleId, status: "active",
    latest_version_number: 1, latest_version_id: `${id}-v1`, version_count: 1,
    document_format: "pdf", byte_size: 1, updated_at: "2026-09-25T00:00:00Z", versions: [],
  } as ResumeView;
}

const roles: TargetRoleView[] = [
  { id: "pm", title: "AI 产品经理", priority: 0, status: "active" },
  { id: "agent", title: "Agent开发", priority: 1, status: "active" },
  { id: "empty", title: "数据分析", priority: 2, status: "active" },
];

describe("groupResumesByRole", () => {
  it("groups in role priority order and keeps a role with no resume", () => {
    const groups = groupResumesByRole(
      [resume("a", "agent", "Agent开发"), resume("p", "pm", "AI 产品经理"), resume("b", "agent", "Agent开发")],
      roles,
    );

    expect(groups.map((group) => [group.title, group.resumes.map((item) => item.id)])).toEqual([
      ["AI 产品经理", ["p"]],
      ["Agent开发", ["a", "b"]],
      ["数据分析", []],
    ]);
  });

  it("never drops a resume whose role is not in the list", () => {
    const groups = groupResumesByRole([resume("x", "gone", "旧岗位")], roles);

    expect(groups.at(-1)).toMatchObject({ roleId: "gone", title: "旧岗位" });
    expect(groups.at(-1)?.resumes.map((item) => item.id)).toEqual(["x"]);
  });
});
