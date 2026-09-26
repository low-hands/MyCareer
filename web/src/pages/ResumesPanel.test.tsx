// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { fetchResumes, fetchTargetRoles, type ResumeView } from "../api/client";
import { ResumesPanel } from "./WorkspaceViews";

vi.mock("../api/client", async (original) => ({
  ...await original<typeof import("../api/client")>(),
  fetchResumes: vi.fn(), fetchTargetRoles: vi.fn(),
}));
const resume: ResumeView = {
  id: "resume-1", name: "产品经理简历", target_role: "产品经理", target_role_id: "role-1", status: "active",
  latest_version_number: 1, latest_version_id: "version-1", version_count: 1,
  document_format: "pdf", byte_size: 100, updated_at: "2026-09-23T00:00:00Z",
  versions: [{ id: "version-1", resume_id: "resume-1", version_number: 1, document_format: "pdf", byte_size: 100, created_at: "2026-09-23T00:00:00Z", change_summary: "初始版本" }],
};
let root: Root;
let container: HTMLDivElement;
const standalone = vi.fn();
beforeEach(() => {
  vi.resetAllMocks();
  vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
  vi.mocked(fetchResumes).mockResolvedValue([resume]);
  vi.mocked(fetchTargetRoles).mockResolvedValue([{ id: "role-1", title: "产品经理", priority: 1, status: "active" }]);
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
});
afterEach(async () => { await act(async () => root.unmount()); container.remove(); vi.unstubAllGlobals(); });

function button(label: string): HTMLButtonElement {
  return Array.from(container.querySelectorAll<HTMLButtonElement>(".resume-version-actions button"))
    .find((node) => node.textContent === label)!;
}

it("offers a critique of a version in its own task, and no experience import", async () => {
  await act(async () => root.render(<ResumesPanel apiBaseUrl="" refreshToken={0} hidden={false} onAskAgent={vi.fn()} onStartStandaloneTask={standalone} />));

  await act(async () => button("简历点评").click());
  expect(standalone).toHaveBeenLastCalledWith(
    expect.stringContaining("点评简历“产品经理简历”的 v1"),
    expect.objectContaining({ resumeVersionId: "version-1" }),
    [],
    "点评简历",
  );
  expect(button("导入经历")).toBeUndefined();
  expect(button("让 Agent 分析")).toBeUndefined();
});
