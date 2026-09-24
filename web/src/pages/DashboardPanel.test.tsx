// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { fetchDashboard, fetchResumes, fetchSavedJobDetail, type Dashboard, type ResumeView } from "../api/client";
import { API_CONTRACT } from "../contracts/api-contract";
import { DashboardPanel } from "./WorkspaceViews";

vi.mock("../api/client", async (original) => ({
  ...await original<typeof import("../api/client")>(),
  fetchDashboard: vi.fn(), fetchResumes: vi.fn(), fetchSavedJobDetail: vi.fn(),
}));
const dashboard = API_CONTRACT.reads.DashboardResponse.examples[0] as unknown as Dashboard;
const resume: ResumeView = {
  id: "resume-1", name: "产品经理简历", target_role: "产品经理", status: "current",
  latest_version_number: 2, latest_version_id: "version-2", version_count: 2,
  document_format: "pdf", byte_size: 100, updated_at: "2026-09-23T00:00:00Z",
  versions: [2, 1].map((n) => ({ id: `version-${n}`, resume_id: "resume-1", version_number: n, document_format: "pdf", byte_size: 100, created_at: "2026-09-23T00:00:00Z", change_summary: n === 1 ? "初始版本" : "新增：新项目" })),
};
let root: Root;
let container: HTMLDivElement;
const standalone = vi.fn();
beforeEach(() => {
  vi.resetAllMocks();
  vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
  HTMLDialogElement.prototype.showModal = function () { this.setAttribute("open", ""); };
  HTMLDialogElement.prototype.close = function () { this.removeAttribute("open"); };
  vi.mocked(fetchDashboard).mockResolvedValue(dashboard);
  vi.mocked(fetchResumes).mockResolvedValue([resume]);
  vi.mocked(fetchSavedJobDetail).mockResolvedValue({ ...dashboard.recent_jobs[0], jd_text: "这是所选岗位的完整 JD", jd_version: 1 });
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
});
afterEach(async () => { await act(async () => root.unmount()); container.remove(); vi.unstubAllGlobals(); });
async function render() {
  await act(async () => root.render(<DashboardPanel apiBaseUrl="" refreshToken={0} hidden={false} onAskAgent={vi.fn()} onStartStandaloneTask={standalone} />));
}
it("opens the clicked JD directly and closes back to its card", async () => {
  await render();
  const card = container.querySelector<HTMLButtonElement>(".dashboard-resource-card")!;
  card.focus();
  await act(async () => card.click());
  expect(fetchSavedJobDetail).toHaveBeenCalledWith(dashboard.recent_jobs[0].id, expect.objectContaining({ signal: expect.any(AbortSignal) }));
  expect(container.querySelector("dialog")?.textContent).toContain("这是所选岗位的完整 JD");
  await act(async () => container.querySelector<HTMLButtonElement>("dialog header button")!.click());
  expect(container.querySelector("dialog")).toBeNull();
  expect(document.activeElement).toBe(card);
});
it("switches the preview and analysis attachment to the exact resume version", async () => {
  await render();
  const card = Array.from(container.querySelectorAll<HTMLButtonElement>(".dashboard-resource-card")).find((node) => node.textContent?.includes(resume.name))!;
  await act(async () => card.click());
  expect(container.querySelector("iframe")?.src).toContain("version-2");
  expect(container.querySelector(".dashboard-version-summary")?.textContent).toContain("新增：新项目");
  await act(async () => container.querySelector<HTMLButtonElement>(".dashboard-detail .resume-version-trigger")!.click());
  const versionOption = Array.from(container.querySelectorAll<HTMLButtonElement>('[role="option"]')).find((node) => node.textContent?.includes("v1"))!;
  await act(async () => versionOption.click());
  expect(container.querySelector("iframe")?.src).toContain("version-1");
  expect(container.querySelector(".dashboard-version-summary")?.textContent).toContain("初始版本");
  expect(container.querySelector(".resume-version-actions button")).toBeNull();
});
