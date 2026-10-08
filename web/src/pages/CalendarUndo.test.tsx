// @vitest-environment jsdom
import { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, expect, it, vi } from "vitest";
import { CalendarPanel } from "./WorkspaceViews";
import { fetchCalendar, fetchInterviews, restoreInterviewCompletion } from "../api/client";

vi.mock("../api/client", async (original) => ({ ...await original<typeof import("../api/client")>(), fetchCalendar: vi.fn(), fetchInterviews: vi.fn(), restoreInterviewCompletion: vi.fn() }));
afterEach(() => { vi.resetAllMocks(); vi.unstubAllGlobals(); });

it("undoes completion from the interview center and surfaces the retro restriction", async () => {
  vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
  vi.mocked(fetchCalendar).mockResolvedValue({ accounts: [], events: [], month: null, timezone: "UTC" });
  const interview = { id: "interview-1", application_id: "app-1", company_name: "Acme", job_title: "Engineer", sequence_number: 1, employer_label: null, scheduled_start: null, status: "completed" };
  vi.mocked(fetchInterviews).mockResolvedValue([interview]);
  vi.mocked(restoreInterviewCompletion).mockRejectedValueOnce(new Error("已有复盘或状态已变化，不能撤销完成。")).mockResolvedValue({ ...interview, status: "scheduled" });
  const container = document.createElement("div");
  const root = createRoot(container);
  try {
    await act(async () => root.render(<CalendarPanel apiBaseUrl="/api" refreshToken={0} hidden={false} onAskAgent={vi.fn()} />));
    const button = () => Array.from(container.querySelectorAll("button")).find((item) => item.textContent === "撤销完成（未复盘时）")!;
    await act(async () => button().click());
    expect(container.textContent).toContain("已有复盘");
    vi.mocked(fetchInterviews).mockResolvedValue([{ ...interview, status: "scheduled" }]);
    await act(async () => button().click());
    expect(restoreInterviewCompletion).toHaveBeenLastCalledWith("interview-1", { apiBaseUrl: "/api" });
    expect(container.textContent).not.toContain("撤销完成（未复盘时）");
  } finally { await act(async () => root.unmount()); }
});
