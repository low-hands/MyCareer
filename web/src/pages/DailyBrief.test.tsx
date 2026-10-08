// @vitest-environment jsdom
import { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, expect, it, vi } from "vitest";
import { DailyBriefPanel } from "./DailyBrief";
import { fetchDailyBrief, fetchRestorableActionItems, restoreActionItem, type ActionItemView } from "../api/client";

vi.mock("../api/client", () => ({ fetchDailyBrief: vi.fn(), fetchRestorableActionItems: vi.fn(), restoreActionItem: vi.fn() }));
afterEach(() => { vi.resetAllMocks(); vi.unstubAllGlobals(); });

it("offers completed and snoozed actions for restoration and refreshes after undo", async () => {
  vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
  vi.mocked(fetchDailyBrief).mockResolvedValue({ timezone: "UTC", generated_at: "2026-10-08T00:00:00Z", overdue: [], due_today: [], upcoming: [], no_due_date: [] });
  const items: ActionItemView[] = ["completed", "snoozed"].map((status) => ({ id: status, status, title: status === "completed" ? "待办一" : "待办二", summary: "说明", action_type: "application_follow_up", source_type: "application", application_id: "app", due_at: null, snoozed_until: null }));
  vi.mocked(fetchRestorableActionItems).mockResolvedValueOnce(items).mockResolvedValue(items.slice(1));
  vi.mocked(restoreActionItem).mockResolvedValue({ ...items[0], status: "open" });
  const container = document.createElement("div");
  const root = createRoot(container);
  try {
    await act(async () => root.render(<DailyBriefPanel apiBaseUrl="/api" refreshToken={0} hidden={false} />));
    expect(container.textContent).toContain("已完成");
    expect(container.textContent).toContain("稍后提醒");
    const button = Array.from(container.querySelectorAll("button")).find((item) => item.textContent === "恢复待办")!;
    await act(async () => button.click());
    expect(restoreActionItem).toHaveBeenCalledWith("completed", { apiBaseUrl: "/api" });
    expect(container.textContent).not.toContain("待办一");
    expect(container.textContent).toContain("待办二");
  } finally { await act(async () => root.unmount()); }
});
