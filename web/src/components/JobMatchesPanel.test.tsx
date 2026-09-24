// @vitest-environment jsdom
import { act, StrictMode } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  fetchJobMatches, fetchReport, fetchResumes, fetchSavedJobs,
  type JobMatchHistory, type ResumeJobMatchView, type SavedJobView,
} from "../api/client";
import { API_CONTRACT } from "../contracts/api-contract";
import { JobsPanel } from "../pages/WorkspaceViews";
import { JobMatchesPanel } from "./JobMatchesPanel";

vi.mock("../api/client", async (importOriginal) => ({
  ...await importOriginal<typeof import("../api/client")>(),
  fetchJobMatches: vi.fn(), fetchReport: vi.fn(), fetchResumes: vi.fn(), fetchSavedJobs: vi.fn(),
}));

const job: SavedJobView = {
  id: "job", title: "Engineer", company_name: "Example", city: null, salary: null,
  source_name: "test", source_url: null, pursuit_status: "open", availability_status: "available",
  captured_at: "2026-09-01T00:00:00Z", last_checked_at: "2026-09-01T00:00:00Z",
  application_status: null, analysis_summary: null, responsibilities: [], required_skills: [],
  preferred_qualifications: [], clarification_questions: [], analyzed_at: null,
  jd_snapshot_id: "jd-v2", jd_version: 2, jd_analysis_status: "none",
  jd_analysis_version: null, jd_analysis: null, resume_match_status: "ready",
  resume_match_fit: "moderate", resume_match_at: null, resume_match_count: 2,
};
const first: ResumeJobMatchView = {
  report_id: "match-v1", job_posting_id: "job", job_title: "Engineer", company_name: "Example",
  jd_snapshot_id: "jd-v1", jd_version: 1, jd_captured_at: "2026-09-01T00:00:00Z",
  jd_available: true, current_jd: false,
  resume_id: "resume", resume_name: "Engineering", resume_version_id: "resume-v1",
  resume_version_number: 1, resume_created_at: "2026-08-01T00:00:00Z", resume_available: true,
  matcher_version: "matcher-v2", created_at: "2026-09-02T00:00:00Z",
  overall_fit: "moderate", summary: "Needs more evidence.",
};
const second = { ...first, report_id: "match-v2", resume_version_id: "resume-v2", resume_version_number: 2 };
let root: Root;
let container: HTMLDivElement;

async function click(text: string): Promise<void> {
  const button = Array.from(container.querySelectorAll("button")).find((item) => item.textContent?.includes(text));
  if (!button) throw new Error(`Missing button: ${text}`);
  await act(async () => button.click());
}

beforeEach(() => {
  vi.resetAllMocks();
  vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
  vi.mocked(fetchSavedJobs).mockResolvedValue([job]);
  vi.mocked(fetchJobMatches).mockResolvedValue({ items: [second, first], total: 2, limit: 20, offset: 0 });
  vi.mocked(fetchReport).mockImplementation(async (_kind, id) => ({
    kind: "resume_job_match", resource_id: id, title: "简历与岗位匹配", subtitle: "moderate",
    body: "## 要求逐项核对\n\n> JD: Python required\n\n简历证据：Built Python services",
    created_at: first.created_at, resume_job_match: id === first.report_id ? first : second,
  }));
  vi.mocked(fetchResumes).mockResolvedValue([{
    id: "resume", name: "Engineering", target_role: "Engineer", status: "active",
    latest_version_number: 2, latest_version_id: "resume-v2", version_count: 2,
    document_format: "text", byte_size: 20, updated_at: first.created_at,
    versions: [2, 1].map((number) => ({
      id: `resume-v${number}`, resume_id: "resume", version_number: number,
      document_format: "text", byte_size: 20, created_at: first.created_at,
      change_summary: number === 1 ? "初始版本" : "新增：新项目",
    })),
  }]);
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
});

afterEach(async () => {
  await act(async () => root.unmount());
  container.remove();
  vi.unstubAllGlobals();
});

describe("job library match history", () => {
  it("searches names and versions, handles no results, and selects a filtered version", async () => {
    const start = vi.fn();
    await act(async () => root.render(<JobMatchesPanel job={job} apiBaseUrl="/api" refreshToken={0} onStartTask={start} />));
    await click("选择简历版本发起匹配");
    await click("请选择简历版本");
    const search = container.querySelector<HTMLInputElement>('input[type="search"]')!;
    expect(document.activeElement).toBe(search);
    const type = async (value: string) => act(async () => {
      Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value")!.set!.call(search, value);
      search.dispatchEvent(new Event("input", { bubbles: true }));
    });
    await type("不存在的简历");
    expect(container.querySelectorAll('[role="option"]')).toHaveLength(0);
    expect(container.textContent).toContain("没有找到对应简历");
    await type(" ENGINEERING v1 ");
    expect(container.querySelectorAll('[role="option"]')).toHaveLength(1);
    await act(async () => search.dispatchEvent(new KeyboardEvent("keydown", { key: "ArrowDown", bubbles: true })));
    expect(document.activeElement?.getAttribute("role")).toBe("option");
    await act(async () => (document.activeElement as HTMLButtonElement).click());
    expect(container.querySelector('[role="listbox"]')).toBeNull();
    await click("开始匹配");
    expect(start).toHaveBeenCalledWith(expect.any(String), expect.objectContaining({ resumeVersionId: "resume-v1" }), expect.any(Array));
  });

  it("opens persisted reports by ID from the job card with exact original version links", async () => {
    const start = vi.fn();
    await act(async () => root.render(
      <StrictMode><JobsPanel apiBaseUrl="/api" hidden={false} refreshToken={0} onAskAgent={vi.fn()} onStartStandaloneTask={start} /></StrictMode>,
    ));
    await click("查看匹配分析（2）");
    expect(container.textContent).toContain("简历 v2 × JD v1");
    expect(container.textContent).toContain("简历 v1 × JD v1");
    const buttons = container.querySelectorAll<HTMLButtonElement>(".report-card-header");
    await act(async () => buttons[1].click());
    expect(fetchReport).toHaveBeenCalledWith("resume_job_match", "match-v1", expect.anything(), expect.anything());
    expect(container.querySelector(".match-provenance")?.textContent).toContain("JD v1");
    expect(container.querySelector(".match-provenance")?.textContent).toContain("简历 v1");
    expect(container.querySelector(".match-provenance")?.textContent).not.toContain("matcher-v2");
    expect(container.querySelector(".match-provenance a")?.getAttribute("href")).toContain("/resume/versions/resume-v1/document");
    expect(container.textContent).toContain("Built Python services");
    expect(start).not.toHaveBeenCalled();
    await click("收起匹配分析");
    await click("查看匹配分析（2）");
    expect(container.querySelectorAll(".match-history-item")).toHaveLength(2);
  });

  it("pins the chosen old resume version and JD even when the job refreshes during selection", async () => {
    vi.mocked(fetchJobMatches).mockResolvedValue({ items: [], total: 0, limit: 20, offset: 0 });
    const start = vi.fn();
    await act(async () => root.render(<JobMatchesPanel job={job} apiBaseUrl="/api" refreshToken={0} onStartTask={start} />));
    expect(container.textContent).toContain("尚未匹配");
    await click("选择简历版本发起匹配");
    await click("请选择简历版本");
    const option = Array.from(container.querySelectorAll<HTMLButtonElement>('[role="option"]')).find((item) => item.textContent?.includes("v1"))!;
    expect(container.querySelectorAll('[role="option"]')).toHaveLength(2);
    await act(async () => {
      option.click();
      root.render(<JobMatchesPanel job={{ ...job, jd_snapshot_id: "jd-v3", jd_version: 3 }} apiBaseUrl="/api" refreshToken={1} onStartTask={start} />);
    });
    await click("开始匹配");
    expect(start).toHaveBeenCalledTimes(1);
    expect(start).toHaveBeenCalledWith(
      expect.stringContaining("JD v2"),
      expect.objectContaining({ resumeVersionId: "resume-v1" }),
      [expect.objectContaining({ jdSnapshotId: "jd-v2" })],
    );
    expect(fetchReport).not.toHaveBeenCalled();
    expect(container.textContent).toContain("匹配任务已排队");
  });

  it("keeps deleted input identities and hides inaccessible document links", async () => {
    vi.mocked(fetchReport).mockResolvedValue({
      kind: "resume_job_match", resource_id: first.report_id, title: "匹配", subtitle: "", body: "Stored report",
      created_at: first.created_at,
      resume_job_match: { ...first, resume_available: false, jd_available: false },
    });
    await act(async () => root.render(<JobMatchesPanel job={job} apiBaseUrl="/api" refreshToken={0} />));
    await click("查看这份匹配报告");
    const provenance = container.querySelector(".match-provenance")!;
    expect(provenance.textContent).toContain("简历 v1");
    expect(provenance.textContent).toContain("JD v1");
    expect(provenance.textContent).toContain("原简历已删除或不可访问");
    expect(provenance.textContent).toContain("原 JD 已删除或不可访问");
    expect(provenance.querySelector("a")).toBeNull();
  });

  it("retries failed history reads and pages without running a matching task", async () => {
    vi.mocked(fetchJobMatches).mockRejectedValueOnce(new Error("offline"));
    await act(async () => root.render(<JobMatchesPanel job={job} apiBaseUrl="/api" refreshToken={0} />));
    expect(container.querySelector('[role="alert"]')?.textContent).toContain("offline");
    vi.mocked(fetchJobMatches).mockResolvedValue({ items: [first], total: 21, limit: 20, offset: 0 });
    await click("刷新匹配历史");
    await click("下一页");
    expect(fetchJobMatches).toHaveBeenLastCalledWith("job", 20, expect.anything());
    expect(fetchReport).not.toHaveBeenCalled();
  });

  it("ignores late history responses from a previously selected job", async () => {
    let complete!: (value: JobMatchHistory) => void;
    vi.mocked(fetchJobMatches).mockReturnValueOnce(new Promise((resolve) => { complete = resolve; }));
    await act(async () => root.render(<JobMatchesPanel job={job} apiBaseUrl="/api" refreshToken={0} />));
    vi.mocked(fetchJobMatches).mockResolvedValue({ items: [], total: 0, limit: 20, offset: 0 });
    await act(async () => root.render(<JobMatchesPanel job={{ ...job, id: "other" }} apiBaseUrl="/api" refreshToken={0} />));
    await act(async () => complete({ items: [first], total: 1, limit: 20, offset: 0 }));
    expect(container.textContent).toContain("尚未匹配");
    expect(container.textContent).not.toContain("Needs more evidence.");
  });

  it("accepts the backend history response contract", () => {
    const examples = API_CONTRACT.reads.JobMatchHistoryResponse.examples satisfies readonly {
      readonly items: readonly ResumeJobMatchView[]; total: number; limit: number; offset: number;
    }[];
    expect(examples[0].items[0].jd_version).toBe(1);
  });
});
