import { describe, expect, it } from "vitest";

import type { MockInterviewSessionView } from "../api/client";
import { practiceRepeat } from "./practiceRepeat";

const base: MockInterviewSessionView = {
  session_id: "s1", status: "cancelled", interview_type: "technical",
  interview_type_label: "技术面", question_count: 2, max_primary_questions: 10,
  max_follow_ups_per_question: 2, report_id: null, summary: null, conversation_id: null,
  created_at: "2026-09-25T00:00:00Z", completed_at: null, updated_at: "2026-09-25T00:00:00Z",
};

describe("practiceRepeat", () => {
  it("repeats a company-only run without a resume in words alone", () => {
    const repeat = practiceRepeat({ ...base, target_company: "字节", target_role: "Java后端" });

    expect(repeat.resource).toBeNull();
    expect(repeat.additionalResources).toEqual([]);
    expect(repeat.prompt).toBe(
      "再来一次模拟面试，设置和上次相同：自由练习，公司「字节」，目标岗位「Java后端」，技术面，10 道主问题，每题最多追问 2 次，不用简历。直接开始。",
    );
  });

  it("attaches the exact JD version and resume version a job run used", () => {
    const repeat = practiceRepeat({
      ...base,
      job_posting_id: "job-1", jd_snapshot_id: "jd-1", jd_version: 2,
      job_title: "Agent开发实习", job_company_name: "量霸科技", target_company: null,
      resume_id: "r1", resume_version_id: "rv1", resume_name: "Agent开发", resume_version_number: 1,
      resume_document_format: "pdf", resume_byte_size: 1234,
    });

    expect(repeat.resource).toMatchObject({ kind: "jd_snapshot", jdSnapshotId: "jd-1", jdVersion: 2 });
    expect(repeat.additionalResources).toEqual([
      expect.objectContaining({ resumeVersionId: "rv1", versionNumber: 1 }),
    ]);
    expect(repeat.prompt).not.toContain("不用简历");
    expect(repeat.prompt).not.toContain("公司「");
  });

  it("repeats an application run through its application only", () => {
    const repeat = practiceRepeat({
      ...base, application_id: "app-1", title: "AI 产品经理", company_name: "阿里巴巴",
      resume_id: "r1", resume_version_id: "rv1", resume_name: "投递版", resume_version_number: 3,
    });

    expect(repeat.resource).toMatchObject({ kind: "application", applicationId: "app-1" });
    // The application carries its own submitted resume; attaching it again would be a second source.
    expect(repeat.additionalResources).toEqual([]);
    expect(repeat.prompt).toContain("针对附带的投递记录");
  });
});
