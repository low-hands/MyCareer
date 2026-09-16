import { describe, expect, it } from "vitest";

import type { SavedJobView } from "../api/client";
import {
  attachmentFromImport,
  attachmentFromSavedJob,
  attachmentFromVersion,
  droppedResumeFile,
  MAX_ATTACHMENTS,
  toInputResources,
  toMessageResources,
  withAttachment,
  type ResumeAttachment,
} from "./attachments";

function attachment(versionId: string, extra: Partial<ResumeAttachment> = {}): ResumeAttachment {
  return {
    resumeVersionId: versionId,
    resumeId: "resume-1",
    name: "主简历",
    versionNumber: 1,
    documentFormat: "pdf",
    byteSize: 2048,
    uploadedAt: null,
    ...extra,
  };
}

describe("resume attachments", () => {
  it("keeps the ids the import returned so the message pins that exact version", () => {
    const attached = attachmentFromImport({
      resume_id: "resume-1",
      resume_version_id: "version-7",
      name: "主简历",
      version_number: 7,
      document_format: "pdf",
      byte_size: 2048,
    });
    expect(attached.resumeId).toBe("resume-1");
    expect(attached.resumeVersionId).toBe("version-7");
    expect(toInputResources([attached])).toEqual([{ kind: "resume_version", id: "version-7" }]);
  });

  it("pins a library version by its own id, not the resume's latest", () => {
    const attached = attachmentFromVersion("主简历", {
      id: "version-2",
      resume_id: "resume-1",
      version_number: 2,
      document_format: "markdown",
      byte_size: 512,
      created_at: "2026-09-13T00:00:00Z",
    });
    expect(toInputResources([attached])).toEqual([{ kind: "resume_version", id: "version-2" }]);
    expect(attached.uploadedAt).toBe("2026-09-13T00:00:00Z");
  });

  it("does not attach the same version twice or more than the server accepts", () => {
    const once = withAttachment([], attachment("v1"));
    expect(withAttachment(once, attachment("v1"))).toBe(once);
    const full = Array.from({ length: MAX_ATTACHMENTS }, (_, index) => attachment(`v${index}`));
    expect(withAttachment(full, attachment("extra"))).toBe(full);
  });

  it("describes the sent chip from metadata only, never the document", () => {
    const [resource] = toMessageResources([attachment("v1", { versionNumber: 3 })]);
    expect(resource).toEqual({
      kind: "resume_version",
      resourceId: "v1",
      resumeId: "resume-1",
      title: "主简历 v3",
      description: "pdf · 2.0 KB",
      available: true,
    });
  });

  it("pins a saved job to its current JD snapshot and sends nothing else", () => {
    const job = {
      id: "job-1",
      title: "AI 产品经理",
      company_name: "示例科技",
      jd_snapshot_id: "snapshot-3",
      jd_version: 3,
    } as SavedJobView;
    const attached = attachmentFromSavedJob(job);
    expect(attached).not.toBeNull();
    expect(toInputResources([attached!])).toEqual([{ kind: "jd_snapshot", id: "snapshot-3" }]);
    expect(toMessageResources([attached!])).toEqual([
      {
        kind: "saved_job",
        resourceId: "snapshot-3",
        jobPostingId: "job-1",
        title: "示例科技 · AI 产品经理",
        description: "JD 快照 v3",
        available: true,
      },
    ]);
    expect(attachmentFromSavedJob({ ...job, jd_snapshot_id: null, jd_version: null })).toBeNull();
  });

  it("accepts only resume documents from a drop", () => {
    const pdf = new File(["%PDF"], "cv.PDF", { type: "application/pdf" });
    const image = new File([""], "photo.png", { type: "image/png" });
    expect(droppedResumeFile([image, pdf])).toBe(pdf);
    expect(droppedResumeFile([image])).toBeNull();
    expect(droppedResumeFile(null)).toBeNull();
  });
});
