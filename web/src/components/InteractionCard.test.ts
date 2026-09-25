import { describe, expect, it } from "vitest";

import { uploadReply } from "./InteractionCard";

describe("uploadReply", () => {
  const imported = {
    resume_id: "resume-1",
    resume_version_id: "resume-version-1",
    name: "正式简历",
    version_number: 2,
    document_format: "pdf",
    byte_size: 10,
  };

  it("answers with the uploaded version attached, not just its name", () => {
    const reply = uploadReply(imported);
    expect(reply.message).toBe("用刚上传的简历《正式简历》v2");
    expect(reply.resources).toEqual([
      expect.objectContaining({ resumeVersionId: "resume-version-1", versionNumber: 2 }),
    ]);
  });

  it("says when the same file was already in the library", () => {
    expect(uploadReply({ ...imported, already_in_library: true }).message).toBe(
      "用简历《正式简历》v2（库里已有这份文件，直接用它）",
    );
  });
});
