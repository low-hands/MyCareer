import { describe, expect, it } from "vitest";

import { newStandaloneAgentTask, standaloneTaskSubject } from "./agentTask";

const label = () => "简历“Library” v1";

describe("standaloneTaskSubject", () => {
  it("names a prompt-only task by its request", () => {
    expect(standaloneTaskSubject(newStandaloneAgentTask("让 Agent 研究这家公司", null), label))
      .toBe("处理“让 Agent 研究这家公司”");
    expect(standaloneTaskSubject(newStandaloneAgentTask("检查我最近的招聘邮件和面试安排，并告诉我哪些需要同步", null), label))
      .toBe("处理“检查我最近的招聘邮件和面试安排，并告…”");
  });

  it("keeps an explicit label and the attachment wording", () => {
    const resume = {
      resumeId: "r", resumeVersionId: "v", name: "Library", versionNumber: 1,
      documentFormat: "pdf", byteSize: 1, uploadedAt: null,
    };
    expect(standaloneTaskSubject(newStandaloneAgentTask("分析", resume), label)).toBe("分析简历“Library” v1");
    expect(standaloneTaskSubject(newStandaloneAgentTask("x", null, [], "再来一次模拟面试"), label)).toBe("再来一次模拟面试");
  });
});
