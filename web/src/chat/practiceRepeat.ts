import type { MockInterviewSessionView } from "../api/client";
import type { ChatAttachment } from "./attachments";

export interface PracticeRepeat {
  prompt: string;
  /** The first resource the new conversation carries, or null when there is none. */
  resource: ChatAttachment | null;
  additionalResources: ChatAttachment[];
  label: string;
}

/**
 * A new practice with the same settings and the same exact inputs.
 *
 * Settings go in the prompt; the resume version, the JD version and the
 * application go as attached resources, so the start rules apply to them as
 * they would to a user's own attachment and nothing is picked by guessing.
 * A run without a resume says so, which is also what the start rules need
 * to skip the resume choice.
 */
export function practiceRepeat(session: MockInterviewSessionView): PracticeRepeat {
  const resources: ChatAttachment[] = [];
  if (session.application_id) {
    resources.push({
      kind: "application",
      applicationId: session.application_id,
      title: [session.company_name, session.title].filter(Boolean).join(" · ") || "投递记录",
      description: "上次练习对应的投递记录",
    });
  } else if (session.jd_snapshot_id && session.job_posting_id && session.jd_version) {
    resources.push({
      kind: "jd_snapshot",
      jdSnapshotId: session.jd_snapshot_id,
      jobPostingId: session.job_posting_id,
      title: session.job_title ?? "岗位",
      companyName: session.job_company_name ?? "",
      jdVersion: session.jd_version,
    });
  }
  if (!session.application_id && session.resume_version_id && session.resume_id && session.resume_name
      && session.resume_version_number) {
    resources.push({
      resumeVersionId: session.resume_version_id,
      resumeId: session.resume_id,
      name: session.resume_name,
      versionNumber: session.resume_version_number,
      documentFormat: session.resume_document_format ?? "pdf",
      byteSize: session.resume_byte_size ?? 0,
      uploadedAt: null,
    });
  }

  const settings = [
    session.application_id ? "针对附带的投递记录" : "自由练习",
    !session.application_id && !session.jd_snapshot_id && session.target_company
      ? `公司「${session.target_company}」` : null,
    session.target_role ? `目标岗位「${session.target_role}」` : null,
    session.interview_type_label,
    `${session.max_primary_questions} 道主问题`,
    `每题最多追问 ${session.max_follow_ups_per_question ?? 0} 次`,
    !session.application_id && !session.resume_version_id ? "不用简历" : null,
  ].filter(Boolean).join("，");
  return {
    prompt: `再来一次模拟面试，设置和上次相同：${settings}。直接开始。`,
    resource: resources[0] ?? null,
    additionalResources: resources.slice(1),
    label: "再来一次模拟面试",
  };
}
