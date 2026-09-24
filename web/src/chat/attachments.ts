import type { ResumeImportResult, ResumeVersionView, SavedJobView } from "../api/client";
import type { TurnInputResource } from "../api/sse";
import type { MessageResource } from "./reducer";

/**
 * A resume version the user has picked for the next message.
 *
 * Only identifiers and display metadata live here: the file itself is already
 * in the resume library, and the message will carry the version id, so the
 * client never holds or re-sends the document.
 */
export interface ResumeAttachment {
  resumeVersionId: string;
  resumeId: string;
  name: string;
  versionNumber: number;
  documentFormat: string;
  byteSize: number;
  uploadedAt: string | null;
}

/**
 * A saved job pinned to one exact JD snapshot, for a task that reads the JD
 * and nothing else. It is never queued with the chat's resume attachments:
 * the only place it travels is a standalone task, so a JD-only analysis
 * cannot silently pick up a resume.
 */
export interface JobAttachment {
  kind: "jd_snapshot";
  jdSnapshotId: string;
  jobPostingId: string;
  title: string;
  companyName: string;
  jdVersion: number;
}

export interface ApplicationAttachment {
  kind: "application";
  applicationId: string;
  title: string;
  description: string;
}

export type ChatAttachment = ResumeAttachment | JobAttachment | ApplicationAttachment;

export function isJobAttachment(item: ChatAttachment): item is JobAttachment {
  return "kind" in item && item.kind === "jd_snapshot";
}

export function isApplicationAttachment(item: ChatAttachment): item is ApplicationAttachment {
  return "kind" in item && item.kind === "application";
}

/** The current JD snapshot of a saved job, or null when none was captured. */
export function attachmentFromSavedJob(job: SavedJobView): JobAttachment | null {
  if (job.jd_snapshot_id === null || job.jd_version === null) return null;
  return {
    kind: "jd_snapshot",
    jdSnapshotId: job.jd_snapshot_id,
    jobPostingId: job.id,
    title: job.title,
    companyName: job.company_name,
    jdVersion: job.jd_version,
  };
}

/** The line shown while a standalone task waits for its own conversation. */
export function attachmentLabel(item: ChatAttachment): string {
  if (isJobAttachment(item)) return `岗位「${item.companyName} · ${item.title}」的 JD v${item.jdVersion}`;
  if (isApplicationAttachment(item)) return `投递记录「${item.title}」`;
  return `简历“${item.name}” v${item.versionNumber}`;
}

export const MAX_ATTACHMENTS = 8;

export const DEFAULT_ATTACHMENT_PROMPT = "帮我分析这份简历";

export function attachmentFromImport(result: ResumeImportResult): ResumeAttachment {
  return {
    resumeVersionId: result.resume_version_id,
    resumeId: result.resume_id,
    name: result.name,
    versionNumber: result.version_number,
    documentFormat: result.document_format,
    byteSize: result.byte_size,
    uploadedAt: null,
  };
}

export function attachmentFromVersion(
  resumeName: string,
  version: ResumeVersionView,
): ResumeAttachment {
  return {
    resumeVersionId: version.id,
    resumeId: version.resume_id,
    name: resumeName,
    versionNumber: version.version_number,
    documentFormat: version.document_format,
    byteSize: version.byte_size,
    uploadedAt: version.created_at,
  };
}

/** Adds `next` unless the same version is already attached or the list is full. */
export function withAttachment(
  current: ResumeAttachment[],
  next: ResumeAttachment,
): ResumeAttachment[] {
  if (current.some((item) => item.resumeVersionId === next.resumeVersionId)) return current;
  if (current.length >= MAX_ATTACHMENTS) return current;
  return [...current, next];
}

/** The structured references sent with the message, one per exact version. */
export function toInputResources(attachments: ChatAttachment[]): TurnInputResource[] {
  return attachments.map((item) =>
    isJobAttachment(item)
      ? { kind: "jd_snapshot", id: item.jdSnapshotId }
      : isApplicationAttachment(item)
        ? { kind: "application", id: item.applicationId }
      : { kind: "resume_version", id: item.resumeVersionId },
  );
}

/** The chips to show on the sent message until the transcript is re-read. */
export function toMessageResources(attachments: ChatAttachment[]): MessageResource[] {
  return attachments.map((item) =>
    isJobAttachment(item)
      ? {
          kind: "saved_job",
          resourceId: item.jdSnapshotId,
          jobPostingId: item.jobPostingId,
          title: `${item.companyName} · ${item.title}`,
          description: `JD 快照 v${item.jdVersion}`,
          available: true,
        }
      : isApplicationAttachment(item)
        ? { kind: "application", resourceId: item.applicationId, title: item.title, description: item.description }
      : {
          kind: "resume_version",
          resourceId: item.resumeVersionId,
          resumeId: item.resumeId,
          title: `${item.name} v${item.versionNumber}`,
          description: `${item.documentFormat} · ${formatBytes(item.byteSize)}`,
          available: true,
        },
  );
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

const RESUME_FILE_PATTERN = /\.(pdf|txt|md|markdown)$/i;

/** The first droppable resume file in a drag payload, if any. */
export function droppedResumeFile(files: FileList | File[] | null | undefined): File | null {
  if (!files) return null;
  for (const file of Array.from(files)) {
    if (RESUME_FILE_PATTERN.test(file.name) || file.type === "application/pdf") return file;
  }
  return null;
}
