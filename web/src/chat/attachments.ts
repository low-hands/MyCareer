import type { ResumeImportResult, ResumeVersionView } from "../api/client";
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
export function toInputResources(attachments: ResumeAttachment[]): TurnInputResource[] {
  return attachments.map((item) => ({ kind: "resume_version", id: item.resumeVersionId }));
}

/** The chips to show on the sent message until the transcript is re-read. */
export function toMessageResources(attachments: ResumeAttachment[]): MessageResource[] {
  return attachments.map((item) => ({
    kind: "resume_version",
    resourceId: item.resumeVersionId,
    resumeId: item.resumeId,
    title: `${item.name} v${item.versionNumber}`,
    description: `${item.documentFormat} · ${formatBytes(item.byteSize)}`,
    available: true,
  }));
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
