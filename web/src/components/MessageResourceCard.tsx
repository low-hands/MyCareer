import { REPORT_KINDS } from "../chat/events";
import type { MessageResource } from "../chat/reducer";
import { ReportCard, type ReportResource } from "./ReportCard";
import { ResumeAttachmentCard } from "./ResumeAttachmentCard";
import { SavedJobCard, type SavedJobResource } from "./SavedJobCard";

const REPORT_RESOURCE_KINDS = new Set<string>([...REPORT_KINDS, "delivered_body"]);

export function isReportResource(resource: MessageResource): resource is ReportResource {
  return REPORT_RESOURCE_KINDS.has(resource.kind);
}

export function isSavedJobResource(resource: MessageResource): resource is SavedJobResource {
  return resource.kind === "saved_job";
}

/**
 * Picks the card for a message resource by its kind, explicitly.
 *
 * Each kind names its card; a kind this build does not know renders a plain
 * chip naming it rather than a report card that would fetch the wrong
 * endpoint and fail.
 */
export function MessageResourceCard({
  resource,
  apiBaseUrl,
}: {
  resource: MessageResource;
  apiBaseUrl: string;
}) {
  if (resource.kind === "resume_version") {
    return <ResumeAttachmentCard resource={resource} apiBaseUrl={apiBaseUrl} />;
  }
  if (isSavedJobResource(resource)) {
    return <SavedJobCard resource={resource} apiBaseUrl={apiBaseUrl} />;
  }
  if (isReportResource(resource)) {
    return <ReportCard resource={resource} apiBaseUrl={apiBaseUrl} />;
  }
  return <UnknownResourceChip kind={resource.kind} title={resource.title} />;
}

function UnknownResourceChip({ kind, title }: { kind: string; title?: string | null }) {
  return (
    <div className="resume-attachment is-unavailable" role="note">
      <span className="resume-attachment-label">
        <strong>{title ?? "未知资源"}</strong>
        <small>当前版本不支持展示此类资源（{kind}）</small>
      </span>
    </div>
  );
}
