import { resumeDocumentUrl } from "../api/client";
import type { MessageResource } from "../chat/reducer";
import { AppIcon } from "./AppIcon";

/**
 * A resume version pinned to a user message.
 *
 * The chip shows the snapshot the transcript kept (name, version, format,
 * size, upload time) and links to the stored original only while the server
 * still reports the version as reachable; once the resume is deleted the
 * snapshot stays but the chip says so instead of offering a dead link.
 */
export function ResumeAttachmentCard({
  resource,
  apiBaseUrl,
}: {
  resource: MessageResource;
  apiBaseUrl: string;
}) {
  const reachable = resource.available !== false && Boolean(resource.resumeId);
  return (
    <div className={`resume-attachment ${reachable ? "" : "is-unavailable"}`}>
      <span className="resume-attachment-icon" aria-hidden="true">
        <AppIcon name="document" size={16} />
      </span>
      <span className="resume-attachment-label">
        <strong>{resource.title ?? "简历版本"}</strong>
        <small>
          {reachable
            ? resource.description ?? "已附带的简历版本"
            : `该简历已删除/不可访问${resource.description ? ` · ${resource.description}` : ""}`}
        </small>
      </span>
      {reachable && resource.resumeId ? (
        <span className="resume-attachment-actions">
          <a
            href={resumeDocumentUrl(resource.resumeId, resource.resourceId, { apiBaseUrl })}
            target="_blank"
            rel="noreferrer"
          >
            查看原文件
          </a>
          <a href={resumeDocumentUrl(resource.resumeId, resource.resourceId, { apiBaseUrl, download: true })}>
            下载
          </a>
        </span>
      ) : null}
    </div>
  );
}
