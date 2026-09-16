import { useEffect, useState } from "react";

import { ApiError, fetchSavedJobSnapshot, type SavedJobSnapshot } from "../api/client";
import type { MessageResource } from "../chat/reducer";
import { AppIcon } from "./AppIcon";

/** A message resource that is a saved job, pinned to one JD version. */
export type SavedJobResource = MessageResource & { kind: "saved_job" };

interface SavedJobCardProps {
  resource: SavedJobResource;
  apiBaseUrl: string;
}

/**
 * The JD behind a reply that only names the job.
 *
 * `resourceId` is the immutable `jd_snapshot_id`, so a card in an old turn
 * keeps opening the version that turn read after the posting is captured
 * again. The heading comes from the reference kept with the message, which is
 * why a deleted posting still shows its name here and only the text is gone.
 */
export function SavedJobCard({ resource, apiBaseUrl }: SavedJobCardProps) {
  const [expanded, setExpanded] = useState(false);
  const [snapshot, setSnapshot] = useState<SavedJobSnapshot | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [removed, setRemoved] = useState(resource.available === false);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    if (!expanded || removed) return;
    const request = new AbortController();
    setSnapshot(null);
    setError(null);
    void fetchSavedJobSnapshot(resource.resourceId, { apiBaseUrl, signal: request.signal })
      .then((value) => {
        if (!request.signal.aborted) setSnapshot(value);
      })
      .catch((cause: unknown) => {
        if (request.signal.aborted) return;
        if (cause instanceof ApiError && cause.status === 404) {
          setRemoved(true);
          return;
        }
        setError(cause instanceof Error ? cause.message : "读取岗位描述失败。");
      });
    return () => request.abort();
  }, [expanded, removed, resource.resourceId, apiBaseUrl]);

  const copyJd = () => {
    if (!snapshot) return;
    void navigator.clipboard.writeText(snapshot.jd_text).then(() => {
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    });
  };

  const heading = snapshot
    ? `${snapshot.company_name}｜${snapshot.title}`
    : resource.title ?? "已保存岗位";
  const subheading = removed
    ? "该岗位已删除或不可访问"
    : snapshot
      ? `JD 第 ${snapshot.jd_version} 版 · ${snapshot.source_name}${
          snapshot.latest_jd_version > snapshot.jd_version
            ? `（当前已更新到第 ${snapshot.latest_jd_version} 版）`
            : ""
        }`
      : resource.description ?? "点开查看完整 JD";

  return (
    <div className={`report-card saved-job-card ${removed ? "is-unavailable" : ""}`}>
      <button
        type="button"
        className="report-card-header"
        aria-expanded={expanded}
        onClick={() => setExpanded((open) => !open)}
      >
        <span className="report-card-icon" aria-hidden="true">
          <AppIcon name="building" size={16} />
        </span>
        <span className="report-card-label">
          <strong>{heading}</strong>
          <small>{subheading}</small>
        </span>
        <span className="report-card-toggle" aria-hidden="true">
          {expanded ? "收起" : "查看完整 JD"}
        </span>
      </button>
      {expanded ? (
        <div className="report-card-body">
          {error ? (
            <div className="error-banner" role="alert">
              {error}
            </div>
          ) : removed ? (
            <div role="status">该岗位已删除或不可访问，无法读取原文。</div>
          ) : snapshot ? (
            <>
              <div className="saved-job-card-actions">
                <button type="button" onClick={copyJd}>
                  {copied ? "已复制" : "复制 JD"}
                </button>
                {snapshot.source_url ? (
                  <a href={snapshot.source_url} target="_blank" rel="noreferrer">
                    打开来源页面
                  </a>
                ) : null}
              </div>
              <pre className="saved-job-card-text">{snapshot.jd_text}</pre>
            </>
          ) : (
            <div className="history-loading">
              <span className="spinner" /> 正在读取 JD…
            </div>
          )}
        </div>
      ) : null}
    </div>
  );
}
