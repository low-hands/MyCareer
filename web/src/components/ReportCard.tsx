import { useEffect, useState } from "react";

import { fetchReport, type ReportView } from "../api/client";
import type { MessageResource } from "../chat/reducer";
import { AppIcon } from "./AppIcon";
import { MarkdownContent } from "./MarkdownContent";

interface ReportCardProps {
  resource: MessageResource;
  userId: string;
  apiBaseUrl: string;
}

const KIND_LABELS: Record<MessageResource["kind"], string> = {
  job_research_report: "公司调研报告",
  mock_interview_report: "模拟面试报告",
  interview_preparation: "面试准备材料",
  interview_retro_report: "真实面试复盘",
  resume_job_match: "简历岗位匹配",
  resume_tailoring_draft: "简历定制草稿",
};

/**
 * The full report behind a message that only summarizes it.
 *
 * Fetched on expand rather than with the transcript: a conversation can hold
 * several reports of a few thousand characters each, and restoring it should
 * not pull all of them. Nothing here is cached across collapses, which keeps a
 * reopened card showing the current stored report rather than a stale copy.
 */
export function ReportCard({ resource, userId, apiBaseUrl }: ReportCardProps) {
  const [expanded, setExpanded] = useState(false);
  const [report, setReport] = useState<ReportView | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!expanded) return;
    const request = new AbortController();
    setError(null);
    void fetchReport(userId, resource.kind, resource.resourceId, {
      statusAtDelivery: resource.statusAtDelivery,
      anchoredByOtherJob: resource.anchoredByOtherJob,
    }, {
      apiBaseUrl,
      signal: request.signal,
    })
      .then(setReport)
      .catch((cause: unknown) => {
        if (!request.signal.aborted) {
          setError(cause instanceof Error ? cause.message : "读取报告失败。");
        }
      });
    return () => request.abort();
  }, [expanded, userId, resource.kind, resource.resourceId, resource.statusAtDelivery, resource.anchoredByOtherJob, apiBaseUrl]);

  return (
    <div className="report-card">
      <button
        type="button"
        className="report-card-header"
        aria-expanded={expanded}
        onClick={() => setExpanded((open) => !open)}
      >
        <span className="report-card-icon" aria-hidden="true">
          <AppIcon name="document" size={16} />
        </span>
        <span className="report-card-label">
          <strong>{report?.title ?? KIND_LABELS[resource.kind]}</strong>
          <small>{report?.subtitle ?? "点开查看完整报告"}</small>
        </span>
        <span className="report-card-toggle" aria-hidden="true">
          {expanded ? "收起" : "展开"}
        </span>
      </button>
      {expanded ? (
        <div className="report-card-body">
          {error ? (
            <div className="error-banner" role="alert">
              {error}
            </div>
          ) : report ? (
            <MarkdownContent content={report.body} className="markdown-content" />
          ) : (
            <div className="history-loading">
              <span className="spinner" /> 正在读取报告…
            </div>
          )}
        </div>
      ) : null}
    </div>
  );
}
