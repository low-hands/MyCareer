import { useEffect, useState } from "react";

import { ApiError, fetchReport, type ReportView } from "../api/client";
import type { ReportResourceKind } from "../chat/events";
import type { MessageResource } from "../chat/reducer";
import { AppIcon } from "./AppIcon";
import { MarkdownContent } from "./MarkdownContent";
import { CompanyReportContent } from "./CompanyReportContent";
import { MatchReportContent } from "./MatchReportContent";
import { MatchProvenance } from "./MatchProvenance";

/** A message resource that is a stored report, as opposed to an attached resume. */
export type ReportResource = MessageResource & { kind: ReportResourceKind };

interface ReportCardProps {
  resource: ReportResource;
  apiBaseUrl: string;
}

const KIND_LABELS: Record<ReportResourceKind, string> = {
  job_research_report: "公司调研报告",
  mock_interview_report: "模拟面试报告",
  interview_preparation: "面试准备材料",
  interview_retro_report: "真实面试复盘",
  resume_job_match: "简历岗位匹配",
  resume_tailoring_draft: "简历定制草稿",
  job_analysis: "岗位 JD 分析",
  delivered_body: "完整内容",
};

const FIT_SUMMARY: Record<string, string> = {
  strong: "匹配度较高", moderate: "有一定匹配", weak: "匹配度偏低", insufficient_evidence: "信息还不够",
};
const FIT_EMOJI: Record<string, string> = { strong: "🎯", moderate: "🟡", weak: "🧩", insufficient_evidence: "🔎" };

/**
 * The full report behind a message that only summarizes it.
 *
 * Fetched on expand rather than with the transcript: a conversation can hold
 * several reports of a few thousand characters each, and restoring it should
 * not pull all of them. Nothing here is cached across collapses, which keeps a
 * reopened card showing the current stored report rather than a stale copy.
 */
export function ReportCard({ resource, apiBaseUrl }: ReportCardProps) {
  const [expanded, setExpanded] = useState(false);
  const [report, setReport] = useState<ReportView | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [removed, setRemoved] = useState(false);

  useEffect(() => {
    if (!expanded) return;
    const request = new AbortController();
    setReport(null);
    setError(null);
    setRemoved(false);
    void fetchReport(resource.kind, resource.resourceId, {
      statusAtDelivery: resource.statusAtDelivery,
      anchoredByOtherJob: resource.anchoredByOtherJob,
    }, {
      apiBaseUrl,
      signal: request.signal,
    })
      .then((value) => {
        if (!request.signal.aborted) setReport(value);
      })
      .catch((cause: unknown) => {
        if (request.signal.aborted) return;
        // The transcript still lists the resource, but the content behind it
        // was deleted along with the job or memory it depended on.
        if (cause instanceof ApiError && cause.status === 404) {
          setRemoved(true);
          return;
        }
        setError(cause instanceof Error ? cause.message : "读取报告失败。");
      });
    return () => request.abort();
  }, [expanded, resource.kind, resource.resourceId, resource.statusAtDelivery, resource.anchoredByOtherJob, apiBaseUrl]);

  return (
    <div className="report-card">
      <button
        type="button"
        className="report-card-header"
        aria-expanded={expanded}
        onClick={() => {
          setReport(null);
          setError(null);
          setRemoved(false);
          setExpanded((open) => !open);
        }}
      >
        <span className="report-card-icon" aria-hidden="true">
          <AppIcon name="document" size={16} />
        </span>
        <span className="report-card-label">
          <strong>{report?.title ?? resource.title ?? KIND_LABELS[resource.kind]}</strong>
          <small>{report?.resume_job_match ? "匹配结论与改进建议" : report?.subtitle ?? (removed ? "内容已删除" : "点开查看完整内容")}</small>
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
          ) : removed ? (
            <div role="status">内容已删除，无法继续查看。</div>
          ) : report?.availability === "expired" ? (
            <div role="status">内容已过期，无法继续查看。</div>
          ) : report ? (
            <>
              {report.resume_job_match ? <div className={`match-report-hero match-report-${report.resume_job_match.overall_fit}`}>
                <div><span className="match-report-eyebrow">匹配结论</span><strong><span aria-hidden="true">{FIT_EMOJI[report.resume_job_match.overall_fit] ?? "📌"}</span> {FIT_SUMMARY[report.resume_job_match.overall_fit] ?? "待评估"}</strong></div>
              </div> : null}
              {resource.kind === "resume_job_match" ? <MatchReportContent body={report.body} hasVerdict={Boolean(report.resume_job_match)} /> : resource.kind === "job_research_report" ? <CompanyReportContent body={report.body} /> : <MarkdownContent content={report.body} className="markdown-content" />}
              {report.resume_job_match ? (
                <MatchProvenance match={report.resume_job_match} apiBaseUrl={apiBaseUrl} />
              ) : null}
            </>
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
