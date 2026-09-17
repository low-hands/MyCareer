import { useEffect, useState } from "react";

import {
  fetchJobMatches, fetchResumes,
  type JobMatchHistory, type ResumeView, type SavedJobView,
} from "../api/client";
import { attachmentFromSavedJob, attachmentFromVersion, type ChatAttachment } from "../chat/attachments";
import { ReportCard } from "./ReportCard";

const FIT_LABELS: Record<string, string> = {
  strong: "强匹配", moderate: "中等匹配", weak: "弱匹配", insufficient_evidence: "证据不足",
};

export function JobMatchesPanel({
  job, apiBaseUrl, refreshToken, onStartTask,
}: {
  job: SavedJobView;
  apiBaseUrl: string;
  refreshToken: number;
  onStartTask?: (prompt: string, resource: ChatAttachment, additionalResources?: ChatAttachment[]) => void;
}) {
  const [history, setHistory] = useState<JobMatchHistory | null>(null);
  const [offset, setOffset] = useState(0);
  const [reload, setReload] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [chooseResume, setChooseResume] = useState(false);

  useEffect(() => {
    const request = new AbortController();
    setHistory(null);
    setError(null);
    void fetchJobMatches(job.id, offset, { apiBaseUrl, signal: request.signal })
      .then((value) => { if (!request.signal.aborted) setHistory(value); })
      .catch((cause: unknown) => {
        if (!request.signal.aborted) setError(cause instanceof Error ? cause.message : "读取匹配历史失败。");
      });
    return () => request.abort();
  }, [job.id, job.resume_match_count, apiBaseUrl, refreshToken, offset, reload]);

  return (
    <section className="job-match-history" aria-label="岗位匹配历史">
      <h3>匹配分析{history ? `（${history.total} 份）` : ""}</h3>
      <p>查看已保存的报告不会重新运行匹配。</p>
      {error ? <p role="alert">{error}</p> : !history ? <p role="status">正在读取匹配历史…</p> : null}
      <button type="button" className="link" onClick={() => setReload((value) => value + 1)}>刷新匹配历史</button>
      {history?.total === 0 ? <p>尚未匹配。请选择一个简历版本发起匹配。</p> : null}
      {history?.items.map((match) => (
        <article key={match.report_id} className="match-history-item">
          <p>
            {match.resume_name ?? "历史简历"} · 简历 v{match.resume_version_number ?? "未知"}
            {" × "}JD v{match.jd_version ?? "未知"}
            {match.current_jd === false ? " · 历史 JD 快照" : ""}
            {" · "}{FIT_LABELS[match.overall_fit] ?? match.overall_fit}
          </p>
          <time>{new Date(match.created_at).toLocaleString("zh-CN")}</time>
          <p>{match.summary}</p>
          <ReportCard
            resource={{ kind: "resume_job_match", resourceId: match.report_id, title: "查看这份匹配报告" }}
            apiBaseUrl={apiBaseUrl}
          />
        </article>
      ))}
      {history && history.total > history.limit ? (
        <div className="job-card-actions">
          <button type="button" disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - history.limit))}>上一页</button>
          <span>{offset + 1}–{Math.min(offset + history.limit, history.total)} / {history.total}</span>
          <button type="button" disabled={offset + history.limit >= history.total} onClick={() => setOffset(offset + history.limit)}>下一页</button>
        </div>
      ) : null}
      <button type="button" className="soft-button" onClick={() => setChooseResume((value) => !value)}>
        {chooseResume ? "取消选择" : "选择简历版本发起匹配"}
      </button>
      {chooseResume ? <MatchResumePicker job={job} apiBaseUrl={apiBaseUrl} onStartTask={onStartTask} /> : null}
    </section>
  );
}

function MatchResumePicker({
  job, apiBaseUrl, onStartTask,
}: Pick<Parameters<typeof JobMatchesPanel>[0], "job" | "apiBaseUrl" | "onStartTask">) {
  const [snapshot] = useState(() => attachmentFromSavedJob(job));
  const [resumes, setResumes] = useState<ResumeView[] | null>(null);
  const [versionId, setVersionId] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [reload, setReload] = useState(0);
  const [started, setStarted] = useState(false);

  useEffect(() => {
    const request = new AbortController();
    setError(null);
    setResumes(null);
    void fetchResumes({ apiBaseUrl, signal: request.signal })
      .then((value) => { if (!request.signal.aborted) setResumes(value); })
      .catch((cause: unknown) => {
        if (!request.signal.aborted) setError(cause instanceof Error ? cause.message : "读取简历版本失败。");
      });
    return () => request.abort();
  }, [apiBaseUrl, reload]);

  const resume = resumes?.find((item) => item.versions.some((version) => version.id === versionId));
  const version = resume?.versions.find((item) => item.id === versionId);
  return (
    <form className="match-resume-picker" onSubmit={(event) => {
      event.preventDefault();
      if (!resume || !version || !snapshot || !onStartTask || started) return;
      setStarted(true);
      onStartTask(
        `请将简历「${resume.name}」v${version.version_number} 与「${snapshot.companyName} · ${snapshot.title}」JD v${snapshot.jdVersion} 进行岗位匹配，使用附带的精确简历版本和 JD 快照，保存匹配报告。`,
        attachmentFromVersion(resume.name, version),
        [snapshot],
      );
    }}>
      <p>本次匹配使用 JD v{snapshot?.jdVersion ?? "未知"}，在独立新对话中运行。</p>
      {!snapshot ? <p role="alert">岗位没有可用的 JD 快照，请先保存完整 JD。</p> : null}
      {!onStartTask ? <p role="alert">当前页面无法启动独立任务，请重新打开工作区。</p> : null}
      {error ? <p role="alert">{error} <button type="button" onClick={() => setReload((value) => value + 1)}>重试</button></p> : null}
      {!resumes && !error ? <p role="status">正在读取简历版本…</p> : null}
      {resumes?.length === 0 ? <p>请先到简历库导入简历。</p> : null}
      <label>匹配使用的简历版本
        <select value={versionId} onChange={(event) => setVersionId(event.target.value)} disabled={started}>
          <option value="">请选择简历版本</option>
          {resumes?.map((item) => (
            <optgroup key={item.id} label={item.name}>
              {item.versions.map((entry) => (
                <option value={entry.id} key={entry.id}>{item.name} · v{entry.version_number} · {new Date(entry.created_at).toLocaleString("zh-CN")}</option>
              ))}
            </optgroup>
          ))}
        </select>
      </label>
      <button type="submit" disabled={!version || !snapshot || !onStartTask || started}>
        {started ? "匹配任务已排队" : "开始匹配"}
      </button>
    </form>
  );
}
