import { useEffect, useId, useRef, useState } from "react";

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
      <div className="match-history-heading"><div><h3>简历匹配{history ? <span>{history.total} 份报告</span> : null}</h3><p>回顾匹配结果，找到下一步改进方向。</p></div>
        <button type="button" className="link" onClick={() => setReload((value) => value + 1)}>刷新匹配历史</button>
      </div>
      {error ? <p role="alert">{error}</p> : !history ? <p role="status">正在读取匹配历史…</p> : null}
      {history?.total === 0 ? <p>尚未匹配。请选择一个简历版本发起匹配。</p> : null}
      {history?.items.map((match) => (
        <article key={match.report_id} className="match-history-item">
          <div className="match-result-heading">
            <div><h4>{match.resume_name ?? "历史简历"}</h4>
              <span className="match-result-meta">简历 v{match.resume_version_number ?? "未知"} × JD v{match.jd_version ?? "未知"}</span>
            </div>
            <span className={`match-fit match-fit-${match.overall_fit}`}>{FIT_LABELS[match.overall_fit] ?? "待评估"}</span>
          </div>
          <time>{new Date(match.created_at).toLocaleString("zh-CN", { dateStyle: "medium", timeStyle: "short" })}</time>
          {match.current_jd === false ? <p className="match-history-notice">岗位描述已更新，这份结果基于更新前的内容。</p> : null}
          <p className="match-result-summary">{match.summary}</p>
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
  const [pickerOpen, setPickerOpen] = useState(false);
  const [search, setSearch] = useState("");
  const searchRef = useRef<HTMLInputElement>(null);
  const pickerRef = useRef<HTMLDivElement>(null);
  const pickerId = useId();
  const triggerRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    if (pickerOpen) { setSearch(""); searchRef.current?.focus(); }
  }, [pickerOpen]);

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

  useEffect(() => {
    if (!pickerOpen) return;
    const close = (event: MouseEvent) => {
      if (pickerRef.current && !pickerRef.current.contains(event.target as Node)) setPickerOpen(false);
    };
    document.addEventListener("mousedown", close);
    return () => document.removeEventListener("mousedown", close);
  }, [pickerOpen]);

  const resume = resumes?.find((item) => item.versions.some((version) => version.id === versionId));
  const version = resume?.versions.find((item) => item.id === versionId);
  const terms = search.trim().toLocaleLowerCase().split(/\s+/).filter(Boolean);
  const filteredVersions = (resumes ?? []).flatMap((item) => item.versions.map((entry) => ({ item, entry })))
    .filter(({ item, entry }) => terms.every((term) => `${item.name} v${entry.version_number}`.toLocaleLowerCase().includes(term)));
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
      <div className="resume-version-picker" ref={pickerRef} onBlur={(event) => {
        if (!event.currentTarget.contains(event.relatedTarget)) setPickerOpen(false);
      }} onKeyDown={(event) => {
        if (event.nativeEvent.isComposing) return;
        if (event.target === searchRef.current && ["Home", "End"].includes(event.key)) return;
        if (event.target === searchRef.current && event.key === "Enter") { event.preventDefault(); return; }
        if (event.key === "Escape") { event.preventDefault(); setPickerOpen(false); triggerRef.current?.focus(); }
        if (["ArrowDown", "ArrowUp", "Home", "End"].includes(event.key)) {
          event.preventDefault();
          if (!pickerOpen) { setPickerOpen(true); return; }
          const options = Array.from(pickerRef.current?.querySelectorAll<HTMLButtonElement>('[role="option"]') ?? []);
          const index = options.indexOf(document.activeElement as HTMLButtonElement);
          const next = index === -1 ? (event.key === "ArrowUp" ? options.length - 1 : 0) : event.key === "Home" ? 0 : event.key === "End" ? options.length - 1 : (index + (event.key === "ArrowDown" ? 1 : -1) + options.length) % options.length;
          options[next]?.focus();
        }
      }}>
        <span className="resume-version-label" id={`${pickerId}-label`}>选择用于匹配的简历</span>
        <button ref={triggerRef} type="button" className={`resume-version-trigger ${pickerOpen ? "is-open" : ""}`} disabled={started || !resumes?.some((item) => item.versions.length)} aria-labelledby={`${pickerId}-label ${pickerId}-value`} aria-controls={pickerOpen ? pickerId : undefined} aria-haspopup="listbox" aria-expanded={pickerOpen} onClick={() => setPickerOpen((value) => !value)}>
          <span className="resume-version-trigger-icon">CV</span>
          <span className="resume-version-trigger-copy" id={`${pickerId}-value`}>
            <strong>{version ? resume?.name : "请选择简历版本"}</strong>
            <small>{version ? `v${version.version_number} · ${new Date(version.created_at).toLocaleDateString("zh-CN")}` : "从已导入的简历中选择"}</small>
          </span>
          <span className="resume-version-chevron" aria-hidden="true">⌄</span>
        </button>
        {pickerOpen ? <div className="resume-version-menu">
          <div className="resume-version-search">
            <input ref={searchRef} type="search" value={search} onChange={(event) => setSearch(event.target.value)} aria-label="搜索简历名称或版本" placeholder="搜索简历名称或版本，如 v2" aria-controls={pickerId} />
          </div>
          <div className="resume-version-results" id={pickerId} role="listbox" aria-label="简历版本">
          {filteredVersions.map(({ item, entry }) => (
            <button type="button" role="option" tabIndex={-1} aria-selected={versionId === entry.id} className={`resume-version-option ${versionId === entry.id ? "is-selected" : ""}`} key={entry.id} onClick={() => { setVersionId(entry.id); setPickerOpen(false); triggerRef.current?.focus(); }}>
              <span className="resume-version-option-icon">CV</span>
              <span className="resume-version-option-copy"><strong>{item.name}</strong><small>v{entry.version_number} · {new Date(entry.created_at).toLocaleDateString("zh-CN")}</small></span>
              {versionId === entry.id ? <span className="resume-version-check" aria-hidden="true">✓</span> : null}
            </button>
          ))}
          </div>
          {filteredVersions.length === 0 ? <p className="resume-version-empty" role="status">没有找到对应简历，试试其他名称或版本。</p> : null}
        </div> : null}
      </div>
      <button type="submit" disabled={!version || !snapshot || !onStartTask || started}>
        {started ? "匹配任务已排队" : "开始匹配"}
      </button>
    </form>
  );
}
