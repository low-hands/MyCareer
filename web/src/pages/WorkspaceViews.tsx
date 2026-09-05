import { useCallback, useEffect, useState } from "react";

import {
  type ApplicationView,
  type ApplicationMockInterviews,
  type CalendarWorkspace,
  type CompanyResearchView,
  type Dashboard,
  type EmailWorkspace,
  type ResumeView,
  type SavedJobView,
  createApplication,
  fetchApplications,
  fetchApplicationMockInterviews,
  fetchCalendar,
  connectQQ,
  disconnectIntegration,
  fetchCompanyResearch,
  fetchDashboard,
  fetchEmailWorkspace,
  fetchResumes,
  deleteSavedJob,
  fetchSavedJobDetail,
  fetchSavedJobs,
  startGoogleConnection,
  setJobAvailability,
  setJobPursuit,
  type SavedJobDetail,
} from "../api/client";
import { AppIcon, type AppIconName } from "../components/AppIcon";
import { ReportCard } from "../components/ReportCard";
import { ResumeImporter } from "../components/ResumeImporter";
import { calendarDays, eventsByDay, monthKey } from "../calendar/month";

type PageProps = {
  apiBaseUrl: string;
  refreshToken: number;
  hidden: boolean;
  onAskAgent: (prompt: string) => void;
};

const STATUS_LABELS: Record<string, string> = {
  submitted: "已投递",
  acknowledged: "已确认",
  interviewing: "面试中",
  offer: "Offer",
  rejected: "未通过",
  withdrawn: "已撤回",
  current: "当前有效",
  outdated: "需要更新",
  superseded: "历史版本",
  active: "已连接",
  cancelled: "已取消",
};

const STALE_AFTER_DAYS = 30;

/** How long since the posting was last seen on the site, when that is worth saying.
 *
 * Not "已下架": nothing here checks whether a posting is still live, and a
 * badge that claims it does would be stating a fact the system cannot observe.
 * What *is* known is when the page was last read — captured, or re-captured on
 * a later visit — so that is what gets said. The reader draws their own
 * conclusion, which is the honest division here and the one this codebase
 * applies everywhere else: report what was observed, never infer the rest.
 */
function stalenessLabel(lastCheckedAt: string): string | null {
  const days = Math.floor((Date.now() - new Date(lastCheckedAt).getTime()) / 86_400_000);
  return days >= STALE_AFTER_DAYS ? `${days} 天未确认` : null;
}

function dateLabel(value: string): string {
  return new Date(value).toLocaleDateString("zh-CN", { month: "short", day: "numeric" });
}

export function safeMeetingUrl(value: string | null | undefined): string | null {
  if (!value) return null;
  try {
    const url = new URL(value);
    return url.protocol === "https:" ? url.href : null;
  } catch {
    return null;
  }
}

function formatBytes(bytes: number): string {
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function usePageData<T>(
  load: (signal: AbortSignal) => Promise<T>,
  refreshToken: number,
  enabled: boolean,
) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [reloadToken, setReloadToken] = useState(0);
  useEffect(() => {
    if (!enabled) return;
    const controller = new AbortController();
    setLoading(true);
    void load(controller.signal)
      .then((next) => {
        setData(next);
        setError(null);
      })
      .catch((cause: unknown) => {
        if (!controller.signal.aborted) {
          setError(cause instanceof Error ? cause.message : "读取数据失败。");
        }
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => controller.abort();
  }, [load, refreshToken, reloadToken, enabled]);
  return { data, error, loading, reload: () => setReloadToken((value) => value + 1) };
}

function PageHeader({
  icon,
  eyebrow,
  title,
  description,
  loading,
  onRefresh,
}: {
  icon: AppIconName;
  eyebrow: string;
  title: string;
  description: string;
  loading: boolean;
  onRefresh: () => void;
}) {
  return (
    <header className="management-header">
      <div className="management-title">
        <span className="page-icon"><AppIcon name={icon} size={25} /></span>
        <div>
          <p className="eyebrow">{eyebrow}</p>
          <h1>{title}</h1>
          <p>{description}</p>
        </div>
      </div>
      <button className="soft-button" type="button" onClick={onRefresh} disabled={loading}>
        <AppIcon name="refresh" size={16} className={loading ? "is-spinning" : undefined} />
        {loading ? "刷新中" : "刷新"}
      </button>
    </header>
  );
}

function EmptyState({
  icon,
  title,
  description,
  action,
  onAction,
}: {
  icon: AppIconName;
  title: string;
  description: string;
  action?: string;
  onAction?: () => void;
}) {
  return (
    <div className="management-empty">
      <span><AppIcon name={icon} size={28} /></span>
      <strong>{title}</strong>
      <p>{description}</p>
      {action && onAction ? <button type="button" onClick={onAction}>{action}</button> : null}
    </div>
  );
}

function ErrorBanner({ message }: { message: string | null }) {
  return message ? <div className="error-banner" role="alert">{message}</div> : null;
}

export function DashboardPanel(props: PageProps) {
  const load = useCallback(
    (signal: AbortSignal) => fetchDashboard({ apiBaseUrl: props.apiBaseUrl, signal }),
    [props.apiBaseUrl],
  );
  const state = usePageData<Dashboard>(load, props.refreshToken, !props.hidden);
  const stats = state.data?.stats;
  const cards: { label: string; value: number; icon: AppIconName; tone: string; hint: string }[] = [
    { label: "岗位库", value: stats?.saved_jobs ?? 0, icon: "search", tone: "blue", hint: "已保存完整 JD" },
    { label: "全部投递", value: stats?.applications ?? 0, icon: "applications", tone: "blue", hint: "已记录的申请" },
    { label: "面试进行中", value: stats?.interviewing ?? 0, icon: "user", tone: "violet", hint: "需要持续准备" },
    { label: "简历版本", value: stats?.resumes ?? 0, icon: "document", tone: "cyan", hint: "按目标岗位管理" },
  ];
  return (
    <section className="management-page dashboard-page" hidden={props.hidden}>
      <PageHeader icon="dashboard" eyebrow="CAREER OVERVIEW" title="求职工作台" description="把投递进度、下一步行动和求职资料放在一个视图里" loading={state.loading} onRefresh={state.reload} />
      <ErrorBanner message={state.error} />
      <div className="stat-grid">
        {cards.map((card) => (
          <article className={`stat-card tone-card-${card.tone}`} key={card.label}>
            <span className="stat-icon"><AppIcon name={card.icon} size={22} /></span>
            <div><small>{card.label}</small><strong>{card.value}</strong><p>{card.hint}</p></div>
          </article>
        ))}
      </div>
      <div className="dashboard-grid">
        <section className="surface-card surface-card-wide">
          <div className="section-heading"><div><small>SAVED JOBS</small><h2>最近保存的 JD</h2></div><span>{stats?.saved_jobs ?? 0} 个岗位</span></div>
          {state.data && state.data.recent_jobs.length > 0 ? (
            <div className="saved-job-grid">
              {state.data.recent_jobs.map((item) => (
                <article key={item.id}>
                  <span className="list-icon tone-blue"><AppIcon name="brief" size={18} /></span>
                  <div>
                    <strong>{item.title}</strong>
                    {/* Staleness rides with the descriptive facts, not in the
                        pill. It answers "how long since this page was read",
                        while the pill answers "where is this in my process" —
                        two axes, and a job can sit anywhere on one regardless
                        of the other. Sharing a slot meant the older fact
                        silently replaced the process state, so "35 天未确认"
                        had to be read as also meaning "未投递". */}
                    <small>{item.company_name} · {[item.city, item.salary].filter(Boolean).join(" · ") || item.source_name}{stalenessLabel(item.last_checked_at) ? ` · ${stalenessLabel(item.last_checked_at)}` : ""}</small>
                  </div>
                  <span className={`status-pill ${item.application_status ? `status-${item.application_status}` : item.availability_status === "closed" ? "status-rejected" : "status-current"}`}>
                    {item.application_status
                      ? STATUS_LABELS[item.application_status] ?? item.application_status
                      : item.availability_status === "closed"
                        ? "已下架"
                        : "未投递"}
                  </span>
                </article>
              ))}
            </div>
          ) : <EmptyState icon="search" title="岗位库还是空的" description="保存你真正看过的完整 JD；它不会因为保存就变成投递记录。" action="去找岗位" onAction={() => props.onAskAgent("帮我打开招聘网站，我想查找并保存感兴趣的岗位")} />}
        </section>
        <section className="surface-card">
          <div className="section-heading"><div><small>APPLICATION PIPELINE</small><h2>投递进展</h2></div><span>{stats?.offers ?? 0} 个 Offer</span></div>
          {state.data && state.data.recent_applications.length > 0 ? (
            <div className="compact-list">
              {state.data.recent_applications.map((item) => (
                <article key={item.id}>
                  <span className="list-icon tone-blue"><AppIcon name="applications" size={18} /></span>
                  <div><strong>{item.title}</strong><small>{item.company_name} · {dateLabel(item.submitted_at)}</small></div>
                  <span className={`status-pill status-${item.status}`}>{STATUS_LABELS[item.status] ?? item.status}</span>
                </article>
              ))}
            </div>
          ) : <EmptyState icon="applications" title="还没有投递记录" description="在对话里保存第一次投递后，进度会出现在这里。" action="记录一次投递" onAction={() => props.onAskAgent("帮我记录一次新的岗位投递")} />}
        </section>
        <section className="surface-card">
          <div className="section-heading"><div><small>NEXT ACTIONS</small><h2>下一步行动</h2></div><span>{state.data?.next_actions.length ?? 0} 项</span></div>
          {state.data && state.data.next_actions.length > 0 ? (
            <div className="action-list">
              {state.data.next_actions.map((item) => (
                <article key={item.id}><span><AppIcon name="check" size={17} /></span><div><strong>{item.title}</strong><small>{item.summary}</small></div></article>
              ))}
            </div>
          ) : <EmptyState icon="check" title="当前没有待办" description="Agent 会根据投递、面试和邮件事件生成行动项。" action="问问 Agent" onAction={() => props.onAskAgent("根据我的求职进度，建议我下一步做什么")} />}
        </section>
      </div>
    </section>
  );
}

const MOCK_INTERVIEW_STATUS_LABELS: Record<string, string> = {
  created: "准备中",
  active: "进行中",
  paused: "已暂停",
  completed: "已完成",
  cancelled: "已取消",
};

function ApplicationPracticePanel({
  application,
  apiBaseUrl,
  refreshToken,
  onAskAgent,
  onClose,
}: {
  application: ApplicationView;
  apiBaseUrl: string;
  refreshToken: number;
  onAskAgent: (prompt: string) => void;
  onClose: () => void;
}) {
  const load = useCallback(
    (signal: AbortSignal) => fetchApplicationMockInterviews(
      application.id,
      { apiBaseUrl, signal },
    ),
    [apiBaseUrl, application.id],
  );
  const state = usePageData<ApplicationMockInterviews>(load, refreshToken, true);
  const startPrompt = `基于投递记录中的「${application.company_name} · ${application.title}」开始一轮模拟面试；请先让我选择面试类型和题目数量。`;

  return (
    <section className="surface-card application-practice-panel">
      <header>
        <div>
          <small>INTERVIEW PRACTICE</small>
          <h2>面试与练习</h2>
          <p>{application.company_name} · {application.title}</p>
        </div>
        <div>
          <button type="button" className="soft-button" onClick={() => onAskAgent(startPrompt)}>
            <AppIcon name="sparkles" size={15} /> 让 Agent 开始模拟面试
          </button>
          <button type="button" className="panel-close-button" onClick={onClose}>关闭</button>
        </div>
      </header>
      <ErrorBanner message={state.error} />
      {state.loading ? (
        <div className="history-loading"><span className="spinner" /> 正在读取练习记录…</div>
      ) : state.data && state.data.sessions.length > 0 ? (
        <div className="mock-session-list">
          {state.data.sessions.map((session) => (
            <article className="mock-session-card" key={session.session_id}>
              <div className="mock-session-heading">
                <div>
                  <span className="card-kicker">{session.interview_type_label}</span>
                  <strong>{session.completed_at ? dateLabel(session.completed_at) : dateLabel(session.created_at)}</strong>
                </div>
                <span className={`status-pill status-${session.status}`}>
                  {MOCK_INTERVIEW_STATUS_LABELS[session.status] ?? session.status}
                </span>
              </div>
              <p>{session.summary ?? (
                session.status === "active"
                  ? "这轮模拟面试仍在原对话中进行。"
                  : session.status === "paused"
                    ? "这轮模拟面试已暂停，已有回答会保留，继续后才会生成最终总结。"
                    : session.status === "cancelled"
                      ? "这轮模拟面试已取消，不会生成最终总结。"
                  : "这轮模拟面试尚未生成最终总结。"
              )}</p>
              <small>已完成 {session.question_count} / {session.max_primary_questions} 道主问题</small>
              {session.report_id ? (
                <ReportCard
                  apiBaseUrl={apiBaseUrl}
                  resource={{
                    kind: "mock_interview_report",
                    resourceId: session.report_id,
                  }}
                />
              ) : null}
              {session.status === "completed" ? (
                <button
                  type="button"
                  className="card-agent-action"
                  onClick={() => onAskAgent(
                    `基于投递记录中的「${application.company_name} · ${application.title}」再进行一轮${session.interview_type_label}模拟面试。`,
                  )}
                >
                  再练一轮同类型
                </button>
              ) : null}
              {session.status === "active" || session.status === "paused" ? (
                <small className="workflow-note">这轮练习只能在最初发起它的对话中继续；此处暂不支持直接跳转。</small>
              ) : null}
            </article>
          ))}
        </div>
      ) : (
        <EmptyState
          icon="sparkles"
          title="还没有模拟面试记录"
          description="模拟面试会在 Agent 对话中进行；完成后，本轮总结和每题反馈会保存在这里。"
          action="让 Agent 开始模拟面试"
          onAction={() => onAskAgent(startPrompt)}
        />
      )}
    </section>
  );
}

export function ApplicationsPanel(props: PageProps) {
  const load = useCallback(async (signal: AbortSignal) => {
    const options = { apiBaseUrl: props.apiBaseUrl, signal };
    const [applications, jobs, resumes] = await Promise.all([
      fetchApplications(options),
      fetchSavedJobs({ ...options, includeIgnored: false }),
      fetchResumes(options),
    ]);
    return { applications, jobs, resumes };
  }, [props.apiBaseUrl]);
  const state = usePageData<{
    applications: ApplicationView[];
    jobs: SavedJobView[];
    resumes: ResumeView[];
  }>(load, props.refreshToken, !props.hidden);
  const [showForm, setShowForm] = useState(false);
  const [jobId, setJobId] = useState("");
  const [resumeVersionId, setResumeVersionId] = useState("");
  const [submittedAt, setSubmittedAt] = useState(() => {
    const now = new Date();
    return new Date(now.getTime() - now.getTimezoneOffset() * 60_000).toISOString().slice(0, 16);
  });
  const [note, setNote] = useState("");
  const [saving, setSaving] = useState(false);
  const [formError, setFormError] = useState<string | null>(null);
  const [practiceApplication, setPracticeApplication] = useState<ApplicationView | null>(null);

  async function submitApplication(): Promise<void> {
    setSaving(true);
    setFormError(null);
    try {
      await createApplication({
        jobPostingId: jobId,
        resumeVersionId,
        submittedAt: new Date(submittedAt).toISOString(),
        note,
      }, { apiBaseUrl: props.apiBaseUrl });
      setShowForm(false);
      setNote("");
      state.reload();
    } catch (cause) {
      setFormError(cause instanceof Error ? cause.message : "无法记录这次投递。");
    } finally {
      setSaving(false);
    }
  }

  return (
    <section className="management-page" hidden={props.hidden}>
      <PageHeader icon="applications" eyebrow="APPLICATIONS" title="投递记录" description="按公司和岗位查看从已投递到 Offer 的完整进度" loading={state.loading} onRefresh={state.reload} />
      <ErrorBanner message={state.error} />
      <div className="management-actions">
        <button type="button" className="soft-button" onClick={() => setShowForm((visible) => !visible)}><AppIcon name="applications" size={16} /> 直接记录投递</button>
        <button type="button" className="soft-button" onClick={() => props.onAskAgent("帮我记录一次新的岗位投递；请根据我已保存的岗位和简历询问并确认必要信息")}><AppIcon name="sparkles" size={16} /> 让 Agent 记录</button>
      </div>
      <ErrorBanner message={formError} />
      {showForm ? <section className="surface-card application-create-form"><div><label>已保存岗位<select value={jobId} onChange={(event) => setJobId(event.target.value)}><option value="">请选择岗位</option>{state.data?.jobs.map((job) => <option key={job.id} value={job.id}>{job.company_name} · {job.title}</option>)}</select></label><label>使用的简历<select value={resumeVersionId} onChange={(event) => setResumeVersionId(event.target.value)}><option value="">请选择简历版本</option>{state.data?.resumes.map((resume) => <option key={resume.latest_version_id} value={resume.latest_version_id}>{resume.name} · v{resume.latest_version_number}</option>)}</select></label><label>实际投递时间<input type="datetime-local" value={submittedAt} onChange={(event) => setSubmittedAt(event.target.value)} /></label><label>备注（可选）<textarea value={note} onChange={(event) => setNote(event.target.value)} maxLength={2000} placeholder="例如：官网投递、内推人等" /></label></div>{state.data && (state.data.jobs.length === 0 || state.data.resumes.length === 0) ? <p>记录前需要至少一个已保存岗位和一个简历版本。可以先前往岗位库/简历管理，或让 Agent 协助。</p> : null}<button type="button" disabled={saving || !jobId || !resumeVersionId || !submittedAt} onClick={() => void submitApplication()}>{saving ? "正在保存…" : "确认已在外部平台投递并记录"}</button></section> : null}
      {state.data && state.data.applications.length > 0 ? <div className="data-table-card"><div className="data-table-head"><span>岗位</span><span>状态</span><span>地点 / 薪资</span><span>投递时间</span></div>{state.data.applications.map((item) => <article className={`data-table-row ${practiceApplication?.id === item.id ? "is-selected" : ""}`} key={item.id}><div><span className="list-icon tone-blue"><AppIcon name="applications" size={18} /></span><span><strong>{item.title}</strong><small>{item.company_name}</small><button type="button" className="row-detail-button" onClick={() => setPracticeApplication(item)}>面试与练习</button></span></div><span><span className={`status-pill status-${item.status}`}>{STATUS_LABELS[item.status] ?? item.status}</span></span><span>{[item.city, item.salary].filter(Boolean).join(" · ") || "未披露"}</span><time>{dateLabel(item.submitted_at)}</time></article>)}</div> : <EmptyState icon="applications" title="还没有投递记录" description="可直接选择已保存岗位和简历记录，也可以让 Agent 通过对话协助。" />}
      {practiceApplication ? (
        <ApplicationPracticePanel
          application={practiceApplication}
          apiBaseUrl={props.apiBaseUrl}
          refreshToken={props.refreshToken}
          onAskAgent={props.onAskAgent}
          onClose={() => setPracticeApplication(null)}
        />
      ) : null}
    </section>
  );
}

export function JobsPanel(props: PageProps) {
  // Ignored jobs are hidden, not gone. Without a way back the button is a
  // one-way door: the row is still there and the store already restores it,
  // but nothing in the product could reach it.
  const [showIgnored, setShowIgnored] = useState(false);
  const load = useCallback(
    (signal: AbortSignal) => fetchSavedJobs({ apiBaseUrl: props.apiBaseUrl, signal, includeIgnored: showIgnored }),
    [props.apiBaseUrl, showIgnored],
  );
  const state = usePageData<SavedJobView[]>(load, props.refreshToken, !props.hidden);
  const [pending, setPending] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  // The opened card's JD, loaded on demand. One at a time on purpose: the point
  // is reading a posting, and keeping several bodies expanded turns the library
  // back into a wall of text.
  const [openJd, setOpenJd] = useState<string | null>(null);
  const [jd, setJd] = useState<SavedJobDetail | null>(null);
  const [jdLoading, setJdLoading] = useState(false);
  const toggleJd = useCallback(
    async (jobPostingId: string) => {
      if (openJd === jobPostingId) {
        setOpenJd(null);
        return;
      }
      setOpenJd(jobPostingId);
      setJd(null);
      setJdLoading(true);
      setActionError(null);
      try {
        setJd(await fetchSavedJobDetail(jobPostingId, { apiBaseUrl: props.apiBaseUrl }));
      } catch (error) {
        setActionError(error instanceof Error ? error.message : "读取 JD 失败");
        setOpenJd(null);
      } finally {
        setJdLoading(false);
      }
    },
    [openJd, props.apiBaseUrl],
  );
  const markClosed = useCallback(
    async (jobPostingId: string) => {
      setPending(jobPostingId);
      setActionError(null);
      try {
        await setJobAvailability(jobPostingId, "closed", { apiBaseUrl: props.apiBaseUrl });
        state.reload();
      } catch (error) {
        setActionError(error instanceof Error ? error.message : "操作失败");
      } finally {
        setPending(null);
      }
    },
    [props.apiBaseUrl, state],
  );
  const restore = useCallback(
    async (jobPostingId: string) => {
      setPending(jobPostingId);
      setActionError(null);
      try {
        await setJobPursuit(jobPostingId, "open", { apiBaseUrl: props.apiBaseUrl });
        state.reload();
      } catch (error) {
        setActionError(error instanceof Error ? error.message : "操作失败");
      } finally {
        setPending(null);
      }
    },
    [props.apiBaseUrl, state],
  );
  const dismiss = useCallback(
    async (jobPostingId: string) => {
      setPending(jobPostingId);
      setActionError(null);
      try {
        await setJobPursuit(jobPostingId, "dismissed", { apiBaseUrl: props.apiBaseUrl });
        // Reload rather than splice the row out locally: the server decides
        // what is on the shortlist, and a list that disagrees with it is worse
        // than one that takes a moment to catch up.
        state.reload();
      } catch (error) {
        setActionError(error instanceof Error ? error.message : "操作失败");
      } finally {
        setPending(null);
      }
    },
    [props.apiBaseUrl, state],
  );
  const remove = useCallback(
    async (item: SavedJobView) => {
      const confirmed = window.confirm(
        `永久删除「${item.company_name} · ${item.title}」？\n\n该岗位、所有 JD 历史快照及其分析都会从数据库中删除，无法恢复。`,
      );
      if (!confirmed) return;
      setPending(item.id);
      setActionError(null);
      try {
        await deleteSavedJob(item.id, { apiBaseUrl: props.apiBaseUrl });
        if (openJd === item.id) {
          setOpenJd(null);
          setJd(null);
        }
        state.reload();
      } catch (error) {
        setActionError(error instanceof Error ? error.message : "删除失败");
      } finally {
        setPending(null);
      }
    },
    [openJd, props.apiBaseUrl, state],
  );
  return (
    <section className="management-page" hidden={props.hidden}>
      <PageHeader icon="search" eyebrow="JOB LIBRARY" title="岗位库" description="查看保存过的完整 JD、分析结果和后续投递状态" loading={state.loading} onRefresh={state.reload} />
      <ErrorBanner message={state.error ?? actionError} />
      <div className="job-library-filter">
        <label>
          <input
            type="checkbox"
            checked={showIgnored}
            onChange={(event) => setShowIgnored(event.target.checked)}
          />
          显示已忽略的岗位
        </label>
      </div>
      {state.data && state.data.length > 0 ? (
        <div className="job-library-grid">
          {state.data.map((item) => (
            <article className="job-analysis-card" key={item.id}>
              <header>
                <span className="job-company-mark"><AppIcon name="brief" size={20} /></span>
                <div>
                  <h2>{item.title}</h2>
                  <p>{item.company_name} · {[item.city, item.salary].filter(Boolean).join(" · ") || item.source_name}</p>
                </div>
                <span className={`status-pill ${item.analysis_summary ? "status-current" : "status-outdated"}`}>
                  {item.analysis_summary ? "已分析" : "待分析"}
                </span>
                {/* Said beside the analysis badge rather than replacing it:
                    "closed", "long unconfirmed" and "not analysed" are
                    unrelated facts, and a single pill would force one to hide
                    the others. The library is where they are acted on, so it
                    is where all of them have to be visible.

                    "已下架" also has to be readable back here, or the button
                    that sets it looks like it did nothing: it hides itself once
                    the job is closed, leaving no trace of the click. */}
                {item.availability_status === "closed" ? (
                  <span className="status-pill status-rejected">已下架</span>
                ) : stalenessLabel(item.last_checked_at) ? (
                  <span className="status-pill status-outdated">{stalenessLabel(item.last_checked_at)}</span>
                ) : null}
              </header>
              {item.analysis_summary ? (
                <div className="job-analysis-body">
                  <p className="job-summary">{item.analysis_summary}</p>
                  {item.required_skills.length > 0 ? (
                    <div className="analysis-section"><strong>必备技能</strong><div className="skill-chips">{item.required_skills.map((skill) => <span key={skill}>{skill}</span>)}</div></div>
                  ) : null}
                  {item.responsibilities.length > 0 ? (
                    <div className="analysis-section"><strong>核心职责</strong><ul>{item.responsibilities.slice(0, 4).map((entry) => <li key={entry}>{entry}</li>)}</ul></div>
                  ) : null}
                  {item.clarification_questions.length > 0 ? (
                    <div className="analysis-question"><AppIcon name="target" size={16} /><span>{item.clarification_questions.length} 个信息点需要进一步确认</span></div>
                  ) : null}
                </div>
              ) : (
                <div className="job-analysis-pending">
                  <p>完整 JD 已保存，但还没有结构化分析。</p>
                  <button type="button" onClick={() => props.onAskAgent(`分析岗位库里的「${item.company_name} · ${item.title}」`)}>让 Agent 分析</button>
                </div>
              )}
              <footer>
                <span>{item.application_status ? STATUS_LABELS[item.application_status] ?? item.application_status : "尚未投递"}</span>
                <time>{dateLabel(item.captured_at)} 保存{item.analyzed_at ? ` · ${dateLabel(item.analyzed_at)} 分析` : ""}</time>
              </footer>
              {/* Three things one card has to support, because they are the
                  three things that happen after reading a JD: go back to the
                  posting, record that you applied, or take it off the list.
                  Without them the library is a place to look at jobs and
                  nothing else, and triage falls back to the conversation —
                  where "the 3rd, 7th and 12th" means counting rows. */}
              {openJd === item.id ? (
                <div className="job-jd-body">
                  {jdLoading ? <p className="job-jd-loading">正在读取 JD…</p> : null}
                  {jd && jd.id === item.id ? (
                    <>
                      <div className="job-jd-meta">
                        <span>JD 快照 v{jd.jd_version}</span>
                        <time>{dateLabel(jd.captured_at)} 抓取</time>
                      </div>
                      {/* Rendered as stored: whitespace preserved, nothing
                          reformatted. A JD read differently from how it was
                          captured is a different JD. */}
                      <pre>{jd.jd_text}</pre>
                    </>
                  ) : null}
                </div>
              ) : null}
              <div className="job-card-actions">
                <button type="button" className="link" onClick={() => toggleJd(item.id)}>
                  {openJd === item.id ? "收起 JD" : "查看完整 JD"}
                </button>
                <button
                  type="button"
                  className="link"
                  onClick={() => props.onAskAgent(`研究岗位库里的「${item.company_name} · ${item.title}」所属公司的业务、产品线和市场情况`)}
                >
                  让 Agent 研究公司
                </button>
              </div>
              {item.pursuit_status === "dismissed" ? (
                <div className="job-card-actions">
                  <span className="ignored-note">已忽略</span>
                  <button
                    type="button"
                    disabled={pending === item.id}
                    onClick={() => restore(item.id)}
                  >
                    {pending === item.id ? "处理中…" : "放回名单"}
                  </button>
                  <button
                    type="button"
                    className="danger"
                    disabled={pending === item.id}
                    onClick={() => remove(item)}
                  >
                    {pending === item.id ? "处理中…" : "永久删除"}
                  </button>
                </div>
              ) : item.application_status ? null : (
                <div className="job-card-actions">
                  {item.source_url ? (
                    // Applying happens on the site, never here: create_application
                    // only records a submission the user reports. So the link is
                    // not a convenience, it is the next step of the flow.
                    <a href={item.source_url} target="_blank" rel="noreferrer noopener">
                      打开原页面
                    </a>
                  ) : null}
                  <button
                    type="button"
                    onClick={() => props.onAskAgent(`我投了岗位库里的「${item.company_name} · ${item.title}」，帮我记录这次投递`)}
                  >
                    我投了
                  </button>
                  {/* Dismissal is a status flip with no other input, so it lands
                      here. Recording an application needs a resume version,
                      which is a decision — that one is handed to the
                      conversation with the job already named. */}
                  {/* Two different facts, so two buttons. "已下架" is what the
                      employer did; "忽略" is what the reader decided. One
                      control for both would make a relisted job indistinguishable
                      from one they changed their mind about. */}
                  {item.availability_status === "closed" ? null : (
                    <button
                      type="button"
                      className="ghost"
                      disabled={pending === item.id}
                      onClick={() => markClosed(item.id)}
                    >
                      已下架
                    </button>
                  )}
                  <button
                    type="button"
                    className="ghost"
                    disabled={pending === item.id}
                    onClick={() => dismiss(item.id)}
                  >
                    {pending === item.id ? "处理中…" : "忽略"}
                  </button>
                  <button
                    type="button"
                    className="danger"
                    disabled={pending === item.id}
                    onClick={() => remove(item)}
                  >
                    {pending === item.id ? "处理中…" : "永久删除"}
                  </button>
                </div>
              )}
            </article>
          ))}
        </div>
      ) : <EmptyState icon="search" title="岗位库还是空的" description="保存完整 JD 后，可以继续分析、匹配简历或记录投递。" action="去找岗位" onAction={() => props.onAskAgent("帮我打开招聘网站，我想查找并保存感兴趣的岗位")} />}
    </section>
  );
}

export function ResumesPanel(props: PageProps) {
  const load = useCallback((signal: AbortSignal) => fetchResumes({ apiBaseUrl: props.apiBaseUrl, signal }), [props.apiBaseUrl]);
  const state = usePageData<ResumeView[]>(load, props.refreshToken, !props.hidden);
  const [showImporter, setShowImporter] = useState(false);
  return (
    <section className="management-page" hidden={props.hidden}>
      <PageHeader icon="document" eyebrow="RESUME LIBRARY" title="简历管理" description="按目标岗位管理简历家族、版本和来源" loading={state.loading} onRefresh={state.reload} />
      <ErrorBanner message={state.error} />
      <div className="management-actions">
        <button type="button" className="soft-button" onClick={() => setShowImporter((open) => !open)}>
          <AppIcon name="plus" size={16} /> {showImporter ? "收起导入" : "导入简历"}
        </button>
      </div>
      {showImporter ? (
        <ResumeImporter
          apiBaseUrl={props.apiBaseUrl}
          onImported={(result) => {
            setShowImporter(false);
            state.reload();
            props.onAskAgent(`刚刚导入了简历“${result.name}”v${result.version_number}，请分析这份简历`);
          }}
        />
      ) : null}
      {state.data && state.data.length > 0 ? (
        <div className="card-grid">
          {state.data.map((item) => (
            <article className="resume-card" key={item.id}>
              <span className="document-preview"><AppIcon name="document" size={28} /><small>{item.document_format.toUpperCase()}</small></span>
              <div className="card-body">
                <span className="card-kicker">{item.target_role}</span>
                <h2>{item.name}</h2>
                <p>最新 v{item.latest_version_number} · 共 {item.version_count} 个版本</p>
                <div className="card-meta"><span>{formatBytes(item.byte_size)}</span><time>{dateLabel(item.updated_at)} 更新</time></div>
                <button type="button" className="card-agent-action" onClick={() => props.onAskAgent(`分析简历库里的“${item.name}”最新版本`)}>让 Agent 分析</button>
              </div>
            </article>
          ))}
        </div>
      ) : !showImporter ? (
        <EmptyState icon="document" title="还没有简历" description="先导入 PDF、TXT 或 Markdown；导入后可立即交给 Agent 分析。" action="导入一份简历" onAction={() => setShowImporter(true)} />
      ) : null}
    </section>
  );
}

export function CalendarPanel(props: PageProps) {
  const [cursor, setCursor] = useState(() => new Date(new Date().getFullYear(), new Date().getMonth(), 1));
  const [selected, setSelected] = useState<CalendarWorkspace["events"][number] | null>(null);
  const [connectionError, setConnectionError] = useState<string | null>(null);
  const timezone = Intl.DateTimeFormat().resolvedOptions().timeZone || "Asia/Shanghai";
  const selectedMonth = monthKey(cursor);
  const load = useCallback((signal: AbortSignal) => fetchCalendar({
    apiBaseUrl: props.apiBaseUrl,
    signal,
    month: selectedMonth,
    timezone,
  }), [props.apiBaseUrl, selectedMonth, timezone]);
  const state = usePageData<CalendarWorkspace>(load, props.refreshToken, !props.hidden);
  const grouped = eventsByDay(state.data?.events ?? [], timezone);
  const days = calendarDays(cursor);
  const now = new Date();
  const todayKey = `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, "0")}-${String(now.getDate()).padStart(2, "0")}`;
  const selectedMeetingUrl = safeMeetingUrl(selected?.meeting_url);
  const selectedExternalUrl = safeMeetingUrl(selected?.external_html_link);

  async function connectGoogle(): Promise<void> {
    try {
      setConnectionError(null);
      window.location.assign(await startGoogleConnection("calendar", { apiBaseUrl: props.apiBaseUrl }));
    } catch (cause) {
      setConnectionError(cause instanceof Error ? cause.message : "无法启动 Google Calendar 授权。");
    }
  }

  return (
    <section className="management-page" hidden={props.hidden}>
      <PageHeader icon="calendar" eyebrow="CALENDAR" title="面试日历" description="查看项目中的面试安排及 Google Calendar 同步状态" loading={state.loading} onRefresh={state.reload} />
      <ErrorBanner message={state.error} />
      <ErrorBanner message={connectionError} />
      <details className="integration-help">
        <summary>如何连接 Google Calendar？</summary>
        <div>
          <p>这是可选功能；不连接时，项目内置月历仍可正常使用。</p>
          <ol>
            <li>管理员先在 <a href="https://console.cloud.google.com/apis/credentials" target="_blank" rel="noreferrer">Google Cloud Console</a> 创建 Web OAuth Client，并启用 Google Calendar API。</li>
            <li>回调地址填写 <code>http://127.0.0.1:8000/v1/connections/google/callback</code>，然后将 Client ID 和 Secret 写入项目根目录 <code>.env</code>。</li>
            <li>重启后端后点击“连接 Google Calendar”，只会申请日历事件权限。</li>
          </ol>
        </div>
      </details>
      <section className="surface-card calendar-accounts">
        <div className="section-heading"><div><small>GOOGLE CALENDAR（可选）</small><h2>日历账号</h2></div><button type="button" onClick={() => void connectGoogle()}>连接 Google Calendar</button></div>
        {state.data?.accounts.map((item) => <article className="integration-account" key={item.id}><span className="list-icon tone-blue"><AppIcon name="calendar" size={18} /></span><div><strong>{item.email_address}</strong><small>{item.calendar_id} · 已连接</small></div><button type="button" className="danger" onClick={() => void disconnectIntegration("calendar", item.id, { apiBaseUrl: props.apiBaseUrl }).then(state.reload)}>断开</button></article>)}
      </section>
      <section className="surface-card calendar-shell">
        <header className="calendar-toolbar">
          <div><button type="button" onClick={() => setCursor(new Date(cursor.getFullYear(), cursor.getMonth() - 1, 1))}>‹</button><button type="button" onClick={() => setCursor(new Date())}>今天</button><button type="button" onClick={() => setCursor(new Date(cursor.getFullYear(), cursor.getMonth() + 1, 1))}>›</button></div>
          <h2>{cursor.getFullYear()} 年 {cursor.getMonth() + 1} 月</h2>
          <button type="button" onClick={() => props.onAskAgent("检查我最近的招聘邮件和面试安排，并告诉我哪些需要同步到日历")}>让 Agent 检查安排</button>
        </header>
        <div className="calendar-month">
          {["周一", "周二", "周三", "周四", "周五", "周六", "周日"].map((day) => <span className="calendar-weekday" key={day}>{day}</span>)}
          {days.map((day) => {
            const events = grouped.get(day.key) ?? [];
            return <div className={`calendar-day ${day.inMonth ? "" : "is-outside"} ${day.key === todayKey ? "is-today" : ""}`} key={day.key}><span>{day.day}</span>{events.slice(0, 2).map((item) => <button type="button" className={`calendar-event-chip status-${item.interview_status}`} key={item.interview_round_id} onClick={() => setSelected(item)}><time>{item.scheduled_start ? new Date(item.scheduled_start).toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit", timeZone: timezone }) : ""}</time>{item.company_name || item.employer_label || "面试"}</button>)}{events.length > 2 ? <small>还有 {events.length - 2} 项</small> : null}</div>;
          })}
        </div>
        {state.data && state.data.events.length === 0 ? <div className="calendar-empty">本月还没有已安排的面试。</div> : null}
      </section>
      {selected ? <aside className="calendar-detail"><button type="button" className="calendar-detail-close" onClick={() => setSelected(null)}>关闭</button><span className="card-kicker">{selected.employer_label || "面试安排"}</span><h2>{selected.company_name || "未关联公司"} · {selected.job_title || "未关联岗位"}</h2><p>{selected.scheduled_start ? new Date(selected.scheduled_start).toLocaleString("zh-CN", { timeZone: selected.timezone || timezone }) : "时间待确认"} · {selected.timezone}</p><dl><div><dt>形式</dt><dd>{selected.interview_format || "unknown"}</dd></div><div><dt>地点</dt><dd>{selected.location || "未提供"}</dd></div><div><dt>同步</dt><dd>{selected.sync_status}</dd></div>{selected.contact_summary ? <div><dt>联系人</dt><dd>{selected.contact_summary}</dd></div> : null}</dl><div className="calendar-detail-actions">{selectedMeetingUrl ? <a href={selectedMeetingUrl} target="_blank" rel="noreferrer">加入会议</a> : null}{selectedExternalUrl ? <a href={selectedExternalUrl} target="_blank" rel="noreferrer">在 Google Calendar 打开</a> : <button type="button" onClick={() => props.onAskAgent(`把 interview_round_id=${selected.interview_round_id} 的面试安排同步到 Google Calendar`)}>让 Agent 同步</button>}</div></aside> : null}
    </section>
  );
}

const EMAIL_EVENT_LABELS: Record<string, string> = {
  acknowledgement: "投递确认",
  interview_invitation: "面试邀请",
  rejection: "未通过",
  offer: "Offer",
  material_request: "材料请求",
  unclear: "待判断",
};

export function EmailPanel(props: PageProps) {
  const load = useCallback((signal: AbortSignal) => fetchEmailWorkspace({ apiBaseUrl: props.apiBaseUrl, signal }), [props.apiBaseUrl]);
  const state = usePageData<EmailWorkspace>(load, props.refreshToken, !props.hidden);
  const pending = state.data?.events.filter((item) => item.status === "pending_confirmation") ?? [];
  const [showQQ, setShowQQ] = useState(false);
  const [qqEmail, setQQEmail] = useState("");
  const [qqCode, setQQCode] = useState("");
  const [connectionError, setConnectionError] = useState<string | null>(null);

  async function connectGoogle(): Promise<void> {
    try {
      setConnectionError(null);
      window.location.assign(await startGoogleConnection("gmail", { apiBaseUrl: props.apiBaseUrl }));
    } catch (cause) {
      setConnectionError(cause instanceof Error ? cause.message : "无法启动 Google 授权。");
    }
  }

  async function submitQQ(): Promise<void> {
    try {
      await connectQQ(qqEmail, qqCode, { apiBaseUrl: props.apiBaseUrl });
      setShowQQ(false);
      setQQCode("");
      state.reload();
    } catch (cause) {
      setConnectionError(cause instanceof Error ? cause.message : "QQ 邮箱连接失败。");
    }
  }

  return (
    <section className="management-page" hidden={props.hidden}>
      <PageHeader icon="mail" eyebrow="EMAIL TRACKING" title="邮件追踪" description="查看招聘邮箱账号、同步状态和 Agent 识别出的求职事件" loading={state.loading} onRefresh={state.reload} />
      <ErrorBanner message={state.error} />
      <ErrorBanner message={connectionError} />
      <div className="management-actions">
        <button type="button" className="soft-button" onClick={() => void connectGoogle()}><AppIcon name="mail" size={16} /> 连接 Gmail</button>
        <button type="button" className="soft-button" onClick={() => setShowQQ((open) => !open)}>连接 QQ 邮箱</button>
        <button type="button" className="soft-button" onClick={() => props.onAskAgent("同步我已连接邮箱中的最新招聘邮件，并列出需要我确认的事件")}>
          <AppIcon name="sparkles" size={16} /> 让 Agent 同步邮箱
        </button>
        {pending.length > 0 ? <button type="button" className="soft-button" onClick={() => props.onAskAgent("逐项处理邮箱里等待确认的求职事件")}>处理 {pending.length} 个待确认事件</button> : null}
      </div>
      <details className="integration-help">
        <summary>Gmail / QQ 邮箱如何连接？</summary>
        <div className="integration-help-columns">
          <section>
            <strong>Gmail</strong>
            <ol>
              <li>管理员在 <a href="https://console.cloud.google.com/apis/credentials" target="_blank" rel="noreferrer">Google Cloud Console</a> 启用 Gmail API，并创建 Web OAuth Client。</li>
              <li>配置项目根目录 <code>.env</code> 后重启后端。</li>
              <li>点击“连接 Gmail”并在 Google 页面授权；这里只申请邮件只读权限。</li>
            </ol>
          </section>
          <section>
            <strong>QQ 邮箱</strong>
            <ol>
              <li>进入 QQ 邮箱网页版的“设置 → 账号与安全 → 安全设置”。</li>
              <li>开启 IMAP/SMTP 服务并生成第三方客户端授权码。</li>
              <li>点击“连接 QQ 邮箱”，填写完整邮箱和授权码；不要填写 QQ 登录密码。</li>
            </ol>
          </section>
        </div>
      </details>
      {showQQ ? <div className="qq-connect-form"><label>QQ 邮箱<input type="email" value={qqEmail} onChange={(event) => setQQEmail(event.target.value)} placeholder="name@qq.com" /></label><label>授权码<input type="password" value={qqCode} onChange={(event) => setQQCode(event.target.value)} /></label><button type="button" disabled={!qqEmail || !qqCode} onClick={() => void submitQQ()}>验证并连接</button><small>使用 QQ 邮箱设置中生成的 IMAP 授权码，不是登录密码。</small></div> : null}
      <div className="dashboard-grid">
        <section className="surface-card">
          <div className="section-heading"><div><small>CONNECTED INBOXES</small><h2>招聘邮箱</h2></div><span>{state.data?.accounts.length ?? 0} 个</span></div>
          {state.data && state.data.accounts.length > 0 ? <div className="compact-list">{state.data.accounts.map((item) => <article key={item.id}><span className="list-icon tone-blue"><AppIcon name="mail" size={18} /></span><div><strong>{item.email_address}</strong><small>{item.provider.toUpperCase()} · {item.last_synced_at ? `${dateLabel(item.last_synced_at)} 同步` : "尚未同步"}</small></div><span className={`status-pill status-${item.status}`}>{STATUS_LABELS[item.status] ?? item.status}</span><button type="button" className="danger" onClick={() => void disconnectIntegration("email", item.id, { apiBaseUrl: props.apiBaseUrl }).then(state.reload)}>断开</button></article>)}</div> : <EmptyState icon="mail" title="尚未连接招聘邮箱" description="连接 Google 或 QQ 邮箱后，Agent 将以只读方式识别招聘事件。凭证不会返回页面或 Agent。" />}
        </section>
        <section className="surface-card">
          <div className="section-heading"><div><small>RECRUITING EVENTS</small><h2>识别事件</h2></div><span>{state.data?.events.length ?? 0} 项</span></div>
          {state.data && state.data.events.length > 0 ? <div className="email-event-list">{state.data.events.map((item) => <article key={item.id}><div><span className="card-kicker">{EMAIL_EVENT_LABELS[item.event_type] ?? item.event_type}</span><strong>{item.company_name && item.application_title ? `${item.company_name} · ${item.application_title}` : "未关联投递"}</strong><p>{item.summary}</p><small>{dateLabel(item.occurred_at)} · 置信度 {Math.round(item.confidence * 100)}%</small></div><span className={`status-pill status-${item.status}`}>{item.status === "pending_confirmation" ? "待确认" : item.status === "applied" ? "已应用" : "已忽略"}</span></article>)}</div> : <EmptyState icon="mail" title="还没有招聘邮件事件" description="Agent 同步后只会在这里展示结构化摘要，不展示邮件正文和凭证。" action="让 Agent 同步邮箱" onAction={() => props.onAskAgent("同步我已连接邮箱中的最新招聘邮件")} />}
        </section>
      </div>
    </section>
  );
}

export function ResearchPanel(props: PageProps) {
  const load = useCallback((signal: AbortSignal) => fetchCompanyResearch({ apiBaseUrl: props.apiBaseUrl, signal }), [props.apiBaseUrl]);
  const state = usePageData<CompanyResearchView[]>(load, props.refreshToken, !props.hidden);
  return (
    <section className="management-page" hidden={props.hidden}>
      <PageHeader icon="building" eyebrow="COMPANY RESEARCH" title="公司研究" description="集中查看用户主动要求生成的业务、产品线和市场资料" loading={state.loading} onRefresh={state.reload} />
      <ErrorBanner message={state.error} />
      <div className="management-actions"><button type="button" className="soft-button" onClick={() => props.onAskAgent("我想研究岗位库里一家目标公司的业务、产品线和市场情况")}><AppIcon name="sparkles" size={16} /> 让 Agent 研究公司</button></div>
      {state.data && state.data.length > 0 ? <div className="research-grid">{state.data.map((item) => <article className="research-card" key={item.id}><div className="research-card-top"><span className="company-avatar">{item.company_name.slice(0, 1)}</span><div><span className="card-kicker">{item.focus || "公司与业务概览"}</span><h2>{item.company_name}</h2></div><span className={`status-pill status-${item.status}`}>{STATUS_LABELS[item.status] ?? item.status}</span></div><p>{item.summary}</p><div className="card-meta"><span>{item.finding_count} 条发现</span><span>{item.source_count} 个来源</span><time>{dateLabel(item.created_at)}</time></div><small>检索锚点：{item.anchor_job_title}</small><ReportCard resource={{ kind: "job_research_report", resourceId: item.id }} apiBaseUrl={props.apiBaseUrl} /></article>)}</div> : <EmptyState icon="building" title="还没有公司研究" description="研究需要以岗位库中的已保存岗位为锚点，由 Agent 确认范围后启动。" action="让 Agent 研究公司" onAction={() => props.onAskAgent("我想研究岗位库里一家目标公司的业务和产品线")} />}
    </section>
  );
}
