import { useCallback, useEffect, useRef, useState, type ReactNode } from "react";

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
  clearApplications,
  deleteResume,
  fetchTargetRoles,
  moveResume,
  type TargetRoleView,
  deleteCompanyResearch,
  fetchApplications,
  fetchApplicationMockInterviews,
  fetchMockInterviews,
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
  resumeDocumentUrl,
  updateApplicationResumeVersion,
  type ResumeVersionView,
} from "../api/client";
import {
  attachmentFromImport,
  attachmentFromSavedJob,
  attachmentFromVersion,
  type ChatAttachment,
  type ResumeAttachment,
} from "../chat/attachments";
import { JobAnalysisDetails } from "../components/JobAnalysisDetails";
import { SearchableSelect } from "../components/SearchableSelect";
import { JobMatchesPanel } from "../components/JobMatchesPanel";
import { AppIcon, type AppIconName } from "../components/AppIcon";
import { ReportCard } from "../components/ReportCard";
import { ResumeImporter } from "../components/ResumeImporter";
import { DropdownMenu } from "../components/DropdownMenu";
import { calendarDays, eventsByDay, monthKey } from "../calendar/month";
import { practiceRepeat } from "../chat/practiceRepeat";
import { groupResumesByRole } from "./resumeGroups";

type PageProps = {
  apiBaseUrl: string;
  refreshToken: number;
  hidden: boolean;
  /** `resource` pins the message to one exact resume version via `input_resources`. */
  onAskAgent: (prompt: string, resource?: ResumeAttachment) => void;
  /**
   * Run one task of its own in a new conversation, instead of inside whatever
   * chat is open: analysing an exact resume version, or the JD of a saved job
   * with nothing but that JD attached.
   */
  onStartStandaloneTask?: StartStandaloneTask;
  onOpenConversation?: (conversationId: string) => void;
  onNavigate?: (view: "dashboard" | "applications" | "jobs" | "brief" | "research" | "resumes") => void;
};

/** Sentinel for "I do not know which resume I sent"; it is sent as null. */
const UNKNOWN_RESUME = "__unknown_resume__";

function resumeVersionOptions(resumes: ResumeView[]) {
  return [
    { value: UNKNOWN_RESUME, label: "不确定用了哪一版", description: "可以之后再补填" },
    ...resumes.flatMap((resume) => resume.versions.map((version) => ({
      value: version.id,
      label: resume.name,
      description: `第 ${version.version_number} 版`,
    }))),
  ];
}

const STATUS_LABELS: Record<string, string> = {
  submitted: "已投递",
  acknowledged: "已确认",
  interviewing: "面试中",
  interview_completed: "面试已完成",
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
  showRefresh = true,
  actions,
}: {
  icon: AppIconName;
  eyebrow: string;
  title: string;
  description: string;
  loading: boolean;
  onRefresh: () => void;
  showRefresh?: boolean;
  actions?: ReactNode;
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
      <div className="management-header-actions">
        {actions}
        {showRefresh ? <button className="soft-button header-refresh-button" type="button" onClick={onRefresh} disabled={loading}>
          <AppIcon name="refresh" size={16} className={loading ? "is-spinning" : undefined} />
          {loading ? "刷新中" : "刷新"}
        </button> : null}
      </div>
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

type DashboardSelection = { kind: "job"; item: SavedJobView } | { kind: "resume"; item: ResumeView };

function DashboardDetail({ selection, props, onClose }: {
  selection: DashboardSelection;
  props: PageProps;
  onClose: () => void;
}) {
  const dialog = useRef<HTMLDialogElement>(null);
  const triggerRef = useRef(document.activeElement);
  const [versionId, setVersionId] = useState(selection.kind === "resume" ? selection.item.latest_version_id : "");
  const load = useCallback((signal: AbortSignal) =>
    selection.kind === "job"
      ? fetchSavedJobDetail(selection.item.id, { apiBaseUrl: props.apiBaseUrl, signal })
      : Promise.resolve(null), [selection, props.apiBaseUrl]);
  const state = usePageData(load, props.refreshToken, true);
  useEffect(() => {
    const element = dialog.current;
    const trigger = triggerRef.current;
    element?.showModal();
    return () => {
      element?.close();
      if (trigger instanceof HTMLElement) trigger.focus();
    };
  }, []);
  const version = selection.kind === "resume" ? selection.item.versions.find((item) => item.id === versionId) : null;
  return (
    <dialog ref={dialog} className="dashboard-detail" aria-labelledby="dashboard-detail-title" onCancel={onClose} onClick={(event) => {
      if (event.target === event.currentTarget) {
        const rect = event.currentTarget.getBoundingClientRect();
        if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) onClose();
      }
    }}>
      <header className="dashboard-detail-header">
        <div><span className="card-kicker">{selection.kind === "job" ? "岗位 JD" : "简历详情"}</span><h2 id="dashboard-detail-title">{selection.kind === "job" ? selection.item.title : selection.item.name}</h2><p>{selection.kind === "job" ? selection.item.company_name : selection.item.target_role}</p></div>
        <button type="button" className="soft-button" onClick={onClose} autoFocus>关闭</button>
      </header>
      <div className="dashboard-detail-content">
        <ErrorBanner message={state.error} />
        {state.error ? <button type="button" className="soft-button" onClick={state.reload}>重试</button> : null}
        {selection.kind === "job" ? <>
          {state.loading ? <p role="status">正在读取 JD…</p> : null}
          {state.data ? <div className="job-jd-body"><div className="job-jd-meta"><span>JD 快照 v{state.data.jd_version}</span><span>{[state.data.city, state.data.salary].filter(Boolean).join(" · ")}</span></div><pre>{state.data.jd_text || "该岗位暂未保存 JD 正文。"}</pre></div> : null}
          {selection.item.jd_analysis ? <JobAnalysisDetails analysis={selection.item.jd_analysis} version={selection.item.jd_analysis_version} stale={selection.item.jd_analysis_status === "stale"} /> : null}
        </> : <>
          <SearchableSelect label="简历版本" placeholder="选择简历版本" searchPlaceholder="搜索版本或日期" value={versionId} onChange={setVersionId} options={selection.item.versions.map((item) => ({ value: item.id, label: `v${item.version_number}${item.id === selection.item.latest_version_id ? " · 最新" : ""}`, description: `${dateLabel(item.created_at)} · ${item.change_summary || "暂无改动概要"}` }))} />
          {version ? <div className="dashboard-version-summary"><strong>v{version.version_number} 改动概要</strong><p>{version.change_summary}</p></div> : null}
          {version ? <>
            <iframe key={version.id} className="dashboard-resume-preview" title={`${selection.item.name} v${version.version_number} 原文件预览`} src={resumeDocumentUrl(selection.item.id, version.id, { apiBaseUrl: props.apiBaseUrl })} />
            <p className="workflow-note">如果浏览器无法预览此格式，可点击“查看原文件”或“下载”。</p>
          </> : <p>暂无可查看的简历版本。</p>}
        </>}
      </div>
    </dialog>
  );
}

export function DashboardPanel(props: PageProps) {
  const load = useCallback(
    (signal: AbortSignal) => fetchDashboard({ apiBaseUrl: props.apiBaseUrl, signal }),
    [props.apiBaseUrl],
  );
  const state = usePageData<Dashboard>(load, props.refreshToken, !props.hidden);
  const loadResumes = useCallback((signal: AbortSignal) => fetchResumes({ apiBaseUrl: props.apiBaseUrl, signal }), [props.apiBaseUrl]);
  const resumes = usePageData(loadResumes, props.refreshToken, !props.hidden);
  const [selection, setSelection] = useState<DashboardSelection | null>(null);
  const stats = state.data?.stats;
  const cards: { label: string; value: number; icon: AppIconName; tone: string; hint: string }[] = [
    { label: "岗位库", value: stats?.saved_jobs ?? 0, icon: "search", tone: "blue", hint: "已保存完整 JD" },
    { label: "全部投递", value: stats?.applications ?? 0, icon: "applications", tone: "blue", hint: "已记录的申请" },
    { label: "面试进行中", value: stats?.interviewing ?? 0, icon: "user", tone: "violet", hint: "需要持续准备" },
    { label: "简历版本", value: stats?.resumes ?? 0, icon: "document", tone: "cyan", hint: "按目标岗位管理" },
  ];
  return (
    <section className="management-page dashboard-page" hidden={props.hidden}>
      <PageHeader icon="dashboard" eyebrow="CAREER OVERVIEW" title="求职工作台" description="把投递进度、下一步行动和求职资料放在一个视图里" loading={state.loading || resumes.loading} onRefresh={() => { state.reload(); resumes.reload(); }} />
      <ErrorBanner message={state.error} />
      <div className="stat-grid">
        {cards.map((card) => (
          <button type="button" className={`stat-card tone-card-${card.tone}`} key={card.label} onClick={() => props.onNavigate?.(card.label === "岗位库" ? "jobs" : card.label === "简历版本" ? "resumes" : "applications")}>
            <span className="stat-icon"><AppIcon name={card.icon} size={22} /></span>
            <div><small>{card.label}</small><strong>{card.value}</strong><p>{card.hint}</p></div><span className="dashboard-arrow" aria-hidden="true">↗</span>
          </button>
        ))}
      </div>
      <div className="dashboard-grid">
        <section className="surface-card">
          <div className="section-heading"><div><small>NEXT ACTIONS</small><h2>下一步行动</h2></div><button type="button" className="section-link" onClick={() => props.onNavigate?.("brief")}>{state.data?.next_actions.length ?? 0} 项 · 查看全部</button></div>
          {state.data && state.data.next_actions.length > 0 ? (
            <div className="action-list">
              {state.data.next_actions.map((item) => (
                <button type="button" className="dashboard-action-row" key={item.id} onClick={() => props.onNavigate?.("brief")}><span><AppIcon name="check" size={17} /></span><div><strong>{item.title}</strong><small>{item.summary}</small></div></button>
              ))}
            </div>
          ) : <EmptyState icon="check" title="当前没有待办" description="Agent 会根据投递、面试和邮件事件生成行动项。" action="问问 Agent" onAction={() => props.onAskAgent("根据我的求职进度，建议我下一步做什么")} />}
        </section>
        <section className="surface-card">
          <div className="section-heading"><div><small>APPLICATION PIPELINE</small><h2>投递进展</h2></div><button type="button" className="section-link" onClick={() => props.onNavigate?.("applications")}>{stats?.offers ?? 0} 个 Offer · 查看全部</button></div>
          {state.data && state.data.recent_applications.length > 0 ? (
            <div className="compact-list">
              {state.data.recent_applications.map((item) => (
                <button type="button" className="dashboard-link-row" key={item.id} onClick={() => props.onNavigate?.("applications")}>
                  <span className="list-icon tone-blue"><AppIcon name="applications" size={18} /></span>
                  <div><strong>{item.title}</strong><small>{item.company_name} · {dateLabel(item.submitted_at)}</small></div>
                  <span className={`status-pill status-${item.status}`}>{STATUS_LABELS[item.status] ?? item.status}</span>
                </button>
              ))}
            </div>
          ) : <EmptyState icon="applications" title="还没有投递记录" description="在对话里保存第一次投递后，进度会出现在这里。" action="记录一次投递" onAction={() => props.onAskAgent("帮我记录一次新的岗位投递")} />}
        </section>
        <section className="surface-card surface-card-wide">
          <div className="section-heading"><div><small>SAVED JOBS</small><h2>最近保存的 JD</h2></div><button type="button" className="section-link" onClick={() => props.onNavigate?.("jobs")}>{stats?.saved_jobs ?? 0} 个岗位 · 查看全部</button></div>
          {state.data && state.data.recent_jobs.length > 0 ? (
            <div className="dashboard-resource-grid">
              {state.data.recent_jobs.map((item) => (
                <button type="button" className="dashboard-resource-card" key={item.id} onClick={() => setSelection({ kind: "job", item })}>
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
                </button>
              ))}
            </div>
          ) : <EmptyState icon="search" title="岗位库还是空的" description="保存你真正看过的完整 JD；它不会因为保存就变成投递记录。" action="去找岗位" onAction={() => props.onAskAgent("帮我打开招聘网站，我想查找并保存感兴趣的岗位")} />}
        </section>
        <section className="surface-card surface-card-wide">
          <div className="section-heading"><div><small>RESUMES</small><h2>我的简历</h2></div><button type="button" className="section-link" onClick={() => props.onNavigate?.("resumes")}>管理全部简历</button></div>
          <ErrorBanner message={resumes.error} />
          {resumes.error ? <button type="button" className="soft-button" onClick={resumes.reload}>重新读取简历</button> : null}
          {resumes.loading && !resumes.data ? <p role="status">正在读取简历…</p> : null}
          {resumes.data?.length ? <div className="dashboard-resource-grid">{resumes.data.map((item) => (
            <button type="button" className="dashboard-resource-card" key={item.id} onClick={() => setSelection({ kind: "resume", item })}>
              <span className="list-icon tone-blue"><AppIcon name="document" size={22} /></span>
              <div><strong>{item.name}</strong><small>{item.target_role || "通用简历"}</small><small>v{item.latest_version_number} · {item.version_count} 个版本 · {dateLabel(item.updated_at)} 更新</small></div>
              <span className="status-pill status-current">{item.document_format.toUpperCase()}</span>
            </button>
          ))}</div> : resumes.data ? <EmptyState icon="document" title="还没有简历" description="导入简历后，可以在这里直接打开并查看各个版本。" action="导入简历" onAction={() => props.onNavigate?.("resumes")} /> : null}
        </section>
      </div>
      {selection && !props.hidden ? <DashboardDetail key={`${selection.kind}:${selection.item.id}`} selection={selection} props={props} onClose={() => setSelection(null)} /> : null}
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
  onOpenConversation,
  onStartStandaloneTask,
  onClose,
}: {
  application: ApplicationView;
  apiBaseUrl: string;
  refreshToken: number;
  onAskAgent: (prompt: string) => void;
  onOpenConversation?: (conversationId: string) => void;
  onStartStandaloneTask?: StartStandaloneTask;
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
  // This panel already has the selected application. Keep the application id
  // in the request so the agent does not need to route through a second
  // natural-language "choose a job" step.
  const startPrompt = `开始一轮模拟面试。目标投递记录 application_id=${application.id}（${application.company_name} · ${application.title}）。只询问我面试类型和题目数量，然后直接开始。`;
  const startPractice = useCallback(() => {
    const resource: ChatAttachment = {
      kind: "application",
      applicationId: application.id,
      title: `${application.company_name} · ${application.title}`,
      description: "已选定的投递记录",
    };
    if (onStartStandaloneTask) onStartStandaloneTask(startPrompt, resource);
    else onAskAgent(startPrompt);
  }, [application.company_name, application.id, application.title, onAskAgent, onStartStandaloneTask, startPrompt]);

  return (
    <section className="surface-card application-practice-panel">
      <header>
        <div>
          <small>INTERVIEW PRACTICE</small>
          <h2>面试与练习</h2>
          <p>{application.company_name} · {application.title}</p>
        </div>
        <div>
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
                  ? "这轮模拟面试仍在原对话中进行。每次回答先判断是否需要追问，整场结束后再统一评分。"
                  : session.status === "paused"
                    ? "这轮模拟面试已暂停，已有回答会保留，继续后才会生成最终总结。"
                    : session.status === "cancelled"
                      ? "这轮模拟面试已取消，不会生成最终总结。"
                  : "这轮模拟面试尚未生成最终总结。"
              )}</p>
              {session.status === "completed" && session.report_id ? (
                <small className="workflow-note">已按题目独立评估（含追问链），再汇总为整场报告。</small>
              ) : null}
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
              <RepeatPracticeButton session={session} onStartStandaloneTask={onStartStandaloneTask} />
              {session.status === "active" || session.status === "paused" ? (
                session.conversation_id && onOpenConversation ? (
                  <button
                    type="button"
                    className="card-agent-action"
                    onClick={() => onOpenConversation(session.conversation_id!)}
                  >
                    回到原对话继续练习
                  </button>
                ) : (
                  <small className="workflow-note">这轮练习只能在最初发起它的对话中继续；没有找到仍持有它的对话。</small>
                )
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
          onAction={startPractice}
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
  const [editingResume, setEditingResume] = useState<string | null>(null);
  const [editedVersion, setEditedVersion] = useState("");
  const [savingResume, setSavingResume] = useState(false);
  const [clearing, setClearing] = useState(false);
  const [actionError, setActionError] = useState<string | null>(null);
  useEffect(() => {
    const open = (event: Event) => { const job = (event as CustomEvent<string>).detail; setJobId(job); setResumeVersionId(""); setShowForm(true); };
    window.addEventListener("open-application-form", open);
    return () => window.removeEventListener("open-application-form", open);
  }, []);
  async function saveResumeVersion(applicationId: string): Promise<void> {
    setSavingResume(true);
    setActionError(null);
    try {
      await updateApplicationResumeVersion(
        applicationId,
        editedVersion === UNKNOWN_RESUME ? null : editedVersion,
        { apiBaseUrl: props.apiBaseUrl },
      );
      setEditingResume(null);
      state.reload();
    } catch (cause) {
      setActionError(cause instanceof Error ? cause.message : "无法更新这次投递使用的简历。");
    } finally {
      setSavingResume(false);
    }
  }

  async function clearAll(): Promise<void> {
    if (!window.confirm("确定清空全部投递记录吗？此操作不可恢复。")) return;
    setClearing(true); setActionError(null);
    try { await clearApplications({ apiBaseUrl: props.apiBaseUrl }); state.reload(); } catch (e) { setActionError(e instanceof Error ? e.message : "清空失败"); } finally { setClearing(false); }
  }

  async function submitApplication(): Promise<void> {
    setSaving(true);
    setFormError(null);
    try {
      await createApplication({
        jobPostingId: jobId,
        resumeVersionId: resumeVersionId && resumeVersionId !== UNKNOWN_RESUME ? resumeVersionId : undefined,
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
      <PageHeader
        icon="applications"
        eyebrow="APPLICATIONS"
        title="投递记录"
        description="按公司和岗位查看从已投递到 Offer 的完整进度"
        loading={state.loading}
        onRefresh={state.reload}
        actions={<div className="management-actions">
          <button type="button" className="soft-button" aria-expanded={showForm} onClick={() => setShowForm((visible) => !visible)}><AppIcon name="applications" size={16} /> {showForm ? "收起记录表单" : "直接记录投递"}</button>
          <button type="button" className="soft-button" onClick={() => props.onAskAgent("帮我记录一次新的岗位投递；请根据我已保存的岗位和简历询问并确认必要信息")}><AppIcon name="sparkles" size={16} /> 让 Agent 记录</button>
          <button type="button" className="danger-action" disabled={clearing || !state.data?.applications.length} onClick={() => void clearAll()}><AppIcon name="trash" size={16} />{clearing ? "正在清空…" : "清空投递记录"}</button>
        </div>}
      />
      <ErrorBanner message={state.error} />
      <ErrorBanner message={actionError} />
      <ErrorBanner message={formError} />
      {showForm ? <section className="surface-card application-create-form"><div><SearchableSelect label="已保存岗位" placeholder="请选择岗位" searchPlaceholder="搜索公司或岗位" value={jobId} onChange={setJobId} disabled={saving} options={(state.data?.jobs ?? []).map((job) => ({ value: job.id, label: job.title, description: job.company_name }))} /><SearchableSelect label="使用的简历" placeholder="请选择简历版本" searchPlaceholder="搜索简历名称或版本" value={resumeVersionId} onChange={setResumeVersionId} disabled={saving} options={resumeVersionOptions(state.data?.resumes ?? [])} /><label>实际投递时间<input type="datetime-local" value={submittedAt} onChange={(event) => setSubmittedAt(event.target.value)} /></label><label>备注（可选）<textarea value={note} onChange={(event) => setNote(event.target.value)} maxLength={2000} placeholder="例如：官网投递、内推人等" /></label></div>{state.data && state.data.jobs.length === 0 ? <p>记录前需要至少一个已保存岗位。可以先前往岗位库，或让 Agent 协助。</p> : null}<button type="button" disabled={saving || !jobId || !resumeVersionId || !submittedAt} onClick={() => void submitApplication()}>{saving ? "正在保存…" : "确认已在外部平台投递并记录"}</button></section> : null}
      {state.data && state.data.applications.length > 0 ? <div className="data-table-card"><div className="data-table-head"><span>岗位</span><span>状态</span><span>地点 / 薪资</span><span>投递时间</span></div>{state.data.applications.map((item) => <article className={`data-table-row ${practiceApplication?.id === item.id ? "is-selected" : ""}`} key={item.id}><div><span className="list-icon tone-blue"><AppIcon name="applications" size={18} /></span><span><strong>{item.title}</strong><small>{item.company_name}</small><small className="application-resume">{item.resume_name ? <>使用简历：{item.resume_name} · 第 {item.resume_version_number} 版{item.resume_deleted ? "（已删除）" : ""}{item.resume_id && item.resume_version_id ? <> · <a href={resumeDocumentUrl(item.resume_id, item.resume_version_id, { apiBaseUrl: props.apiBaseUrl })} target="_blank" rel="noreferrer noopener">查看</a></> : null}</> : "未记录简历版本"} · <button type="button" className="link-button" onClick={() => { setEditingResume(item.id); setEditedVersion(item.resume_version_id ?? UNKNOWN_RESUME); }}>{item.resume_version_id ? "更换" : "补填"}</button></small>{editingResume === item.id ? <span className="application-resume-editor"><SearchableSelect label="投递时用的简历" placeholder="请选择简历版本" searchPlaceholder="搜索简历名称或版本" value={editedVersion} onChange={setEditedVersion} disabled={savingResume} options={resumeVersionOptions(state.data?.resumes ?? [])} /><button type="button" disabled={savingResume || !editedVersion} onClick={() => void saveResumeVersion(item.id)}>{savingResume ? "正在保存…" : "保存"}</button><button type="button" className="link-button" disabled={savingResume} onClick={() => setEditingResume(null)}>取消</button></span> : null}<button type="button" className="row-detail-button" onClick={() => setPracticeApplication(item)}>面试与练习</button></span></div><span><span className={`status-pill status-${item.status}`}>{STATUS_LABELS[item.status] ?? item.status}</span>{item.interview_round_number ? <small className="application-round-label">{item.interview_round_label || `第 ${item.interview_round_number} 场`}</small> : null}</span><span>{[item.city, item.salary].filter(Boolean).join(" · ") || "未披露"}</span><time>{dateLabel(item.submitted_at)}</time></article>)}</div> : <EmptyState icon="applications" title="还没有投递记录" description="可直接选择已保存岗位和简历记录，也可以让 Agent 通过对话协助。" />}
      {practiceApplication ? (
        <ApplicationPracticePanel
          application={practiceApplication}
          apiBaseUrl={props.apiBaseUrl}
          refreshToken={props.refreshToken}
          onAskAgent={props.onAskAgent}
          onOpenConversation={props.onOpenConversation}
          onStartStandaloneTask={props.onStartStandaloneTask}
          onClose={() => setPracticeApplication(null)}
        />
      ) : null}
    </section>
  );
}

const DERIVED_STATUS_LABELS: Record<SavedJobView["jd_analysis_status"], string> = {
  none: "待分析",
  ready: "已完成",
  stale: "当前版本待分析",
};

export function derivedStatusLabel(status: SavedJobView["jd_analysis_status"]): string {
  return DERIVED_STATUS_LABELS[status];
}

function derivedStatusClass(status: SavedJobView["jd_analysis_status"]): string {
  return status === "ready" ? "status-current" : "status-outdated";
}

/** The message that starts a JD-only analysis of one saved job. */
export function jobAnalysisPrompt(job: Pick<SavedJobView, "title" | "company_name">): string {
  return `请仅基于 JD 文本分析岗位库里的「${job.company_name} · ${job.title}」，不要结合简历。`;
}

export function JobsPanel(props: PageProps) {
  // JD analysis is a task of its own: a new conversation with the exact JD
  // snapshot attached and nothing else, so an open chat's resume, job or
  // interview state cannot leak into what is meant to be a reading of the JD.
  // Without a snapshot (nothing captured yet) it falls back to naming the job.
  const analyzeJob = (item: SavedJobView) => {
    const prompt = jobAnalysisPrompt(item);
    const attachment = attachmentFromSavedJob(item);
    if (attachment && props.onStartStandaloneTask) {
      props.onStartStandaloneTask(prompt, attachment);
      return;
    }
    props.onAskAgent(prompt);
  };
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
  const [openMatches, setOpenMatches] = useState<string | null>(null);
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
                {/* Two pills for two independent facts: whether the JD itself
                    has been analysed, and whether a resume was matched against
                    it. "stale" means the JD was captured again after the
                    result; the old result stays as history, the pill says the
                    current version has not been read yet. */}
                <span className={`status-pill ${derivedStatusClass(item.jd_analysis_status)}`}>
                  JD 分析·{derivedStatusLabel(item.jd_analysis_status)}
                </span>
                <span className={`status-pill ${derivedStatusClass(item.resume_match_status)}`}>
                  简历匹配·{item.resume_match_status === "none" ? "尚未匹配" : derivedStatusLabel(item.resume_match_status)}
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
              {item.jd_analysis ? (
                <div className="job-analysis-body">
                  <p className="job-summary">{item.jd_analysis.summary}</p>
                  <JobAnalysisDetails
                    analysis={item.jd_analysis}
                    version={item.jd_analysis_version}
                    stale={item.jd_analysis_status === "stale"}
                  />
                  {item.jd_analysis_status === "stale" ? (
                    <button type="button" className="link" onClick={() => analyzeJob(item)}>
                      分析当前版本 JD
                    </button>
                  ) : null}
                </div>
              ) : item.analysis_summary ? (
                <div className="job-analysis-body">
                  <p className="job-summary">{item.analysis_summary}</p>
                  {item.required_skills.length > 0 ? (
                    <div className="analysis-section"><strong>必备技能</strong><div className="skill-chips">{item.required_skills.map((skill) => <span key={skill}>{skill}</span>)}</div></div>
                  ) : null}
                  {item.responsibilities.length > 0 ? (
                    <div className="analysis-section"><strong>核心职责</strong><ul>{item.responsibilities.slice(0, 4).map((entry) => <li key={entry}>{entry}</li>)}</ul></div>
                  ) : null}
                  {item.clarification_questions.length > 0 ? (
                    <details className="analysis-question">
                      <summary><AppIcon name="target" size={16} /><span>{item.clarification_questions.length} 个信息点需要进一步确认</span><span className="analysis-question-toggle"><span className="when-closed">展开查看</span><span className="when-open">收起</span></span></summary>
                      <ol>{item.clarification_questions.map((question, index) => <li key={index}>{question}</li>)}</ol>
                    </details>
                  ) : null}
                </div>
              ) : (
                <div className="job-analysis-pending">
                  <p>完整 JD 已保存，但还没有结构化分析。</p>
                  <button type="button" onClick={() => analyzeJob(item)}>让 Agent 分析</button>
                </div>
              )}
              <footer>
                <span>{item.application_status ? STATUS_LABELS[item.application_status] ?? item.application_status : "尚未投递"}</span>
                <time>
                  {dateLabel(item.captured_at)} 保存
                  {item.analyzed_at ? ` · ${dateLabel(item.analyzed_at)} 分析` : ""}
                  {item.resume_match_at ? ` · ${dateLabel(item.resume_match_at)} 匹配` : ""}
                </time>
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
                <button
                  type="button"
                  className="link"
                  aria-expanded={openMatches === item.id}
                  onClick={() => setOpenMatches(openMatches === item.id ? null : item.id)}
                >
                  {openMatches === item.id ? "收起匹配分析" : `查看匹配分析（${item.resume_match_count ?? 0}）`}
                </button>
                <button type="button" className="job-primary-action" aria-expanded={openJd === item.id} onClick={() => toggleJd(item.id)}>
                  {openJd === item.id ? "收起 JD" : "查看完整 JD"}
                </button>
                <button type="button" className="link" onClick={() => analyzeJob(item)}>
                  让 Agent 分析
                </button>
                <button
                  type="button"
                  className="link"
                  onClick={() => props.onAskAgent(`研究岗位库里的「${item.company_name} · ${item.title}」所属公司的业务、产品线和市场情况`)}
                >
                  让 Agent 研究公司
                </button>
              </div>
              {openMatches === item.id && !props.hidden ? (
                <JobMatchesPanel
                  job={item}
                  apiBaseUrl={props.apiBaseUrl}
                  refreshToken={props.refreshToken}
                  onStartTask={props.onStartStandaloneTask}
                />
              ) : null}
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
                    onClick={() => window.dispatchEvent(new CustomEvent("open-application-form", { detail: item.id }))}
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
  const load = useCallback(async (signal: AbortSignal) => {
    const [resumes, roles] = await Promise.all([
      fetchResumes({ apiBaseUrl: props.apiBaseUrl, signal }),
      fetchTargetRoles({ apiBaseUrl: props.apiBaseUrl, signal }),
    ]);
    return { resumes, roles };
  }, [props.apiBaseUrl]);
  const state = usePageData<{ resumes: ResumeView[]; roles: TargetRoleView[] }>(load, props.refreshToken, !props.hidden);
  // null: closed; "" : import with the importer's own role choice; otherwise into that role.
  const [importRoleId, setImportRoleId] = useState<string | null>(null);
  const analyzeResume = props.onStartStandaloneTask ?? props.onAskAgent;
  const [actionError, setActionError] = useState<string | null>(null);
  const [deletingResume, setDeletingResume] = useState<string | null>(null);
  const [movingResume, setMovingResume] = useState<string | null>(null);
  async function removeResume(id: string, name: string): Promise<void> {
    if (!window.confirm(`确定删除简历“${name}”及其全部版本吗？`)) return;
    setDeletingResume(id); setActionError(null);
    try { await deleteResume(id, { apiBaseUrl: props.apiBaseUrl }); state.reload(); } catch (e) { setActionError(e instanceof Error ? e.message : "删除失败"); } finally { setDeletingResume(null); }
  }
  async function relocateResume(id: string, targetRoleId: string): Promise<void> {
    setMovingResume(id); setActionError(null);
    try { await moveResume(id, targetRoleId, { apiBaseUrl: props.apiBaseUrl }); state.reload(); } catch (e) { setActionError(e instanceof Error ? e.message : "移动失败"); } finally { setMovingResume(null); }
  }
  const roles = state.data?.roles ?? [];
  const groups = state.data ? groupResumesByRole(state.data.resumes, roles) : [];
  const importer = (roleId: string) => (
    <ResumeImporter
      key={roleId || "any"}
      apiBaseUrl={props.apiBaseUrl}
      initialTargetRoleId={roleId || undefined}
      onCancel={() => setImportRoleId(null)}
      onImported={(result) => {
        setImportRoleId(null);
        state.reload();
        analyzeResume("帮我分析这份简历", attachmentFromImport(result));
      }}
    />
  );
  return (
    <section className="management-page" hidden={props.hidden}>
      <PageHeader icon="document" eyebrow="RESUME LIBRARY" title="简历管理" description="按目标岗位管理简历家族、版本和来源" loading={state.loading} onRefresh={state.reload} actions={<div className="management-actions"><button type="button" className="soft-button" onClick={() => setImportRoleId((open) => open === "" ? null : "")}><AppIcon name="plus" size={16} /> {importRoleId === "" ? "收起导入" : "导入简历"}</button></div>} />
      <ErrorBanner message={state.error} />
      <ErrorBanner message={actionError} />
      {importRoleId === "" ? importer("") : null}
      {state.data && state.data.resumes.length > 0 ? groups.map((group) => (
        <section className="resume-role-group" key={group.roleId} aria-label={`目标岗位：${group.title}`}>
          <div className="resume-role-heading">
            <h2>{group.title}</h2>
            <span>{group.resumes.length} 份</span>
            <button type="button" className="link-button" onClick={() => setImportRoleId((open) => open === group.roleId ? null : group.roleId)}>
              {importRoleId === group.roleId ? "收起导入" : "导入到这个岗位"}
            </button>
          </div>
          {importRoleId === group.roleId ? importer(group.roleId) : null}
          {group.resumes.length > 0 ? (
            <div className="card-grid">
              {group.resumes.map((item) => {
                const [latest, ...older] = item.versions;
                const otherRoles = roles.filter((role) => role.id !== item.target_role_id);
                return (
                  <article className="resume-card" key={item.id}>
                    <span className="document-preview"><AppIcon name="document" size={28} /><small>{item.document_format.toUpperCase()}</small></span>
                    <div className="card-body">
                      <h2>{item.name}</h2>
                      <p>最新 v{item.latest_version_number} · 共 {item.version_count} 个版本</p>
                      <div className="card-meta"><span>{formatBytes(item.byte_size)}</span><time>{dateLabel(item.updated_at)} 更新</time></div>
                      {latest ? (
                        <ResumeVersionRow resumeName={item.name} version={latest} apiBaseUrl={props.apiBaseUrl} onAnalyze={analyzeResume} latest />
                      ) : null}
                      {older.length > 0 ? (
                        <details className="resume-version-history">
                          <summary>历史版本（{older.length}）</summary>
                          {older.map((version) => (
                            <ResumeVersionRow key={version.id} resumeName={item.name} version={version} apiBaseUrl={props.apiBaseUrl} onAnalyze={analyzeResume} />
                          ))}
                        </details>
                      ) : null}
                      <div className="resume-card-actions">
                        {otherRoles.length > 0 ? (
                          <DropdownMenu
                            label={movingResume === item.id ? "正在移动…" : "移动到…"}
                            ariaLabel={`把“${item.name}”移动到其他岗位`}
                            disabled={movingResume === item.id}
                            items={otherRoles.map((role) => ({ value: role.id, label: role.title }))}
                            onSelect={(roleId) => void relocateResume(item.id, roleId)}
                          />
                        ) : null}
                        <button type="button" className="danger-action" disabled={deletingResume === item.id} onClick={() => void removeResume(item.id, item.name)}><AppIcon name="trash" size={15} />{deletingResume === item.id ? "正在删除…" : "删除简历"}</button>
                      </div>
                    </div>
                  </article>
                );
              })}
            </div>
          ) : importRoleId !== group.roleId ? <p className="resume-role-empty">这个岗位下还没有简历。</p> : null}
        </section>
      )) : importRoleId === null ? (
        <EmptyState icon="document" title="还没有简历" description="先导入 PDF、TXT 或 Markdown；导入后可立即交给 Agent 分析。" action="导入一份简历" onAction={() => setImportRoleId("")} />
      ) : null}
    </section>
  );
}

/**
 * One stored version: when it was uploaded, its format and size, and the
 * three things a person can do with it. "让 Agent 分析" sends this exact
 * version's id, so the conversation stays pinned to it even after a newer
 * version is imported.
 */
function ResumeVersionRow({
  resumeName,
  version,
  apiBaseUrl,
  onAnalyze,
  latest = false,
  showAnalyze = true,
}: {
  resumeName: string;
  version: ResumeVersionView;
  apiBaseUrl: string;
  onAnalyze: (prompt: string, resource: ResumeAttachment) => void;
  latest?: boolean;
  showAnalyze?: boolean;
}) {
  return (
    <div className={`resume-version-row ${latest ? "is-latest" : ""}`}>
      <div>
        <strong>v{version.version_number}{latest ? " · 最新" : ""}</strong>
        <small>{version.document_format.toUpperCase()} · {formatBytes(version.byte_size)} · {new Date(version.created_at).toLocaleString("zh-CN")} 上传</small>
        <p className="resume-version-summary"><strong>改动概要</strong>{version.change_summary || "暂无可用的版本摘要"}</p>
      </div>
      <div className="resume-version-actions">
        <a href={resumeDocumentUrl(version.resume_id, version.id, { apiBaseUrl })} target="_blank" rel="noreferrer">查看原文件</a>
        <a href={resumeDocumentUrl(version.resume_id, version.id, { apiBaseUrl, download: true })}>下载</a>
        {showAnalyze ? <button type="button" onClick={() => onAnalyze(`帮我分析简历“${resumeName}”的 v${version.version_number}`, attachmentFromVersion(resumeName, version))}>让 Agent 分析</button> : null}
      </div>
    </div>
  );
}

type StartStandaloneTask = (
  prompt: string,
  resource: ChatAttachment | null,
  additionalResources?: ChatAttachment[],
  label?: string,
) => void;

/** Start the same practice again, in a new conversation, from a finished or cancelled run. */
function RepeatPracticeButton({
  session,
  onStartStandaloneTask,
}: {
  session: MockInterviewSessionView;
  onStartStandaloneTask?: StartStandaloneTask;
}) {
  if (!onStartStandaloneTask || (session.status !== "cancelled" && session.status !== "completed")) return null;
  return (
    <button
      type="button"
      className="card-agent-action"
      onClick={() => {
        const repeat = practiceRepeat(session);
        onStartStandaloneTask(repeat.prompt, repeat.resource, repeat.additionalResources, repeat.label);
      }}
    >
      再来一次
    </button>
  );
}

export function CalendarPanel(props: PageProps) {
  const [activeSection, setActiveSection] = useState<"schedule" | "practice">("schedule");
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
  const state = usePageData<CalendarWorkspace>(load, props.refreshToken, !props.hidden && activeSection === "schedule");
  const loadPractice = useCallback(async (signal: AbortSignal) => {
    const options = { apiBaseUrl: props.apiBaseUrl, signal };
    const history = await fetchMockInterviews(options);
    return history.sessions.sort((left, right) => Date.parse(right.updated_at) - Date.parse(left.updated_at));
  }, [props.apiBaseUrl]);
  const practice = usePageData(loadPractice, props.refreshToken, !props.hidden && activeSection === "practice");
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
      <PageHeader
        icon="calendar"
        eyebrow="INTERVIEW CENTER"
        title="面试中心"
        description="统一管理真实面试日程、模拟练习与面试复盘"
        loading={activeSection === "schedule" ? state.loading : practice.loading}
        onRefresh={activeSection === "schedule" ? state.reload : practice.reload}
      />
      <div className="interview-tabs" role="tablist" aria-label="面试中心内容">
        <button type="button" role="tab" aria-selected={activeSection === "schedule"} className={activeSection === "schedule" ? "is-active" : ""} onClick={() => setActiveSection("schedule")}>
          <AppIcon name="calendar" size={16} /> 面试日程
        </button>
        <button type="button" role="tab" aria-selected={activeSection === "practice"} className={activeSection === "practice" ? "is-active" : ""} onClick={() => setActiveSection("practice")}>
          <AppIcon name="sparkles" size={16} /> 模拟面试
        </button>
      </div>
      {activeSection === "schedule" ? <>
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
      {selected ? <aside className="calendar-detail"><button type="button" className="calendar-detail-close" onClick={() => setSelected(null)}>关闭</button><span className="card-kicker">{selected.employer_label || "面试安排"}</span><h2>{selected.company_name || "未关联公司"} · {selected.job_title || "未关联岗位"}</h2><p>{selected.scheduled_start ? new Date(selected.scheduled_start).toLocaleString("zh-CN", { timeZone: selected.timezone || timezone }) : "时间待确认"} · {selected.timezone}</p><dl><div><dt>形式</dt><dd>{selected.interview_format || "unknown"}</dd></div><div><dt>地点</dt><dd>{selected.location || "未提供"}</dd></div><div><dt>日历状态</dt><dd>{selected.sync_status === "not_synced" ? "尚未添加到 Google Calendar" : selected.sync_status}</dd></div>{selected.contact_summary ? <div><dt>联系人</dt><dd>{selected.contact_summary}</dd></div> : null}</dl><div className="calendar-detail-actions">{selectedMeetingUrl ? <a href={selectedMeetingUrl} target="_blank" rel="noreferrer">加入会议</a> : null}{selectedExternalUrl ? <a href={selectedExternalUrl} target="_blank" rel="noreferrer">在 Google Calendar 打开</a> : <button type="button" onClick={() => props.onAskAgent(`把 interview_round_id=${selected.interview_round_id} 的面试安排添加到 Google Calendar`)}>添加到 Google Calendar</button>}</div></aside> : null}
      </> : <>
        <ErrorBanner message={practice.error} />
        <section className="surface-card interview-practice-hero">
          <div>
            <span className="list-icon tone-blue"><AppIcon name="sparkles" size={21} /></span>
            <div><small>PRACTICE HISTORY</small><h2>模拟面试复盘</h2><p>这里集中查看挂投递岗位的练习和自由练习。你可以从投递记录开始，也可以直接开启一场自由练习。</p></div>
          </div>
          <button type="button" className="soft-button" onClick={() => props.onAskAgent("开始一场自由模拟面试，先询问我的目标岗位和题目数量")}><AppIcon name="sparkles" size={16} /> 开始自由练习</button>
        </section>
        {practice.loading && !practice.data ? <div className="history-loading"><span className="spinner" /> 正在汇总练习记录…</div> : null}
        {practice.data && practice.data.length > 0 ? (
          <section className="interview-practice-history">
            <div className="section-heading"><div><small>HISTORY</small><h2>全部练习记录</h2></div><span>{practice.data.length} 轮</span></div>
            <div className="mock-session-list">
              {practice.data.map((session) => (
                <article className="mock-session-card interview-session-card" key={session.session_id}>
                  <div className="mock-session-heading">
                    <div><span className="card-kicker">{session.interview_type_label}</span><strong>{[session.company_name, session.title].filter(Boolean).join(" · ")}</strong></div>
                    <span className={`status-pill status-${session.status}`}>{MOCK_INTERVIEW_STATUS_LABELS[session.status] ?? session.status}</span>
                  </div>
                  <p>{session.summary ?? (session.status === "active" ? "这轮模拟面试仍在原对话中进行。每次回答先判断是否需要追问，整场结束后再统一评分。" : session.status === "paused" ? "练习已暂停，可以回到原对话继续。" : session.status === "cancelled" ? "这轮练习已取消。" : "练习尚未生成最终总结。")}</p>
                  {session.status === "completed" && session.report_id ? <small className="workflow-note">已按题目独立评估（含追问链），再汇总为整场报告。</small> : null}
                  <small>{dateLabel(session.completed_at ?? session.created_at)} · 已完成 {session.question_count} / {session.max_primary_questions} 道主问题</small>
                  <small className="application-resume">{session.resume_name ? <>使用简历：{session.resume_name} · 第 {session.resume_version_number} 版{session.resume_deleted ? "（已删除）" : ""}{session.resume_id && session.resume_version_id ? <> · <a href={resumeDocumentUrl(session.resume_id, session.resume_version_id, { apiBaseUrl: props.apiBaseUrl })} target="_blank" rel="noreferrer noopener">查看</a></> : null}</> : "未使用简历"}</small>
                  {session.report_id ? <ReportCard apiBaseUrl={props.apiBaseUrl} resource={{ kind: "mock_interview_report", resourceId: session.report_id }} /> : null}
                  <RepeatPracticeButton session={session} onStartStandaloneTask={props.onStartStandaloneTask} />
                  {(session.status === "active" || session.status === "paused") && session.conversation_id && props.onOpenConversation ? <button type="button" className="card-agent-action" onClick={() => props.onOpenConversation?.(session.conversation_id!)}>回到原对话继续练习</button> : null}
                </article>
              ))}
            </div>
          </section>
        ) : practice.data ? <section className="surface-card"><EmptyState icon="sparkles" title="还没有模拟面试记录" description="可以从投递记录开始岗位练习，也可以点击“开始自由练习”直接开始。" action="开始自由练习" onAction={() => props.onAskAgent("开始一场自由模拟面试，先询问我的目标岗位和题目数量")} /></section> : null}
      </>}
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
      <PageHeader icon="mail" eyebrow="EMAIL TRACKING" title="邮件追踪" description="查看招聘邮箱账号、同步状态和 Agent 识别出的求职事件" loading={state.loading} onRefresh={state.reload} actions={<div className="management-actions"><button type="button" className="soft-button" onClick={() => void connectGoogle()}><AppIcon name="mail" size={16} /> 连接 Gmail</button><button type="button" className="soft-button" onClick={() => setShowQQ((open) => !open)} aria-expanded={showQQ}>{showQQ ? "收起 QQ 邮箱连接" : "连接 QQ 邮箱"}</button><button type="button" className="soft-button" onClick={() => props.onAskAgent("同步我已连接邮箱中的最新招聘邮件，并列出需要我确认的事件")}><AppIcon name="sparkles" size={16} /> 让 Agent 同步邮箱</button>{pending.length > 0 ? <button type="button" className="soft-button" onClick={() => props.onAskAgent("逐项处理邮箱里等待确认的求职事件")}>处理 {pending.length} 个待确认事件</button> : null}</div>} />
      <ErrorBanner message={state.error} />
      <ErrorBanner message={connectionError} />
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
  const [actionError, setActionError] = useState<string | null>(null);
  const [deletingReport, setDeletingReport] = useState<string | null>(null);
  async function removeReport(id: string): Promise<void> {
    if (!window.confirm("确定删除这份公司研究及其来源吗？此操作不可恢复。")) return;
    setDeletingReport(id); setActionError(null);
    try { await deleteCompanyResearch(id, { apiBaseUrl: props.apiBaseUrl }); state.reload(); } catch (e) { setActionError(e instanceof Error ? e.message : "删除失败"); } finally { setDeletingReport(null); }
  }
  return (
    <section className="management-page" hidden={props.hidden}>
      <PageHeader icon="building" eyebrow="COMPANY RESEARCH" title="公司研究" description="集中查看用户主动要求生成的业务、产品线和市场资料" loading={state.loading} onRefresh={state.reload} actions={<div className="management-actions"><button type="button" className="soft-button" onClick={() => props.onAskAgent("我想研究岗位库里一家目标公司的业务、产品线和市场情况")}><AppIcon name="sparkles" size={16} /> 让 Agent 研究公司</button></div>} />
      <ErrorBanner message={state.error} />
      <ErrorBanner message={actionError} />
      {state.data && state.data.length > 0 ? <div className="research-grid">{state.data.map((item) => <article className="research-card" key={item.id}><div className="research-card-top"><span className="company-avatar">{item.company_name.slice(0, 1)}</span><div><span className="card-kicker">{item.focus || "公司与业务概览"}</span><h2>{item.company_name}</h2></div><span className={`status-pill status-${item.status}`}>{STATUS_LABELS[item.status] ?? item.status}</span></div><p>{item.summary}</p><div className="card-meta"><span>{item.finding_count} 条发现</span><span>{item.source_count} 个来源</span><time>{dateLabel(item.created_at)}</time></div><small>关联岗位 · {item.anchor_job_title}</small><ReportCard resource={{ kind: "job_research_report", resourceId: item.id }} apiBaseUrl={props.apiBaseUrl} /><div className="research-card-actions"><button type="button" className="danger-action" disabled={deletingReport === item.id} onClick={() => void removeReport(item.id)}><AppIcon name="trash" size={15} />{deletingReport === item.id ? "正在删除…" : "删除研究"}</button></div></article>)}</div> : <EmptyState icon="building" title="还没有公司研究" description="研究需要以岗位库中的已保存岗位为锚点，由 Agent 确认范围后启动。" action="让 Agent 研究公司" onAction={() => props.onAskAgent("我想研究岗位库里一家目标公司的业务和产品线")} />}
    </section>
  );
}
