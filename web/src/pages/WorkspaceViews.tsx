import { useCallback, useEffect, useState } from "react";

import {
  type ApplicationView,
  type CalendarWorkspace,
  type CompanyResearchView,
  type Dashboard,
  type ResumeView,
  type SavedJobView,
  fetchApplications,
  fetchCalendar,
  fetchCompanyResearch,
  fetchDashboard,
  fetchResumes,
  fetchSavedJobs,
} from "../api/client";
import { AppIcon, type AppIconName } from "../components/AppIcon";

type PageProps = {
  userId: string;
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

function dateLabel(value: string): string {
  return new Date(value).toLocaleDateString("zh-CN", { month: "short", day: "numeric" });
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
  action: string;
  onAction: () => void;
}) {
  return (
    <div className="management-empty">
      <span><AppIcon name={icon} size={28} /></span>
      <strong>{title}</strong>
      <p>{description}</p>
      <button type="button" onClick={onAction}>{action}</button>
    </div>
  );
}

function ErrorBanner({ message }: { message: string | null }) {
  return message ? <div className="error-banner" role="alert">{message}</div> : null;
}

export function DashboardPanel(props: PageProps) {
  const load = useCallback(
    (signal: AbortSignal) => fetchDashboard(props.userId, { apiBaseUrl: props.apiBaseUrl, signal }),
    [props.userId, props.apiBaseUrl],
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
                    <small>{item.company_name} · {[item.city, item.salary].filter(Boolean).join(" · ") || item.source_name}</small>
                  </div>
                  <span className={`status-pill ${item.application_status ? `status-${item.application_status}` : item.availability_status === "closed" ? "status-rejected" : "status-current"}`}>
                    {item.application_status ? STATUS_LABELS[item.application_status] ?? item.application_status : item.availability_status === "closed" ? "已下架" : "未投递"}
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

export function ApplicationsPanel(props: PageProps) {
  const load = useCallback((signal: AbortSignal) => fetchApplications(props.userId, { apiBaseUrl: props.apiBaseUrl, signal }), [props.userId, props.apiBaseUrl]);
  const state = usePageData<ApplicationView[]>(load, props.refreshToken, !props.hidden);
  return (
    <section className="management-page" hidden={props.hidden}>
      <PageHeader icon="applications" eyebrow="APPLICATIONS" title="投递记录" description="按公司和岗位查看从已投递到 Offer 的完整进度" loading={state.loading} onRefresh={state.reload} />
      <ErrorBanner message={state.error} />
      {state.data && state.data.length > 0 ? <div className="data-table-card"><div className="data-table-head"><span>岗位</span><span>状态</span><span>地点 / 薪资</span><span>投递时间</span></div>{state.data.map((item) => <article className="data-table-row" key={item.id}><div><span className="list-icon tone-blue"><AppIcon name="applications" size={18} /></span><span><strong>{item.title}</strong><small>{item.company_name}</small></span></div><span><span className={`status-pill status-${item.status}`}>{STATUS_LABELS[item.status] ?? item.status}</span></span><span>{[item.city, item.salary].filter(Boolean).join(" · ") || "未披露"}</span><time>{dateLabel(item.submitted_at)}</time></article>)}</div> : <EmptyState icon="applications" title="还没有投递记录" description="投递记录会关联具体岗位、简历版本和后续邮件事件。" action="通过对话记录" onAction={() => props.onAskAgent("帮我记录一次新的岗位投递")} />}
    </section>
  );
}

export function JobsPanel(props: PageProps) {
  const load = useCallback(
    (signal: AbortSignal) => fetchSavedJobs(props.userId, { apiBaseUrl: props.apiBaseUrl, signal }),
    [props.userId, props.apiBaseUrl],
  );
  const state = usePageData<SavedJobView[]>(load, props.refreshToken, !props.hidden);
  return (
    <section className="management-page" hidden={props.hidden}>
      <PageHeader icon="search" eyebrow="JOB LIBRARY" title="岗位库" description="查看保存过的完整 JD、分析结果和后续投递状态" loading={state.loading} onRefresh={state.reload} />
      <ErrorBanner message={state.error} />
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
            </article>
          ))}
        </div>
      ) : <EmptyState icon="search" title="岗位库还是空的" description="保存完整 JD 后，可以继续分析、匹配简历或记录投递。" action="去找岗位" onAction={() => props.onAskAgent("帮我打开招聘网站，我想查找并保存感兴趣的岗位")} />}
    </section>
  );
}

export function ResumesPanel(props: PageProps) {
  const load = useCallback((signal: AbortSignal) => fetchResumes(props.userId, { apiBaseUrl: props.apiBaseUrl, signal }), [props.userId, props.apiBaseUrl]);
  const state = usePageData<ResumeView[]>(load, props.refreshToken, !props.hidden);
  return (
    <section className="management-page" hidden={props.hidden}>
      <PageHeader icon="document" eyebrow="RESUME LIBRARY" title="简历管理" description="按目标岗位管理简历家族、版本和来源" loading={state.loading} onRefresh={state.reload} />
      <ErrorBanner message={state.error} />
      {state.data && state.data.length > 0 ? <div className="card-grid">{state.data.map((item) => <article className="resume-card" key={item.id}><span className="document-preview"><AppIcon name="document" size={28} /><small>{item.document_format.toUpperCase()}</small></span><div className="card-body"><span className="card-kicker">{item.target_role}</span><h2>{item.name}</h2><p>最新 v{item.latest_version_number} · 共 {item.version_count} 个版本</p><div className="card-meta"><span>{formatBytes(item.byte_size)}</span><time>{dateLabel(item.updated_at)} 更新</time></div></div></article>)}</div> : <EmptyState icon="document" title="还没有简历" description="上传原始简历后，Agent 可以分析、润色并生成新的不可变版本。" action="导入一份简历" onAction={() => props.onAskAgent("我想导入并分析一份简历")} />}
    </section>
  );
}

export function CalendarPanel(props: PageProps) {
  const load = useCallback((signal: AbortSignal) => fetchCalendar(props.userId, { apiBaseUrl: props.apiBaseUrl, signal }), [props.userId, props.apiBaseUrl]);
  const state = usePageData<CalendarWorkspace>(load, props.refreshToken, !props.hidden);
  return (
    <section className="management-page" hidden={props.hidden}>
      <PageHeader icon="calendar" eyebrow="CALENDAR" title="面试日历" description="查看已连接账号以及经过确认后同步的面试事件" loading={state.loading} onRefresh={state.reload} />
      <ErrorBanner message={state.error} />
      <div className="dashboard-grid">
        <section className="surface-card"><div className="section-heading"><div><small>CONNECTED ACCOUNTS</small><h2>日历账号</h2></div><span>{state.data?.accounts.length ?? 0} 个</span></div>{state.data && state.data.accounts.length > 0 ? <div className="compact-list">{state.data.accounts.map((item) => <article key={item.id}><span className="list-icon tone-blue"><AppIcon name="calendar" size={18} /></span><div><strong>{item.email_address}</strong><small>Google Calendar · {item.calendar_id}</small></div><span className="status-pill status-active">已连接</span></article>)}</div> : <EmptyState icon="calendar" title="尚未连接日历" description="连接后，面试时间变更仍会先让你确认再写入。" action="连接 Calendar" onAction={() => props.onAskAgent("帮我连接 Google Calendar")} />}</section>
        <section className="surface-card"><div className="section-heading"><div><small>SYNCED EVENTS</small><h2>已同步事件</h2></div><span>{state.data?.events.length ?? 0} 项</span></div>{state.data && state.data.events.length > 0 ? <div className="compact-list">{state.data.events.map((item) => <article key={item.id}><span className="list-icon tone-violet"><AppIcon name="brief" size={18} /></span><div><strong>面试日程</strong><small>{dateLabel(item.updated_at)} 更新</small></div><span className={`status-pill status-${item.status}`}>{STATUS_LABELS[item.status] ?? item.status}</span></article>)}</div> : <EmptyState icon="brief" title="还没有同步事件" description="识别到面试邀请后，可以在对话中确认写入日历。" action="查看面试安排" onAction={() => props.onAskAgent("检查我最近的面试安排，并告诉我哪些需要同步到日历")} />}</section>
      </div>
    </section>
  );
}

export function ResearchPanel(props: PageProps) {
  const load = useCallback((signal: AbortSignal) => fetchCompanyResearch(props.userId, { apiBaseUrl: props.apiBaseUrl, signal }), [props.userId, props.apiBaseUrl]);
  const state = usePageData<CompanyResearchView[]>(load, props.refreshToken, !props.hidden);
  return (
    <section className="management-page" hidden={props.hidden}>
      <PageHeader icon="building" eyebrow="COMPANY RESEARCH" title="公司研究" description="集中查看用户主动要求生成的业务、产品线和市场资料" loading={state.loading} onRefresh={state.reload} />
      <ErrorBanner message={state.error} />
      {state.data && state.data.length > 0 ? <div className="research-grid">{state.data.map((item) => <article className="research-card" key={item.id}><div className="research-card-top"><span className="company-avatar">{item.company_name.slice(0, 1)}</span><div><span className="card-kicker">{item.focus || "公司与业务概览"}</span><h2>{item.company_name}</h2></div><span className={`status-pill status-${item.status}`}>{STATUS_LABELS[item.status] ?? item.status}</span></div><p>{item.summary}</p><div className="card-meta"><span>{item.finding_count} 条发现</span><span>{item.source_count} 个来源</span><time>{dateLabel(item.created_at)}</time></div><small>检索锚点：{item.anchor_job_title}</small></article>)}</div> : <EmptyState icon="building" title="还没有公司研究" description="公司研究是可选能力，只在你主动要求时检索公开业务资料。" action="研究一家公司" onAction={() => props.onAskAgent("我想研究一个目标公司的业务和产品线")} />}
    </section>
  );
}
