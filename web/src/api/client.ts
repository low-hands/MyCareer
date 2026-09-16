import type {
  PublicStreamEvent,
  ReportDeliveryStatus,
  ReportResourceKind,
} from "../chat/events";

export interface ActionItemView {
  id: string;
  action_type: string;
  source_type: string;
  application_id: string | null;
  title: string;
  summary: string;
  due_at: string | null;
  status: string;
  snoozed_until: string | null;
}

export interface DailyBrief {
  timezone: string;
  generated_at: string;
  overdue: ActionItemView[];
  due_today: ActionItemView[];
  upcoming: ActionItemView[];
  no_due_date: ActionItemView[];
}

export interface ApplicationView {
  id: string;
  status: string;
  title: string;
  company_name: string;
  city: string | null;
  salary: string | null;
  submitted_at: string;
  updated_at: string;
}

export interface MockInterviewSessionView {
  session_id: string;
  status: string;
  interview_type: string;
  interview_type_label: string;
  question_count: number;
  max_primary_questions: number;
  report_id: string | null;
  summary: string | null;
  conversation_id: string | null;
  created_at: string;
  completed_at: string | null;
  updated_at: string;
}

export interface ApplicationMockInterviews {
  application_id: string;
  title: string;
  company_name: string;
  sessions: MockInterviewSessionView[];
}

export type JobSeniority = "fresh_graduate" | "junior" | "mid" | "senior" | "lead";
export type RequirementTier = "S" | "A" | "B" | "C";
export type RequirementKind = "fact" | "inference";

export interface TieredRequirement {
  text: string;
  tier: RequirementTier;
  kind: RequirementKind;
  jd_quote: string;
}

export interface QuotedFinding {
  text: string;
  jd_quote: string;
}

/** JD-only analysis: grounded in the posting text alone, never in a resume. */
export interface JobAnalysisResult {
  core_objective: string;
  seniority: JobSeniority;
  requirements: TieredRequirement[];
  core_competencies: string[];
  implicit_requirements: QuotedFinding[];
  ats_keywords: string[];
  hr_focus: string[];
  hiring_manager_focus: string[];
  likely_interview_topics: string[];
  red_flags: QuotedFinding[];
  information_gaps: string[];
  summary: string;
}

/** `stale`: the JD has a newer snapshot than the one this result was computed on. */
export type JobDerivedStatus = "none" | "ready" | "stale";

export interface SavedJobView {
  id: string;
  title: string;
  company_name: string;
  city: string | null;
  salary: string | null;
  source_name: string;
  source_url: string | null;
  pursuit_status: "open" | "dismissed";
  availability_status: string;
  captured_at: string;
  last_checked_at: string;
  application_status: string | null;
  analysis_summary: string | null;
  responsibilities: string[];
  required_skills: string[];
  preferred_qualifications: string[];
  clarification_questions: string[];
  analyzed_at: string | null;
  jd_snapshot_id: string | null;
  jd_version: number | null;
  jd_analysis_status: JobDerivedStatus;
  jd_analysis_version: number | null;
  jd_analysis: JobAnalysisResult | null;
  resume_match_status: JobDerivedStatus;
  resume_match_fit: string | null;
  resume_match_at: string | null;
}

export interface ConversationView {
  id: string;
  status: string;
  title: string;
  last_message_preview: string;
  message_count: number;
  active_workflow: string | null;
  phase: string | null;
  created_at: string;
  last_active_at: string;
}

export type MessageResourceKind = ReportResourceKind | "resume_version" | "saved_job";

export interface ConversationResourceView {
  kind: MessageResourceKind;
  resource_id: string;
  status_at_delivery: ReportDeliveryStatus | null;
  anchored_by_other_job: boolean | null;
  title: string | null;
  description?: string | null;
  /** Whether the referenced asset can still be opened; null when not tracked. */
  available?: boolean | null;
  /** The owning resume of a `resume_version` that is still accessible. */
  resume_id?: string | null;
}

export interface ConversationMessageView {
  role: "user" | "assistant";
  content: string;
  created_at: string;
  /** Plural: one turn can store two reports, and each gets its own card. */
  resources: ConversationResourceView[];
}

export interface ReportView {
  availability?: "available" | "expired";
  kind: string;
  resource_id: string;
  title: string;
  subtitle: string;
  body: string;
  created_at: string;
}

export interface ConversationTranscript {
  messages: ConversationMessageView[];
  active_workflow: string | null;
  phase: string | null;
  pending_interaction: Extract<
    PublicStreamEvent,
    { type: "interaction_required" }
  > | null;
  pending_interaction_body: string | null;
}

export interface ResumeVersionView {
  id: string;
  resume_id: string;
  version_number: number;
  document_format: string;
  byte_size: number;
  created_at: string;
}

export interface ResumeView {
  id: string;
  name: string;
  target_role: string;
  status: string;
  latest_version_number: number;
  latest_version_id: string;
  version_count: number;
  document_format: string;
  byte_size: number;
  updated_at: string;
  /** Newest first. */
  versions: ResumeVersionView[];
}

export interface TargetRoleView {
  id: string;
  title: string;
  priority: number;
  status: string;
}

export interface ResumeImportResult {
  resume_id: string;
  resume_version_id: string;
  name: string;
  version_number: number;
  document_format: string;
  byte_size: number;
}

export interface EmailAccountView {
  id: string;
  provider: string;
  email_address: string;
  status: string;
  connection_status: string;
  needs_reauthorization: boolean;
  last_synced_at: string | null;
}

export interface EmailEventView {
  id: string;
  event_type: string;
  status: string;
  summary: string;
  confidence: number;
  application_id: string | null;
  application_title: string | null;
  company_name: string | null;
  occurred_at: string;
}

export interface EmailWorkspace {
  accounts: EmailAccountView[];
  events: EmailEventView[];
}

export interface CalendarAccountView {
  id: string;
  provider: string;
  email_address: string;
  calendar_id: string;
  status: string;
  connection_status: string;
  needs_reauthorization: boolean;
  updated_at: string;
}

export interface CalendarEventView {
  id: string;
  interview_round_id: string;
  application_id: string | null;
  company_name: string | null;
  job_title: string | null;
  employer_label: string | null;
  sequence_number: number | null;
  interview_status: string | null;
  scheduled_start: string | null;
  scheduled_end: string | null;
  timezone: string | null;
  interview_format: string | null;
  location: string | null;
  meeting_url: string | null;
  contact_summary: string | null;
  sync_status: string;
  status: string;
  external_html_link: string | null;
  updated_at: string;
}

export interface CalendarWorkspace {
  accounts: CalendarAccountView[];
  events: CalendarEventView[];
  month: string | null;
  timezone: string;
}

export interface CompanyResearchView {
  id: string;
  company_name: string;
  anchor_job_title: string;
  status: string;
  focus: string | null;
  summary: string;
  finding_count: number;
  source_count: number;
  created_at: string;
}

export interface Dashboard {
  stats: {
    saved_jobs: number;
    applications: number;
    interviewing: number;
    offers: number;
    resumes: number;
    research_reports: number;
    calendar_accounts: number;
  };
  application_statuses: Record<string, number>;
  recent_jobs: SavedJobView[];
  recent_applications: ApplicationView[];
  next_actions: ActionItemView[];
}

export class ApiError extends Error {
  readonly status: number | null;

  constructor(message: string, status: number | null = null) {
    super(message);
    this.status = status;
  }
}

export async function getJson<T>(
  path: string,
  params: Record<string, string>,
  options: { apiBaseUrl: string; signal?: AbortSignal },
): Promise<T> {
  const query = new URLSearchParams(params).toString();
  const suffix = query ? `?${query}` : "";
  const response = await fetch(`${options.apiBaseUrl}${path}${suffix}`, {
    headers: { Accept: "application/json" },
    signal: options.signal,
  });
  if (!response.ok) {
    throw new ApiError(`${path} 返回 ${response.status}`, response.status);
  }
  return (await response.json()) as T;
}

export function fetchDailyBrief(
  options: { apiBaseUrl: string; signal?: AbortSignal },
): Promise<DailyBrief> {
  return getJson<DailyBrief>("/v1/daily-brief", {}, options);
}

type ReadOptions = { apiBaseUrl: string; signal?: AbortSignal };

export function fetchDashboard(options: ReadOptions): Promise<Dashboard> {
  return getJson<Dashboard>("/v1/dashboard", {}, options);
}

export function fetchApplications(options: ReadOptions): Promise<ApplicationView[]> {
  return getJson<ApplicationView[]>("/v1/applications", {}, options);
}

export function fetchApplicationMockInterviews(
  applicationId: string,
  options: ReadOptions,
): Promise<ApplicationMockInterviews> {
  return getJson<ApplicationMockInterviews>(
    `/v1/applications/${encodeURIComponent(applicationId)}/mock-interviews`,
    {},
    options,
  );
}

export async function createApplication(
  values: {
    jobPostingId: string;
    resumeVersionId: string;
    submittedAt?: string;
    note?: string;
  },
  options: ReadOptions,
): Promise<ApplicationView> {
  const response = await fetch(`${options.apiBaseUrl}/v1/applications`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "application/json" },
    body: JSON.stringify({
      job_posting_id: values.jobPostingId,
      resume_version_id: values.resumeVersionId,
      submitted_at: values.submittedAt || null,
      note: values.note?.trim() || null,
    }),
    signal: options.signal,
  });
  if (!response.ok) throw new ApiError(`记录投递失败：${response.status}`);
  return (await response.json()) as ApplicationView;
}

export function fetchSavedJobs(
  options: ReadOptions & { includeIgnored?: boolean },
): Promise<SavedJobView[]> {
  return getJson<SavedJobView[]>(
    "/v1/jobs",
    { include_dismissed: String(Boolean(options.includeIgnored)) },
    options,
  );
}

export interface SavedJobDetail {
  id: string;
  title: string;
  company_name: string;
  city: string | null;
  salary: string | null;
  source_name: string;
  source_url: string | null;
  availability_status: string;
  pursuit_status: "open" | "dismissed";
  jd_text: string;
  jd_version: number;
  captured_at: string;
}

/** The JD itself, fetched when a card is opened rather than with the list.
 *
 * Same snapshot the agent reads back in conversation, so the two ways of
 * reaching a posting cannot disagree about what it says.
 */
export function fetchSavedJobDetail(
  jobPostingId: string,
  options: ReadOptions,
): Promise<SavedJobDetail> {
  return getJson<SavedJobDetail>(
    `/v1/jobs/${encodeURIComponent(jobPostingId)}`,
    {},
    options,
  );
}

export interface SavedJobSnapshot {
  jd_snapshot_id: string;
  job_posting_id: string;
  title: string;
  company_name: string;
  source_name: string;
  source_url: string | null;
  jd_version: number;
  latest_jd_version: number;
  jd_text: string;
  captured_at: string;
}

/** One immutable JD version, as a conversation card opens it.
 *
 * Addressed by snapshot rather than posting so a card in an old turn keeps
 * opening the text that turn read after the posting was captured again.
 */
export function fetchSavedJobSnapshot(
  jdSnapshotId: string,
  options: ReadOptions,
): Promise<SavedJobSnapshot> {
  return getJson<SavedJobSnapshot>(
    `/v1/jd-snapshots/${encodeURIComponent(jdSnapshotId)}`,
    {},
    options,
  );
}

/** Record a closure by hand, for when the extension cannot.
 *
 * The same field the extension writes. It was never a separate source of
 * truth — it reports what the person looking at the page saw — so this is that
 * report without the shortcut, for the ordinary cases where the shortcut is
 * unavailable: the site reworded its notice, the page needs a login, the
 * extension is not installed here.
 */
export async function setJobAvailability(
  jobPostingId: string,
  availabilityStatus: "active" | "closed" | "unknown",
  options: { apiBaseUrl: string; signal?: AbortSignal },
): Promise<void> {
  const response = await fetch(
    `${options.apiBaseUrl}/v1/jobs/${encodeURIComponent(jobPostingId)}/availability`,
    {
      method: "PUT",
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json",
      },
      body: JSON.stringify({ availability_status: availabilityStatus }),
      signal: options.signal,
    },
  );
  if (!response.ok) {
    throw new ApiError(`更新岗位状态失败：${response.status}`);
  }
}

/** Rule a saved job out of the shortlist, or put it back.
 *
 * The only shortlist fact the server cannot derive. Everything else — saved,
 * applied to, closed by the employer — already follows from stored records;
 * "I looked at this and I am not going to apply" exists nowhere until the
 * person says so, and this is where they say it.
 */
export async function setJobPursuit(
  jobPostingId: string,
  pursuitStatus: "open" | "dismissed",
  options: { apiBaseUrl: string; signal?: AbortSignal },
): Promise<void> {
  const response = await fetch(
    `${options.apiBaseUrl}/v1/jobs/${encodeURIComponent(jobPostingId)}/pursuit`,
    {
      method: "PUT",
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json",
      },
      body: JSON.stringify({ pursuit_status: pursuitStatus }),
      signal: options.signal,
    },
  );
  if (!response.ok) {
    throw new ApiError(`更新岗位状态失败：${response.status}`);
  }
}

/** Permanently remove a saved job and its JD history.
 *
 * Unlike pursuit dismissal this has no restore path. The server refuses jobs
 * already referenced by an application so their historical input stays
 * readable.
 */
export async function deleteSavedJob(
  jobPostingId: string,
  options: { apiBaseUrl: string; signal?: AbortSignal },
): Promise<void> {
  const response = await fetch(
    `${options.apiBaseUrl}/v1/jobs/${encodeURIComponent(jobPostingId)}`,
    {
      method: "DELETE",
      headers: { Accept: "application/json" },
      signal: options.signal,
    },
  );
  if (!response.ok) {
    let message = `删除岗位失败：${response.status}`;
    try {
      const payload = (await response.json()) as { detail?: { message?: string } };
      if (payload.detail?.message) message = payload.detail.message;
    } catch {
      // Keep the status-based fallback when an intermediary returns no JSON.
    }
    throw new ApiError(message);
  }
}

export function fetchResumes(options: ReadOptions): Promise<ResumeView[]> {
  return getJson<ResumeView[]>("/v1/resumes", {}, options);
}

export function fetchTargetRoles(options: ReadOptions): Promise<TargetRoleView[]> {
  return getJson<TargetRoleView[]>("/v1/target-roles", {}, options);
}

export async function createTargetRole(
  title: string,
  options: ReadOptions,
): Promise<TargetRoleView> {
  const response = await fetch(`${options.apiBaseUrl}/v1/target-roles`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "application/json" },
    body: JSON.stringify({ title }),
    signal: options.signal,
  });
  if (!response.ok) throw new ApiError(`创建目标岗位失败：${response.status}`);
  return (await response.json()) as TargetRoleView;
}

export async function importResume(
  values: {
    file: File;
    name?: string;
    resumeId?: string;
    targetRoleId?: string;
    /**
     * Sent as `Idempotency-Key`. A retry of the same upload (same file, name,
     * role and destination) returns the version already created instead of a
     * second one; a different request under the same key is refused.
     */
    idempotencyKey?: string;
  },
  options: ReadOptions,
): Promise<ResumeImportResult> {
  const body = new FormData();
  body.set("file", values.file);
  if (values.name) body.set("name", values.name);
  if (values.resumeId) body.set("resume_id", values.resumeId);
  if (values.targetRoleId) body.set("target_role_id", values.targetRoleId);
  const headers: Record<string, string> = { Accept: "application/json" };
  if (values.idempotencyKey) headers["Idempotency-Key"] = values.idempotencyKey;
  const response = await fetch(`${options.apiBaseUrl}/v1/resumes/import`, {
    method: "POST",
    headers,
    body,
    signal: options.signal,
  });
  if (!response.ok) {
    let message = `导入简历失败：${response.status}`;
    try {
      const payload = (await response.json()) as { detail?: { message?: string } };
      if (payload.detail?.message) message = payload.detail.message;
    } catch {
      // Keep the status fallback when an intermediary returns no JSON.
    }
    throw new ApiError(message);
  }
  return (await response.json()) as ResumeImportResult;
}

/** The stored original of one resume version, served by the owner-checked route. */
export function resumeDocumentUrl(
  resumeId: string,
  versionId: string,
  options: { apiBaseUrl: string; download?: boolean },
): string {
  const base = options.apiBaseUrl.replace(/\/$/, "");
  const path = `${base}/v1/resumes/${encodeURIComponent(resumeId)}/versions/${encodeURIComponent(versionId)}/document`;
  return options.download ? `${path}?download=true` : path;
}

export function fetchEmailWorkspace(options: ReadOptions): Promise<EmailWorkspace> {
  return getJson<EmailWorkspace>("/v1/email", {}, options);
}

export function fetchCalendar(
  options: ReadOptions & { month?: string; timezone?: string },
): Promise<CalendarWorkspace> {
  const params: Record<string, string> = {};
  if (options.month) params.month = options.month;
  if (options.timezone) params.timezone = options.timezone;
  return getJson<CalendarWorkspace>("/v1/calendar", params, options);
}

export async function startGoogleConnection(
  kind: "gmail" | "calendar",
  options: ReadOptions,
): Promise<string> {
  const response = await fetch(`${options.apiBaseUrl}/v1/connections/google/${kind}/start`, {
    method: "POST",
    headers: { Accept: "application/json" },
    signal: options.signal,
  });
  if (!response.ok) {
    let message = `启动 Google 授权失败：${response.status}`;
    try {
      const payload = (await response.json()) as {
        detail?: string | { code?: string; message?: string };
      };
      if (
        typeof payload.detail === "object"
        && payload.detail?.code === "GOOGLE_OAUTH_NOT_CONFIGURED"
      ) {
        message = "尚未配置 Google OAuth。请按页面中的连接说明，在项目根目录 .env 添加 Client ID 和 Client Secret，然后重启后端。";
      }
      const detail = typeof payload.detail === "string"
        ? payload.detail
        : payload.detail?.message;
      if (detail && message.startsWith("启动 Google 授权失败")) message = detail;
    } catch {
      // Keep the status-based fallback when the response is not JSON.
    }
    throw new ApiError(message);
  }
  const payload = (await response.json()) as { authorization_url: string };
  return payload.authorization_url;
}

export async function connectQQ(
  emailAddress: string,
  authorizationCode: string,
  options: ReadOptions,
): Promise<void> {
  const response = await fetch(`${options.apiBaseUrl}/v1/connections/qq`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "application/json" },
    body: JSON.stringify({
      email_address: emailAddress,
      authorization_code: authorizationCode,
    }),
    signal: options.signal,
  });
  if (!response.ok) throw new ApiError(`连接 QQ 邮箱失败：${response.status}`);
}

export async function disconnectIntegration(
  kind: "email" | "calendar",
  accountId: string,
  options: ReadOptions,
): Promise<void> {
  const response = await fetch(
    `${options.apiBaseUrl}/v1/connections/${kind}/${encodeURIComponent(accountId)}`,
    { method: "DELETE", headers: { Accept: "application/json" }, signal: options.signal },
  );
  if (!response.ok) throw new ApiError(`断开账号失败：${response.status}`);
}

export function fetchCompanyResearch(options: ReadOptions): Promise<CompanyResearchView[]> {
  return getJson<CompanyResearchView[]>("/v1/company-research", {}, options);
}

export function fetchConversations(options: ReadOptions): Promise<ConversationView[]> {
  return getJson<ConversationView[]>("/v1/conversations", {}, options);
}

/** A job saved from the agent's own BOSS search, waiting to re-enter its conversation. */
export interface JobCapturedEventView {
  id: string;
  conversation_id: string;
  job_posting_id: string;
  jd_snapshot_id: string;
  title: string;
  company_name: string;
  created_at: string;
}

export function fetchPendingJobCaptures(options: ReadOptions): Promise<JobCapturedEventView[]> {
  return getJson<{ events: JobCapturedEventView[] }>("/v1/job-captures/events", {}, options).then(
    (payload) => payload.events,
  );
}

export async function acknowledgeJobCapture(
  eventId: string,
  options: ReadOptions,
): Promise<boolean> {
  const response = await fetch(
    `${options.apiBaseUrl}/v1/job-captures/events/${encodeURIComponent(eventId)}/ack`,
    { method: "POST", headers: { Accept: "application/json" }, signal: options.signal },
  );
  if (!response.ok) throw new ApiError(`确认岗位采集事件失败：${response.status}`, response.status);
  const payload = (await response.json()) as { acknowledged: boolean };
  return payload.acknowledged;
}

export function fetchConversationMessages(
  conversationId: string,
  options: ReadOptions,
): Promise<ConversationTranscript> {
  return getJson<ConversationTranscript>(
    `/v1/conversations/${encodeURIComponent(conversationId)}/messages`,
    {},
    options,
  );
}

export async function deleteConversation(
  conversationId: string,
  options: ReadOptions,
): Promise<void> {
  const response = await fetch(
    `${options.apiBaseUrl}/v1/conversations/${encodeURIComponent(conversationId)}`,
    {
      method: "DELETE",
      headers: { Accept: "application/json" },
      signal: options.signal,
    },
  );
  if (!response.ok) {
    let message = `删除会话失败：${response.status}`;
    try {
      const payload = (await response.json()) as { detail?: { message?: string } };
      if (payload.detail?.message) message = payload.detail.message;
    } catch {
      // Keep the status-based fallback when an intermediary returns no JSON.
    }
    throw new ApiError(message);
  }
}

export function fetchReport(
  kind: ReportResourceKind,
  resourceId: string,
  deliveryContext: {
    statusAtDelivery?: ReportDeliveryStatus | null;
    anchoredByOtherJob?: boolean | null;
  },
  options: ReadOptions,
): Promise<ReportView> {
  const params: Record<string, string> = {};
  if (deliveryContext.statusAtDelivery) {
    params.status_at_delivery = deliveryContext.statusAtDelivery;
  }
  if (deliveryContext.anchoredByOtherJob != null) {
    params.anchored_by_other_job = String(deliveryContext.anchoredByOtherJob);
  }
  return getJson<ReportView>(
    `/v1/reports/${encodeURIComponent(kind)}/${encodeURIComponent(resourceId)}`,
    params,
    options,
  );
}
