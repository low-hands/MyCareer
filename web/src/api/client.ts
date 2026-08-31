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

export interface SavedJobView {
  id: string;
  title: string;
  company_name: string;
  city: string | null;
  salary: string | null;
  source_name: string;
  source_url: string | null;
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

export interface ConversationMessageView {
  role: "user" | "assistant";
  content: string;
  created_at: string;
}

export interface ConversationTranscript {
  messages: ConversationMessageView[];
  active_workflow: string | null;
  phase: string | null;
}

export interface ResumeView {
  id: string;
  name: string;
  target_role: string;
  status: string;
  latest_version_number: number;
  version_count: number;
  document_format: string;
  byte_size: number;
  updated_at: string;
}

export interface CalendarAccountView {
  id: string;
  provider: string;
  email_address: string;
  calendar_id: string;
  status: string;
  updated_at: string;
}

export interface CalendarEventView {
  id: string;
  interview_round_id: string;
  status: string;
  external_html_link: string | null;
  updated_at: string;
}

export interface CalendarWorkspace {
  accounts: CalendarAccountView[];
  events: CalendarEventView[];
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

export class ApiError extends Error {}

export async function getJson<T>(
  path: string,
  params: Record<string, string>,
  options: { apiBaseUrl: string; signal?: AbortSignal },
): Promise<T> {
  const query = new URLSearchParams(params).toString();
  const response = await fetch(`${options.apiBaseUrl}${path}?${query}`, {
    headers: { Accept: "application/json" },
    signal: options.signal,
  });
  if (!response.ok) {
    throw new ApiError(`${path} 返回 ${response.status}`);
  }
  return (await response.json()) as T;
}

export function fetchDailyBrief(
  userId: string,
  options: { apiBaseUrl: string; signal?: AbortSignal },
): Promise<DailyBrief> {
  return getJson<DailyBrief>("/v1/daily-brief", { user_id: userId }, options);
}

type ReadOptions = { apiBaseUrl: string; signal?: AbortSignal };

export function fetchDashboard(userId: string, options: ReadOptions): Promise<Dashboard> {
  return getJson<Dashboard>("/v1/dashboard", { user_id: userId }, options);
}

export function fetchApplications(userId: string, options: ReadOptions): Promise<ApplicationView[]> {
  return getJson<ApplicationView[]>("/v1/applications", { user_id: userId }, options);
}

export function fetchSavedJobs(userId: string, options: ReadOptions): Promise<SavedJobView[]> {
  return getJson<SavedJobView[]>("/v1/jobs", { user_id: userId }, options);
}

export function fetchResumes(userId: string, options: ReadOptions): Promise<ResumeView[]> {
  return getJson<ResumeView[]>("/v1/resumes", { user_id: userId }, options);
}

export function fetchCalendar(userId: string, options: ReadOptions): Promise<CalendarWorkspace> {
  return getJson<CalendarWorkspace>("/v1/calendar", { user_id: userId }, options);
}

export function fetchCompanyResearch(userId: string, options: ReadOptions): Promise<CompanyResearchView[]> {
  return getJson<CompanyResearchView[]>("/v1/company-research", { user_id: userId }, options);
}

export function fetchConversations(userId: string, options: ReadOptions): Promise<ConversationView[]> {
  return getJson<ConversationView[]>("/v1/conversations", { user_id: userId }, options);
}

export function fetchConversationMessages(
  userId: string,
  conversationId: string,
  options: ReadOptions,
): Promise<ConversationTranscript> {
  return getJson<ConversationTranscript>(
    `/v1/conversations/${encodeURIComponent(conversationId)}/messages`,
    { user_id: userId },
    options,
  );
}
