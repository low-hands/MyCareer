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
