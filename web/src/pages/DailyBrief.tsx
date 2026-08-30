import { useCallback, useEffect, useState } from "react";

import { ActionItemView, DailyBrief, fetchDailyBrief } from "../api/client";

const BUCKETS: { key: keyof DailyBrief; label: string; tone: string }[] = [
  { key: "overdue", label: "已逾期", tone: "overdue" },
  { key: "due_today", label: "今天", tone: "today" },
  { key: "upcoming", label: "接下来", tone: "upcoming" },
  { key: "no_due_date", label: "没有期限", tone: "someday" },
];

function dueLabel(item: ActionItemView, timezone: string): string {
  if (!item.due_at) return "无期限";
  return new Date(item.due_at).toLocaleString("zh-CN", {
    timeZone: timezone,
    month: "numeric",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function DailyBriefPanel({
  userId,
  apiBaseUrl,
  refreshToken,
  hidden,
}: {
  userId: string;
  apiBaseUrl: string;
  /** Changes when a chat turn finishes, so acting in chat updates the board. */
  refreshToken: number;
  hidden: boolean;
}) {
  const [brief, setBrief] = useState<DailyBrief | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const load = useCallback(
    (signal?: AbortSignal) => {
      setLoading(true);
      return fetchDailyBrief(userId, { apiBaseUrl, signal })
        .then((next) => {
          setBrief(next);
          setError(null);
        })
        .catch((cause: unknown) => {
          if (signal?.aborted) return;
          setError(cause instanceof Error ? cause.message : "读取日报失败。");
        })
        .finally(() => {
          if (!signal?.aborted) setLoading(false);
        });
    },
    [userId, apiBaseUrl],
  );

  useEffect(() => {
    const controller = new AbortController();
    void load(controller.signal);
    return () => controller.abort();
  }, [load, refreshToken]);

  const total = brief
    ? BUCKETS.reduce((sum, bucket) => sum + (brief[bucket.key] as ActionItemView[]).length, 0)
    : 0;

  return (
    <section className="brief-panel" hidden={hidden} aria-label="今日待办">
      <header className="brief-header">
        <div>
          <p className="eyebrow">DAILY BRIEF</p>
          <h1>今天该推进什么</h1>
        </div>
        <button type="button" onClick={() => void load()} disabled={loading}>
          {loading ? "刷新中…" : "刷新"}
        </button>
      </header>

      {error ? <div className="error-banner" role="alert">{error}</div> : null}

      {brief && total === 0 && !error ? (
        <div className="brief-empty">
          <strong>没有待办。</strong>
          <p>
            投递、面试和邮件事件会自动生成这里的条目。还没有内容，通常是因为
            还没有记录过任何投递。
          </p>
        </div>
      ) : null}

      {brief
        ? BUCKETS.map((bucket) => {
            const items = brief[bucket.key] as ActionItemView[];
            if (items.length === 0) return null;
            return (
              <div className={`brief-bucket brief-${bucket.tone}`} key={bucket.key}>
                <h2>
                  {bucket.label}
                  <span>{items.length}</span>
                </h2>
                <ul>
                  {items.map((item) => (
                    <li key={item.id}>
                      <div className="brief-item-head">
                        <strong>{item.title}</strong>
                        <time>{dueLabel(item, brief.timezone)}</time>
                      </div>
                      <p>{item.summary}</p>
                    </li>
                  ))}
                </ul>
              </div>
            );
          })
        : null}

      <p className="brief-footnote">
        这些条目由投递、面试和邮件事件推导而来，改动仍然通过对话完成。
      </p>
    </section>
  );
}
