import { describe, expect, it } from "vitest";

import type { CalendarEventView } from "../api/client";
import { calendarDays, eventDayKey, monthKey } from "./month";

const event = (start: string): CalendarEventView => ({
  id: "event-1",
  interview_round_id: "round-1",
  application_id: null,
  company_name: "Acme",
  job_title: "Engineer",
  employer_label: "Technical interview",
  sequence_number: 1,
  interview_status: "scheduled",
  scheduled_start: start,
  scheduled_end: null,
  timezone: "UTC",
  interview_format: "video",
  location: null,
  meeting_url: null,
  contact_summary: null,
  sync_status: "not_synced",
  status: "scheduled",
  external_html_link: null,
  updated_at: start,
});

describe("calendar month helpers", () => {
  it("builds a Monday-first six-week grid", () => {
    const days = calendarDays(new Date(2026, 8, 1));
    expect(days).toHaveLength(42);
    expect(days[0].date.getDay()).toBe(1);
    expect(monthKey(new Date(2026, 8, 1))).toBe("2026-09");
  });

  it("groups cross-day UTC events in the requested timezone", () => {
    expect(eventDayKey(event("2026-09-05T17:30:00Z"), "Asia/Shanghai")).toBe(
      "2026-09-06",
    );
  });
});
