import type { CalendarEventView } from "../api/client";

export interface CalendarDay {
  date: Date;
  key: string;
  day: number;
  inMonth: boolean;
}

export function monthKey(date: Date): string {
  return `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, "0")}`;
}

export function calendarDays(cursor: Date): CalendarDay[] {
  const first = new Date(cursor.getFullYear(), cursor.getMonth(), 1);
  const mondayOffset = (first.getDay() + 6) % 7;
  const start = new Date(first);
  start.setDate(first.getDate() - mondayOffset);
  return Array.from({ length: 42 }, (_, index) => {
    const date = new Date(start);
    date.setDate(start.getDate() + index);
    return {
      date,
      key: `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, "0")}-${String(date.getDate()).padStart(2, "0")}`,
      day: date.getDate(),
      inMonth: date.getMonth() === cursor.getMonth(),
    };
  });
}

export function eventDayKey(event: CalendarEventView, timezone: string): string | null {
  if (!event.scheduled_start) return null;
  const parts = new Intl.DateTimeFormat("en-CA", {
    timeZone: timezone,
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  }).formatToParts(new Date(event.scheduled_start));
  const value = Object.fromEntries(parts.map((part) => [part.type, part.value]));
  return `${value.year}-${value.month}-${value.day}`;
}

export function eventsByDay(
  events: CalendarEventView[],
  timezone: string,
): Map<string, CalendarEventView[]> {
  const grouped = new Map<string, CalendarEventView[]>();
  for (const event of events) {
    const key = eventDayKey(event, timezone);
    if (!key) continue;
    grouped.set(key, [...(grouped.get(key) ?? []), event]);
  }
  return grouped;
}
