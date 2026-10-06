# Mistaken-record recovery audit — 2026-10-06

| Operation | Visible after the turn | Correction path | Conversation receipt |
| --- | --- | --- | --- |
| Create application | Application list shows company, role, status and submission time | Correct its status directly, or delete that one local record from “投递记录”; an attached interview, practice session or email event blocks deletion until the attachment is handled | Names the company, role and status and points to the list |
| Create interview | “面试中心” now lists **all** rounds, including those without a scheduled time; scheduled rounds also appear on the calendar | Delete that local round; a calendar proposal/link or mock interview session blocks deletion; local preparation, retrospective and derived reminders are removed with it | Names the round and time and points to the center |
| Dismiss action item | “今日待办 → 已忽略” now lists dismissed items | Restore that item to `open`; the event history records `reopened` | Names the dismissed item and its restore location |

Deletion is scoped to the authenticated user. It removes only this project's
records; it does not retract a real job application or delete an external
calendar event. The HTTP surface returns 404 for another user's record and 409
when linked data prevents a safe local delete. These corrections are direct UI
actions and do not change the Agent's approval policy.
Bulk clearing also refuses to proceed while this user has calendar proposals
or links, so it cannot bypass the individual-delete guard.

The stores are separate SQLite files. For an individual interview round, the
round and its same-file history are deleted in one transaction; only after a
successful delete are the derived reminders and preparation removed. A
cross-file cleanup failure can leave derived rows for later reconciliation.
Linked-record checks cannot be atomic across files, so a concurrent link
created between the check and delete remains a low-probability race in this
local single-user workflow. When deleting the last
round, the application status returns from `interviewing` to its prior
`submitted` or `acknowledged` status only if the latest status event is the
automatic update made when the round was created. That update now carries the
structured reason `interview_created`; the v2→v3 migration backfills existing
events. Later user or email status updates remain untouched. A direct status
correction resets the follow-up clock; at the next daily-brief refresh, stale
follow-up reminders become obsolete and new ones use the corrected status.

Remaining limitation: a record with an external calendar operation or mock
practice history is retained. A linked application also retains its associated
email facts. The user can see why deletion was blocked; those linked records
must be resolved separately. No automatic deletion of external or user-authored
history is implied by “undo”.
