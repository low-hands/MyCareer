import { useState, type SetStateAction } from "react";

import type { ResumeAttachment } from "./attachments";

interface ComposerDraft {
  draft: string;
  attachments: ResumeAttachment[];
}

function emptyDraft(): ComposerDraft {
  return { draft: "", attachments: [] };
}

export function useConversationComposer(conversationId: string) {
  const [drafts, setDrafts] = useState(new Map<string, ComposerDraft>());
  const current = drafts.get(conversationId) ?? emptyDraft();

  function update(change: (draft: ComposerDraft) => ComposerDraft): void {
    setDrafts((previous) => {
      const next = new Map(previous);
      next.set(conversationId, change(previous.get(conversationId) ?? emptyDraft()));
      return next;
    });
  }

  function setDraft(draft: string): void {
    update((current) => ({ ...current, draft }));
  }

  function setAttachments(value: SetStateAction<ResumeAttachment[]>): void {
    update((current) => ({
      ...current,
      attachments: typeof value === "function" ? value(current.attachments) : value,
    }));
  }

  function discardDraft(id: string): void {
    setDrafts((previous) => {
      const next = new Map(previous);
      next.delete(id);
      return next;
    });
  }

  return { ...current, setDraft, setAttachments, discardDraft };
}
