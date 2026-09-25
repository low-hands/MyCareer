import type { ChatAttachment } from "./attachments";

/**
 * A task started from a workspace page that must run in its own, new
 * conversation: analysing a resume version picked in the library, or the JD
 * of a saved job with no resume alongside it. It is
 * created once, switches the visible chat only when that chat is idle, and
 * sends only after the new conversation's (empty) transcript has been
 * hydrated, so the request cannot land in the old conversation or be wiped
 * by hydration.
 */
export interface StandaloneAgentTask {
  conversationId: string;
  prompt: string;
  /** Null for a task that carries only its prompt, such as repeating a practice run with no resume or job. */
  resource: ChatAttachment | null;
  additionalResources?: ChatAttachment[];
  /** What the waiting notice says; defaults to analysing the attached resources. */
  label?: string;
}

export function standaloneTaskResources(task: StandaloneAgentTask): ChatAttachment[] {
  return [task.resource, ...(task.additionalResources ?? [])].filter(
    (item): item is ChatAttachment => item !== null,
  );
}

export interface AgentTaskChatState {
  conversationId: string;
  /** The conversation whose transcript the chat currently shows. */
  hydratedConversationId: string | null;
  busy: boolean;
  historyLoading: boolean;
}

export type AgentTaskStep = "wait" | "switch" | "send";

export function newStandaloneAgentTask(
  prompt: string,
  resource: ChatAttachment | null,
  additionalResources: ChatAttachment[] = [],
  label?: string,
): StandaloneAgentTask {
  return { conversationId: `conversation-${crypto.randomUUID()}`, prompt, resource, additionalResources, label };
}

/**
 * What the chat should do next for a queued task. It never interrupts a
 * running turn or a transcript load; a task waits through both.
 */
export function standaloneAgentTaskStep(
  task: StandaloneAgentTask,
  chat: AgentTaskChatState,
): AgentTaskStep {
  if (chat.busy || chat.historyLoading) return "wait";
  if (chat.conversationId !== task.conversationId) return "switch";
  if (chat.hydratedConversationId !== task.conversationId) return "wait";
  return "send";
}
