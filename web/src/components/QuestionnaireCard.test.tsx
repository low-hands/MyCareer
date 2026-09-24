// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, expect, it, vi } from "vitest";

import type { InteractionRequiredEvent } from "../chat/types";
import { QuestionnaireCard } from "./QuestionnaireCard";

const interaction: InteractionRequiredEvent = {
  type: "interaction_required",
  interaction_id: "interaction_1234567890abcdef1234",
  kind: "questionnaire",
  scope: "questionnaire",
  prompt: "补充简历信息",
  options: [],
  allow_free_text: false,
  questions: [1, 2, 3, 4, 5].map((number) => ({
    question_id: `q${number}`,
    prompt: `问题 ${number}`,
    kind: "single",
    options: [{ value: "yes", label: `回答 ${number}`, meaning: "choice" }],
    allow_free_text: false,
    allow_skip: true,
  })),
};

let root: Root;
let container: HTMLDivElement;
beforeEach(() => {
  sessionStorage.clear();
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
});
afterEach(async () => {
  await act(async () => root.unmount());
  container.remove();
});

function button(label: string): HTMLButtonElement {
  const found = [...container.querySelectorAll("button")].find((item) => item.textContent === label);
  if (!found) throw new Error(`button missing: ${label}`);
  return found;
}

it("shows five questions in sequence and sends one bound response", async () => {
  const onReply = vi.fn();
  await act(async () => root.render(<QuestionnaireCard interaction={interaction} disabled={false} onReply={onReply} />));
  for (let number = 1; number <= 5; number += 1) {
    expect(container.querySelector("h3")?.textContent).toBe(`问题 ${number}`);
    expect(container.textContent).toContain(`补充信息 · ${number}/5`);
    await act(async () => button(`回答 ${number}`).click());
  }
  expect(container.textContent).not.toContain("待回答");
  expect(container.textContent).not.toContain("已回答");
  expect(onReply).not.toHaveBeenCalled();
  await act(async () => button("提交").click());
  expect(onReply).toHaveBeenCalledTimes(1);
  expect(onReply.mock.calls[0][0].interactionResponse).toEqual({
    interaction_id: interaction.interaction_id,
    scope: "questionnaire",
    action: "submit",
    answers: [1, 2, 3, 4, 5].map((number) => ({
      question_id: `q${number}`, selected_values: ["yes"], free_text: null, skipped: false,
    })),
  });
});

it("keeps local answers across refresh and disables submission while busy", async () => {
  const onReply = vi.fn();
  await act(async () => root.render(<QuestionnaireCard interaction={interaction} disabled={false} onReply={onReply} />));
  await act(async () => button("回答 1").click());
  await act(async () => root.unmount());
  root = createRoot(container);
  await act(async () => root.render(<QuestionnaireCard interaction={interaction} disabled={true} onReply={onReply} />));
  expect(button("回答 1").getAttribute("aria-checked")).toBe("true");
  expect(button("回答 1").disabled).toBe(true);
  expect(onReply).not.toHaveBeenCalled();
});

it("renders an obvious editable field for open-text questions", async () => {
  const onReply = vi.fn();
  const openTextInteraction: InteractionRequiredEvent = {
    ...interaction,
    interaction_id: "interaction_open_text_123456",
    questions: interaction.questions!.map((question) => ({
      ...question,
      kind: "free_text" as const,
      options: [],
      allow_free_text: true,
      allow_skip: false,
    })),
  };
  await act(async () => root.render(
    <QuestionnaireCard interaction={openTextInteraction} disabled={false} onReply={onReply} />,
  ));
  const input = container.querySelector("textarea");
  expect(input).not.toBeNull();
  expect(input?.disabled).toBe(false);
  expect(input?.placeholder).toBe("请在这里填写回答…");
  await act(async () => {
    if (!input) return;
    const setter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value")?.set;
    setter?.call(input, "2026 年 10 月 8 日 14:00，Asia/Shanghai");
    input.dispatchEvent(new Event("input", { bubbles: true }));
  });
  expect(button("下一步").disabled).toBe(false);
});
