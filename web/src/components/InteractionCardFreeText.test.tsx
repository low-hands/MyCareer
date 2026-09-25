// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, expect, it, vi } from "vitest";

import type { InteractionRequiredEvent } from "../chat/types";
import { InteractionCard } from "./InteractionCard";

let root: Root;
let container: HTMLDivElement;
beforeEach(() => {
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
});
afterEach(async () => {
  await act(async () => root.unmount());
  container.remove();
});

it("offers free text as one more option that opens in place and sends on Enter", async () => {
  const onReply = vi.fn();
  const interaction: InteractionRequiredEvent = {
    type: "interaction_required",
    interaction_id: "interaction_1234567890abcdef1234",
    kind: "single_selection",
    prompt: "用哪份简历？",
    options: [{ label: "不用简历", value: "without_resume" }],
    allow_free_text: true,
  };
  await act(async () => root.render(
    <InteractionCard interaction={interaction} disabled={false} apiBaseUrl="" onReply={onReply} />,
  ));
  expect(container.querySelector("textarea")).toBeNull();

  const other = [...container.querySelectorAll("button")].find((item) => item.textContent === "其他，直接输入…")!;
  await act(async () => other.click());
  const input = container.querySelector<HTMLInputElement>(".option-input input")!;
  const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value")!.set!;
  await act(async () => {
    setter.call(input, "用我昨天发的那份");
    input.dispatchEvent(new Event("input", { bubbles: true }));
  });
  await act(async () => {
    input.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter", bubbles: true }));
  });

  expect(onReply).toHaveBeenCalledWith({ message: "用我昨天发的那份" });
});
