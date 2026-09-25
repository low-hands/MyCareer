// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, expect, it, vi } from "vitest";

import { DropdownMenu } from "./DropdownMenu";

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

const items = [{ value: "pm", label: "AI 产品经理" }, { value: "agent", label: "Agent开发" }];

function key(target: Element, name: string) {
  target.dispatchEvent(new KeyboardEvent("keydown", { key: name, bubbles: true }));
}

it("opens a list, picks with the keyboard, and closes", async () => {
  const onSelect = vi.fn();
  await act(async () => root.render(<DropdownMenu label="移动到…" items={items} onSelect={onSelect} />));
  const trigger = container.querySelector<HTMLButtonElement>(".dropdown-menu-trigger")!;
  expect(container.querySelector("[role=menu]")).toBeNull();

  await act(async () => trigger.click());
  expect(trigger.getAttribute("aria-expanded")).toBe("true");
  const menu = container.querySelector("[role=menu]")!;
  await act(async () => key(menu, "ArrowDown"));
  expect(document.activeElement?.textContent).toBe("Agent开发");
  await act(async () => (document.activeElement as HTMLButtonElement).click());

  expect(onSelect).toHaveBeenCalledWith("agent");
  expect(container.querySelector("[role=menu]")).toBeNull();
});

it("closes on Escape without choosing", async () => {
  const onSelect = vi.fn();
  await act(async () => root.render(<DropdownMenu label="移动到…" items={items} onSelect={onSelect} />));
  await act(async () => container.querySelector<HTMLButtonElement>(".dropdown-menu-trigger")!.click());
  await act(async () => key(container.querySelector("[role=menu]")!, "Escape"));

  expect(container.querySelector("[role=menu]")).toBeNull();
  expect(onSelect).not.toHaveBeenCalled();
});
