// @vitest-environment jsdom
import { act, useState } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { useConfirmDialog } from "./ConfirmDialog";

let container: HTMLDivElement;
let root: Root;
const results: boolean[] = [];

function Harness() {
  const { confirm, confirmDialog } = useConfirmDialog();
  const [, setTick] = useState(0);
  return (
    <>
      <button type="button" id="ask" onClick={async () => {
        results.push(await confirm({ title: "删除简历「A」？", body: "无法恢复。" }));
        setTick((value) => value + 1);
      }}>ask</button>
      {confirmDialog}
    </>
  );
}

beforeEach(() => {
  vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
  results.length = 0;
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
});
afterEach(() => {
  act(() => root.unmount());
  container.remove();
});

async function ask() {
  await act(async () => container.querySelector<HTMLButtonElement>("#ask")!.click());
  return container.querySelector("dialog.confirm-dialog")!;
}

describe("useConfirmDialog", () => {
  it("shows the question with focus on cancel and resolves true only on confirm", async () => {
    await act(async () => root.render(<Harness />));
    const dialog = await ask();
    expect(dialog.textContent).toContain("删除简历「A」？");
    expect(dialog.textContent).toContain("无法恢复。");
    expect(document.activeElement?.textContent).toBe("取消");
    const confirm = [...dialog.querySelectorAll("button")].find((button) => button.textContent === "删除")!;
    await act(async () => confirm.click());
    expect(results).toEqual([true]);
    expect(container.querySelector("dialog")).toBeNull();
  });

  it("resolves false on cancel and on Escape", async () => {
    await act(async () => root.render(<Harness />));
    let dialog = await ask();
    await act(async () => [...dialog.querySelectorAll("button")].find((button) => button.textContent === "取消")!.click());
    dialog = await ask();
    await act(async () => dialog.dispatchEvent(new Event("cancel", { cancelable: true })));
    expect(results).toEqual([false, false]);
    expect(container.querySelector("dialog")).toBeNull();
  });
});
