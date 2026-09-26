import { useCallback, useEffect, useRef, useState, type ReactNode } from "react";

export interface ConfirmOptions {
  title: string;
  body?: string;
  confirmLabel?: string;
}

/**
 * The app's own confirmation for destructive actions, in place of the
 * browser's `window.confirm`. `confirm(options)` resolves true only when the
 * reader presses the confirm button; Escape, the backdrop and "取消" resolve
 * false. Focus starts on "取消", so a stray Enter never deletes anything.
 */
export function useConfirmDialog(): {
  confirm: (options: ConfirmOptions) => Promise<boolean>;
  confirmDialog: ReactNode;
} {
  const [pending, setPending] = useState<ConfirmOptions | null>(null);
  const resolver = useRef<((value: boolean) => void) | null>(null);

  const settle = useCallback((value: boolean) => {
    resolver.current?.(value);
    resolver.current = null;
    setPending(null);
  }, []);
  const confirm = useCallback((options: ConfirmOptions) => {
    resolver.current?.(false);
    setPending(options);
    return new Promise<boolean>((resolve) => { resolver.current = resolve; });
  }, []);
  useEffect(() => () => resolver.current?.(false), []);

  return {
    confirm,
    confirmDialog: pending ? <ConfirmDialog options={pending} onSettle={settle} /> : null,
  };
}

function ConfirmDialog({ options, onSettle }: { options: ConfirmOptions; onSettle: (value: boolean) => void }) {
  const dialog = useRef<HTMLDialogElement>(null);
  const cancel = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    const element = dialog.current;
    const trigger = document.activeElement;
    if (element && typeof element.showModal === "function") element.showModal();
    else element?.setAttribute("open", "");
    cancel.current?.focus();
    return () => {
      if (element?.open && typeof element.close === "function") element.close();
      if (trigger instanceof HTMLElement) trigger.focus();
    };
  }, []);
  return (
    <dialog
      ref={dialog}
      className="confirm-dialog"
      aria-labelledby="confirm-dialog-title"
      onCancel={(event) => { event.preventDefault(); onSettle(false); }}
      onClick={(event) => { if (event.target === event.currentTarget) onSettle(false); }}
    >
      <div className="confirm-dialog-body">
        <h2 id="confirm-dialog-title">{options.title}</h2>
        {options.body ? <p>{options.body}</p> : null}
      </div>
      <div className="confirm-dialog-actions">
        <button ref={cancel} type="button" className="soft-button" onClick={() => onSettle(false)}>取消</button>
        <button type="button" className="confirm-dialog-danger" onClick={() => onSettle(true)}>
          {options.confirmLabel ?? "删除"}
        </button>
      </div>
    </dialog>
  );
}
