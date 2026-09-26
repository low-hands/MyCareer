import { useEffect, useRef } from "react";

/**
 * A list row that asks before deleting, in place of the row itself: no modal
 * for a one-line action. Focus starts on "取消"; Escape, or moving focus out
 * of the row, cancels.
 */
export function InlineDeleteConfirm({
  question,
  note,
  busy,
  className = "",
  onConfirm,
  onCancel,
}: {
  question: string;
  note?: string;
  busy: boolean;
  className?: string;
  onConfirm: () => void;
  onCancel: () => void;
}) {
  const root = useRef<HTMLDivElement>(null);
  const cancel = useRef<HTMLButtonElement>(null);
  useEffect(() => { cancel.current?.focus(); }, []);
  return (
    <div
      ref={root}
      className={`inline-delete-confirm ${className}`.trim()}
      role="group"
      aria-label={question}
      onKeyDown={(event) => {
        if (event.key === "Escape") { event.preventDefault(); onCancel(); }
      }}
      onBlur={(event) => {
        if (!busy && !root.current?.contains(event.relatedTarget as Node | null)) onCancel();
      }}
    >
      <div className="inline-delete-copy">
        <strong>{question}</strong>
        {note ? <small>{note}</small> : null}
      </div>
      <div className="inline-delete-actions">
        <button ref={cancel} type="button" disabled={busy} onClick={onCancel}>取消</button>
        <button type="button" className="is-danger" disabled={busy} onClick={onConfirm}>
          {busy ? "删除中…" : "删除"}
        </button>
      </div>
    </div>
  );
}
