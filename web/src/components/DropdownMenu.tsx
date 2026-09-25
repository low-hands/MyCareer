import { useEffect, useId, useRef, useState } from "react";

export interface DropdownMenuItem {
  value: string;
  label: string;
}

/**
 * A small action menu: a trigger that opens a list and runs `onSelect` with
 * the picked value. Used where a native <select> would only be a disguised
 * command ("移动到…"), so it styles like the app's other menus and supports
 * the keys a menu is expected to: arrows, Home/End, Enter, Escape.
 */
export function DropdownMenu({
  label,
  items,
  disabled = false,
  ariaLabel,
  onSelect,
}: {
  label: string;
  items: DropdownMenuItem[];
  disabled?: boolean;
  ariaLabel?: string;
  onSelect: (value: string) => void;
}) {
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(0);
  const root = useRef<HTMLDivElement>(null);
  const trigger = useRef<HTMLButtonElement>(null);
  const options = useRef<(HTMLButtonElement | null)[]>([]);
  const menuId = useId();

  useEffect(() => {
    if (open) options.current[active]?.focus();
  }, [open, active]);
  useEffect(() => {
    if (disabled) setOpen(false);
  }, [disabled]);

  function close(refocus: boolean) {
    setOpen(false);
    if (refocus) trigger.current?.focus();
  }

  return (
    <div
      className="dropdown-menu"
      ref={root}
      onBlur={(event) => {
        if (!root.current?.contains(event.relatedTarget as Node | null)) setOpen(false);
      }}
      onKeyDown={(event) => {
        if (!open) return;
        if (event.key === "Escape") { event.preventDefault(); close(true); return; }
        const last = items.length - 1;
        const next = { ArrowDown: Math.min(active + 1, last), ArrowUp: Math.max(active - 1, 0), Home: 0, End: last }[event.key];
        if (next !== undefined) { event.preventDefault(); setActive(next); }
      }}
    >
      <button
        ref={trigger}
        type="button"
        className={`dropdown-menu-trigger${open ? " is-open" : ""}`}
        aria-label={ariaLabel}
        aria-haspopup="menu"
        aria-expanded={open}
        aria-controls={open ? menuId : undefined}
        disabled={disabled || items.length === 0}
        onClick={() => { setActive(0); setOpen((value) => !value); }}
      >
        <span>{label}</span>
        <svg aria-hidden="true" width="12" height="12" viewBox="0 0 12 12">
          <path d="M3 4.5 6 7.5 9 4.5" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </button>
      {open ? (
        <div className="dropdown-menu-list" role="menu" id={menuId}>
          {items.map((item, index) => (
            <button
              key={item.value}
              ref={(node) => { options.current[index] = node; }}
              type="button"
              role="menuitem"
              tabIndex={index === active ? 0 : -1}
              className="dropdown-menu-item"
              onClick={() => { close(true); onSelect(item.value); }}
            >
              {item.label}
            </button>
          ))}
        </div>
      ) : null}
    </div>
  );
}
