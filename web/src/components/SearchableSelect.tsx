import { useEffect, useId, useRef, useState } from "react";

interface SelectOption { value: string; label: string; description?: string }

export function SearchableSelect({ label, placeholder, searchPlaceholder, value, options, onChange, disabled = false }: {
  label: string; placeholder: string; searchPlaceholder: string; value: string;
  options: SelectOption[]; onChange: (value: string) => void; disabled?: boolean;
}) {
  const id = useId();
  const root = useRef<HTMLDivElement>(null);
  const trigger = useRef<HTMLButtonElement>(null);
  const search = useRef<HTMLInputElement>(null);
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const selected = options.find((option) => option.value === value);
  const terms = query.trim().toLocaleLowerCase().split(/\s+/).filter(Boolean);
  const filtered = options.filter((option) => terms.every((term) => `${option.label} ${option.description ?? ""}`.toLocaleLowerCase().includes(term)));
  useEffect(() => {
    if (!open) return;
    setQuery("");
    search.current?.focus();
    const close = (event: PointerEvent) => {
      if (!root.current?.contains(event.target as Node)) setOpen(false);
    };
    document.addEventListener("pointerdown", close);
    return () => document.removeEventListener("pointerdown", close);
  }, [open]);
  useEffect(() => { if (disabled) setOpen(false); }, [disabled]);
  return <div className="searchable-select resume-version-picker" ref={root} onBlur={(event) => {
    if (!event.currentTarget.contains(event.relatedTarget)) setOpen(false);
  }} onKeyDown={(event) => {
    if (event.nativeEvent.isComposing) return;
    if (event.key === "Escape") { event.preventDefault(); setOpen(false); trigger.current?.focus(); }
    if (event.target === search.current && event.key === "Enter") { event.preventDefault(); return; }
    if (event.target === search.current && ["Home", "End"].includes(event.key)) return;
    if (!["ArrowDown", "ArrowUp", "Home", "End"].includes(event.key)) return;
    event.preventDefault();
    if (!open) { setOpen(true); return; }
    const buttons = Array.from(root.current?.querySelectorAll<HTMLButtonElement>('[role="option"]') ?? []);
    const index = buttons.indexOf(document.activeElement as HTMLButtonElement);
    const next = event.key === "Home" ? 0 : event.key === "End" ? buttons.length - 1 : index === -1 ? (event.key === "ArrowUp" ? buttons.length - 1 : 0) : (index + (event.key === "ArrowDown" ? 1 : -1) + buttons.length) % buttons.length;
    buttons[next]?.focus();
  }}>
    <span className="resume-version-label" id={`${id}-label`}>{label}</span>
    <button ref={trigger} type="button" className={`resume-version-trigger ${open ? "is-open" : ""}`} aria-labelledby={`${id}-label ${id}-value`} aria-haspopup="listbox" aria-expanded={open} aria-controls={open ? id : undefined} disabled={disabled} onClick={() => setOpen(!open)}>
      <span className="resume-version-trigger-copy" id={`${id}-value`}><strong>{selected?.label ?? placeholder}</strong>{selected?.description ? <small>{selected.description}</small> : null}</span>
      <span className="resume-version-chevron" aria-hidden="true">⌄</span>
    </button>
    {open ? <div className="resume-version-menu">
      <div className="resume-version-search"><input ref={search} type="search" aria-label={searchPlaceholder} placeholder={searchPlaceholder} value={query} onChange={(event) => setQuery(event.target.value)} aria-controls={id} /></div>
      <div id={id} className="resume-version-results" role="listbox" aria-label={label}>
        {filtered.map((option) => <button type="button" role="option" tabIndex={-1} aria-selected={option.value === value} className={`resume-version-option ${option.value === value ? "is-selected" : ""}`} key={option.value} onClick={() => { onChange(option.value); setOpen(false); trigger.current?.focus(); }}>
          <span className="resume-version-option-copy"><strong>{option.label}</strong>{option.description ? <small>{option.description}</small> : null}</span>
          {option.value === value ? <span className="resume-version-check" aria-hidden="true">✓</span> : null}
        </button>)}
      </div>
      {!filtered.length ? <p className="resume-version-empty" role="status">{options.length ? "没有找到匹配项，试试其他关键词。" : "暂无可选内容"}</p> : null}
    </div> : null}
  </div>;
}
