import { useEffect, useRef } from "react";

import { isImeKeyEvent } from "../chat/keyboard";

/**
 * The "type your own answer" row of a choice list.
 *
 * Closed, it reads like any other option; opened, the same row becomes a
 * one-line input, so free text costs no more space than the choice it
 * replaces. Enter confirms and Escape closes it again, as in the choice
 * prompts of terminal agents.
 */
export function InlineOtherOption({
  label,
  hint,
  placeholder,
  value,
  open,
  disabled,
  submitLabel = "确定",
  onOpen,
  onClose,
  onChange,
  onSubmit,
}: {
  label: string;
  hint?: string;
  placeholder: string;
  value: string;
  open: boolean;
  disabled: boolean;
  submitLabel?: string;
  onOpen: () => void;
  onClose?: () => void;
  onChange: (value: string) => void;
  onSubmit: () => void;
}) {
  const input = useRef<HTMLInputElement>(null);
  useEffect(() => {
    if (open) input.current?.focus();
  }, [open]);

  if (!open) {
    return (
      <button type="button" className="option-button option-other" disabled={disabled} onClick={onOpen}>
        <span>{label}</span>
        {hint ? <small>{hint}</small> : null}
      </button>
    );
  }
  return (
    <div className="option-input">
      <input
        ref={input}
        aria-label={label}
        value={value}
        maxLength={1000}
        disabled={disabled}
        placeholder={placeholder}
        onChange={(event) => onChange(event.target.value)}
        onKeyDown={(event) => {
          if (isImeKeyEvent(event)) return;
          if (event.key === "Enter") {
            event.preventDefault();
            if (value.trim()) onSubmit();
          } else if (event.key === "Escape" && onClose) {
            event.preventDefault();
            onClose();
          }
        }}
      />
      <button type="button" disabled={disabled || !value.trim()} onClick={onSubmit}>
        {submitLabel}
      </button>
    </div>
  );
}
