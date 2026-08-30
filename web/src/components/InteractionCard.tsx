import { useMemo, useState } from "react";

import type { InteractionRequiredEvent } from "../chat/types";

interface InteractionCardProps {
  interaction: InteractionRequiredEvent;
  disabled: boolean;
  onReply: (message: string) => void;
}

function optionValue(option: InteractionRequiredEvent["options"][number]): string {
  if (option.selection_index !== undefined) return String(option.selection_index);
  return option.value ?? option.label;
}

export function InteractionCard({ interaction, disabled, onReply }: InteractionCardProps) {
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const isMultiple = interaction.kind === "multiple_selection";
  const selectedMessage = useMemo(() => [...selected].join(", "), [selected]);

  if (interaction.kind === "free_text") {
    return (
      <div className="interaction-card" aria-labelledby={interaction.interaction_id}>
        <p id={interaction.interaction_id}>{interaction.prompt}</p>
        <span className="interaction-hint">请在下方输入框中回答后继续。</span>
      </div>
    );
  }
  if (interaction.kind === "file_upload") {
    return (
      <div className="interaction-card" aria-labelledby={interaction.interaction_id}>
        <p id={interaction.interaction_id}>{interaction.prompt}</p>
        <span className="interaction-hint">当前客户端尚未接入文件上传，请先使用文本说明。</span>
      </div>
    );
  }

  return (
    <div className="interaction-card" aria-labelledby={interaction.interaction_id}>
      <p id={interaction.interaction_id}>{interaction.prompt}</p>
      <div className="interaction-options">
        {interaction.options.map((option) => {
          const value = optionValue(option);
          const active = selected.has(value);
          return (
            <button
              type="button"
              className={active ? "option-button is-selected" : "option-button"}
              key={`${option.label}-${value}`}
              disabled={disabled}
              aria-pressed={isMultiple ? active : undefined}
              onClick={() => {
                if (!isMultiple) {
                  onReply(value);
                  return;
                }
                setSelected((current) => {
                  const next = new Set(current);
                  if (next.has(value)) next.delete(value);
                  else next.add(value);
                  return next;
                });
              }}
            >
              <span>{option.label}</span>
              {option.description ? <small>{option.description}</small> : null}
            </button>
          );
        })}
      </div>
      {isMultiple ? (
        <button
          type="button"
          className="confirm-button"
          disabled={disabled || selected.size === 0}
          onClick={() => onReply(selectedMessage)}
        >
          确认选择
        </button>
      ) : null}
    </div>
  );
}
