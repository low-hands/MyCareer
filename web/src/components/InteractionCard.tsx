import { useMemo, useState } from "react";

import type { InteractionRequiredEvent } from "../chat/types";
import type { InteractionResponse } from "../api/sse";
import { ResumeImporter } from "./ResumeImporter";
import { QuestionnaireCard } from "./QuestionnaireCard";

export interface InteractionReply {
  message: string;
  interactionResponse?: InteractionResponse;
}

interface InteractionCardProps {
  interaction: InteractionRequiredEvent;
  disabled: boolean;
  apiBaseUrl: string;
  onReply: (reply: InteractionReply) => void;
}

function optionValue(option: InteractionRequiredEvent["options"][number]): string {
  if (option.selection_index != null) return String(option.selection_index);
  return option.value ?? option.label;
}

export function InteractionCard({ interaction, disabled, apiBaseUrl, onReply }: InteractionCardProps) {
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const isMultiple = interaction.kind === "multiple_selection";
  const selectedMessage = useMemo(() => [...selected].join(", "), [selected]);

  if (interaction.kind === "questionnaire") {
    return <QuestionnaireCard key={interaction.interaction_id} interaction={interaction} disabled={disabled} onReply={onReply} />;
  }

  if (interaction.kind === "free_text") {
    return (
      <div className="interaction-card" aria-labelledby={interaction.interaction_id}>
        <span className="interaction-kicker">需要你的回答</span>
        <p id={interaction.interaction_id}>{interaction.prompt}</p>
      </div>
    );
  }
  if (interaction.kind === "file_upload") {
    return (
      <div className="interaction-card" aria-labelledby={interaction.interaction_id}>
        <span className="interaction-kicker">需要文件</span>
        <p id={interaction.interaction_id}>{interaction.prompt}</p>
        <ResumeImporter
          apiBaseUrl={apiBaseUrl}
          onImported={(result) => onReply({
            message: `已导入简历“${result.name}”v${result.version_number}，请继续处理。`,
          })}
        />
      </div>
    );
  }

  return (
    <div className="interaction-card" aria-labelledby={interaction.interaction_id}>
      <span className="interaction-kicker">请选择后继续</span>
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
                  // Any scoped interaction binds its confirm/cancel buttons to
                  // the durable interaction id; the parser has already limited
                  // scope to the values the server routes on.
                  onReply({
                    message: option.label,
                    interactionResponse:
                      interaction.scope != null &&
                      (value === "confirm" || value === "cancel")
                        ? {
                            interaction_id: interaction.interaction_id,
                            scope: interaction.scope,
                            action: value,
                          }
                        : undefined,
                  });
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
          onClick={() => onReply({ message: selectedMessage })}
        >
          确认选择
        </button>
      ) : null}
    </div>
  );
}
