import { useMemo, useState } from "react";

import type { InteractionRequiredEvent } from "../chat/types";
import type { InteractionResponse } from "../api/sse";
import type { ResumeImportResult } from "../api/client";
import { attachmentFromImport, type ChatAttachment } from "../chat/attachments";
import { InlineOtherOption } from "./InlineOtherOption";
import { ResumeImporter } from "./ResumeImporter";
import { QuestionnaireCard } from "./QuestionnaireCard";

export interface InteractionReply {
  message: string;
  interactionResponse?: InteractionResponse;
  /** Sent with the reply instead of whatever the composer has queued. */
  resources?: ChatAttachment[];
}

interface InteractionCardProps {
  interaction: InteractionRequiredEvent;
  disabled: boolean;
  apiBaseUrl: string;
  onReply: (reply: InteractionReply) => void;
}

/** Answer with the uploaded version attached, which is what the server starts on. */
export function uploadReply(result: ResumeImportResult): InteractionReply {
  const label = `《${result.name}》v${result.version_number}`;
  return {
    message: result.already_in_library
      ? `用简历${label}（库里已有这份文件，直接用它）`
      : `用刚上传的简历${label}`,
    resources: [attachmentFromImport(result)],
  };
}

function optionValue(option: InteractionRequiredEvent["options"][number]): string {
  if (option.selection_index != null) return String(option.selection_index);
  return option.value ?? option.label;
}

export function InteractionCard({ interaction, disabled, apiBaseUrl, onReply }: InteractionCardProps) {
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [freeText, setFreeText] = useState("");
  const [uploading, setUploading] = useState(false);
  const [typing, setTyping] = useState(false);
  const isMultiple = interaction.kind === "multiple_selection";
  const selectedMessage = useMemo(() => [...selected].join(", "), [selected]);

  if (interaction.kind === "questionnaire") {
    return <QuestionnaireCard key={interaction.interaction_id} interaction={interaction} disabled={disabled} onReply={onReply} />;
  }

  if (interaction.kind === "free_text") {
    // The question is already the last message; the composer takes the answer.
    return null;
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
      {interaction.accepts_upload === "resume" ? (
        <div className="interaction-upload">
          {uploading ? (
            <ResumeImporter
              apiBaseUrl={apiBaseUrl}
              submitLabel="上传并使用"
              onCancel={() => setUploading(false)}
              onImported={(result) => onReply(uploadReply(result))}
            />
          ) : (
            <button type="button" className="option-button" disabled={disabled} onClick={() => setUploading(true)}>
              <span>上传一份简历</span>
              <small>会存入简历库；和库里某一版完全相同时直接用那一版</small>
            </button>
          )}
        </div>
      ) : null}
      {interaction.allow_free_text ? (
        <div className="interaction-options">
          <InlineOtherOption
            label="其他，直接输入…"
            placeholder="输入你的回答，回车发送"
            value={freeText}
            open={typing}
            disabled={disabled}
            submitLabel="发送"
            onOpen={() => setTyping(true)}
            onClose={() => setTyping(false)}
            onChange={setFreeText}
            onSubmit={() => onReply({ message: freeText.trim() })}
          />
        </div>
      ) : null}
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
