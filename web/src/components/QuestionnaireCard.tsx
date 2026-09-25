import { useEffect, useRef, useState } from "react";

import type { InteractionRequiredEvent } from "../chat/types";
import { InlineOtherOption } from "./InlineOtherOption";
import type { InteractionReply } from "./InteractionCard";

type DraftAnswer = { selected_values: string[]; free_text: string; skipped: boolean };

interface Props {
  interaction: InteractionRequiredEvent;
  disabled: boolean;
  onReply: (reply: InteractionReply) => void;
}

function storedAnswers(key: string): Record<string, DraftAnswer> {
  try {
    const value = JSON.parse(sessionStorage.getItem(key) ?? "{}");
    if (!value || typeof value !== "object" || Array.isArray(value)) return {};
    const restored: Record<string, DraftAnswer> = {};
    for (const [id, answer] of Object.entries(value)) {
      if (!answer || typeof answer !== "object" || Array.isArray(answer)) continue;
      const entry = answer as Record<string, unknown>;
      if (!Array.isArray(entry.selected_values)
        || !entry.selected_values.every((item) => typeof item === "string")
        || typeof entry.free_text !== "string"
        || typeof entry.skipped !== "boolean") continue;
      restored[id] = {
        selected_values: entry.selected_values,
        free_text: entry.free_text.slice(0, 1000),
        skipped: entry.skipped,
      };
    }
    return restored;
  } catch {
    return {};
  }
}

export function QuestionnaireCard({ interaction, disabled, onReply }: Props) {
  const questions = interaction.questions ?? [];
  const storageKey = `career-questionnaire:${interaction.interaction_id}`;
  const [index, setIndex] = useState(0);
  const [answers, setAnswers] = useState<Record<string, DraftAnswer>>(() => storedAnswers(storageKey));
  const heading = useRef<HTMLHeadingElement>(null);
  const [noting, setNoting] = useState(false);
  useEffect(() => { heading.current?.focus(); setNoting(false); }, [index]);
  useEffect(() => {
    try { sessionStorage.setItem(storageKey, JSON.stringify(answers)); } catch { /* In-memory draft still works. */ }
  }, [answers, storageKey]);
  const question = questions[index];
  if (!question) return null;
  const answer = answers[question.question_id] ?? { selected_values: [], free_text: "", skipped: false };
  const selected = new Set(answer.selected_values);
  const otherSelected = question.options.some((option) => option.meaning === "other" && selected.has(option.value));
  const valid = answer.skipped ? question.allow_skip : question.kind === "free_text"
    ? answer.free_text.trim().length > 0
    : question.kind === "single" ? selected.size === 1 && (!otherSelected || answer.free_text.trim().length > 0)
      : selected.size > 0 && (!otherSelected || answer.free_text.trim().length > 0);
  const allValid = questions.every((item) => {
    const entry = answers[item.question_id];
    if (!entry) return false;
    if (entry.skipped) return item.allow_skip;
    if (item.kind === "free_text") return entry.free_text.trim().length > 0;
    const other = item.options.some((option) => option.meaning === "other" && entry.selected_values.includes(option.value));
    return (item.kind === "single" ? entry.selected_values.length === 1 : entry.selected_values.length > 0)
      && (!other || entry.free_text.trim().length > 0);
  });
  function update(next: DraftAnswer) {
    setAnswers((current) => ({ ...current, [question.question_id]: next }));
  }
  function choose(value: string, meaning: string) {
    if (question.kind === "single" || meaning === "none") {
      update({ ...answer, selected_values: [value], skipped: false,
        free_text: meaning === "other" ? answer.free_text : "" });
      if (meaning !== "other" && index < questions.length - 1) {
        setIndex(index + 1);
      }
      return;
    }
    const withoutNone = question.options.filter((item) => item.meaning === "none").map((item) => item.value);
    const next = selected.has(value)
      ? answer.selected_values.filter((item) => item !== value)
      : [...answer.selected_values.filter((item) => !withoutNone.includes(item)), value];
    const retainsOther = question.options.some((item) => item.meaning === "other" && next.includes(item.value));
    update({ ...answer, selected_values: next, skipped: false,
      free_text: question.allow_free_text || retainsOther ? answer.free_text : "" });
  }
  /** Enter in an inline answer: move on, or submit on the last question. */
  function advance() {
    if (!valid) return;
    if (index < questions.length - 1) setIndex(index + 1);
    else submit();
  }
  function submit() {
    if (!allValid || disabled) return;
    onReply({
      message: "已提交当前任务问卷回答。",
      interactionResponse: {
        interaction_id: interaction.interaction_id,
        scope: "questionnaire",
        action: "submit",
        answers: questions.map((item) => {
          const entry = answers[item.question_id];
          return {
            question_id: item.question_id,
            selected_values: entry.selected_values,
            free_text: entry.free_text.trim() || null,
            skipped: entry.skipped,
          };
        }),
      },
    });
  }
  return (
    <section className="interaction-card questionnaire-card" aria-labelledby="questionnaire-title">
      <span className="interaction-kicker">补充信息{questions.length > 1 ? ` · ${index + 1}/${questions.length}` : ""}</span>
      <p>{interaction.prompt}</p>
      <h3 id="questionnaire-title" ref={heading} tabIndex={-1}>{question.prompt}</h3>
      {question.kind === "free_text" ? (
        <textarea aria-label={question.prompt} value={answer.free_text} disabled={disabled || answer.skipped}
          rows={4} maxLength={1000} placeholder="请在这里填写回答…"
          onChange={(event) => update({ ...answer, free_text: event.target.value, skipped: false })} />
      ) : (
        <div className="interaction-options" role={question.kind === "single" ? "radiogroup" : "group"}
          aria-label={question.prompt}>
          {question.options.map((option) => option.meaning === "other" ? (
            // "Other" is answered in its own row: choosing it opens the input there.
            <InlineOtherOption key={option.value} label={option.label} placeholder={`${option.label}：请输入，回车确认`}
              value={answer.free_text} open={selected.has(option.value)} disabled={disabled}
              onOpen={() => choose(option.value, option.meaning)}
              onChange={(text) => update({ ...answer, free_text: text, skipped: false })}
              onSubmit={advance} />
          ) : (
            <button key={option.value} type="button" disabled={disabled}
              className={selected.has(option.value) ? "option-button is-selected" : "option-button"}
              role={question.kind === "single" ? "radio" : "checkbox"}
              aria-checked={selected.has(option.value)}
              onClick={() => choose(option.value, option.meaning)}>{option.label}</button>
          ))}
          {question.allow_free_text && !question.options.some((option) => option.meaning === "other")
            && !question.options.some((option) => option.meaning === "none" && selected.has(option.value)) ? (
            <InlineOtherOption label="补充说明（可选）" placeholder="补充说明，回车确认"
              value={answer.free_text} open={noting || answer.free_text.length > 0} disabled={disabled}
              onOpen={() => setNoting(true)}
              onClose={() => { if (!answer.free_text) setNoting(false); }}
              onChange={(text) => update({ ...answer, free_text: text, skipped: false })}
              onSubmit={advance} />
          ) : null}
        </div>
      )}
      <div className="questionnaire-actions">
        <button type="button" disabled={disabled || index === 0} onClick={() => setIndex(index - 1)}>上一步</button>
        {question.allow_skip ? <button type="button" disabled={disabled}
          onClick={() => { update({ selected_values: [], free_text: "", skipped: true });
            if (index < questions.length - 1) setIndex(index + 1); }}>跳过</button> : null}
        {index < questions.length - 1 && question.kind !== "single" ? (
          <button type="button" disabled={disabled || !valid} onClick={() => setIndex(index + 1)}>下一步</button>
        ) : index === questions.length - 1 ? (
          <button type="button" className="questionnaire-submit" disabled={disabled || !allValid} onClick={submit}>提交</button>
        ) : null}
      </div>
    </section>
  );
}
