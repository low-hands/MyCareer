# Interview reporting guidance

Use this reference only after the workflow supplies per-question evaluations.
Each question was scored on its own chain; you see those evaluations, not the
answers. The report is a practice artifact, not an employer decision,
personality profile, or hiring forecast.

## Synthesize evidence, do not average prose

- Base every conclusion on one or more supplied question evaluations.
- Per-question results (question, rating, summary, follow-up count) are
  assembled by the workflow from those evaluations; do not restate them.
- Do not let one excellent or poor answer silently determine unrelated areas.
- Repeated patterns across different questions deserve more weight than a single
  stylistic moment.

## Distinguish four kinds of conclusion

1. `strengths`: capabilities demonstrated with specific observed evidence;
2. `development_areas`: material gaps, errors, or weak patterns actually observed;
3. `practice_actions`: concrete next exercises linked to those observations;
4. `limitations`: important capabilities not tested, weak source grounding,
   early termination, insufficient answers, or other limits on interpretation.

Do not convert an untested area into a weakness. Do not convert a candidate's
new answer-only claim into confirmed career history. Avoid statements about
personality, employability, pass likelihood, or what the employer will decide.

## Check consistency across questions

Each evaluation lists `key_facts` the candidate stated. Compare them across
questions and report in `consistency_issues` only facts that cannot both be
true, naming both questions, for example "第1题说团队5人，第4题说团队3人".
Different facts about different projects are not a contradiction, and a detail
given in one answer but not another is not one either. Leave the list empty
when nothing conflicts.

## Prescribe deliberate practice

Each practice action should name:

- the capability to practice;
- a concrete exercise or revision;
- what improved performance would look like.

Prefer a short prioritized list over one recommendation per minor issue. Examples
include rebuilding a system-design answer around explicit constraints and failure
modes, rewriting a project explanation to separate personal ownership from team
work, or rehearsing a truthful transition explanation in a concise form. Do not
write a fabricated model answer on the candidate's behalf unless the product has
explicitly requested a separate coaching operation.

## Completion-aware reporting

- `plan_completed`: summarize the planned coverage and still list untested areas.
- `user_ended` or `time_limit`: report only completed evidence and identify the
  remaining coverage as a limitation, not a failure.
- `safety_stop`: keep the report minimal and do not restate unsafe or sensitive
  content unnecessarily.
- If `question_chain_limitations` says an answer was truncated or omitted,
  treat that question as partially observed: do not infer that the missing
  portion was absent, and mention the limitation when it materially affects
  confidence in the rating.

Write for the candidate. Never name workflow fields or data structures
(`confirmed_resume_facts`, `key_facts`, `question_chain`, and the like); say
what they mean instead, for example "简历中的经历尚未经你确认".

The top-level summary should be candid and compact: overall observed pattern,
strongest demonstrated capability, highest-priority development area, and the
scope limitation that most affects confidence.
