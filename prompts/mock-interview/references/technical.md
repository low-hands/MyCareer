# Technical and role-specific guidance

Use this reference for knowledge, problem-solving, system-design, project deep-dive, and professional scenario questions.

## Question design

- Tie the question to a quoted JD requirement, a grounded resume claim, or both.
- For resume deep-dives, ask the candidate to explain their actual contribution, constraints, reasoning, trade-offs, validation, and result.
- For knowledge questions, prefer application and comparison over isolated definitions.
- For system design, state only the minimum scenario and constraints needed to begin; let the candidate ask clarifying questions.
- For problem solving, ask for assumptions, decomposition, chosen approach, and
  how the candidate would verify the result. Do not reward guessing the hidden
  preferred solution.
- For role scenarios, test a realistic decision the role may face, but do not
  pretend the hypothetical is known to occur at the target employer.
- Do not assume a scale, architecture, incident, or metric that is absent from the supplied material.
- Ask one main question at a time. Put alternative designs, failure modes, and
  evidence checks into possible follow-ups rather than stacking them into the
  opening question.

## Evaluation

Select only dimensions relevant to the question:

- `accuracy`: technical or professional correctness.
- `relevance`: whether the answer addresses the question and constraints.
- `specificity`: concrete mechanisms, examples, ownership, or evidence.
- `reasoning`: assumptions, decomposition, trade-offs, and validation.
- `communication`: clarity and ability to make uncertainty explicit.

Do not penalize a candidate for choosing a reasonable alternative design when they explain its assumptions and trade-offs. Separate incorrect claims from missing detail.

Use the score scale consistently:

- `1`: absent, materially incorrect, or unrelated evidence;
- `2`: partial understanding with important errors or unexplained leaps;
- `3`: workable answer covering the main requirement with some missing depth;
- `4`: correct and specific reasoning with useful trade-offs or validation;
- `5`: unusually strong, coherent treatment of constraints, alternatives,
  failure modes, and evidence. Do not require perfection for a `5`.

Derive the overall rating from the answer, not from style alone:

- `strong`: the central answer is correct or professionally sound and well supported;
- `adequate`: the core approach is workable, with bounded gaps;
- `weak`: substantial correction or missing reasoning is needed;
- `insufficient_evidence`: the response does not contain enough relevant material
  to assess the intended capability. This is not a synonym for technically wrong.

When evaluating a project claim, distinguish what the resume already confirms,
what the current answer newly alleges, and what remains unverifiable. Do not
silently promote answer-only metrics or ownership claims into confirmed evidence.

## Follow-up

Use one focused follow-up when it can clarify a material gap, such as:

- an unexplained technical choice;
- unclear personal ownership in a resume project;
- missing failure mode, trade-off, measurement, or validation;
- a suspiciously broad claim that needs evidence.

Prefer the follow-up with the highest information value: the answer to it should
be capable of changing the assessment. Move on when further questioning would
merely request the same answer in different words, test trivia outside the plan
focus, or exceed the session's configured follow-up budget.
