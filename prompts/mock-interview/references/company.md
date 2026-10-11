# Company-aware interview guidance

Use this reference to adapt preparation emphasis and interviewer style when the
workflow supplies an explicit target company. A company profile is a preparation
prior, not a promise about a particular team, interviewer, or current hiring
process. The exact JD and role title always outrank a generic company profile.

## Evidence and precedence

Apply signals in this order:

1. explicit responsibilities and requirements in the immutable JD;
2. explicit role title, function, level, and business area;
3. current sourced employer-process or values research, if the workflow supplies it;
4. the broad preparation profiles below;
5. general role practice when the company is unknown or ambiguous.

Never claim that a topic is guaranteed, assign interview probabilities, quote
pass rates, or state a current number of rounds from these profiles. Subsidiaries,
business groups, geographies, levels, and interviewers can differ materially.

## Choosing a profile

- When `company_style_profile` is set, the workflow has already matched the
  employer: use exactly that profile.
- When it is null but `target_company` is set, the name is not in the workflow's
  alias list. Use a profile only if you are confident the employer is that
  company or one of its businesses, and return its exact `###` heading in the
  plan's `company_style_profile`. The candidate is told which profile was used.
- Otherwise, or when unsure, apply no profile and return null. Never apply a
  profile to a company that merely resembles one below.
- During `ask`, follow the profile the plan recorded, or none.

## Company business context

`company_business_context`, when present, is the candidate's own research report
on the company: what it does and where it stands, with each finding marked
`fact`, `inference`, or `unknown` and a confidence. It is not evidence about how
the company interviews; interview style comes only from the profiles below.

- Use it for motivation and business-understanding questions, for example why
  this company or how the candidate reads one of its named businesses.
- Quote only what a `fact` states. Put an `inference` to the candidate as an
  open premise ("有公开信息显示……，你怎么看"), never as settled truth, and do
  not build a question on an `unknown`.
- When `outdated` is true, avoid time-sensitive details or say they may have
  changed.
- It never outranks the JD, and it cannot make a business line part of this
  role unless the JD says so.
- Without it, ask motivation questions without asserting anything about the
  company's business.

## How company context may change practice

Company context may influence:

- which relevant JD capability receives scarce plan coverage;
- whether follow-ups emphasize fundamentals, product impact, operational rigor,
  customer value, execution, scale, collaboration, or leadership judgment;
- the wording and pace of a question;
- which alternative angle to use when two questions are equally grounded.

It must not change factual correctness, excuse weak evidence, demand loyalty
performances, or turn cultural stereotypes into personality judgments.

## Preparation profiles

### Alibaba and related businesses

When relevant to the JD, emphasize business context, customer value, ownership,
cross-team execution, architectural trade-offs, reliability, and why a choice was
made over alternatives. For behavioral items, ask for concrete coordination and
decision evidence rather than slogans or memorized values language.

### Tencent and related businesses

When relevant, emphasize technical fundamentals, depth beneath framework-level
answers, user or product impact, data-informed decisions, and the actual mechanics
of a project. Product and game roles should follow their supplied business context;
do not assume all Tencent teams share one interview style.

### ByteDance and related businesses

When relevant, emphasize concise problem framing, applied problem solving,
measurable impact, learning speed, project depth, and implementation detail.
For technical roles, algorithmic or systems reasoning may be useful when supported
by the JD, but never claim every interview or round contains a coding problem.

### Baidu

When relevant, emphasize foundations, technical depth, experimentation, and the
path from research or models to reliable production outcomes. AI, search, ranking,
or recommendation topics belong only when the role or JD supports them.

### Meituan

When relevant, emphasize practical execution, operational constraints, data-based
diagnosis, reliability, efficiency, and demonstrated business outcomes. Use the
specific business domain in the JD rather than assuming every role concerns the
same local-commerce scenario.

### Huawei

When relevant, emphasize engineering fundamentals, disciplined reasoning,
quality, reliability, delivery constraints, and clear individual responsibility.
Hardware, embedded, telecom, or process-heavy scenarios must come from the role
context, not merely from the employer name.

### JD.com

When relevant, emphasize reliable engineering, business operations, scale,
customer outcomes, and end-to-end ownership. Supply-chain, logistics, retail, or
Java-specific questions require corresponding JD evidence.

### PDD Holdings and related businesses

When relevant, emphasize prioritization under ambiguity, execution, efficiency,
data, business judgment, and concrete outcomes. Do not reproduce stereotypes
about working hours, pressure tolerance, or personal sacrifice.

### Google

When relevant, emphasize clear problem decomposition, correctness, complexity,
testing, scalable design, collaboration, and the ability to clarify ambiguity.
Coding and design emphasis should follow the supplied role and level; do not
present a generic profile as a current official loop.

### Meta

When relevant, emphasize executable problem solving, product or user impact,
scale-aware design, prioritization, ownership, and behavioral evidence. Keep
coding, product, and system-design coverage proportional to the actual role.

### Amazon and AWS

When relevant, emphasize customer outcomes, ownership, depth of investigation,
trade-offs, delivery, and decisions supported by observable results. Leadership
themes may guide behavioral practice, but do not demand canned principle names
or treat a memorized framework as evidence.

### Microsoft

When relevant, emphasize collaborative engineering, structured problem solving,
design judgment, customer impact, learning from feedback, and working across team
boundaries. Adapt technical depth to the specific product area and role.

## Unknown and startup employers

For an employer without a reliable profile, infer only from the JD and role:

- product stage and likely ambiguity from explicit wording;
- expected ownership and collaboration from stated responsibilities;
- domain, scale, or regulation only when stated;
- gaps that should be recorded as limitations rather than filled with assumptions.

A startup label alone does not justify assuming chaos, long hours, broad authority,
or weak process. Ask realistic role scenarios grounded in the supplied materials.

## Quality gate

Before using a company-flavored item or question, verify that:

- it remains useful if the company profile is wrong or outdated;
- it is anchored to the JD, role, resume, or clearly labeled practice hypothesis;
- it avoids guarantees about rounds, question frequency, culture, or hiring outcomes;
- it tests a job-relevant capability rather than conformity to a stereotype;
- the same objective evaluation rubric can still be applied to the answer.
