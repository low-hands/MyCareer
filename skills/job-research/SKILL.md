---
name: job-research
description: Research public company, product-line, business, market, and competitor context for one saved job when the user explicitly requests it. Do not use automatically during job matching, resume tailoring, or interview preparation, and do not claim to recover a specific team's private work from a generic JD.
---

# Job Research

Research public context that helps the user understand a saved job. Treat the
exact JD as a source of search anchors, not proof that public company material
describes this specific role or team. External sources add context but must not
silently rewrite or strengthen the JD.

This is an optional, user-requested activity. Its existence does not make it a
required step in job matching, resume tailoring, application tracking, interview
preparation, or mock interviews.

The research scope may contain `user_provided_context`, such as product or
business clues the user reports hearing in an interview and explicitly agrees to
use for research. Treat it only as an unverified search lead. It is not a public
source, cannot support a `fact`, and cannot become confirmed merely because a
search result contains similar words. Do not expose private names, contact
details, or unrelated interview recollections in queries or findings.

## Research priorities

Spend limited source capacity in this order:

1. official material about a product or business line explicitly named in the JD
   or supplied by the user;
2. current primary product, business, investor, regulatory, or technical material
   about that public context;
3. reputable secondary reporting about relevant market changes or competitors;
4. company-level background only when no narrower public anchor exists or it
   materially helps the user's stated question.

Search for products, customers, business models, public initiatives, market
conditions, competitors, and technical context when the JD or user request gives
a defensible link. Do not spend sources on generic history, slogans, rankings,
or interview-question lists unless the requested focus makes them directly
useful.

If the JD contains no product-line or business anchor, do not manufacture one.
Limit the result to the company-level question the user actually asked, state
that its relationship to the role cannot be confirmed, and put the missing
role-specific context in `open_questions` or `limitations`.

## Chinese-market sources

Generic web search underserves Chinese employers: the useful primary material is
concentrated in places a broad query rarely surfaces. When the company is
Chinese, spend source capacity on these before falling back to general results.

| Question | Where the primary material is |
|---|---|
| Products, architecture, technical direction | the company's own engineering blog, its open-source repositories, conference talks |
| Business lines, funding, ownership, scale | 招股书, 年报, 投资方公告, 工商信息 (天眼查, 企查查) |
| Market position, competitors, recent moves | 36氪, 晚点LatePost, 界面新闻, 第一财经, and the company's own announcements |
| Regulatory or licensing constraints | the issuing body's own publication |

Prefer the company's own material and the issuing body's own publication over
reporting about them. Chinese tech media frequently republishes the same wire
copy; three outlets carrying one story are one source, not three, and citing
them as three overstates the support behind a finding.

### Self-reported content is not a primary source

Anonymous or pseudonymous posts on career and social platforms — 脉脉, 牛客网,
知乎, 小红书, CSDN, and similar — are individual recollections. They may not
support a `fact` under any circumstances. At most they support an `inference`
with `low` confidence, phrased as what some people report rather than as what is
the case, and only when several independent posts agree and nothing primary
contradicts them.

### What belongs to interview preparation instead

Compensation bands, interview loops, question banks, and hiring-process
timelines are out of scope here even when such posts are easy to find. Job
research answers what the company does and where it stands; the material a
candidate needs to walk into a specific interview is a different request with a
different consumer. Do not widen a company-research run into it, and do not
report those numbers as company facts.

## Evidence boundaries

- A `fact` states only what cited sources support.
- An `inference` connects cited evidence to the role and must be phrased as a
  possibility or reasoned interpretation, not employer-confirmed truth.
- An `unknown` records a material unanswered question and cites no source as if
  absence proved the answer.
- Company-wide information does not prove that a particular role or team works
  on that product, follows the same process, uses the same stack, operates at the
  same scale, or shares the same culture.
- Do not search for or infer private team roadmaps, reporting lines, headcount,
  staffing reasons, current projects, or day-to-day responsibilities.
- Never claim likely interview questions, hiring stages, headcount, internal
  plans, compensation, or team practices without suitable current evidence.
- Conflicting sources remain a limitation; do not average them into a false fact.

## Sources

Use only sources actually returned by web research in this run. Give every source
a short stable key such as `S1`, preserve its canonical public URL, title,
publisher when known, publication date when supported, and a short relevant
excerpt. The excerpt is evidence, not a place to copy an article.

Every `fact` and `inference` must cite at least one returned source key. Do not
include unused sources, duplicate URLs, search-result pages, inaccessible URLs,
or a source that merely mentions the company without supporting a finding.

## Output quality

Focus the summary on the public company, product, and business context the user
asked for. Keep findings individually testable, separate fact from inference,
surface unanswered questions, and state limitations caused by stale, missing,
conflicting, or company-wide-only evidence. Return only the configured structured
response.
