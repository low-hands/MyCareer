# Capability search example queries — 2026-10-05

## Data and review

The example author worked in a separate agent context. It received only an
export of the 70 searchable tools' names, namespaces, summaries, parameter
schemas, and precondition descriptions. It did not receive the development
cases, either holdout, Qwen baseline, or existing Chinese aliases. It wrote
350 Chinese queries (five per tool) as JSON outside the repository. The
reviewed set is stored in `capabilities/example_queries.py`.

I reviewed every query for tool ownership, one-action wording, read/write
distinctions, and write requests that were too weak. Fourteen queries were
rewritten: nine for meaning or wording, and five after a similarity audit.
The audit compared normalized query text with every original user message in
the development set and both holdouts using `SequenceMatcher` at 0.65, plus
long substring matches. It flagged four development overlaps and one original
holdout overlap; all five were rewritten. Re-running the audit found **zero**
matches at that threshold in all three sets. The generated JSON remained
outside the repository.

The catalogue validates 5–10 queries of 4–40 characters for every searchable
tool, rejects exact copies of the tool name or its aliases, and forbids
examples on runtime-only tools, `search_capabilities`, and
`route_to_capability`. Examples are retrieval metadata only. Registered and
legacy-visible schema hashes and prompt fingerprints remain the values in the
stage 1–2 report; the search tool is still hidden from every legacy profile.

## Index rule

BM25 still scores names and aliases at weight 5, summaries at 2, and parameter
names at 1. Each example is tokenized and scored as a separate short document;
the strongest example for a tool contributes at weight 3. Taking the maximum
prevents tools with more examples from receiving more votes or a longer BM25
document. Generic two-character overlaps are ignored: an example needs at
least two matching tokens and 25% token coverage. For a `WRITE` tool it needs
50% coverage, so an incidental mention of a job or interview does not promote
a write. These thresholds were set using the development set only. The
optional semantic index still uses the earlier catalogue text; adding examples
to its embedding or averaging example vectors is deferred until a real
embedding provider can be compared.

## First-step query recall

The original user message is used as the search query. A hit means one of the
accepted first-step business tools is in the first 1, 3, or 5 results. Control
cases with no positive first-step demand are excluded from recall.

| Set | Configuration | Top 1 | Top 3 | Top 5 |
| --- | --- | ---: | ---: | ---: |
| Development (43 demands) | Before examples | 14/43 | 23/43 | 30/43 (69.8%) |
| Development (43 demands) | With examples | 16/43 | 32/43 | **35/43 (81.4%)** |
| Original holdout (18 demands) | Before examples | 5/18 | 8/18 | 12/18 (66.7%) |
| Original holdout (18 demands) | With examples | 7/18 | 12/18 | **15/18 (83.3%)** |
| Independent holdout (12 demands) | Before examples | 2/12 | 6/12 | 8/12 (66.7%) |
| Independent holdout (12 demands) | With examples | 2/12 | 5/12 | **9/12 (75.0%)** |

The development set alone was used for the index thresholds. Each holdout was
scored once after those values and the audited examples were fixed. The
independent holdout's top-3 result fell by one; its top-5 gain is only one
case, so broader independent samples are needed before treating this as a
stable gain. No holdout score is locked in tests.

Independent holdout top-5 misses with examples:

| Case | Expected first tool | Top-five results |
| --- | --- | --- |
| `independent_holdout_saved_job_requirements` | `get_saved_job` | `find_saved_jobs`, `correct_job_requirement_tier`, `analyze_job`, `update_owner_settings` |
| `independent_holdout_email_to_interview` | `sync_application_emails` | `list_applications`, `update_interview`, `list_email_events`, `get_interview`, `search_career_episodes` |
| `independent_holdout_tailor_review_export` | `draft_resume_tailoring` | `get_resume_job_match`, `export_resume_artifact`, `review_resume_tailoring`, `analyze_job`, `finalize_resume_tailoring` |

## Noise and write exposure

All **141/141** existing Chinese aliases still retrieve their owner in the
top five. `天气怎么样`, `的`, and `zzzxxyyunknownword` returned no result in
the initial run. The development baseline without examples was 30/43.

For control cases, the number of `WRITE` tool names in the first five was
**6 → 6** on development, **9 → 7** on the original holdout, and **5 → 5**
on the independent holdout. This is a retrieval proxy, not authorization;
existing execution checks still decide whether a tool can run. The
development counts are locked by a test. Both holdouts are only tested for
successful search execution, with no score assertions.

## Frequency-gate revision after the holdout run

The initial index excluded `怎么` and `么样` by hand. Those exceptions were
removed. A Chinese bigram now loses its *evidence vote* for the two-token
example-match gate when it occurs in more than 15% of tools' aliases/examples
**and** spans more than 60% of tool namespaces. The namespace condition keeps
frequent but meaningful terms such as `岗位` and `面试` from being treated as
general filler. Common terms still contribute to BM25 with their normal IDF;
they are not deleted from query or document tokens. This rule is derived from
the catalogue and changes automatically as example metadata changes.

On the development set, the revised index gives **17/43** at top 1,
**31/43** at top 3, and **35/43** at top 5. Development control write
exposure remains **6**, and all 141 aliases still retrieve their owner in
the top five. `天气怎么样`, `这个怎么弄`, and `帮我看一下` return no result;
`今天吃什么` returns only the read tool `get_daily_brief`. These are new
noise probes rather than a claim of perfect out-of-domain detection.

**The holdouts were not rescored for this revision.** Their numbers above
describe the initial `0c07bce` index only and do not establish revised-index
holdout performance. The test run for this revision explicitly deselected the
test that invokes both frozen holdouts.

## Cross-field action gate — 2026-10-06

The example-only evidence gate left an alias and summary path for action
tools. A query such as `帮我看看我的经历` could therefore rank fact confirmation
and deletion proposals from shared object words. `WRITE` tools and
`propose_*` tools now need at least two non-common matched terms across all
indexed fields: name, aliases, summary, parameter names, and examples. The
same eligibility check filters semantic candidates. Exact tool names,
namespaces, and full aliases remain explicit discovery requests.

Two object terms alone can still describe a read request, for example
`投递记录`. For non-exact requests the gate also requires an explicit action
term, or a request form without a read cue. This additional conservative
check is shared by every action tool; it is not a special case for career
memory. No example query was added or changed in this revision.

On the development set, top-1/top-3/top-5 recall is now **16/43, 30/43,
33/43**, compared with 17/43, 31/43, 35/43 before this gate. Control-case
`WRITE` results fell from **6 to 2**. `帮我看看我的经历`, `看看我的面试安排`,
`我的投递记录有哪些`, and `这个岗位不错` return no `WRITE` or `propose_*`
tool in the top five, even when synthetic semantic scores favor those tools.
`我想把这个岗位加入投递` still returns `create_application` in the top five.
All 141 exact aliases remain retrievable. The career-memory read recall was
not tuned with new examples; its behavior should be checked with model-written
queries in the next phase.

**Neither holdout was rescored for this gate.** All holdout metrics earlier
in this report remain historical observations for `0c07bce`, not claims about
the current index.
