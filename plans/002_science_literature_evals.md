# Science-literature evals

Source assessment: `../dr-benchmark-review/benchmark-review.md`.

## Thesis

The leverage is a reusable citation-grounding scorer, not any single benchmark.
The benchmarks are delivery vehicles for it. Build the scorer once, then onboard
benchmarks that exercise it across domains and task shapes.

## Sequence and status

### 1. Extract citation scorer — DONE

Lifted the citation precision/recall machinery out of `astabench_sqa.py` into
`src/olmo_eval/common/scorers/citation.py`:

- `CITATION_GROUP_PROMPT`, `JUST_HAS_A_TITLE`, `clean_sentence`, the
  snippet-filter helpers, `compute_citation_scores_from_groups`,
  `score_citation_group`.
- New entry point `score_citations_for_sections(judge_fn, parsed_response)`,
  extracted from the old `AstaBenchSQA._score_citations` section/table walk.
- Exported from `common/scorers/__init__.py`. `astabench_sqa` delegates; behavior
  unchanged.

Pure refactor; parity verified by review. `common/scorers` does not import from
`evals/tasks` (clean layering).

### 2. ExpertQA — DONE

`src/olmo_eval/evals/tasks/expertqa.py`, registered as `expertqa`.

- HF `cmalaviya/expertqa`, config `main`, 2,177 rows, single train split. Keeps
  `field` / `specific_field` metadata for the 32-field breakdown.
- Generation into the JSON-sections cited-report format, then graded on
  `citation_precision`, `citation_recall` (shared scorer) and `answer_precision`
  (irrelevant-paragraph judge). `global_avg` is the mean of the three.
- Reuses generation prompt, parsing, precision judge, and score-reading metric
  classes from `astabench_sqa` (the `aime`->`minerva_math`, `naturalqs`->`drop`
  cross-task pattern).
- Wired into `science:research` and `science:judge` (flows into `science:all`
  via judge; absent from `science:nojudge`).

Design decisions locked:
- No `ingredient_recall`: ExpertQA has no per-question rubric.
- No reference grading: ExpertQA's annotations grade its own pre-generated
  answers, so they cannot grade a fresh model answer. Factuality-vs-reference is
  deferred (see open items).

### 3. LitSearch — agentic, low-lift — TODO

Decision: agentic definition, not the published Recall@5/@20. The harness has no
retrieval / embedding / ANN / Recall@k infrastructure, so the published metric
would be a separate retriever sub-project. Deferred.

Plan:
- Give the model the `semantic_scholar_snippet_search` tool; the task succeeds
  if the gold paper surfaces in the returned results. Metric is a found@k /
  hit-rate.
- Confirm the harness's agentic/tool-use path first (the `AstaExternalEval` /
  inspect_ai sandbox route is the likely vehicle; verify in-loop vs external).
- Docstring + metric name must state plainly: this measures the S2 API + agent
  loop, not the model's retrieval against gold corpus IDs. Not comparable to
  published LitSearch numbers.
- Wire into `science:research` (+ `science:judge` if tool/judge-dependent), with
  the matching `test_science.py` update. Add task tests.

Open before writing: gold-match criterion (S2 corpus-ID / arXiv-ID exact vs
title fuzzy) and `k` (tool returns top-5).

### 4. DeepScholar-Bench — track, not hillclimb — TODO

External sandboxed eval (mechanism #3): vendor/port the retrieval + synthesis +
verifiability harness, manage live arXiv-by-date-range refresh, parse results
back. Weeks, not days. Cheaper now that the citation scorer exists.

Caveats: live data is contamination-resistant but operationally
non-reproducible (URL rot, no fixed snapshot). The ~31% geomean ceiling means
floor effects for an early OLMo. Treat as a north-star tracker, not a gradient
source, until the model clears the floor.

## Cross-cutting

### Judge cost / variance — decide before several mechanism-#2 tasks run together

Every mechanism-#2 task hardcodes a `gpt-4o-mini` judge. ExpertQA alone is
2,177 generations x several judge calls each. With more landing, judge cost and
run-to-run variance become a recurring tax. Make a deliberate call on judge model
+ response caching before three or four of these run together.

### Deferred: promote shared attributed-QA helpers (review item)

`PRECISION_EVAL_PROMPT`, `compute_precision_score`, `format_report`, and the
score-reading metric classes are now shared by two tasks via importing from
`astabench_sqa`, making it a de facto shared module. Acceptable for two
consumers. When a third lands that needs them (likely DeepScholar's precision /
generation path), lift precision + generation + parsing into a `common/scorers`
attributed-QA module in one focused move. Not worth doing speculatively;
agentic LitSearch will not touch these.
