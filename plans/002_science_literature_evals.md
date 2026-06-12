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

### 3. LitSearch — agentic, low-lift — DONE

`src/olmo_eval/evals/tasks/litsearch.py`, registered as `litsearch`. Agentic
definition, not the published Recall@5/@20 (the harness has no retrieval / ANN /
Recall@k infrastructure; that would be a separate retriever project).

- HF `princeton-nlp/LitSearch`, config `query`, split `full`, 597 queries. Gold
  is Semantic Scholar `corpusids`.
- The model is given the `semantic_scholar_snippet_search` tool and searches the
  live S2 API. Scoring reads `response.trajectory`: parse corpus IDs from tool
  results and intersect with the query's gold IDs. `found_rate` = queries with
  >=1 gold ID surfaced; `gold_recall` = mean fraction of gold surfaced. Primary
  is `found_rate`.
- Exact ID matching (not title fuzzing): added `corpusId` to the S2 query
  `fields` and a `Corpus ID:` line per result in `harness/tools/search.py`
  (additive; the tool is shared by the `dr_tulu` preset etc., input schema
  unchanged).

Constraints (stated in the task docstring):
- Only produces signal under a tool-providing agentic harness (a scaffold that
  executes tool calls, with `semantic_scholar_snippet_search` available, e.g.
  `dr_tulu`). Run without tools, the trajectory is empty and every query scores
  zero (logged as a warning).
- Measures the live S2 API + agent loop, not retrieval over the fixed LitSearch
  corpus. NOT comparable to published LitSearch numbers.

Suite wiring: `science:research` only. Deliberately NOT in `science:judge` /
`science:nojudge` / `science:all` — it needs tools, not a judge, so it does not
fit the judge/nojudge execution split and would score zero in a routine
`science:all` run.

Tools are selected at the harness level (`HarnessConfig.tools`), not per-task, so
the task cannot attach its own tool; the run must use a tool-providing harness.

### 4. DeepScholar-Bench — track, not hillclimb — SKELETON (unvalidated)

External sandboxed eval (mechanism #3). A registered skeleton is committed; full
details and open items live in `plans/003_deepscholar_bench.md`.

- `src/olmo_eval/evals/external/benchmarks/deepscholar/` — `DeepScholarExternalEval`
  (`SandboxedExternalEval`, modeled on tau2), registered as `deepscholar_bench`.
- Pins the repo's shipped dataset snapshot rather than the live arXiv pipeline,
  which resolves the reproducibility concern (no URL rot). Live refresh is out of
  scope.
- Model under test plugs in via the LOTUS generation config (litellm
  `hosted_vllm/<model>` + api_base). Judge stays gpt-4o; needs OPENAI + TAVILY keys.

NOT validated: never run end to end (no Docker/GPU/keys at authoring time). The
LOTUS config schema, results.csv schema, eval/mode enums, and limit flag are all
unconfirmed (see plan 003). Needs a beaker run to validate and iterate; not wired
into any suite until then. Expect possible null signal for an early OLMo (may not
drive the pipeline at all), consistent with track-not-hillclimb.

## Cross-cutting

### Judge cost / variance — decide before several mechanism-#2 tasks run together

Every mechanism-#2 task hardcodes a `gpt-4o-mini` judge. ExpertQA alone is
2,177 generations x several judge calls each. With more landing, judge cost and
run-to-run variance become a recurring tax. Make a deliberate call on judge model
+ response caching before three or four of these run together.

## Scorer validation (the gate)

The strategic review's top point: the shared scorer must not become an
unvalidated oracle that rewards citation-shaped prose. Validating it against
human citation-faithfulness labels is the gate before adding more judged tasks
(source assessment, benchmark-review.md:328). First step shipped:

- `src/olmo_eval/common/scorers/citation_validation.py` — adversarial cases
  (supporting control, topical-but-non-supporting, title-only half-credit,
  citation-stuffed, shuffled, uncited) with known-correct expected scores. Two
  layers: a deterministic oracle-judge layer proving the scoring pipeline
  penalizes bad citations GIVEN a correct judge (CI:
  tests/core/test_citation_validation.py, includes a guard that an
  over-permissive judge fails), and a real-judge kill test
  (`python -m olmo_eval.common.scorers.citation_validation`, needs OPENAI_API_KEY)
  testing whether gpt-4o-mini IS a correct judge on these cases.

Real-judge ladder run (--repeat 5) and the resulting decision:
- gpt-4o-mini (the old hardcoded judge) FAILS the score-inflating cases
  (topical_non_supporting, shuffled_citations) reliably -> retired.
- gpt-5.5:medium is the cheapest config that passes all six cases 6/6 across 5
  runs. Now the default judge, via `build_default_judge_fn` in llm_judge.py;
  ExpertQA and AstaBench SQA use it. Override with `$OLMO_EVAL_JUDGE` (e.g.
  `gpt-5-mini` for cheap iteration, reliable on the inflating cases and losing
  only the conservative stuffed-recall edge; or `gpt-5.5:high`).
- Pareto frontier and per-model notes are in citation_validation.py near
  DEFAULT_JUDGE_LADDER.

Still open (needs people / data):
- Human-agreement study: 50-100 labeled ScholarQA/ExpertQA outputs across 2-3
  models; measure scorer-human agreement, ranking stability, judge variance.
  Scaffolded (`LabeledCitationExample`, `AUDIT_SET`, `citation_scorer_agreement`)
  but unpopulated.

Review-driven hygiene (not yet done): stop hillclimbing on `global_avg` /
geomean (report scorer tiers separately); relabel ExpertQA as
attribution/on-topicness and agentic LitSearch as a retrieval smoke test.
Deferred bigger items: fixed-corpus LitSearch Recall@k (clean C1), a non-MCQ
physical-science / figure-table eval for breadth.

### Deferred: promote shared attributed-QA helpers (review item)

`PRECISION_EVAL_PROMPT`, `compute_precision_score`, `format_report`, and the
score-reading metric classes are now shared by two tasks via importing from
`astabench_sqa`, making it a de facto shared module. Acceptable for two
consumers. When a third lands that needs them (likely DeepScholar's precision /
generation path), lift precision + generation + parsing into a `common/scorers`
attributed-QA module in one focused move. Not worth doing speculatively;
agentic LitSearch will not touch these.
