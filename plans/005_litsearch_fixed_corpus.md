# LitSearch fixed-corpus retrieval + reranking

Closes the gap in plan 002 step 3: the current `litsearch` task is an agentic
smoke test over the live Semantic Scholar API (does a gold paper surface in tool
results), NOT the published fixed-corpus Recall@k. It measures the agent loop +
live API, scores zero without a tool harness, drifts as S2 changes, and never
scores the model's actual selection.

Decision (user): build **both** — a reproducible retriever baseline AND a
model-as-reranker eval on the same fixed corpus.

## Status (code done; one offline build + a GPU run pending)

DONE and verified locally (no GPU, no corpus download):
- `scripts/internal/build_litsearch_pools.py` — BM25 build over `corpus_clean`,
  reports retriever Recall@5/@20 + the rerank-pool ceiling, writes the
  self-contained pool artifact. `bm25s` API usage smoke-tested on a toy corpus.
- `src/olmo_eval/evals/tasks/litsearch_rerank.py` — `litsearch_rerank` task +
  `RecallAtKMetric` (recall@5 primary, recall@20). Registered via autodiscovery.
- Wired into `science:research` + `science:nojudge` (so it flows into
  `science:all`); kept out of `science:judge`. Verified via `expanded_tasks`.
- `tests/evals/tasks/test_litsearch_rerank.py` — 18 tests (parse, recall@k
  cutoff, process_doc, format_request, score_responses dedup/out-of-range). All
  pass; full suite/registry/science regression green (110 tests).
- `uv add bm25s`; ruff + ty clean.

PENDING (needs resources this laptop lacks):
1. Run the build once on a faster connection or beaker to download
   `corpus_clean`, produce `litsearch_rerank_pools.jsonl`, and record the BM25
   Recall@5/@20 baseline. Commit/HF-host the artifact. Pin the HF revision.
2. Add the retriever-baseline regression test once the real number is known
   (assert BM25 Recall@5 in its measured band).
3. Beaker run for real OLMo Recall@k (the task scores zero-shaped until the pool
   artifact exists; until then `instances` raises a clear "build the pools"
   error by design).

## Source

LitSearch (Ajith et al., EMNLP 2024, arXiv:2407.18940). HF
`princeton-nlp/LitSearch`:
- `query` (597 queries, gold `corpusids`, specificity/quality tags) — already used.
- `corpus_clean` (64,183 papers: title, abstract, citation IDs) — the fixed
  corpus we index. Frozen on HF, so pinning a revision gives reproducibility.

Published anchors for sanity checks: BM25 ≈ 50% R@5; dense (GritLM) 74.8% R@5;
+4.4% with GPT-4o reranking.

## Architecture

Two artifacts, one shared offline build.

### A. Offline build script — `scripts/internal/build_litsearch_pools.py`

Run once, output committed/pinned; not run at eval time.

1. Load `corpus_clean` (pin the HF revision).
2. Build a BM25 index over title + abstract (dep: `bm25s`, pure-CPU, fast at
   64K docs; `rank_bm25` is the fallback).
3. For each of the 597 queries, retrieve the top-N ranked candidates (N=100).
4. Compute and log the **retriever baseline**: BM25 Recall@5 / Recall@20 over
   the full corpus ranking (expect ~0.5 R@5 — the guard band).
5. Write a compact pinned artifact `litsearch_rerank_pools.jsonl`:
   `{query_id, query, gold_corpusids, candidate_corpusids}` — ranked candidate
   IDs only, no text (keeps it <1 MB and committable). Abstracts are re-joined
   from `corpus_clean` at task load, so the corpus stays the single source of
   truth.
6. Dense/ANN retriever is a **follow-on** (needs an encoder + FAISS; better as a
   one-off beaker embedding pass). BM25 ships first; the script leaves a hook to
   add a dense ranking column to the same artifact.

The retriever Recall@k is reported by the build script and frozen by a
regression test (below); it is not a harness `Task` because the model under test
is not involved.

### B. Reranking task — `litsearch_rerank`

`src/olmo_eval/evals/tasks/litsearch_rerank.py`. The existing agentic `litsearch`
stays as-is (distinct task, distinct signal); this is additive.

- Data source: the committed `litsearch_rerank_pools.jsonl`. On first
  `process_doc`, lazy-load `corpus_clean` into a `{corpusid: (title, abstract)}`
  lookup cached on the task instance, and join the candidate text.
- `process_doc` -> `Instance`: stash `gold_corpusids` and the ordered candidate
  list (id, title, abstract) in metadata.
- `format_request` -> CHAT: a numbered candidate list (title + abstract
  truncated to a fixed char budget) and "select the k most relevant papers,
  most-relevant first; output their numbers as JSON". Default pool size in the
  prompt is capped (e.g. 50) to stay within context; configurable via variant.
- `extract_answer`: parse the JSON list of selected candidate numbers -> the
  corresponding corpus IDs (tolerant parsing, like the SQA path).
- Scoring (deterministic, no judge, no tools): Recall@k of the model's selected
  set against gold. Custom `RecallAtK` metric(s) reported at k=5 and k=20.
  Because the model returns a set/ranking, "Recall@k" = fraction of gold in the
  model's top-k selection. Report both; primary = Recall@5.

This addresses every gap: fixed reproducible corpus (no live API), a real top-k
cutoff, and it scores the model's actual selection (discernment), not what a
tool surfaced.

### Metrics

`RecallAtKMetric(k)` in a task-local module (mirrors LitSearch's existing
`_LitSearchMetricBase` pattern: precompute per-instance recall in
`score_responses`, average in `Metric.compute`). Overrides
`supports_pairwise_scorer_fallback() -> False`. Two instances: k=5, k=20.

### Suite wiring

- Add `litsearch_rerank` to `science:research`.
- It is judge-free and tool-free, so unlike agentic `litsearch` it CAN go in
  `science:nojudge` / `science:all`.
- Document in `science.py` what it measures (model reranking Recall@k over a
  fixed corpus) vs the agentic `litsearch` (live-API retrieval smoke test).

## Tests

- **Retriever baseline regression** (`tests/.../test_litsearch_pools.py`):
  assert BM25 Recall@5 over the corpus is within an expected band (~0.45-0.55),
  guarding the build. Runs against a small committed fixture or a marked
  slow/integration test if it needs the full corpus.
- **Reranker scoring unit test**: oracle selection (all gold) -> Recall@k = 1.0;
  empty/wrong selection -> 0.0; partial -> correct fraction. Pure logic, no
  model, no network.
- **Pool construction test**: artifact rows well-formed; candidate lists are the
  configured length; gold IDs present in the pool whenever the retriever ranked
  them (documents the recall ceiling the model can reach).

## Boundaries / what needs the user

- BM25 build + pool generation + all scoring/tests run on CPU locally; I can
  build and validate these end to end with oracle/mock selections.
- Actual OLMo Recall@k requires a **beaker run the user launches** (no local
  GPU); I validate the harness path with a mock model.
- Dense/ANN retriever (the stronger baseline) is deferred to a follow-on with an
  encoder; BM25 ships first and is enough to close the reproducibility gap.

## Dependencies

`uv add bm25s` (pure-Python/numpy, CPU). Dense follow-on would add
`faiss-cpu` + an encoder; not in v1.

## Sequence

1. `uv add bm25s`; write `build_litsearch_pools.py`; download `corpus_clean`
   (pin revision); build BM25; log retriever R@5/@20; emit + commit
   `litsearch_rerank_pools.jsonl`.
2. Implement `litsearch_rerank` task + `RecallAtKMetric`; register it.
3. Wire into `science:research` (+ `science:nojudge` / `science:all`); document
   the split from agentic `litsearch` in `science.py`.
4. Tests: retriever baseline regression, reranker scoring, pool construction.
5. Lint + type + unit tests. Smoke-run the task path with a mock model to
   confirm prompt build + selection parsing + Recall@k wiring; hand off to a
   beaker run for real OLMo numbers.
