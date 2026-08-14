# CTC suite: 22 long-context tasks, ladders from 2k to 1M tokens

Each task puts a corpus of N documents in-prompt and asks a question whose difficulty scales with
how much of the corpus must be tracked *simultaneously* — from O(N) retrieval, through O(N²)
relational tasks (find every contradicting pair), to O(NM)/O(N³) structural ones (cluster
everything, find planted triples). Every task has a context ladder; a rung label is the measured
median prompt length through the reference prompt path, not a document count.

## Quickstart

```bash
export HF_TOKEN=...        # needs read access to PrasannSinghal/ctc-suite-eval (private)

# preview without a model
uv run olmo-eval run -m mock -t ctc_contradiction:r32k --dry-run

# one task, one rung
uv run olmo-eval run -m <model> -t ctc_nq:r64k --save-predictions

# suites
uv run olmo-eval run -m <model> -t ctc:figure     # all 22 tasks, 2k-32k grid (108 runs)
uv run olmo-eval run -m <model> -t ctc:r128k      # every task at 128k (one x-axis column)
uv run olmo-eval run -m <model> -t ctc:oolong     # one task's whole ladder
uv run olmo-eval run -m <model> -t ctc:xlong      # everything above 32k (69 runs)
uv run olmo-eval run -m <model> -t ctc            # all 177 task x rung combinations
```

A bare task name (`-t ctc_nq`) evaluates the 32k rung. Suite aggregation is DISPLAY_ONLY on
purpose: the metrics are heterogeneous (f1, pair f1, kendall tau, ce_pos_recall, partial credit)
and a cross-task average would be meaningless.

## The 22 tasks

| task | class | metric | ladder top | notes |
|---|---|---|---|---|
| ctc_fiqa, ctc_nq, ctc_hpqa, ctc_msmarco, ctc_scifact, ctc_obliq, ctc_niah | O(N) | f1 (gold ids) | 1M (msmarco 512k) | retrieval family |
| ctc_rerank | O(N) | **ce_pos_recall** | 512k | see metric note below |
| ctc_oolong | O(N) | partial credit | 1M | aggregate questions over a line stream |
| ctc_outlier_amzn, ctc_outlier_fixedm | O(N) | f1 (set) | 1M / 512k | fixed-K controls |
| ctc_outlier | O(NM) | f1 (set) | 1M | K grows with N (~n/9) — the scale-K row |
| ctc_qdmatch_fiqa/nq/hpqa | O(N²) | pair f1 | 512k / 1M / 256k | query↔document matching |
| ctc_contradiction | O(N²) | f1 (pairs) | 1M | PubMed claims, IID realistic mode |
| ctc_absence, ctc_xabsence | O(N²) | f1 (set) | 16k / 32k | deletion / exact-copy-orphan detection |
| ctc_strmatch | O(N²) | f1 (pairs) | 32k | planted shared word-runs |
| ctc_reorder | O(N²) | kendall tau | 16k | restore reading order |
| ctc_grouping | O(NM) | pairwise f1 | 32k | cluster abstracts |
| ctc_textgroups | O(N³) | f1 (groups) | 32k | planted feature-sum triples |

Ladder tops below 1M are **source-corpus ceilings, not laziness** — e.g. qdmatch_hpqa exhausts all
4,000 labeled HotpotQA units at 256k; absence/reorder are bounded by contiguous-book length. Each
cap is documented on its RosterRow.

## Numbers that must travel with results

- Rungs ≥256k hold **125 examples** (seeded subsample; SE ≈ ±0.041 at f1≈0.7). `ctc_scifact` is
  300 and `ctc_obliq` 126 at every rung. Everything else is 500. Quote sizes inline.
- **rerank's metric is ce_pos_recall**: the fraction of documents with cross-encoder score > 0
  (median 3, p90 5 per example) present in the model's first 10 emitted ids. Single-qrel MRR@10
  saturates at ~0.98 and is emitted only as a secondary. Relevance in this data is bimodal
  (nothing between CE −5 and 0), which is also why an NDCG@10 over CE gains would collapse to
  the top-3 — measured before this metric was chosen.
- `ctc_parse_ok` is stored per output: a parse-rate collapse is a decoding/stopping regression
  wearing an accuracy drop's clothes. Check it before believing a low score.
- Contexts ≥256k exceed most models' native windows; the serving side (YaRN etc.) is the caller's
  responsibility and belongs next to any reported number.

## Design and provenance

- Prompt templates, parsers, metrics, gold-index conventions and stop rules are **vendored
  byte-faithful** under `_vendor/` from the `ctc` package (AI2 OLMo-core branch `prasann/ctc`),
  where they are golden-fixture-tested against the implementation that produced the suite's
  published numbers. Fix upstream and re-vendor; do not edit `_vendor/` (it is ruff-excluded to
  stay diffable).
- Scoring mirrors the reference runner call-for-call: stop-rule cleanup →
  `spec.parse(text, n_docs)` → `spec.score(parsed, gold)` with the spec's declared gold field.
- Gold conventions are pinned by test: the pair family stores 1-based indices, the retrieval
  family 0-based, and answering with the wrong base scores zero silently — the class of bug the
  vendoring exists to prevent (`tests/evals/tasks/test_ctc_suite.py`).
- Data: `PrasannSinghal/ctc-suite-eval` (private HF; parquet, one config per task, one split per
  rung). `CTC_SUITE_DATA_ROOT=/path` substitutes a local `<subset>/rung_<tokens>.jsonl` tree at
  load time. Hub copies strip builder-metadata keys and serialize the free-form `meta` dict to a
  JSON string (schema stability across splits); grading reads neither.
- Data generation, ladder builders, and per-rung realized-token measurements live in the OLMo-core
  working repo (`debug/ctc_1m_ladders/REPORT.md` is the build provenance record).
