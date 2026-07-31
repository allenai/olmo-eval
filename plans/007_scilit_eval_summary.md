# Sci-lit eval sweep: results summary charts

Status: PROTOTYPE. Script runs end-to-end against a snapshot of the tracking
sheet. Numbers are provisional — the sweep is ~49% complete and four scores are
flagged suspect pending a Beaker check.

## What this is

A repeatable way to turn the sci-lit eval tracking sheet into a handful of
slide-ready charts. Not a one-off deck: results are still landing (the Qwen
DeepScholar runs are being redone), so the deliverable is a script that can be
re-run against a fresh pull and regenerate everything.

Source sheet: `1mfqNt9QoZlNz08GwpAlB0ESr90G2QE7SjUkLnWfF6sA` ("SciLit Eval
Matrix"). The first tab is long-form `Model Name | Eval Name | Metric | Score |
Beaker Run ID | Notes`, 9 models x 9 evals = 81 rows.

## Data flow

    sheet -> scripts/analysis/data/eval_matrix.csv   (manual/assisted pull)
          -> scripts/analysis/metadata.toml          (checked in, hand-maintained)
          -> scripts/analysis/eval_summary.py
          -> scripts/analysis/out/*.{html,png} + summary.md

The script declares its own dependencies with PEP 723 inline metadata, so
`uv run scripts/analysis/eval_summary.py` builds an isolated environment and
the project's `pyproject.toml` and lockfile stay untouched. Charting libraries
have no business in the eval package's dependency graph.

The CSV export endpoint returns only the sheet's first tab and needs auth, so
the pull is not part of the script. Two options, either fine:

- Ask Claude to refresh the CSV via the Drive integration (one call).
- `gspread` with OAuth, ~20 lines, one-time browser consent. Worth adding only
  if the script needs to run unattended.

The two reference tabs (model -> HF ref, eval -> olmo-eval ref) are static
metadata, so they live in `metadata.toml` rather than being re-fetched. That
file also carries what the sheet cannot: the sci-lit vs sentinel split, display
ordering, and the suspect-run flags.

## Row status

"Missing" is not one state. The script derives four, and the coverage chart
shows all of them:

| status | condition | meaning |
|---|---|---|
| `ok` | score + run id | usable |
| `suspect` | flagged in `metadata.toml` | score exists but is probably a harness artifact |
| `no-provenance` | score, no run id | real number, cannot be traced back to a run |
| `not-run` | no score | not yet attempted |

A cell with a run id but no score folds into `not-run` — the run id is treated
as a transcription slip rather than as evidence a run happened. Only one cell
sits there today (Qwen3.5 9B / ExpertQA), which did not pay for a category of
its own. Revisit if failed-but-recorded runs become common.

`no-provenance` stays separate: OLMo-3 7B Instruct has three scores with no run
id, and those numbers still rank.

`ok` and `no-provenance` both feed ranks and composites — a missing run id is a
traceability gap, not a validity gap. Withholding them was the first thing
tried and it silently dropped OLMo-3 7B Instruct, the reference baseline, out
of the summary entirely on the strength of three untraceable-but-real scores.

`suspect` rows are the only ones excluded. They are still drawn, in grey, with
the reason attached — dropping them silently would hide that a model was
attempted. Since a suspect score often rounds to ~0 and draws no visible bar,
the value label carries the word "suspect" rather than relying on bar color.

## Suspect scores (all unverified)

Flagged from inspection, not yet confirmed against the runs:

- Olmo-3 7B Think — ExpertQA 0.008, LitSearch-rerank 0.000. Near-total failure
  on tasks its Instruct sibling handles; smells like thinking-trace extraction.
- Qwen3.5-35B-A3B — IFEval 0.050, against 0.377 for Qwen3.5 9B. A 35B model at
  one eighth its 9B sibling is a harness bug, not a capability gap.
- GLM-4.1V-9B-Thinking — LitSearch-open 0.000. The sheet note says the task
  scores zero without `semantic_scholar_snippet_search`. The same note sits on
  Qwen3.5 9B's 0.350, so one of the two is misattributed.
- Gemma4 26B-A4B — MMLU 0.557, below OLMo-3 7B's 0.642.

Resolving these needs the Beaker logs; every one has a run id. Until then the
flags carry `verified = false`.

## Scoring

Per-eval bars show raw scores. Each panel has its own axis, so the SAGE-open
(0.02-0.08) vs MMLU (0.56-0.85) magnitude gap never collides.

Cross-eval aggregation uses percentile rank within eval, not min-max. Min-max is
unstable while runs are landing: one new model setting a new max retroactively
moves every other bar, so last week's slide won't reproduce for reasons
unrelated to the models. Rank moves too, but it reads as ordinal, so nobody
mistakes it for a score.

Sentinels (IFEval, MMLU, MATH-500) are never averaged into the sci-lit
composite. Per the sheet's own eval tab they are regression monitors, not the
objective.

Composites are only computed for models clearing `--min-coverage` sci-lit evals
(default 5 of 6). Below that, a composite over 1 eval and one over 6 are not the
same quantity and ranking them together is misleading. Excluded models are
listed by name in `summary.md` rather than dropped.

Rank resolution is the current weak point, and it is a coverage problem rather
than a method problem. Each eval has only 3-6 usable results, so a percentile
rank can take only 3-6 values and the profile lines swing between 0 and 1 on
small score differences. The per-eval `n` is printed on the profile axis for
exactly this reason. It also means the present top three (0.611 / 0.606 /
0.600) are a three-way tie, not a ranking — the summary chart says so in its
subtitle.

## Charts

1. `coverage.png` — model x eval grid, five statuses, score in-cell. Doubles as
   the run queue.
2. `scores_sci-lit.png` — 6 faceted bar panels, raw score, fixed model order,
   suspect bars greyed with reason.
3. `scores_sentinel.png` — same, 3 panels.
4. `profile.png` — percentile rank across sci-lit evals, one line per
   sufficiently-covered model, direct-labelled.
5. `summary.png` — mean sci-lit percentile rank, `n=k/6` on every bar, with a
   sentinel panel alongside.

Everything is folded into `out/index.html`, a 7-slide deck with the PNGs
inlined as data URIs — no network, no sibling assets, opens over `file://`.
Arrow keys or space to advance. A title slide carries the coverage stat tiles
and a final slide carries the standing, suspect and needs-attention tables.
`@media print` unhides every slide with page breaks, so browser print-to-PDF
gives a shareable copy. Individual `.html` charts keep their Vega tooltips for
anyone who wants to hover the numbers.

Colors follow the `dataviz` skill: status palette for the coverage grid, single
hue plus grey for the per-eval bars (one series, emphasis), the validated
4-slot categorical for the profile lines. Light mode only for now; the palette
is a dict at the top of the script, so dark is a swap rather than a rewrite.

## Deferred

- Beaker check on the four suspects.
- DeepScholar-Bench sub-metric breakdown. Only one row carries a metric name
  (`geomean_fixed`), but the components (organization, nugget_coverage,
  reference_coverage, cite_p) exist upstream and would make a good extra panel.
- Dark mode steps.
- `gspread` fetch, if unattended runs are wanted.

## Note

An earlier attempt added the charting libraries to the `analysis` extra with
`uv add`. That re-resolved the whole lockfile as a side effect — downgrading
numpy 2.4.4 -> 2.2.6, rich 14.3.4 -> 13.9.4 and textual 8.2.4 -> 6.2.1, none of
which the new packages depend on. `uv remove` plus a later sync put both
`pyproject.toml` and `uv.lock` back; neither shows a diff now. Worth knowing
that `uv add` in this repo can move unrelated pins.
