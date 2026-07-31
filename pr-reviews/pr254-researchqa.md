# PR 254 review — Add ResearchQA scholarly-QA benchmark

**Branch:** `yilun/researchqa` → `main`
**Files:** `src/olmo_eval/evals/tasks/researchqa.py` (new), `tests/evals/tasks/test_researchqa.py` (new)
**Verdict:** Two new files, no changes to shared code, so the blast radius is contained to whoever runs `researchqa`. The port of the official coverage scorer looks faithful and the judge wiring checks out. No blocking issues.

Verified against the codebase:
- `build_openai_judge_fn` accepts `model`, `scorer_name`, `max_tokens`, `temperature`, `reasoning_effort` (`llm_judge.py:93`) — all args used here are valid.
- The default judge `gpt-5.4-mini` is a recognized id in the repo's judge ladder (`citation_validation.py:327`), so the default resolves.
- `reasoning_effort=(effort if separator else None)` correctly yields `None` for a bare `gpt-5.4-mini` and `"medium"` for `gpt-5.5:medium`.
- Per-answer aggregation: `_scores_from_items` divides `sum(item_scores)` by `len(rubric)`, and `item_scores` always has one entry per rubric item (failed batches extend with `0.0`), so the denominator is right even with parse failures.

## Must fix before merging

- None.

## Could fix if time

- **`researchqa.py` `build_researchqa_judge_fn` line ~146 (`max_tokens=4096`).** Large for a judge whose entire expected output is a newline-separated list of five-word labels. As the inline comment acknowledges, a reasoning judge can spend the budget and truncate the label list, which then fails `parse_coverage_labels` (count mismatch) and scores the whole batch 0.0. That is the intended fail-safe, but 4096 is generous enough that a flaky reasoning judge could silently zero batches on long rubrics. Consider a tighter cap for non-reasoning judges, or surface the per-batch failure rate more prominently than a single aggregate warning.

- **`researchqa.py` `_score_single` retry loop lines ~420–426.** Up to 3 attempts per batch with `2**attempt` sleep, all sequential per response. With batches of 8 over 776 mini instances this is a lot of serial judge calls on the failure path. Fine for correctness; just a throughput note if parse failures turn out common with a given judge.

## Small nit but probably fine

- **`researchqa.py` `ResearchQAScorer.score` lines ~172–173.** Reads `output.metadata.get("rubric_coverage")`, but the task stores per-output values under `output.metadata["score:rubric_coverage"]`. This looks like a mismatch, but it is inert: it mirrors the existing `SQAJudgeScorer` placeholder idiom, and pairwise analysis reads `response.scores[metric.name]` first (`metrics/base.py:52`), which the task *does* populate — so the scorer's `score()` is never the source of truth. No action needed; noting it so a future reader doesn't "fix" it into a real code path.

- **`researchqa.py` per-type metrics (`CoverageTypeMetric.compute`, lines ~209–216).** Pools rubric items across answers (item-weighted), while the primary metric is an unweighted per-answer mean. This divergence is deliberate and documented in the module docstring; just be aware the per-type tiers and the headline number aggregate differently.

- **Independent of #253.** `retrieval_date_cutoff` is set in `process_doc` and is a harmless no-op without the #253 harness change, and the prompt's "Prefer sources on or before {date}" is a soft instruction that stands alone. Good decoupling.
