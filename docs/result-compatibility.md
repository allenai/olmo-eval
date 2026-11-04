# Result compatibility log

This log records merged changes that alter what an existing task spec or suite measures,
so a result stored before the change can be identified and interpreted. New tasks and new
variants do not need an entry. See
[Changing or retiring an existing spec](../CONTRIBUTING.md#changing-or-retiring-an-existing-spec)
for when a change is allowed in place.

Add an entry in the PR that makes the change, newest first. Each entry states:

- **Affected**: task specs, suites, or shared code paths, including suite membership or
  aggregation changes.
- **Change**: what is measured differently.
- **Identity**: whether the task hash changes. When it does not, results from before and
  after share a hash and can only be told apart by commit or date.
- **Comparability**: which earlier results remain comparable and what replaces the old
  behavior, if anything.

## 2026-11-03: Auditable IFEval per-question exports

- **Affected**: predictions produced with the four IFEval metrics, including `ifeval`,
  `ifeval_ood` (IFBench), and all `ifeval_mt_*` specs and their variants. Consumers of
  those predictions, including uploaded per-instance metrics, inherit the old error.
  Multi-IF uses separate metrics and is not changed.
- **Change**: all four exact metric keys previously inherited the `ifeval` scorer's
  prompt-level loose scalar. They now use their own strict/loose instruction outcomes.
  Predictions retain the boolean lists and any verifier construction errors in
  `model_output[i].scorer_results.ifeval`. List order follows the request's
  `doc.instruction_id_list`. Like the aggregate metrics, per-question metrics use
  output zero. No verifier, extraction, or aggregate calculation changes.
- **Identity**: task hashes are unchanged. Identify fixed exports by evaluator revision
  and the retained instruction evidence, not by the presence of an exact metric key:
  old exports already contained incorrectly labeled exact keys.
- **Aggregation**: prompt accuracy is the fraction of questions with a nonempty list
  whose entries all pass. Instruction accuracy is the total number of passing entries
  divided by the total number of entries. Do not average per-question instruction
  accuracies without weighting by list length: scores of 1/2 and 1/1 pool to 2/3,
  not 3/4. An explicitly empty list contributes zero instructions and a failed prompt;
  an all-empty instruction population scores zero, matching the aggregate metric.
- **Missing evidence**: fresh responses without the relevant list omit that metric;
  only prompt-level loose may use the scorer scalar. Missing or malformed lists are
  unavailable, not zero. Calling `augment_prediction_instance_metrics` on an older
  prediction recomputes IF values from retained lists if present; otherwise it removes
  unsupported strict/instruction keys, including old exact-looking keys. A scalar-only
  exact key cannot distinguish a valid value from the historical alias. The valid loose
  prompt value remains usable. This helper changes the supplied object only: it does
  not migrate existing files or database rows, and uploading an old file does not repair it.
- **Comparability and recovery**: valid saved aggregate scores remain comparable.
  Historical strict per-question gains, regressions, and paired uncertainty calculated
  from the aliased values are invalid. Original scored `Response` metadata or retained
  per-instruction verdicts can reconstruct all four metrics exactly without verification.
  Scorer scalars alone cannot recover strict outcomes or partial instruction success.
  Keep original artifacts immutable and write any repaired export separately.

Rechecking saved answers is a new measurement unless the original verifier state can be
reproduced. In the mixing baseline (`01M6C80EK9Y2BT5RM131DCFWF1`), the completed attempt
for `ifeval_mt_ood_wildchat_unused_withRewrite` has 1,387 rows: every exported IF metric
averages 0.33813987022350395, while aggregate strict is 0.2581110310021629. All 28 checked
IF task/snapshot artifacts (28,014 rows) contain the alias and lack instruction verdicts.
Their saved loose prompt values reconcile; the existing answer reconstructions fully
reconcile in only 15 of 28 cases. Even full aggregate reconciliation does not prove each
historical strict label: opposite row errors can cancel.

CPU probes in the original evaluator image reproduce two sources of uncertainty:
IFBench `b281f8d4` randomly fills unspecified parameters (the multi-turn letter-frequency
rows omit `let_relation`), and its unseeded `langdetect` checks can vary even on the same
verifier and answer. Both explicit language instructions and English-capitalization
checks use that detector. Pinning the image and Python's random seed alone is insufficient;
runtime-downloaded NLTK resources are another environment dependency to record. These
probes establish replay variability, not a complete attribution of every historical
mismatch or a recovery of the original random choices. Prefer original scored metadata;
otherwise retain the reconciled loose analysis and keep reconstructed strict labels
separate and provisional. Do not choose seeds or parameters to fit aggregate totals.


## 2026-10-22: Scoring failures are left out and budgeted, not scored as zero (#492)

- **Affected**: every task run by `AsyncEvalRunner` whose scoring failed for some output:
  a scorer that raised, a judge with no verdict after its retries, or a lost or missing
  sandbox. Runs where every output was scored are unchanged.
- **Change**: a failed output used to be recorded as 0 and kept in the metrics. It now
  has no score. An instance with no scored output is left out of the metrics, and an
  instance with some scored outputs is scored from those, including pass@k and pass^k.
  Unscored instances and instances with no output share the failure budget
  (`max_hard_failure_rate`, 5% by default); a task over it, or with nothing scored,
  fails and publishes no metrics. The rule that one infrastructure scoring error (a
  judge that gave no verdict, a lost or missing sandbox) withheld the whole task is
  gone, so such a task now publishes when its failures fit the budget. Judge tasks no
  longer raise when every sample of an instance fails.
- **Identity**: task hashes are unchanged.
- **Comparability**: an earlier result is affected only if its predictions carry
  `scoring_errors`. Leaving out a zero raises a mean of non-negative scores: a
  higher-is-better metric that averaged those zeros in comes out better, and a
  lower-is-better one, such as an error rate, comes out worse. Metrics that already
  excluded failed outputs, such as the judged-task, omniscience and Arena-Hard metrics,
  are unchanged, except that a run with a few judge failures is no longer withheld.
  Counts that tasks keep of their own errors, such as omniscience's
  `judge_parsing_errors`, now count only failures that did not raise.

## 2026-10-16: `ctc_contradiction` scores its high-low gold pairs (#490)

- **Affected**: `ctc_contradiction` at rungs `r64k`, `r128k`, `r256k`, `r512k` and `r1m`,
  and the `ctc` suites that include them. Rungs `r2k` to `r32k` are unchanged.
- **Change**: gold pairs are sorted low-high when the data loads. On the affected rungs,
  87–91% of examples hold at least one high-low pair, and the vendored scorer raised on
  them. The runner recorded those examples as 0 on every metric, so a model's score was
  bounded by the few examples without such a pair. Those examples are now scored.
- **Identity**: task hashes are unchanged. An earlier result can be told apart by its
  predictions: the unscored examples carry a `ValueError` under `scoring_errors`, which the
  dashboard stores as `has_scoring_error` on each instance.
- **Comparability**: results for the affected rungs from before the change are not
  measurements and should be discarded. Results for `r2k` to `r32k` remain comparable.

## 2026-10-02: Chat-style `ANSWER:` lines in multiple-choice extraction (#433)

- **Affected**: generation variants that extract an answer letter with the shared MCQ
  extractor (`evals/extract/mcq.py`) or `ANSWER_LINE_PATTERN`, including GPQA, LAB-Bench,
  MedQA, BBQ, and WMDP. Log-likelihood variants are unaffected.
- **Change**: answers formatted with markdown emphasis, parentheses, or the letter on the
  next line (`**Answer:** B`, `Answer:\n\nC`) are now extracted instead of falling through
  to later patterns or scoring as unparsed.
- **Identity**: task hashes are unchanged.
- **Comparability**: earlier scores can differ for models that use those answer formats,
  since a later fallback pattern may have picked a different letter or none. Compare runs
  on the same side of the commit.

## 2026-09-24: MMLU-Pro suites use instance-weighted aggregation (#385)

- **Affected**: suites `mmlu_pro`, `mmlu_pro:bpb`, and `mmlu_pro:cot`.
- **Change**: suite aggregation changed from `AVERAGE` (equal weight per category) to
  `WEIGHTED_AVERAGE` (weighted by instance count). Per-task results are unchanged.
- **Identity**: task hashes are unchanged; only the suite-level number differs.
- **Comparability**: suite averages computed before the change are macro averages and are
  not comparable with later suite averages. Per-category task results remain comparable.

## 2026-09-24: `gsm8k` honors `num_fewshot=0` (#335)

- **Affected**: `gsm8k` runs with `num_fewshot` explicitly set to 0.
- **Change**: a zero-shot override previously still included the full few-shot prompt;
  it now includes no examples. Runs at the default or a positive few-shot count are
  unchanged.
- **Identity**: the stored config already recorded `num_fewshot: 0`, so before and after
  share a hash.
- **Comparability**: earlier "zero-shot" `gsm8k` results were few-shot results and are not
  comparable with later zero-shot runs.
