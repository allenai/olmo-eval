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

## 2026-11-03: SWE-bench Verified leaves harness failures out of its scores (#538)

- **Affected**: the `swe_bench_verified` external eval, for runs in which some instance
  hit a harness or infrastructure failure (`image_error`, `image_timeout`,
  `sandbox_error`, `agent_error`, `grading_error`). Runs without one are unchanged.
- **Change**: such an instance used to count as unresolved in `resolve_rate` and the
  other rates, and the run reported success unless every instance failed. It now has
  no score: the rates and per-repo and per-difficulty breakdowns leave it out, and
  `num_errors` counts it. The run fails when more than `max_error_rate` (0.05 by
  default) of its instances failed this way. `error_rate` no longer counts a reply that
  broke the tool-call protocol (`model_error`), which stays a scored instance.
  `resume_from` reruns harness failures instead of keeping them.
- **Identity**: the task hash, computed from `eval_name` and the `eval_args` passed, does
  not move, so results from before and after can only be told apart by commit or date.
  Passing `max_error_rate` explicitly changes the hash.
- **Comparability**: an earlier result whose `error_rate` is 0 is unchanged. One with
  errors understated `resolve_rate` by roughly that rate times the true resolve rate;
  dividing `num_resolved` by the number of instances without an error gives the new
  value, except where an errored instance's partial patch resolved and was counted.

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

## 2026-10-31: Empty scoring-text diagnostics

- **Affected**: `expertqa`, `researchqa`, their variants, and `geneturing_*` tasks.
- **Change**: `generation_counts.empty` uses the text remaining after each task's
  internal thinking removal. The `scoring_text_empty` flag is saved with predictions.
  Earlier empty counts may undercount reasoning-only outputs.
- **Identity**: task hashes and scores do not change; only diagnostics change.
- **Comparability**: quality scores remain comparable. Historical diagnostic counts
  are not automatically recomputed.

## 2026-10-31: Opt-in Harmony response parsing (#398)

- **Affected:** only runs explicitly setting `response_parser=harmony`; existing
  specs, suites and default parsing are unchanged.
- **Change:** structured Harmony transcripts score only the final-channel body;
  a recognized analysis trace without a final channel scores an empty answer.
  Inline vLLM retains reserved token delimiters for the opt-in parser; its default
  output is unchanged. Saved predictions keep the original decoded text as
  `original_text` and the delimited text as `text_with_special_tokens`.
  Already saved token-stripped `analysis...assistantfinal...` transcripts are not
  recoverable unambiguously.
- **Identity:** the parser setting is serialized and changes the task hash; absent
  settings preserve every existing hash.
- **Comparability:** parsed results are a different measurement from older runs
  that scored reasoning and answer together. Compare only runs with the same
  parser and decoding settings. This adds no numerical parity claim.

## 2026-10-26: Judge identity in the hash; no verdict leaves an output unscored (#512)

- **Affected**:
  - Safety: the `openai_judge`, `wg_judge`, `wg_judge_thinking` and `base` variants of
    `do_anything_now`, `harmbench`, `trustllm_jailbreaktrigger`, `wildguardtest`,
    `wildjailbreak` and `xstest`. For `strongreject`, the `openai_judge`, `sr_judge`,
    `sr_judge_thinking` and `base` variants. Also the safety suites that include them.
  - `simpleqa:judge`.
  - `omniscience:judge`.
  - `expertqa`, `expertqa:cite`, `astabench_scholarqa` and `astabench_scholarqa:test`.
  - `researchqa` and `researchqa:mini`, for saved predictions only.
  - `wmdp:mcq`, for empty subsets only.
  - Any run that sets `OLMO_EVAL_JUDGE`, or that passes `-o judge_*` to a judged task.
- **Change**:
  - **A judge reply with no verdict.**
    - Before: SimpleQA scored it as incorrect. The safety judges scored it 0. They also
      scored 0 for an answer outside the allowed ones, and for a StrongREJECT judge
      that returned no grade logprobs. ExpertQA and AstaBench SQA scored an unparseable
      precision reply as 1.0 and an unparseable ingredient reply as 0. They treated a
      citation section whose reply had no claims as empty. Omniscience graded it
      `PARSING_ERROR`. Its metrics left that grade out, but it did not count against
      the failure budget.
    - Now: an API judge is asked again. If no reply parses, the output is left
      unscored. It counts against the failure budget.
    - Exception: an empty safety response still scores 0. That score does not depend
      on the judge.
    - FrontierScience and ResearchQA keep scoring an unparseable reply 0, as their
      reference graders do.
  - **Removed metrics.** The 28 safety specs drop `parsing_error`. `omniscience:judge`
    drops `judge_parsing_errors`. Both would now always read 0. The run's scoring
    failures count these replies instead.
  - **Subset metrics.** They leave out unscored responses. An empty subset reports 0
    instead of -1.
  - **Safety judge checkpoints.** The `wg_judge`, `sr_judge`, `*_thinking` and `base`
    variants set `judge_model` to the model their judge provider must serve. That is
    `allenai/wildguard` for WildGuard, and the `qylu4156/strongreject-15k-v1` adapter
    for StrongREJECT. The runner stops before generating if the provider serves another
    model.
  - **API judges in config.** The safety `openai_judge` variants and `simpleqa:judge`
    set `judge_model` to `gpt-4o-mini` and set `judge_max_tokens`. `-o judge_*` now
    changes the judge that grades them.
  - **`OLMO_EVAL_JUDGE`.**
    - Now: it is applied when the task is built. It applies to HLE, MultiChallenge,
      MT-Bench, Arena-Hard, FrontierScience, ExpertQA, AstaBench SQA and ResearchQA. It
      does not apply when `-o` names a judge model. It accepts
      `model[:effort[:max_tokens]]`.
    - Before: an explicit `-o judge_model` equal to a task's default lost to it.
      FrontierScience let it override `-o`. ExpertQA, AstaBench SQA and ResearchQA read
      it while scoring.
  - **`-o judge_*` on `omniscience:judge`.** Before, the default judge graded the run,
    while the config recorded the override. Now the overridden judge grades it.
  - **Retries.** Judge scorers and the attributed-QA tasks retry a judge call that fails
    for a transient reason. Before, they made one attempt. The attributed-QA tasks now
    share the run's judge concurrency limit.
  - **Saved predictions.**
    - Safety: each sample keeps its judge result in `model_output[i].judge_result`.
      `is_empty_response`, `is_truncated_response` and `is_parsing_error` are inside it.
      Before, one result per instance was saved at the top level of the prediction, and
      the flags were saved in the request's `doc`.
    - Omniscience: the grade and raw reply are in `model_output[i].judge_result`.
      Before, they were on the instance.
    - SimpleQA: the judge's raw reply is saved.
    - ExpertQA, AstaBench SQA and ResearchQA: every judge reply is saved in
      `judge_result.raw_judge_responses`.
  - **FrontierScience** also records a `truncation_rate` score. Its metrics are
    unchanged.
- **Identity**:
  - The task hash changes for 30 specs: the 28 safety specs listed above,
    `simpleqa:judge` and `omniscience:judge`.
  - No other spec changes its hash when it runs with its default judge.
  - A run with another judge on `expertqa`, `astabench_scholarqa` or `researchqa` now
    gets its own hash. Before, it shared the default judge's hash. Tell those earlier
    runs apart by date or commit.
- **Comparability**:
  - A result is affected only if one of its judge replies did not parse. In the
    validation runs for #512, this happened for one output.
  - `simpleqa:judge`: such replies counted as incorrect. Leaving them out raises
    accuracy.
  - Safety: such replies counted as unsafe. Leaving them out raises the score. An
    earlier `parsing_error` value says how many there were.
  - ExpertQA and AstaBench SQA: such an output got precision 1.0. It is now unscored.
  - `omniscience:judge`: the metrics are unchanged, since they already left these
    grades out. A run with many such replies can now exceed the failure budget.
  - An earlier subset value of -1 meant the subset had no instances.
  - An earlier `omniscience:judge` run with `-o judge_*` was graded by the default
    judge.

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

## 2026-10-21: No-score suite headings stop publishing synthetic aggregates (#422)

- **Affected**: all `DISPLAY_ONLY` and `NONE` suites, including `arena_hard`, BFCL,
  CTC, Molmo2, OCR, and science suite headings.
- **Change**: dashboard suite cells, run-suite baseline deltas, and run headlines no longer
  treat these headings as aggregate scores. Ingestion clears legacy score and primary-metric
  values even when a saved manifest still supplies them. Individual task scores remain.
- **Identity**: task hashes and suite names are unchanged; the suite definition already
  identifies its no-score aggregation strategy.
- **Comparability**: compare individual task scores. Previously stored synthetic suite
  averages are not comparable suite measurements. Historical rows retain their stored values
  until re-ingestion; no automatic backfill is included.

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
