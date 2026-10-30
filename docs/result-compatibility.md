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
