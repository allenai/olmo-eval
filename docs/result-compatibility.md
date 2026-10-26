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

## 2026-10-26: One judge base: judge identity in the hash, no verdict leaves an output unscored (#PR)

- **Affected**:
  - `simpleqa:judge`.
  - The `openai_judge`, `wg_judge` (`sr_judge` for strongreject), `*_thinking` and `base`
    variants of `do_anything_now`, `harmbench`, `strongreject`,
    `trustllm_jailbreaktrigger`, `wildguardtest`, `wildjailbreak` and `xstest`, and the
    safety suites that include them.
  - `wmdp:mcq`, whose subset metrics change for empty subsets only.
  - Runs that set `OLMO_EVAL_JUDGE`, or `-o judge_*` on `omniscience:judge`, `expertqa`,
    `astabench_scholarqa` and `researchqa`.
- **Change**:
  - A judge reply that names no verdict leaves the output unscored, counted against the
    failure budget; an API judge (SimpleQA, the safety `openai_judge` variants) is asked
    again first. Before, SimpleQA scored it as incorrect and the safety judges scored it,
    an answer outside the allowed ones, or a StrongREJECT judge with no grade logprobs,
    as 0.0.
  - The safety variants drop their `parsing_error` metric, which would now always read
    0; the run's scoring failures count these replies instead. Subset metrics leave out
    unscored responses, and an empty subset reports 0.0 instead of -1.
  - Saved predictions keep each safety sample's judge result under its
    `model_output[i].judge_result`, with `is_empty_response`, `is_truncated_response`
    and `is_parsing_error` inside it. Before, one result per instance was saved at the
    top level of the prediction and the flags in the request's `doc`. SimpleQA
    predictions now keep the judge's raw reply.
  - The safety variants that judge with an auxiliary provider record the model it must
    serve as `judge_model` (`allenai/wildguard`, or the `qylu4156/strongreject-15k-v1`
    adapter), and the runner refuses a provider that serves another model.
  - `OLMO_EVAL_JUDGE` is applied when the task is built, to every judged task whose `-o`
    overrides name no judge model, and takes `model[:effort[:max_tokens]]`. Before, an
    explicit `-o judge_model` equal to a task's default lost to it, FrontierScience let
    it override `-o`, and ExpertQA, AstaBench SQA and ResearchQA read it while scoring.
  - `-o judge_*` on `omniscience:judge` now changes the judge that scores the run. Before,
    the runner applied it after building the scorer, so the default judge scored while
    the config recorded the override.
  - Judge API calls are retried on transient failures for judge scorers and the
    attributed-QA tasks, which made one attempt before.
- **Identity**: the task hash changes for the 28 safety specs listed above. It does not
  change for any other spec run with its default judge, including `simpleqa:judge` and
  `wmdp:mcq`. Runs with another judge on `expertqa`, `astabench_scholarqa` and
  `researchqa` now get their own hash; before, they shared the default judge's hash.
- **Comparability**: a `simpleqa:judge` result is affected only if its judge gave a reply
  with no grade; leaving those out raises its accuracy. Safety results before this change
  counted unparseable judge replies as unsafe, so a run with such replies scores higher
  now; earlier `parsing_error` values say how many there were. An earlier subset value of
  -1 meant the subset had no instances. An earlier `omniscience:judge` run with
  `-o judge_*` was scored by the default judge.

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
