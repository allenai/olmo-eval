# Final-answer-only correctness evaluation

Thinking models can run out of tokens before delivering an answer. Some benchmark
extractors still find a correct answer token inside that unfinished reasoning. The
opt-in `reward_final_answer_only` task option measures correctness of delivered answers
using the same answer-region rule as MILES training: score only the text after the last
`</think>`, and assign zero when that delimiter or a nonblank final section is missing.
Existing task defaults, extractors, scores, and hashes remain unchanged.

## Enable the option

Use the ordinary task configuration field or a task CLI override:

```bash
uv run olmo-eval run -m YOUR_MODEL -t gpqa_diamond:cot \
  -o reward_final_answer_only=true --no-upload
```

```python
from olmo_eval.evals.tasks.common import get_task

task = get_task("gpqa_diamond:cot", {"reward_final_answer_only": True})
```

The option defaults to `False`. As with other optional scoring fields, the serialized
configuration omits it when false to preserve existing hashes; absence means false.
When enabled, `reward_final_answer_only: true` is serialized and changes task/result,
progress-cache, and artifact identity. Do not pool the two scoring configurations.

| Original response | Scoring behavior when enabled |
| --- | --- |
| `reason</think> A ` | Score ` A `, including the surrounding whitespace |
| `reason</think>B</think>A` | Score `A` after the last delimiter |
| `reason</think>` or whitespace-only suffix | Explicit zero |
| Unfinished reasoning without `</think>` | Explicit zero |
| Ordinary non-thinking answer `A` without `</think>` | Explicit zero |

This is an answer-region rule. It does not require an opening tag, exactly one closing
tag, a boxed answer, a particular finish reason, or a specific answer format. Existing
benchmark extraction still runs on eligible final text. GPQA's multiple-choice
extraction rules, including their permissive boundaries, are unchanged. A token-capped
response with a nonblank final section remains eligible.

This option takes precedence over `strip_thinking`, `strip_unclosed_thinking`, and
`thinking_prefilled`. Processing always starts from `original_text` when available,
so repeated processing and prior thinking stripping do not lose the final section.

## Supported combinations

Both the metric and the concrete scorer must be supported. Unsupported combinations
raise a configuration error before task data is loaded by the runner.

| Component | Supported |
| --- | --- |
| Requests | Single-turn text completion or chat |
| Metrics | `AccuracyMetric`, `F1Metric`, `SQuADF1Metric`, `PassAtKMetric`, `PassPowKMetric` |
| Scorers | `ExactMatchScorer`, `ExactMatchFlexScorer`, `MultipleChoiceScorer`, `F1Scorer`, `SQuADF1Scorer`, `GPQACoTExactMatchScorer` |
| GPQA | Generative default and `:cot` tasks, including subject subsets |

Logprob/MC-continuation and BPB metrics, refusal and safety metrics, inverted metrics,
LLM judges, process/sandbox scorers (including symbolic math and code execution),
agent/tool trajectories, and task-specific scoring/aggregation overrides are unsupported.
The runner validates before calling a task's `score_responses` override, so an override
cannot silently bypass the gate. An audited scorer explicitly declares
`supports_final_answer_only = True`; subclasses must declare their own support.
Extending support requires checking both the scorer's answer inputs and the metric's
meaning of zero.

Missing answers bypass extraction and scoring and receive an explicit per-output zero.
They remain in accuracy and pass-metric denominators, including all-missing responses
and mixed valid/missing multi-sample responses. `first` and `max` aggregation retain their
usual meanings. Scorer exceptions remain scoring errors; they are not converted into
missing-answer zeros.

## Saved diagnostics

Enabled outputs preserve the raw generation in `original_text`, including outputs
without a delimiter. Token counts, logprob metadata, and finish reasons continue to
describe the full generation. Each saved output includes:

```json
{"final_answer_gate": {"applied": true, "reason": "missing_delimiter", "zeroed": true}}
```

`reason` is `final_answer`, `missing_delimiter`, or `empty_final_answer`. A missing
answer has a numeric zero in its sample/instance metrics and no gate-induced
`scoring_errors`. Disabled fresh scoring produces no gate metadata. The option does
not alter trajectories or requests.

## Offline rescoring

The `rescore` command reads local predictions and their saved requests, joins by
`native_id`, and scores against the saved labels. It never initializes an inference
provider, downloads a dataset, calls a judge, or uploads results.

```bash
uv run olmo-eval rescore --task gpqa_diamond:cot \
  --predictions /path/to/gpqa-predictions.jsonl \
  --requests /path/to/gpqa-requests.jsonl \
  --reward-final-answer-only --output-dir /tmp/gpqa-final-only
```

Omit `--reward-final-answer-only` (or use `--no-reward-final-answer-only`) to rescore
with the task's existing default behavior. Use the original task spec, including
variants. The command uses that spec's current scoring configuration; it does not
reconstruct unrecorded task overrides or a historical extractor revision. Only the
supported synchronous correctness combinations above can be rescored.

Every output must have string-valued `original_text`, even when it is empty. Missing
original text is an error. Older writers saved `original_text` only when thinking was
stripped. For such artifacts, add `--text-is-original` **only after verifying that each
output lacking `original_text` has unmodified `text`**. This assertion is recorded in
the results; the command never guesses from an absent delimiter. Already stripped
answers without recoverable raw text cannot be reliably rescored.

Results and predictions are written to separate task-hash-qualified files. Existing
files are not overwritten. Results record source paths and SHA-256 digests, configuration,
task hash, instance count, and aggregate metrics. Predictions preserve native/document
IDs and raw responses, replace old extracted answers/scores with newly computed values,
and retain generation token counts and finish reasons. Saved requests must match the
task name and labels; duplicate IDs, missing requests, and absent outputs fail clearly.
GPQA's numeric labels are known choice indices. Other tasks with numeric labels must
provide explicit string-valued `gold_answer` because an index alone does not identify
whether the scorer expected a letter or answer text. Enabled predictions save this field.

## GPQA audit validation

Offline rescoring of the saved November 4 science investigation reproduced all original
per-example scores and the expected final-answer-only totals. No inference was run.
The audit's original evaluator revision was `ede61eb914c0cae312c9c8112389afbe524fb096`.
Legacy outputs without `original_text` were confirmed to retain their unmodified text.

| Diamond model | Instances | Original correct | Final-answer-only correct |
| --- | ---: | ---: | ---: |
| SFT | 198 | 69 | 37 |
| Calendar | 198 | 51 | 40 |

| Artifact | SHA-256 |
| --- | --- |
| SFT predictions | `33fb3f74b0a6954ef34f726534cea5fd1b7307ab9140e5e639e52e1a6305cdef` |
| Calendar predictions | `f4e15b7bdf60369fc5527ce054920eb81601789eb4cb7a7c6023fb906a7cb67a` |
| Requests (both) | `9dc6a5ad2443ceecb2d2e368e4f20dbbbff90fab4ba11cf0e0da48c06b609e27` |

Source runs: [SFT](https://beaker.org/ex/01M6C80EK9Y2BT5RM131DCFWF1) and
[Calendar evaluation](https://beaker.org/ex/01M6GCKM9C05W0PQDFP50BSK6H).
