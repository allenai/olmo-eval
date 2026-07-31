# MMLU logprob failure: assessment

Date: 2026-07-23
Author: investigation for the eval run plan (see `EVAL_RUN_PLAN.md`)
Constraint: diagnose and fix **without modifying any `src/` code** on this
branch. All proposed fixes are launch/config-only (provider overrides).

## TL;DR

The Gemma4 26B-A4B MMLU job produced a ~0.069 score (below the 0.25 random
floor). This is not a model result and not a scoring-logic bug. The vLLM engine
died with a **CUDA out-of-memory error during the logprob (`prompt_logprobs`)
phase**, so every request after the crash returned HTTP 500 / connection errors.
The harness replaced each failed request with an empty output, which the scorer
recorded as `logprob = 0.0`. The uniform near-zero score across all 57 subjects
is the signature of "engine died early, everything downstream is empty".

The most likely reason we hit OOM is that we launched MMLU **the same way we
launch the agentic/generation tasks**, which is wrong for a logprob task:

1. `tensor_parallel_size = 1` even though the job was given 2 GPUs, so the 26B
   model ran on a single H100 with the second GPU idle.
2. `max_model_len = 32768` while MMLU prompts are ~500 tokens, so the KV-cache
   reservation ate the memory headroom the `prompt_logprobs` path needs.
3. Default `gpu_memory_utilization = 0.9` plus default concurrency, so the
   transient `prompt_logprobs` allocation had nowhere to go.

Generation tasks (ExpertQA, LitSearch-rerank, IFEval, MATH-500) survived the
same config because generation never allocates the prompt-logprob logits over
the full prompt. MMLU is the only task in the set that uses the logprob path.

## Evidence

Experiment: `gemma4-26b-a4b-it-mmlu-20260723-171131`
- Beaker experiment: `01KY7ZCCR9GVPEC2KZM13SRA8B`
- Result dataset: `01KY7ZCCRNFEAK0XH1ND5E3427`
- vLLM server log in dataset: `logs/vllm_server_google_gemma-4-26B-A4B-it/vllm_server_36949.log`

Job exit code was 0 and `metrics.json.errors` was `[]`, so nothing surfaced the
failure at the top level. All 57 subjects / 14,042 instances "completed".

Predictions (e.g. `mmlu_college_mathematics_...-predictions.jsonl`):

```json
{"doc_id": 0, "model_output": [], "instance_metrics": {"logprob": {"logprob": 0.0}}, "label": 1}
```

`model_output` is empty for every instance, and the logprob is exactly `0.0`.

Harness-side (job log): during the logprob phase the dispatcher logged a burst
of failures, then the server became unreachable:

```
[WARNING] dispatch: process_fn raised HTTPStatusError: Server error '500 Internal Server Error' for url 'http://127.0.0.1:36949/v1/completions'
... (many)
[WARNING] dispatch: process_fn raised ConnectError: All connection attempts failed
[WARNING] dispatch: process_fn raised ReadError:
```

Engine-side (vLLM server log), the fatal error at 17:26:24:

```
(EngineCore pid=1406) ERROR ... [dump_input.py:79] Dumping scheduler output ...
    SchedulerStats(num_running_reqs=12, ..., kv_cache_usage=0.112...)
(EngineCore pid=1406) ERROR ... [core.py:1110] EngineCore encountered a fatal error.
(EngineCore pid=1406) torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 3.00 GiB.
    GPU 0 has a total capacity of 79.19 GiB of which 2.89 GiB is free.
    Including non-PyTorch memory, this process has 76.29 GiB memory in use.
    Of the allocated memory 72.23 GiB is allocated by PyTorch ...
(APIServer pid=992) ERROR ... vllm.v1.engine.exceptions.EngineDeadError:
    EngineCore encountered an issue. See stack trace (above) for the root cause.
```

Engine config from the same dump (the "using it wrong" evidence):

```
tensor_parallel_size=1, max_seq_len=32768, enforce_eager=False,
enable_prefix_caching=True, enable_chunked_prefill=True, dtype=torch.bfloat16
```

The model is a 26B MoE (`fused_moe`, `E=128`) running on a single H100 (79 GiB).
Weights + KV reservation left only ~2.9 GiB free; a 3.0 GiB transient
allocation from the prompt-logprob step tipped it over.

## How the failure maps to code (read-only, no changes proposed to it)

- MMLU uses `MultipleChoiceLogprobFormatter` + `LogprobMCAccuracyMetric` +
  `LogprobScorer` (confirmed in `src/olmo_eval/evals/tasks/mmlu.py` and the job
  config dump). This routes to the provider's logprob path, not generation.
- `VLLMServerProvider._logprobs_single_impl`
  (`src/olmo_eval/inference/providers/vllm_server.py:970`) scores each
  continuation by POSTing token IDs to `/v1/completions` with
  `"prompt_logprobs": self._prompt_logprobs` (default 5, set at line 318). This
  is the request that OOMs the engine.
- On failure, `alogprobs` (line 1130) uses `dispatch_concurrent`; failed
  requests come back as `None` and are converted to `[]`
  (line 1165: `[r if r is not None else [] for r in results]`). That empty list
  is the `model_output: []` we see, and the scorer then yields `logprob 0.0`.
- GPU→TP mapping: `tensor_parallel_size` defaults to 1
  (`gpu_planner.py:105`, provider `__init__` default). The launcher's `--gpus N`
  sets the Beaker allocation (`launch/config.py`, `launcher.py`), not the vLLM
  `--tensor-parallel-size`. Nothing in `launch_safe_evals.sh`'s provider
  overrides set TP, so it stayed 1.
- All knobs we need are config-settable: `_build_server_command`
  (`vllm_server_utils.py:268`) exposes `--tensor-parallel-size`,
  `--gpu-memory-utilization`, `--max-model-len`, and passes any extra
  `provider.kwargs` through as `--flag value` (lines 371-383). So
  `tensor_parallel_size`, `gpu_memory_utilization`, `max_num_seqs` are all
  reachable via `--override provider.kwargs.<name>=...`.

## What this is NOT (ruled out)

- Not a scoring bug: the scorer never received logprobs; it defaulted to 0.0.
- Not model capability: the score is an artifact of the crash, not the model.
- Not a tool-call / reasoning-parser issue: MMLU uses none of that.
- Not a plain HTTP 500 that retried cleanly: the engine died
  (`EngineDeadError`) and never recovered within the job.
- Not "prompt logprobs unsupported": early `/v1/completions` calls returned
  200 OK before the crash, so the endpoint and payload shape are accepted.

## Leading hypothesis

Insufficient GPU memory headroom for the `prompt_logprobs` step, caused by
launching MMLU with generation-task settings (TP=1, 32k context, 0.9 util).
The fix is config, not code.

## Proposed experiments (config-only, ordered by expected value)

Run a small MMLU canary (a few subjects or `--limit`) between changes so the
loop is cheap. Each override is added to the existing launch; none touch `src/`.

1. **Use both GPUs.** `--override provider.kwargs.tensor_parallel_size=2`
   (job already requests `--gpus 2`). Sharding the 26B weights across two H100s
   roughly doubles free memory for the logprob spike. Highest-leverage single
   change.
2. **Right-size the context.** `--override provider.max_model_len=4096`
   (MMLU prompts are ~500 tokens). Shrinks the KV-cache reservation and frees
   several GiB. Safe because no MMLU prompt approaches 4k.
3. **Lower memory utilization.**
   `--override provider.kwargs.gpu_memory_utilization=0.8` (or 0.7). Leaves more
   unreserved GPU memory for the transient allocation.
4. **Cap concurrent prefills.** `--override provider.kwargs.max_num_seqs=8`
   (12 were running at crash time). Reduces the peak simultaneous
   prompt-logprob allocation.
5. If still tight, reduce the returned logprob breadth via
   `--override provider.kwargs.prompt_logprobs=1` (scorer needs the target
   token's logprob; fewer returned entries lowers memory pressure). Verify the
   scored numbers are unchanged versus a known-good baseline before trusting it.

Expectation: (1)+(2) alone should resolve it. Apply them first; add (3)/(4) only
if a canary still OOMs.

## Open questions for a second opinion (codx)

1. Is `vllm_server` the intended provider for logprob/MC tasks like MMLU, or was
   the historical/upstream reference the in-process `vllm` provider? The
   server-path docstring says it "matches the inline vLLM provider's behavior
   exactly", which implies inline is canonical. If MMLU is meant to run under a
   different provider profile, that is a cleaner fix than tuning memory.
2. Does `enable_prefix_caching=True` interact badly with `prompt_logprobs` (the
   logprob path sends raw token IDs and `add_special_tokens=false`)? Worth
   checking whether disabling prefix caching for the MMLU job changes peak
   memory or correctness.
3. Should the harness treat an all-empty logprob task as a hard error rather
   than reporting exit 0 with `errors: []`? Right now a dead engine produces a
   plausible-looking 0.069 number. (Flagging only; any fix here is a `src/`
   change and out of scope for this branch.)
4. Is there a canonical `gpu_memory_utilization` / `max_model_len` profile for
   logprob tasks elsewhere in the repo or oe-eval-internal that we should match
   rather than hand-tuning?

## Reproduction pointers

- Server log (engine OOM): dataset `01KY7ZCCRNFEAK0XH1ND5E3427`, file
  `logs/vllm_server_google_gemma-4-26B-A4B-it/vllm_server_36949.log`, timestamp
  17:26:24.
- Job log (dispatch 500s → ConnectError): `beaker experiment logs 01KY7ZCCR9GVPEC2KZM13SRA8B`.
- Predictions with empty `model_output`: same dataset, `predictions/google_gemma-4-26B-A4B-it/mmlu_*-predictions.jsonl`.
- Launch path: `scripts/beaker/launch_safe_evals.sh --only mmlu` (provider
  overrides currently: `max_model_len=32768`, `startup_timeout=1800`,
  `language_model_only=true`).
