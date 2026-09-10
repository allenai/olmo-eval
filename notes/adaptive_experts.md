# Adaptive numbers of experts

## Objective

Measure the quality/compute curve from changing Qwen3-MoE's active experts, then train a
single model that remains capable across several expert budgets and eventually chooses its
own budget. The fixed routing sweep uses `K = {1, 2, 4, 8, 16}`; the checkpoint default is
`K = 8` out of 128 experts.

Changing `K` reduces or increases expert computation, but does not change model memory:
all 30.5B parameters remain resident. Approximate activated parameter counts are 1.71B,
1.94B, 2.39B, 3.30B, and 5.11B for the five settings.

## Infrastructure

- Cluster: `ai2/jupiter`
- Workspace: `ai2/olmo-instruct`
- Priority: `urgent`
- Beaker group: `adaptive-experts-qwen3-30b-20260710` by default
- Job ledger: `notes/beaker_jobs.jsonl`
- Result storage: the persistent Beaker result dataset mounted from `/results`
- Base checkpoint: `Qwen/Qwen3-30B-A3B-Base`
- First post-trained checkpoint: `Qwen/Qwen3-30B-A3B` with thinking enabled
- Inference: vLLM server with `hf_overrides.num_experts_per_tok=K`
- Primary post-training output cap: 32,768 tokens with a 40,960-token server context

Every full sweep must pin the same model revision, task definitions, random seeds, number of
inference replicas, and serving settings at all values of `K`. Explicit `K = 8` must match an
unmodified-default smoke run before larger launches.

The launcher copies the exact local `src/` tree to a content-addressed snapshot under
`../.run_snapshots/olmo-eval/` before submission. Beaker jobs import that snapshot through
`PYTHONPATH`; its path, content hash, and base Git commit are recorded in the ledger. Submission
itself uses a temporary clean worktree because Gantry requires a clean, remotely available Git
commit. This preserves reproducibility without requiring an intermediate commit or push.

The launcher currently uses `--no-store`: `ai2/olmo-instruct` does not have the
`olmo_eval_DB_SECRET_ARN` and `olmo_eval_PGHOST` secrets required by olmo-eval's PostgreSQL/S3
`--store` path. Outputs, predictions, metrics, and logs still persist in each experiment's Beaker
result dataset and are addressable through the recorded experiment ID. Switch to `--store` only
after those two workspace secrets are provisioned.

## Evaluation sequence

### 1. Smoke tests

Use one H100 replica per job and test `K = {1, 8, 16}` on small, limited slices for both
checkpoints. Confirm model loading, top-16 routing, reasoning parsing, scoring, stored results,
and inference metrics. Do not launch the full sweeps until these jobs pass.

### 2. Base model

Run the canonical OLMoBase suites:

- `olmobase:easy:code:bpb`
- `olmobase:easy:math:bpb`
- `olmobase:easy:qa:bpb`
- `olmobase:easy:qa:rc`
- `olmobase:gen`
- `olmobase:math`
- `olmobase:mcqa_non_stem`
- `olmobase:mcqa_stem`

Use one Beaker experiment with eight one-GPU inference replicas per value of `K`. Each experiment
passes all eight OLMoBase suites to olmo-eval together; the runner deduplicates overlapping expanded
tasks while preserving a summary metric for each requested suite. This matches successful reference
experiment `01KX4XBGFP5A0XE1JFQT2NWCVM`: it uses the default harness and does not include the
execution-based `olmobase:code` suite, so no sandboxes are needed. Running all five settings
concurrently requires 40 H100s; sequential execution requires eight.

### 3. Hybrid-thinking model

The initial pilot is `adaptive_experts:hybrid_pilot`:

- MATH-500 in chat format, 500 instances
- GPQA Diamond, 198 instances
- IFEval OOD, 300 instances

Use four one-GPU inference replicas per value of `K`. All three tasks use the Qwen3 thinking
sampling recommendation and a 32,768-token output cap. This follows the checkpoint's official
recommendation for most queries and matches the OLMo post-training comparison; Qwen recommends
38,912 for especially difficult math and programming benchmarks. Treat suite aggregation as
display-only and inspect the capability-specific results. Retain 8192-token results as a separate
constrained-serving curve, not the primary quality result.

After the initial `K = {1, 2, 4, 8, 16}` sweep, run the same post-training pilot at `K = 32` as
an upper-compute extension. The base/pretraining sweep remains `K = {1, 2, 4, 8, 16}`.

HumanEval+ is excluded from the initial full sweep because its smoke run exposed an unresolved
answer-extraction or scoring integration issue. MATH, GPQA, and IFEval provide the first quality
curve without blocking on that harness issue.

## Analysis

For every task and coherent suite, report:

- Absolute primary score at each `K`
- Paired per-instance delta against `K = 8`
- Shared sample size, wins/losses/ties, uncertainty, and MDE80
- Prompt tokens, completion/reasoning tokens, and truncation rate
- Output tokens/s, wall-clock time, GPU-hours, utilization, memory, and power
- Quality versus approximate activated parameters and measured compute

Use the same prompts and seeds so paired analysis is valid. Completion length must be treated as
a possible mediator: a setting that generates more reasoning tokens may improve quality while
looking slower for reasons beyond per-token expert cost.

## Post-training directions

Randomly sampling `K` during training teaches robustness to budgets, not adaptivity. A controller
or explicit budget condition is required for the model to choose compute.

1. **Budget-conditioned SFT.** Sample one `K` per microbatch, communicate it with a budget token
   or embedding, and combine supervised loss with token-level KL distillation from `K = 8` or
   `K = 16`. Start with homogeneous-budget microbatches because current MoE kernels assume a
   fixed top-k shape. Preserve per-budget load-balancing losses and oversample low budgets.
2. **Sequence-level controller.** Predict one budget for an entire sequence and optimize
   `task_loss + lambda * mean_active_experts`. This is the easiest form to translate into real
   serving savings. Compare it with fixed-`K` baselines at the same average expert count.
3. **RL.** Generate grouped rollouts of the same prompt at several budgets. Reward task success
   minus compute cost and use the within-prompt counterfactuals to train the controller. Initialize
   from budget-conditioned SFT to reduce variance.
4. **Finer adaptivity.** Move from sequence-level to layer-level and finally token-level budgets.
   Add consistency or monotonicity regularization so increasing budget does not systematically
   degrade predictions. Measure realized kernel time because irregular token-level routing may
   reduce FLOPs without reducing wall time.
5. **Pretraining.** Introduce the same budget condition and sampled-budget curriculum during
   pretraining, with load-balancing statistics maintained separately by budget. Begin near the
   checkpoint's native `K = 8`, then widen the mixture.

Useful non-training baselines include choosing `K` from router entropy, the margin between routed
experts, or the cumulative router probability mass. Also compute an oracle that selects the best
fixed budget per instance; this bounds the potential value of an adaptive controller.

## Launch policy

Use `scripts/adaptive_experts/launch.sh`. It records each successfully submitted Beaker experiment
immediately in `notes/beaker_jobs.jsonl`. A successful submission without a captured experiment ID
is treated as a launch failure so untracked jobs cannot silently accumulate.

### Smoke launch history

The first hybrid smoke attempts on 2026-07-10 exposed an obsolete vLLM flag: vLLM 0.19.1 no
longer accepts `--enable-reasoning`. Those K=1/8/16 experiment IDs remain in the ledger; K=8 was
stopped early and the others exited during server startup. The `retry1` launches remove that flag
and retain `reasoning_parser=qwen3` plus `enable_thinking=true`.

The corrected 8192-token smoke sweep used the same 16 MATH-500 and 16 GPQA Diamond examples at
every budget:

| K | MATH acc. | MATH final / trunc. | GPQA acc. | GPQA final / trunc. |
|---:|----------:|--------------------:|----------:|--------------------:|
| 1 | 0.0% | 0 / 12 | 0.0% | 0 / 12 |
| 2 | 0.0% | 0 / 16 | 0.0% | 0 / 14 |
| 4 | 68.75% | 14 / 4 | 50.0% | 11 / 5 |
| 8 | 87.5% | 16 / 2 | 68.75% | 14 / 3 |
| 16 | 87.5% | 15 / 1 | 68.75% | 14 / 3 |

“Final” counts responses with non-empty post-reasoning content; “trunc.” counts responses that hit
the output cap. K=1 and K=2 spend nearly all their budget in the reasoning channel without a
scorable final answer.

The corrected 16384-token sensitivity was:

| K | MATH acc. | MATH final / trunc. | GPQA acc. | GPQA final / trunc. |
|---:|----------:|--------------------:|----------:|--------------------:|
| 2 | 0.0% | 0 / 15 | 0.0% | 0 / 16 |
| 4 | 75.0% | 14 / 3 | 68.75% | 16 / 0 |
| 8 | 87.5% | 15 / 1 | 68.75% | 16 / 1 |

K=2 inference time increased from about 2.9 to 7.8 minutes without producing a single final
answer. K=4 benefited from the longer cap, while the K=8 scores were unchanged. These results make
8k useful as a constrained-serving curve but too censored for the primary post-training result:
at the checkpoint-default K=8, 5/32 responses hit the 8k cap, versus 2/32 at 16k. Use 32k for the
primary pilot and report final-answer and truncation rates at every budget.

The K=8 32k smoke completed with no censoring: all 32 responses had final content and none reached
the cap. The longest response was 8,987 tokens on MATH and 12,997 on GPQA. MATH accuracy was 93.75%
and GPQA accuracy was 81.25%, versus 87.5% and 68.75% in the 8k run. Sampling is stochastic, so
the score difference is not a pure causal estimate of the cap, but the finish-length evidence is
unambiguous. Qwen's model card recommends 32,768 output tokens for most queries and 38,912 for
difficult math/programming benchmarks:
<https://huggingface.co/Qwen/Qwen3-30B-A3B>.

The first 16k submission exposed a runtime bug in nested task overrides: `sampling_params` remained
a plain dictionary and could not be hashed by the batcher. K=2 failed and the still-loading K=4/8
jobs were stopped; all IDs remain in the ledger. The runner now extracts nested sampling values
into a `SamplingParams` dataclass, with a regression test, and the `16k-retry1` jobs produced the
table above.

### Aborted split full-sweep launch

An initial full-sweep submission on 2026-07-10 was pinned to source snapshot
`2ee1e187aa1e2f6cc4ab7fb1bc025a5b0f07c643b571b63d2b923637b0404497`:

- Post-training: five experiments, four H100s each, covering the complete MATH-500, GPQA Diamond,
  and IFEval OOD datasets with a 32,768-token output cap. HumanEval+ is excluded.
- Pretraining/base: 35 experiments, eight H100s each, covering all seven OLMoBase launch groups
  at every value of K.

The pretraining grouping was incorrect: olmo-eval can run all OLMoBase suites together in one
experiment. All 39 still-active experiments from this batch were stopped after the issue was
identified; `adaptive-qwen3-base-k1-easy-math-code-full-20260710` had already succeeded. The 40
experiment IDs remain in `notes/beaker_jobs.jsonl` for a complete audit trail. The corrected
launcher submits five pretraining experiments total, one per K, with all suites passed to
each experiment.

### First consolidated full-sweep launch

The corrected sweep was submitted on 2026-07-10 as ten experiments total:

- Five post-training experiments, one per K, each containing the three-task 32k hybrid pilot.
- Five pretraining experiments, one per K, each containing all nine OLMoBase suites. Each Beaker
  experiment expands to 296 unique task specs and preserves the nine requested suite summaries.

All ten jobs use source snapshot
`2ee1e187aa1e2f6cc4ab7fb1bc025a5b0f07c643b571b63d2b923637b0404497`. At
2026-07-10T05:08:05Z, all five post-training experiments were running and all five pretraining
experiments were scheduled, with no failures. Their IDs and exact commands are recorded in
`notes/beaker_jobs.jsonl` under run tags `consolidated32k-20260710` and
`consolidated-20260710`.

The five base experiments in this batch incorrectly included the execution-based `olmobase:code`
suite. The launcher also copied a 64-sandbox/56-minimum setting intended for remote Modal sandboxes,
but the workspace ran them as local subcontainers. All eight vLLM replicas in every experiment
became ready in 86–122 seconds; sandbox startup then stalled with only 30–32 of 64 monitors running
and no evaluation work beginning. K=1 was manually canceled and the remaining four base jobs were
stopped. The five post-training experiments were unaffected. The corrected base configuration
matches reference experiment `01KX4XBGFP5A0XE1JFQT2NWCVM`: eight non-execution OLMoBase suites,
the default harness, and no sandboxes.

### Reference-shaped base relaunch

The base sweep was relaunched under run tag `reference-default-20260710` with the exact task and
harness shape of successful reference experiment `01KX4XBGFP5A0XE1JFQT2NWCVM`: one experiment
per K, eight OLMoBase suites, 279 unique expanded tasks, eight inference replicas, the default
harness, and no sandboxes. The five replacement experiment IDs are recorded in
`notes/beaker_jobs.jsonl`. At 2026-07-10T05:51:24Z, K=1/2/4 were running and K=8/16 were scheduled.

## Variance and transition sweep

Treat the completed runs as replicate 1. Only `olmobase:math` is stochastic in the base suite:
its 11 tasks sample at temperature 0.6 with four or eight samples. The other seven OLMoBase suite
summaries are deterministic. Therefore:

- At base K=1/2/4/8/16, run two additional `olmobase:math` replicates only.
- At base K=5/6/7, run the full deterministic suite once plus two additional math-only replicates.
- At post-training K=1/2/4/8/16/32, run two additional full-suite replicates.
- At post-training K=5/6/7, run three full-suite replicates.

This yields three stochastic measurements at every requested K without rerunning deterministic
base metrics. Repeat identity is recorded explicitly in the ledger's `run_tag` field.

## GLM-4.5-Air GPU sizing

On 2026-07-11, `zai-org/GLM-4.5-Air` was smoke-tested on H100s in `ai2/jupiter` with its default
K=8 routing. The standard checkpoint loads as BF16 and has roughly 106B parameters, so its weights
alone require about 212 GB before KV cache and runtime overhead.

| Tensor parallelism | Test shape | Outcome |
|---:|---|---|
| 2 | Not launched | Cannot fit the approximately 212 GB of BF16 weights in 160 GB of aggregate H100 memory. |
| 3 | 4,096-token context, one sequence | Failed during model initialization because vLLM requires the 151,552-token vocabulary to be divisible by TP=3. [Experiment](https://beaker.org/ex/01KX7S3C7VJ1FFHT2B8XG4D7ZA) |
| 4 | 4,096-token context, one sequence | Passed end to end after increasing the startup timeout; the provider became ready in 206.9 seconds. [Experiment](https://beaker.org/ex/01KX7ST2ZZMYH3B4PWTVPHR8BR) |
| 4 | 40,960-token context, 16 sequences | Passed end to end; the provider became ready in 193.1 seconds and completed the test inference. [Experiment](https://beaker.org/ex/01KX7T6AHTN39E1DVR4RQSDD3N) |

Use **4 H100s per vLLM engine** for GLM-4.5-Air. The production-shaped configuration is
`tensor_parallel_size=4`, `max_model_len=40960`, `max_num_seqs=16`,
`gpu_memory_utilization=0.95`, and `startup_timeout=900`. TP=4 is both the minimum viable layout
and sufficient for the planned 32k post-training evaluations. A 900-second startup timeout is
recommended because a prior TP=4 attempt with the default 300-second timeout expired while its
workers were still alive, even though subsequent launches became ready in roughly 3–3.5 minutes.

## GPT-OSS-120B production sweep

GPT-OSS-120B fits one H100 per vLLM engine at TP=1, but the MXFP4 load path initially failed while
reformatting MoE weights because the CUDA allocator had enough reserved memory but no sufficiently
large contiguous free block. Setting `PYTORCH_ALLOC_CONF=expandable_segments:True` fixed that
fragmentation without changing model arithmetic. Four concurrent engines then exposed a separate
Harmony tokenizer cache race; the official `o200k_base.tiktoken` and `cl100k_base.tiktoken` files
were predownloaded under `.runtime_assets/tiktoken_encodings`, and every job now sets
`TIKTOKEN_ENCODINGS_BASE` to that shared read-only directory. One subsequent K=1 endpoint attempt
received a transient HTTP/2 502 during model startup; its identical retry succeeded on the same
node.

The validated production configuration is four independent TP=1 vLLM engines on four H100s,
`max_model_len=40960`, `max_num_seqs=64`, `max_num_batched_tokens=1024`,
`gpu_memory_utilization=0.95`, prefix caching disabled, and a 900-second startup timeout. The
batched-token setting limits scheduler work per iteration, not the total response length; each of
MATH-500, GPQA Diamond, and IFEval OOD retains a 32,768-token generation allowance. K=1 brought all
four engines online in about 105 seconds, while K=16 did so in about 159 seconds. Both immediately
started real 64-example batches; K=16 processed its initial batches at roughly 2.4 items/second.

The intended production sweep is three replicates at K=1--8, submitted on 2026-07-11 at urgent
priority to `ai2/jupiter` in `ai2/olmo-instruct`. Replicate 1 uses the successful K=1 endpoint retry
plus run tag `full-c64-harmony-r1-20260711` for K=2--8. Replicates 2 and 3 use run tags
`full-c64-harmony-r2-20260711` and `full-c64-harmony-r3-20260711`, respectively, for K=1--8. This
is 24 usable full-suite experiments total.

K=9--16 was mistakenly submitted in the same launch wave. On 2026-07-11, all 23 production jobs
above K=8 and all seven earlier GPT-OSS K=16 diagnostic experiments were permanently deleted from
Beaker. Their experiment IDs and original commands remain in `notes/beaker_jobs.jsonl` as audit
history, but they are not part of the 3x8 result matrix.

### GPT-OSS non-power-of-two TP=2 reruns

The original TP=1 K=3/5/6/7 runs were not usable: each engine passed startup but OOM'd on its first
real 64-example batch, producing empty outputs. A first TP=2 diagnostic wave then exposed two
distinct vLLM startup failures. The default custom all-reduce path raised a CUDA invalid-argument
error; disabling it in favor of NCCL allowed startup to proceed, but CUDA-graph capture then OOM'd.
Those failed diagnostics remain in the append-only job ledger.

The validated TP=2 configuration disables custom all-reduce and enables eager execution. Both a
one-example smoke and a production-shaped 64-example smoke completed end to end with this setup.
The final K=3/5/6/7 rerun matrix uses four TP=2 engines (eight H100s per experiment),
`max_num_seqs=64`, `max_num_batched_tokens=1024`, and `gpu_memory_utilization=0.90`. Three
replicates per K were submitted with tags `full-c64-harmony-tp2-eager-r1-20260711` through
`full-c64-harmony-tp2-eager-r3-20260711`, at urgent priority on `ai2/jupiter` in
`ai2/olmo-instruct`. Eager execution is expected to be slower, but preserves the full 32,768-token
generation allowance. After launch, the first production K=3/5/7 jobs progressed into real
64-request generation batches at roughly 0.25--0.32 requests/second, confirming that the corrected
configuration also works outside the smoke suite.

### GLM replicate expansion

Seven additional GLM-4.5-Air experiments were submitted at urgent priority on `ai2/jupiter` in
`ai2/olmo-instruct`, retaining two TP=4 engines and the complete three-task post-training suite:

- K=4 and K=8 received two additional runs each, bringing each point to three total runs when
  combined with its original pilot.
- K=2 received three new runs.

The new GLM tags are `expanded-r1-20260711` through `expanded-r3-20260711`; r1 contains K=2,
while r2 and r3 contain K=2/4/8.

## AIME 2026 pass@32 sweep

`aime_2026` is implemented and the public `MathArena/aime_2026` train split loaded successfully
with 30 problems. The bare task generates once per problem at temperature 0, but the standard
post-training variant `aime_2026:pass_at_32` automatically generates **32 independent outputs for
the same fixed prompt** at temperature 0.6 and top-p 0.95. It uses a 32,768-token output cap and
reports accuracy plus pass@1/4/8/16/32; pass@1 is the primary metric. The prompt is the problem
followed by: `Please reason step by step, and put your final answer within \\boxed{}.` There are no
separate prompt variants.

Because the variant already supplies 32 stochastic samples per problem, only one Beaker experiment
was submitted per chat checkpoint/K. The run tag is `aime2026-pass32-20260711`; all experiments use
urgent priority on `ai2/jupiter` in `ai2/olmo-instruct`.

| Checkpoint | K values | Jobs | Serving layout |
|:-----------|:---------|-----:|:---------------|
| Qwen3-30B-A3B hybrid-thinking | 1--16, 32 | 17 | 4 one-GPU engines |
| GPT-OSS-120B | 1--8 | 8 | K=1/2/4/8: 4 TP=1 engines; K=3/5/6/7: 4 TP=2 eager/NCCL engines |
| GLM-4.5-Air | 2/4/8 | 3 | 2 TP=4 engines |

Thirteen Qwen3 Base AIME jobs were initially submitted due to an overly broad interpretation of
"all models." They were permanently deleted from Beaker on 2026-07-11 after the scope was clarified
to chat models only. Their IDs and commands remain in the append-only ledger as audit history, but
they are excluded from the retained sweep.

The retained sweep contains 28 experiments, all recorded in `notes/beaker_jobs.jsonl`. At
2026-07-12T02:09Z, 26 had finished and two Qwen jobs (K=2 and K=32) were still running. Both
subsequently completed and were collected: K=2 scored 0.00% at pass@1 and pass@32, while K=32
scored 18.54% pass@1 and 53.33% pass@32. Of the
finished jobs, GPT-OSS K=1 is invalid: all requests returned empty outputs following Harmony decode
errors. The other 25 completed bundles are valid and collected.

## Post-training collection and GLM curve extension

On 2026-07-11, all 12 corrected GPT-OSS K=3/5/6/7 TP=2 runs and all seven GLM K=2/4/8
replicate-expansion runs completed successfully. Their 19 result datasets were downloaded under
`results/adaptive_experts/`, checked for task errors, incorporated into the Markdown tables, and
added to the per-model plots. GPT-OSS now has three valid runs at every K=1--8; GLM has three valid
runs at K=2/4/8.

The GLM post-training curve was then extended to K=1/3/5/7, with three runs per K. These 12 jobs
retain the same complete three-task suite, two TP=4 engines, 40,960-token server context,
32,768-token output allowance, and urgent Jupiter/`ai2/olmo-instruct` placement. Replicate tags are
`curve-r1-20260711`, `curve-r2-20260711`, and `curve-r3-20260711`. All 12 completed and were
collected by 2026-07-12. One K=5 r3 execution was preempted/canceled once, then automatically
retried and succeeded; its final result is valid.

The refreshed plots include the complete GLM post-training curve and three new AIME figures (one
per chat model) showing pass@1/4/8/16/32. Qwen AIME K=2 and K=32 are now complete and included;
GPT-OSS AIME K=1 is excluded rather than plotted as a zero score.

Per the plotting convention established on 2026-07-12, invalid results are omitted entirely from
curves. GPT-OSS AIME K=1 remains documented in the Markdown status table but is absent from the
plotted data.

## Dolci-think SFT checkpoint pilot

An initial post-training sweep was submitted for the local checkpoint:

`/weka/oe-adapt-default/jacobm/olmoe3/post-training/checkpoints/qwen3-30b-a3b-dolci-think-olmo-core-sft-100k-20260710-174355-hf`

The checkpoint is a complete 13-shard `Qwen3MoeForCausalLM` HF export with default K=8 and a custom
chat template that opens an assistant `<think>` block. The initial launch used the Qwen3 reasoning
parser, a 40,960-token server context, `max_num_seqs=16`, and four one-GPU vLLM engines. Each
experiment contained the complete MATH-500, GPQA Diamond, and IFEval OOD post-training suite with a
32,768-token generation allowance.

One initial run was launched at each K=2/4/6/8 with tag `initial-r1-20260712`, urgent priority on
Jupiter, workspace `ai2/olmo-instruct`, and group
`adaptive-experts-qwen3-dolci-think-sft100k-20260712`. All four Beaker jobs later reported success,
but output validation found them unusable: each engine crashed at position 32,768 because this
checkpoint has a native `max_position_embeddings=32768` and no RoPE scaling. Output coverage was
only 2--46/198 for GPQA, 3--64/300 for IFEval, and 4--120/500 for MATH-500. The resulting empty and
zero scores are excluded from plots.

Four replacements were submitted under tag `native32k-r1-20260712`. They set the server's total
context to 32,768 and each task's generation cap to 30,000, leaving room for the prompt while
preserving a long reasoning allowance. The replacement IDs are
`01KXA7852F42HNHM4KAV5AGY4V` (K=2), `01KXA7BCCAXGD1DS1EY6ZW6AQQ` (K=4),
`01KXA7CB3SCFYB63VVEHPCCPYK` (K=6), and `01KXA74QPCECFSKGCBXVRT4AGE` (K=8). All launch IDs and
exact commands are in the append-only ledger.

The native-context replacements still used `reasoning_parser=qwen3`, which is incompatible with the
OLMo thinking template used for SFT. Because the prompt already ends in `<think>`, Qwen3 treats any
response lacking `</think>` as reasoning-only and exposes empty answer content. The OLMo3 parser
falls back to exposing the full response as content when the closing tag is absent. The Qwen3-parser
results are superseded and excluded from plots; its still-running K=2 job was stopped.

A second K=2/4/6/8 sweep was launched with `reasoning_parser=olmo3` under tag
`olmo3-parser-r1-20260712`. The experiment IDs are `01KXAA986DGEJB76QXZDH4N9E1` (K=2),
`01KXAA9GRJJXDV4CH0FT3MBH8D` (K=4), `01KXAA9S8J21JTCQCXCJG8W4PW` (K=6), and
`01KXAAA1ME9WC283P25QBE0MMV` (K=8). All four completed, were collected, passed engine-health and
prediction-row validation, and are plotted. Macro scores for K=2/4/6/8 are respectively
7.96%, 46.02%, 52.58%, and 51.46%. At K=2, 843 of 997 generated responses reached the 30,000-token
cap, providing strong evidence that the low-budget score is a genuine truncation-heavy collapse.

## Candidate next research directions (2026-07-15)

The motivating result is that off-the-shelf MoEs often preserve much of their evaluated capability
when active K is reduced, even though Qwen's experts ranked 5--8 collectively receive roughly 36%
of its normalized K=8 gate weight. Router weight therefore may not measure an expert's marginal
functional value; expert outputs may be redundant, aligned, or canceling.

Promising experimental directions, in rough priority order:

1. **Marginal expert-value analysis:** at fixed hidden states, compare rank truncation, individual
   rank ablations, random subsets of the selected experts, and renormalized versus unnormalized
   recombination. Measure hidden-state change and next-token KL, not only router probability.
2. **Layer-adaptive K:** test coarse schedules such as K=8 in broadly routed early layers and K=4
   or K=2 in more concentrated late layers. Compare policies at equal mean expert calls.
3. **Token-adaptive escalation:** begin at a small K and increase it only when router mass, margin,
   entropy, or expert-contribution disagreement indicates uncertainty. Compare against fixed K at
   the same average compute.
4. **Prompt- or capability-adaptive budgets:** choose K once per request using prompt features or
   early model signals; this is simpler to serve than token-level dynamic routing and may capture
   task-dependent sensitivity.
5. **Elastic SFT:** train one checkpoint across randomly sampled K values, potentially distilling
   low-K logits or hidden states from its K=8 execution. Begin with frozen experts and router/LoRA
   adaptation before allowing full-model updates.
6. **Selected-expert dropout:** randomly remove routed experts during post-training to encourage
   robustness across active-expert budgets.
7. **Compute-aware RL:** let a policy select K while charging for expert calls, after supervised
   experiments establish stable actions and useful decision signals.
8. **Variable-K pretraining:** sample K across batches, tokens, or layers, potentially conditioned
   on an explicit compute-budget token, to create a natively elastic or slimmable MoE.
9. **Expert pruning or merging:** pursue this separately if weight memory, rather than only active
   FLOPs, is a target; lowering K alone leaves every expert parameter resident.

Across these experiments, report quality against actual expert calls and measured serving
throughput, include top-ranked versus random-subset controls, and retain cap-hit/final-answer
diagnostics so degenerate long generations cannot masquerade as ordinary accuracy loss.

### Expert-activation magnitude and functional contribution

Router weight alone is insufficient because Qwen normalizes the selected gate weights but does not
normalize each expert's output before combining them. For routed experts, the local update is
approximately `sum_i g_i E_i(h)`: a low-weight expert can still have a large contribution if its
output norm is large, and vector direction determines whether expert contributions reinforce,
duplicate, or cancel one another. An implementation audit of the Transformers
`Qwen3MoeSparseMoeBlock` used by this checkpoint found no separate shared-expert branch; the relevant
comparison is therefore the routed MoE update against the decoder residual stream.

A contribution-focused pass should record unweighted expert-output norms, weighted contribution
norms, pairwise cosine alignment, alignment with the full MoE update, and the effect of removing
each expert with and without renormalizing the survivors. Local hidden-state cosine/L2 changes and
next-token KL are more useful measures of marginal value than gate mass alone. Full regeneration
can follow on the small set of layers and prompts where local effects are largest.

The completed 60-response fixed-trajectory experiment does not support activation magnitude as a
hidden source of lower-ranked-expert importance. Mean raw expert-output norm declines with router
rank; rank 8 is 58.1% of rank 1. Consequently, the top four rise from 63.3% of K=8 gate mass to
69.2% of summed contribution norm. Boundary reordering is common but limited: a bottom-four expert
beats some top-four expert on 56.9% of token-layer observations, while only 9.8% of all cross-half
pairs are inverted. The detailed design, bootstrap intervals, task/phase/layer results, and plots
are in [the Qwen expert-contribution experiment note](qwen_expert_contribution_experiment.md).

### IFEval examples that change across the K=8 boundary

Build a prompt-by-K-by-replicate matrix using IFEval's prompt-level and individual-constraint
checks. Identify robust threshold examples that pass most runs above K=8 and fail most runs below
K=8, alongside matched always-correct, always-wrong, and non-monotonic controls. First separate
completed behavioral failures from cap hits, missing parsed answers, and repetitive degeneration.

For the retained examples, compare which exact constraints fail—formatting, required or forbidden
phrases, length, language, and multi-instruction composition—and inspect several stochastic
responses rather than attributing one textual difference to K. A later paired mechanistic pass
should use matched seeds or a shared teacher-forced prefix, locate the first meaningful logit or
token divergence, and compare router selections and weighted expert contributions by layer. This
would connect the behavioral threshold to expert function rather than only cataloguing output
differences.
