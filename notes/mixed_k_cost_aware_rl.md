# Mixed-K cost-aware RL implementation and prompt-policy analysis

Last updated: 2026-08-03

## Objective

Train Qwen3-30B-A3B-Base on DAPO-Math with multiple expert budgets represented
inside each prompt's rollout group. A correct low-K response receives a slightly
higher shaped reward than a correct high-K response, while every incorrect response
keeps the same failure reward. This directly exposes the shared model to the
quality/compute tradeoff without pretending that a constant prompt-group compute
penalty can survive GRPO's within-group centering.

Both planned 100-step pilots completed successfully after a rollout-only smoke and a
two-step learner smoke.

## Topology

The prior Qwen3 DAPO runs used one 8-GPU B200 node with:

- an eight-GPU Megatron actor/learner;
- one eight-GPU SGLang rollout engine (EP=8); and
- colocation/offloading, so these were the same eight physical GPUs rather than 16.

The mixed-K implementation preserves the one-node, eight-physical-GPU envelope and
all eight logical rollout GPU slots. It changes the rollout layout to homogeneous EP=2
engines so simultaneous K policies can share current actor weights:

| Named rollout model | K | Engines | GPUs per engine | Logical rollout GPUs |
|---|---:|---:|---:|---:|
| `k4` | 4 | 1 | 2 | 2 |
| `k6` | 6 | 1 | 2 | 2 |
| `k8` | 8 | 2 | 2 | 4 |

The extra K=8 engine gives the highest-budget arm twice the serving capacity. All four
engines use the same TP/EP topology, which permits rank-local expert weight transfer
from the learner to every engine.

## First reward condition

Each prompt has nine sibling generations, split evenly and deterministically across
K={4,6,8}. The initial correctness-only bonuses are:

| K | Correct base reward | Bonus | Shaped correct reward |
|---:|---:|---:|---:|
| 4 | 1.0 | +0.10 | 1.10 |
| 6 | 1.0 | +0.05 | 1.05 |
| 8 | 1.0 | +0.00 | 1.00 |

Incorrect responses remain at -1.0 regardless of K. Dynamic sampling still decides
whether to retain a prompt using unshaped correctness, and reward normalization occurs
within the nine-response prompt group after shaping.

## Implementation

- A custom rollout function assigns sibling samples to named K=4/6/8 SGLang routers
  and records K in sample and training metadata.
- The standard SGLang generator now honors an optional named-router field while keeping
  the default single-router behavior unchanged.
- Weight synchronization aggregates every named deployment marked `update_weights`.
- Routing replay decodes each rollout's true expert width, pads K=4/K=6 IDs to the
  learner's fixed K=8 shape, and masks padded slots to exactly zero in the learner's MoE
  combination.
- Reference scaling uses the native top-eight denominator in both SGLang and Megatron.
- Per-K rollout accuracy, response length, truncation, routing-weight sums, and debug
  samples are persisted for smoke validation.

The current learner still executes a fixed eight expert slots and masks the inactive
ones. This establishes the correct learning objective and numerics, but does not yet
claim learner-side MoE FLOP savings. Rollout generation uses the actual selected K.

Targeted local validation currently passes 13 tests with one Ray-dependent aggregation
test skipped outside the Slime container. The tests cover K assignment, correctness-only
reward shaping, per-prompt normalization, variable-width replay padding/masking, and the
existing reference-scaling paths. The metrics-hook regression test uses the same bound
`data_source.get_samples` method shape that Slime supplies in production.

## Smoke tests

The first rollout attempt reached the three named routers and GPU offsets 0, 2, and 4,
then failed before loading weights because the YAML supplied SGLang's JSON model override
as a parsed dictionary instead of the required JSON string. This was a configuration-only
failure and is retained in the job ledger:

- failed configuration smoke: [01KZ4GMEN2M2C60DFSSCK661CH](https://beaker.org/ex/01KZ4GMEN2M2C60DFSSCK661CH)
- failed overlapping-port smoke: [01KZ4H6Z5YMZMBAAEVAEPKEJTW](https://beaker.org/ex/01KZ4H6Z5YMZMBAAEVAEPKEJTW)
- failed telemetry-counter smoke: [01KZ4HHF5WT12BPPHBF7XPXE03](https://beaker.org/ex/01KZ4HHF5WT12BPPHBF7XPXE03)
- failed bound-method telemetry smoke: [01KZ4JQ9046197DJKGKAB0K61A](https://beaker.org/ex/01KZ4JQ9046197DJKGKAB0K61A)
- corrected rollout smoke: [01KZ4K413JA8D5SJ0P7MMM8D9B](https://beaker.org/ex/01KZ4K413JA8D5SJ0P7MMM8D9B)

The third and fourth attempts verified the intended GPU placement (base GPU IDs 0, 2,
4, and 6), K
overrides (4, 6, 8, and 8), disjoint distributed initialization ports, healthy generation
from every engine, and real DAPO reward extraction. They then exposed two variants of the
same optional-metrics issue: the data source has no generic `metadata` dictionary, and the
hook is passed its bound `get_samples` method rather than the instance. The hook now stores
its private counter on the bound method's owner and has an exact regression test.

The corrected rollout smoke completed successfully. It generated 18 responses (six per
K), wrote a 4/6/8-wide routing tensor for the corresponding samples, and exited with code
zero. There were no truncated samples. During its real forward passes, mean selected
router mass was 0.770 at K=4, 0.903 at K=6, and 1.000 at K=8. This confirms both true
rollout sparsity and the intended reference-scaled semantics. The two-step learner smoke
was launched only after these checks passed:

- two-step learner smoke: [01KZ4KHT4EY24SW342KJW1D13D](https://beaker.org/ex/01KZ4KHT4EY24SW342KJW1D13D)

The two-step learner smoke also completed successfully with exit code zero. Both steps
performed reference forwards, actor backwards, optimizer updates, and synchronization to
all four rollout engines. Gradient norms were finite (0.7575 and 0.7102); step 1 reported
KL 0.0221 and a policy-gradient loss of approximately -1.3e-8. A near-zero on-policy
scalar loss is expected after within-prompt advantage centering, while the nonzero gradient
shows that the cost-shaped rewards produce a learning signal. The final distributed
checkpoint was saved at iteration 1 (399 GiB) under:

`/weka/oe-adapt-default/jacobm/olmoe3/adaptive-compute/slime-runs/qwen3-30b-a3b-base-dapo/checkpoints/slime-qwen3-base-dapo-mixed-k4-k6-k8-reference-rl-smoke-2step-8gpu-v1-20260803`

The candidate telemetry also preserved exact arm balance before dynamic filtering. The
first candidate pool contained 144 samples (48 per K), and the second contained 90 (30
per K). This confirms that dynamic sampling sees the intended nine-sibling prompt groups
without biasing the arm allocation.

## Completed 100-step pilots

The cost-shaped production pilot keeps the validated
K={4,6,8} topology and reward bonuses, uses 16 prompt groups per rollout with nine sibling
responses per prompt (global batch 144), permits 20,480 response tokens, and uses dynamic
sampling with an oversampling batch of 64. The learning rate remains 1e-6, and distributed
checkpoints are saved every 20 steps. It completed with exit code zero as
[01KZ4RMSMCY4JFP52GV84TJYZG](https://beaker.org/ex/01KZ4RMSMCY4JFP52GV84TJYZG).

A matched neutral mixed-K control uses the same topology, sampling, optimizer, and
checkpoint schedule, but sets the correct-response bonuses to zero for every K. This
separates the effect of mixed-K training exposure from the explicit preference for a
correct lower-compute response. Both production pilots are placed in
`ai2/OLMo-3-moe-experiments` on `ai2/titan`, urgent and unallocated.
The neutral control also completed with exit code zero as
[01KZ4RVKFNWM8CABP7FMPVSBT7](https://beaker.org/ex/01KZ4RVKFNWM8CABP7FMPVSBT7).

## Checkpoint evaluation plan

Export steps 20, 40, 60, 80, and 100 from both pilots to HF format. Cross-evaluate
every checkpoint with reference-scaled K={4,6} and native K=8. Each condition uses
the exact earlier RL learning-curve recipe: one combined
MATH-500 + AIME 2025 pass@32 evaluation, 20,480 maximum generated tokens for each
task, 32,768 total model context, four TP=1 vLLM engines, and the pinned olmo-eval
snapshot with no RoPE override. MATH-500 therefore retains temperature 0.6, top-p
0.95, and top-k 20; AIME retains temperature 0.6, top-p 0.95, and 32 samples. These
first-pass trajectory evaluations are single runs, matching the earlier sanity sweep;
MATH-500 can be backfilled to three replicates after the initial curves are inspected.

Both five-checkpoint exports completed successfully. The complete 30-condition matrix
(two training conditions x five checkpoints x three inference K values) was submitted
and recorded with phase `slime-mixed-k-checkpoint-eval`; all jobs use the group
[adaptive-compute-slime-qwen3-base-mixed-k-eval-sweep-20260804](https://beaker.org/orgs/ai2/workspaces/holmes-testing/groups/01KZ5M9TV93EXP487YKP3HEVWH).

The three initially failed evaluations later succeeded after Beaker resumed them, so no
conditions are missing from this first-pass matrix. MATH-500 is being backfilled with two
matched replicates using explicit task seeds 43 and 44; together with the original
seed-42 run, this gives three MATH-500 samples per condition. The 60 backfill jobs are in
[adaptive-compute-slime-qwen3-base-mixed-k-eval-replicates-20260804](https://beaker.org/orgs/ai2/workspaces/holmes-testing/groups/01KZ726SQSY3GQE6D8TSAR1ZQZ).
AIME remains a single pass@32 run per condition rather than being replicated.

## Step-100 to step-500 continuations

Both pilots were resumed from their iteration-99 distributed checkpoints (the checkpoint
after 100 completed updates) with the original mixed-K routing, optimizer, batch, and
sampling settings. The optimizer scheduler horizon is extended while restoring optimizer
moments and training progress. Both continuations target 500 total updates and save at
steps 150, 200, 250, 300, 350, 400, 450, and 500. They run as unallocated urgent jobs in
`ai2/OLMo-3-moe-experiments` on `ai2/titan`:

- cost-shaped continuation:
  [01KZ5R5K6DQ0YVKW3F3BTTMV7Y](https://beaker.org/ex/01KZ5R5K6DQ0YVKW3F3BTTMV7Y)
- neutral continuation:
  [01KZ5R5XXD2ER107N6CMQK8S66](https://beaker.org/ex/01KZ5R5XXD2ER107N6CMQK8S66)

The cost-shaped continuation was intentionally stopped after its step-400 checkpoint was
successfully written. The completed `iter_0000399` save contains all 16 distributed shards,
`common.pt`, `metadata.json`, and the distributed-checkpoint `.metadata` file. Beaker records the
manual cancellation at 2026-08-05 17:12:41 UTC. The neutral continuation remains active toward
step 500.

New checkpoints are written into each pilot's existing distributed-checkpoint directory,
while continuation candidate telemetry and rollout details use separate
`mixed-k-continuations/` output directories. This preserves a single checkpoint sequence
without appending the step-101+ diagnostic records to the original 100-step files.

Both continuations reached their step-250 save point. Steps 150, 200, and 250 were
exported successfully to HF format for both objectives using:

- cost-shaped export: [01KZ72M08NZ89Q9D5YHGSKTYNG](https://beaker.org/ex/01KZ72M08NZ89Q9D5YHGSKTYNG)
- neutral export: [01KZ72M3GWM9K380T3WKS412QM](https://beaker.org/ex/01KZ72M3GWM9K380T3WKS412QM)

Each of these six new checkpoints is evaluated at reference-scaled K={4,6} and native
K=8. For each checkpoint/K condition, the submitted recipe is one seed-42 combined
MATH-500 + AIME 2025 pass@32 job plus seed-43 and seed-44 MATH-only jobs. This is 18
conditions, 54 jobs, three MATH-500 samples per condition, and one AIME pass@32 run per
condition. These jobs share the replicate group linked above and retain the corrected
20,480-token generation cap, 32,768-token context, four TP=1 vLLM engines, and pinned
olmo-eval snapshot.

## Step-300/350 continuation evaluations

The next available continuation checkpoints were cost-shaped step 300 and neutral steps
300 and 350. All three were exported successfully to HF format with one-GPU Titan jobs:

- cost step 300 export: [01KZ86MB10SD5AD5MK6B8Z89E2](https://beaker.org/ex/01KZ86MB10SD5AD5MK6B8Z89E2)
- neutral steps 300 and 350 export: [01KZ86MAZZKXNS1DY6GW73DCZ2](https://beaker.org/ex/01KZ86MAZZKXNS1DY6GW73DCZ2)

For each checkpoint, the same reference-scaled K={4,6} and native K=8 matrix was
submitted: seed-42 combined MATH-500 + AIME 2025 pass@32, plus seed-43 and seed-44
MATH-only runs. This adds 27 tracked eval jobs to
[adaptive-compute-slime-qwen3-base-mixed-k-step300-350-evals-20260805](https://beaker.org/orgs/ai2/workspaces/holmes-testing/groups/01KZ86Z8MGXNAQH4K8DDYRVKFJ).

All 27 step-300/350 eval jobs completed successfully. The restarted neutral step-250
native-K=8 combined job also succeeded on its second execution, filling its third
MATH-500 seed and AIME point. The current learning-curve artifacts therefore cover the
cost-shaped run through step 300 and the neutral run through step 350. Every plotted
MATH-500 point is the mean of three evaluations with observed min--max shading; every
AIME point is Pass@1 estimated from one pass@32 evaluation. Artifacts:

- `notes/plots/slime_qwen3_base_dapo_mixed_k_math500_by_step.{png,svg}`
- `notes/plots/slime_qwen3_base_dapo_mixed_k_aime_by_step.{png,svg}`
- `notes/slime_qwen3_base_dapo_mixed_k_checkpoint_evals.csv`

## Neutral mixed-K versus fixed-K training

The neutral mixed-K checkpoint can be compared directly with the earlier fixed-K training arms
by holding inference routing constant: reference-scaled K=4 against fixed reference-scaled K=4,
reference-scaled K=6 against fixed reference-scaled K=6, and native K=8 against fixed native K=8.
At step 350, the three-run MATH-500 means for neutral mixed-K versus fixed-K are 81.60% versus
81.33% at K=4, 82.80% versus 80.20% at K=6, and 84.00% versus 84.07% at K=8. Thus one neutral
mixed-K checkpoint matches the specialized K=4 and K=8 checkpoints and exceeds the specialized
K=6 checkpoint on MATH-500. AIME Pass@1 is +0.63, +1.77, and -3.54 points respectively, but each
AIME point comes from only one pass@32 evaluation and is noisier than the three-run MATH mean.

The comparison plot is
`notes/plots/slime_qwen3_base_dapo_neutral_mixed_vs_fixed_k.{png,svg}`. This is not a perfectly
compute-matched experiment: each mixed-K step trains on 144 responses (three responses at each K
for 16 prompts), while each earlier fixed-K step trains on 128 responses all generated at one K.

The complementary inference-K view is
`notes/plots/slime_qwen3_base_dapo_neutral_vs_k8_step100_350_inference_plateau.{png,svg}`. It
overlays the neutral step-350 K={4,6,8} points on the fixed-K8-trained checkpoint's complete
reference-scaled K=4--8 sweep. At step 350 the curves are nearly identical at their shared points:
81.60% versus 80.93% at K=4, 82.80% versus 82.93% at K=6, and 84.00% versus 84.07% at K=8. The
neutral step-100 curve is also shown. The fixed-K8 step-100 checkpoint is shown only at native K=8
because no matched reference-scaled K=4--7 sweep was run for that earlier checkpoint.

## Existing prompt-level evidence

The matched analysis uses the same 500 MATH-500 prompts at every K, all three stochastic
replicates, the three step-350 RL checkpoints, and reference-scaled inference routing.
Its detailed table, per-prompt CSV, and heatmap are in
[the prompt transition note](slime_qwen3_base_dapo_prompt_k_transitions.md).

The short version for K=4/6/8 is:

- 73.6--78.0% of prompts are majority-correct at all three K values;
- 10.6--12.6% are majority-wrong at all three;
- only 4.2--6.8% show a clean monotonic transition that needs K=6 or K=8;
- 5.4--7.0% show a non-monotonic majority pattern; and
- 27.8--38.6% have at least one K where the three generations disagree.

Prompt difficulty is not random—the empirical K=4/K=8 success correlations are
0.735--0.797—but a hard minimum-K label is both imbalanced and noisy. Before building a
classifier, the next offline step should compute an oracle quality-versus-average-K curve
from multi-sample probabilities. A learned selector should use probabilistic targets or
confidence bounds, not the minimum K that happened to solve one sampled generation.

At the three-sample probability level, K=4 and K=8 tie on 69--79% of prompts. K=8 is
better on 11.6--16.0%, but K=4 is also better on 7.8--15.0%; the net mean K=8 gain is
only 0.9--3.1 points. This reinforces that a prompt classifier has a small, noisy
addressable slice unless it predicts success probabilities rather than a hard minimum K.
