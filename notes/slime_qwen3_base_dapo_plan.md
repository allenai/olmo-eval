# Slime baseline: Qwen3-30B-A3B Base on DAPO-Math-17k

Last updated: 2026-07-27

## Goal

Establish a fixed-K=8 RL baseline for `Qwen/Qwen3-30B-A3B-Base` before changing K during
post-training. The baseline must validate the complete Beaker path: data, Qwen chat formatting,
SGLang rollouts, DAPO reward parsing, MoE routing replay, Megatron optimization, checkpointing,
and W&B logging.

Placement is `ai2/linear-rnns`, unallocated on `ai2/titan`, at urgent priority. Slime's current
quick-start documentation explicitly supports B200 with the same setup as H-series GPUs; these
smokes are the empirical Blackwell gate before a pilot. Training logs to
the `ai2-llm/adaptive-compute` W&B project. Credentials are injected as Beaker environment
secrets. Never pass the W&B credential with Slime's `--wandb-key` option: Ray prints the full
entrypoint and Slime copies parsed arguments into the W&B config. The environment variable alone
is sufficient for the W&B SDK.

## Prepared assets

The preparation job [01KYBFV9S9QDMR5J9VS127W42Y](https://beaker.org/ex/01KYBFV9S9QDMR5J9VS127W42Y)
completed successfully and created:

- HF checkpoint: `/weka/oe-adapt-default/jacobm/olmoe3/adaptive-compute/slime-runs/qwen3-30b-a3b-base-dapo/hf/Qwen3-30B-A3B-Base`
- Megatron distributed checkpoint: `/weka/oe-adapt-default/jacobm/olmoe3/adaptive-compute/slime-runs/qwen3-30b-a3b-base-dapo/megatron/Qwen3-30B-A3B-Base_torch_dist`
- deduplicated data: `/weka/oe-adapt-default/jacobm/olmoe3/adaptive-compute/slime-runs/qwen3-30b-a3b-base-dapo/data/dapo-math-17k.unique.jsonl`

The current Hugging Face snapshot contains 1,791,700 rows but only 17,917 distinct official
`extra_info.index` values, i.e. each intended problem is repeated about 100 times. The local JSONL
keeps the first copy of every official index.

The preparation gate also verified that the model's chat template converts the official prompt
messages to Qwen ChatML and appends the assistant generation prompt. The model EOS token is
`<|endoftext|>` (ID 151643).

## Reward and prompt decisions

- Use `--apply-chat-template` with the checkpoint's own template.
- Keep the official DAPO instruction requiring a final `Answer: ...` line.
- Use Slime's `--rm-type dapo`, which extracts the last `Answer:` value, applies the DAPO/Minerva
  integer normalization, and returns +1 for correct and -1 for incorrect.
- Do not use Slime's `deepscaler` reward for this dataset: it expects a different reasoning/final
  format and boxed-answer behavior.
- Use temperature 1 and top-p 1.0. The short reward smoke uses an 8,192-token response cap; the
  meaningful pilot should use the official DAPO-scale 20,480-token cap.
- Keep K=8 in both Megatron (`--moe-router-topk 8`) and SGLang for the baseline.
- Enable rollout routing replay so the learner can replay the rollout expert choices rather than
  silently recomputing a different MoE route. Use Slime's `object-store` rollout-data transport:
  the one-node NIXL path failed while initializing UCX on Titan, while object-store completed
  rollout, optimization, weight synchronization, and checkpointing.

## Initial optimizer/training plan

The first end-to-end smoke uses two rollout steps, four prompts per step, eight samples per prompt,
global batch 32, GRPO clipping 0.20/0.28, constant LR `1e-6`, weight decay 0.1, and CPU-offloaded
Adam. Dynamic sampling is deliberately disabled so the raw base-model reward distribution can be
measured.

Non-zero reward is principally a rollout/template/reward-parser property, not something a chosen
number of optimizer steps can manufacture. If the base model produces no positive samples, GRPO
has no useful within-prompt signal; fix formatting, response length, or initialization before a
long run. If the smoke has mixed-reward groups, use a 100-step pilot at LR `1e-6`, enable Slime's
non-zero-reward-variance dynamic-sampling filter, evaluate every 20 steps, and stop early if reward
or format accuracy does not move. Only choose a longer run after seeing the reward and throughput
curves from that pilot.

### Recipe provenance and scope

This pilot is a resource-scaled combination of two public reference recipes rather than an exact
reproduction of either one:

- Slime's shipped Qwen3-30B-A3B recipe supplies the Megatron model/layout arguments, GRPO loss,
  constant `1e-6` learning rate, Adam settings, `0.20/0.28` clipping, eight samples per prompt,
  dynamic token batching, and the one-rollout/one-update structure. Slime's quick-start example
  uses the same 16 prompts x 8 samples = 128-sequence global batch as this pilot.
- The official DAPO recipe motivates DAPO-Math-17k, dynamic non-zero-variance filtering, the
  20,480-token generation ceiling, zero KL coefficient, and Clip-Higher at `0.20/0.28`.

Accordingly, call the current run a **DAPO-style GRPO pilot**, not a faithful full-DAPO
reproduction. Full DAPO uses 512 rollout prompts, 16 responses per prompt, 16 optimizer updates per
rollout, a 20-step linear warmup, token-level policy-gradient reduction, and soft overlong reward
shaping. The one-node pilot deliberately uses 16 retained prompts, eight responses per prompt, and
one update per rollout; it currently has no LR warmup, token-level loss, or soft overlong penalty.
Those are follow-up knobs after the fixed-K systems and held-out-evaluation path are validated.

## GPU floor

Slime's shipped Qwen3-30B-A3B recipe is an eight-H100 colocated job with TP=4 and EP=8; eight GPUs
is therefore the supported/default baseline. Because model/optimizer state includes all 30.5B
parameters even though about 3.3B are active per token, active parameter count alone is not a
memory estimate.

The lower-footprint layouts were tested empirically without changing K:

| GPUs | Learner layout | SGLang layout | Meaning |
|--:|:--|:--|:--|
| 8 | TP=4, EP=8 | TP=8 | supported/default baseline; practical minimum |
| 4 | TP=4, EP=4 | TP=4 | healthy reduced-footprint probe |
| 2 | TP=2, EP=2 | TP=2 | fits, but learner throughput is pathological |

The four- and two-GPU layouts are adaptations, not the upstream default. Even if they fit, their
throughput may make eight GPUs the practical minimum.

The original Jupiter submissions were stopped while queued and replaced on Titan. Qwen3 MoE on
B200 requires SGLang's Triton MoE runner; the default backend failed during kernel selection. The
final clean smokes use learner CP=1, SGLang EP equal to the GPU count, SGLang memory fraction 0.7,
object-store transport, and the Triton MoE backend:

| GPUs | Beaker / W&B | Result |
|--:|:--|:--|
| 4 | [Beaker](https://beaker.org/ex/01KYBMRSNVBZHMMQ8B1AWD053K) / [W&B](https://wandb.ai/ai2-llm/adaptive-compute/runs/vq515qrr) | Two rollout/train steps and a checkpoint completed. Step-0 actor training took about 105 s; rollout-0 aggregate throughput was about 1,475 generated tokens/s. |
| 8 | [Beaker](https://beaker.org/ex/01KYBMRYD1SSABSYBP6FV8D3P7) / [W&B](https://wandb.ai/ai2-llm/adaptive-compute/runs/8x0oc8hp) | Two rollout/train steps and a checkpoint completed. Step-0 actor training took about 51 s; rollout-0 aggregate throughput was about 1,542 generated tokens/s. |
| 2 | [Beaker](https://beaker.org/ex/01KYBNKPCW1VS0N9S2JHZR72AE) / [W&B](https://wandb.ai/ai2-llm/adaptive-compute/runs/t1e0889j) | Fits and completes updates, but actor training took about 305 s on the first step and 91 s once warm. Aggregate rollout throughput was about 1,197 generated tokens/s on rollout 0. |

The 4- and 8-GPU smokes both produced mixed-reward groups and non-zero positive rewards from the
base model, validating the prompt and DAPO reward path. Eight GPUs are the minimum reasonable
production layout: compared with four, total rollout throughput is similar but learner updates are
about twice as fast. Two GPUs save capacity but were about six times slower on the first learner
update and about three times slower on the warm second update than eight GPUs.

The 100-step baseline pilot is now running as
[Beaker 01KYBPVETBTR3JF19T1NFNB9AK](https://beaker.org/ex/01KYBPVETBTR3JF19T1NFNB9AK) with
[W&B run 1vi98nq6](https://wandb.ai/ai2-llm/adaptive-compute/runs/1vi98nq6). It uses one
8xB200 Titan node, K=8, 16 prompts x 8 samples (global batch 128), a 20,480-token response cap,
dynamic sampling, LR `1e-6`, and a 20-step checkpoint interval. It logs to
`ai2-llm/adaptive-compute` and is tracked in `beaker_jobs.jsonl`.

The first production step passed the complete runtime gate. Dynamic sampling generated a
64-prompt candidate pool, dropped 30 all-wrong groups, and retained 16 mixed-reward groups (128
samples). The retained batch had 23/128 correct samples (mean raw reward `-0.640625`), mean/maximum
response lengths of 665/2,215 tokens, and no truncation. Rollout/filtering took 180.5 s, reference
log probabilities took 22.8 s, and the actor update took 49.8 s. Step 0 had finite loss, gradient
norm 0.528, and successfully synchronized the updated weights back to SGLang before rollout 1.

## SGLang K override

The pinned SGLang 0.5.15.post1 exposes `--json-model-override-args`; Slime automatically exposes
SGLang `ServerArgs` with the `--sglang-` prefix. A future rollout override is therefore:

```bash
--sglang-json-model-override-args '{"num_experts_per_tok": 4}'
```

For RL, changing only SGLang is insufficient. Megatron must receive the same
`--moe-router-topk K`, and any routing-replay assumptions must be revalidated. Native lower-K
Qwen routing is normalized. Reproducing the K=8-reference-preserving intervention requires a
small SGLang router-weight patch; the JSON override alone does not preserve the removed mass.

SGLang also exposes routed-expert return telemetry. The one-B200 native-normalized K=4 smoke
[01KYBKH1QJPDQ5PWH8E2TCDA4Y](https://beaker.org/ex/01KYBKH1QJPDQ5PWH8E2TCDA4Y) completed on
Titan with the Triton MoE runner and directly returned four expert IDs per routed layer/token.
This validates the native normalized K override. Reproducing the reference-preserving policy still
requires a router-weight patch.

The full K=4 rollout/learner path was subsequently validated in
[Beaker 01KYBSE88B0JN9GH941NRZB722](https://beaker.org/ex/01KYBSE88B0JN9GH941NRZB722) with
[W&B run 4tzuowfl](https://wandb.ai/ai2-llm/adaptive-compute/runs/4tzuowfl). Megatron resolved
`moe_router_topk=4`, SGLang resolved `num_experts_per_tok=4`, and routing-replay storage had shape
`[..., 48, 4]`. Two rollout/train/weight-sync cycles and the final distributed checkpoint
completed. Both tiny four-prompt batches happened to be all-wrong, so the smoke validated the
systems path but not a non-zero-gradient K=4 update.

The matched 100-step K=4 dynamic-sampling pilot is
[Beaker 01KYBTPHHBYTKK0GZ58AJGHY86](https://beaker.org/ex/01KYBTPHHBYTKK0GZ58AJGHY86). Its
64-prompt candidate pool and 16 retained mixed-reward groups provide the meaningful K=4 learning
gate while keeping every other training setting matched to the K=8 pilot.

The K=4 pilot was manually canceled on 2026-07-25 at 07:06 UTC, while it was still in a rollout
wave and before it wrote its first checkpoint. It was replaced for the overnight test by the
matched native-normalized K=6 pilot
[Beaker 01KYC1KAWSQD6SYSKTJZ5252XK](https://beaker.org/ex/01KYC1KAWSQD6SYSKTJZ5252XK).
K=6 keeps the same 100 updates, 16 retained prompt groups, eight samples per prompt, 64-prompt
dynamic-sampling candidate pool, 128-sample global batch, 20,480-token response cap, `1e-6`
learning rate, 20-step checkpoint interval, eight B200 GPUs, and routing replay as K=8. Only the
matched Megatron and SGLang router K changes from eight to six.

The K=6 pilot passed its initial live-training gate on 2026-07-25. Optimizer steps 0, 1, and 2
all completed with finite losses (`-1.26e-8`, `-5.59e-9`, and `-1.30e-8`) and finite gradient
norms (`1.039`, `2.031`, and `1.016`). Each update synced all six weight groups back to SGLang in
8.6--10.1 seconds, and rollout 3 began with the step-2 weights. The first three dynamically
sampled batches had raw rewards `-0.672`, `-0.594`, and `-0.656` (approximately 16.4%, 20.3%, and
17.2% correct), discarded 59, 46, and 36 homogeneous prompt groups, and had zero truncations.
Rollout throughput after the cold first batch was 58.6--69.7 generated tokens/GPU/s. No OOM,
NaN, routing-replay mismatch, or failed learner-to-SGLang synchronization appeared. Some sampled
incorrect completions were strongly incoherent, which is a model-quality signal to inspect in the
held-out K=6 evaluation rather than a systems failure; the DAPO scorer correctly rejected them.

This K=4 condition is **native normalized K=4**, not K=8-reference-scaled K=4. SGLang retains
Qwen's `norm_topk_prob=true`, and Megatron applies softmax to the selected top-K logits. Both the
rollout and learner therefore combine four experts with weights that sum to one.

### K=4 with K=8-reference-scaled weights

The reference-scaled policy is now implemented as an explicit, opt-in Slime routing mode. For
router logits `z`, target K=4, and reference K=8, it selects the same four highest-ranked experts
but uses

```text
w_i = exp(z_i) / sum(exp(z_j) for j in top-8)
```

instead of softmax-normalizing only the selected top four. Thus the four weights preserve their
relative values from native K=8 and normally sum to less than one. This is the exact intervention
used by the earlier reference-scaled inference evaluations. Merely setting Megatron's
`--moe-router-pre-softmax` or SGLang's `renormalize=false` would normalize over all 128 experts and
would not reproduce it.

The implementation is shared across both sides of RL:

- SGLang's fused top-K path temporarily obtains the top eight weights, retains the first four
  expert IDs, and divides their weights by the top-eight mass. The patch is injected into SGLang's
  spawned model workers through an opt-in Python startup hook.
- Megatron uses pre-softmax router probabilities and applies the same top-eight mass denominator
  before the selected rollout expert IDs are replayed. This keeps rollout generation, reference
  log probabilities, and policy training on the same routing policy.
- The mode is selected with `MOE_ROUTER_WEIGHT_MODE=reference_scaled` and
  `MOE_ROUTER_REFERENCE_TOPK=8`; native normalized routing remains the default.

Seven numerical unit-test cases cover exact K={4,6}/reference-K=8 scaling, K=8 identity,
Megatron replay scaling, and unsorted SGLang candidate handling. The successful one-B200 SGLang smoke is
[Beaker 01KYC55K4GQRG7TFFD2AMDG8XA](https://beaker.org/ex/01KYC55K4GQRG7TFFD2AMDG8XA).
It returned exactly four expert IDs per token/layer and directly measured a selected-weight sum
of 0.7273 on average across the six probe rows (range 0.6657--0.7841), confirming that the K=4
weights are not renormalized to one. Two preceding smokes failed before generation because the
patch path did not reach SGLang's separately spawned model workers; the scoped startup hook fixed
that propagation issue.

The matched two-step, eight-B200 end-to-end RL smoke is
[Beaker 01KYC5BZCJAV4WQSW0JHYR1SWM](https://beaker.org/ex/01KYC5BZCJAV4WQSW0JHYR1SWM) with
[W&B run mxn4uc5d](https://wandb.ai/ai2-llm/adaptive-compute/runs/mxn4uc5d). Its first dynamically
sampled batch retained 16 mixed-reward groups (128 samples), had mean raw reward `-0.640625`, and
had no truncations. Reference-log-prob replay completed, optimizer step 0 had finite loss and
gradient norm 1.990, learner/rollout mean log-prob absolute difference was 0.0130, and all updated
weights synchronized back to SGLang before the second rollout. No OOM, NaN, routing-shape mismatch,
or failed weight update appeared. Step 1 independently completed with mean raw reward `-0.6875`,
zero truncations, learner/rollout log-prob difference 0.0125, and finite gradient norm 0.968. The
two-step smoke therefore exercised two complete generation, routing-replay, reference-forward,
backpropagation, optimizer, and weight-synchronization cycles.

The matched 100-step single-node run is
[Beaker 01KYC6EM1D4SDNWS8P3RJQEBPX](https://beaker.org/ex/01KYC6EM1D4SDNWS8P3RJQEBPX).
It uses one 8xB200 Titan node at urgent priority, K=4 with K=8-reference-scaled weights, 16 retained
prompts x 8 samples, a 64-prompt dynamic-sampling candidate pool, global batch 128, 20,480-token
response cap, constant LR `1e-6`, and checkpoints every 20 steps. Checkpoints are written under:

`/weka/oe-adapt-default/jacobm/olmoe3/adaptive-compute/slime-runs/qwen3-30b-a3b-base-dapo/checkpoints/slime-qwen3-base-dapo-k4-reference-k8-pilot-100step-8gpu-v1-20260725`

## Step-20 K=8 export and held-out evaluation

The first 20-update K=8 checkpoint is Megatron iteration 19 (`iter_0000019`, because rollout IDs
are zero based). It was frozen behind an immutable checkpoint-root pointer and exported to HF at:

`/weka/oe-adapt-default/jacobm/olmoe3/adaptive-compute/slime-runs/qwen3-30b-a3b-base-dapo/hf-checkpoints/slime-qwen3-base-dapo-k8-pilot-100step-8gpu-v5-20260725-step20-hf`

The successful direct distributed-checkpoint export is
[Beaker 01KYBYYPT1NT4Y8B5BB9R8Y48V](https://beaker.org/ex/01KYBYYPT1NT4Y8B5BB9R8Y48V). The
export has exactly the same 18,867 HF parameter keys and 61,064,245,248 model-weight bytes as the
source Qwen checkpoint. Its copied config retains K=8 and `norm_topk_prob=true`.

The held-out math evaluation is
[Beaker 01KYBZAC2NQJ2E243F1PBJEAZJ](https://beaker.org/ex/01KYBZAC2NQJ2E243F1PBJEAZJ), in
`ai2/linear-rnns` on `ai2/jupiter` at urgent priority. One four-H100 olmo-eval job runs
MATH-500 (one sample) and AIME 2025 (32 samples, reporting pass@1/4/8/16/32), with a 32,768-token
generation cap and native normalized K=8. All four vLLM servers successfully loaded the exported
checkpoint before evaluation began.

That first evaluation is invalid. The checkpoint has native `max_position_embeddings=32768` with
default RoPE and no scaling, but the server had been forced to a 40,960-token context so that a
32,768-token response could follow the prompt. The explicit override bypassed vLLM's length guard
without extending its 32,768-entry rotary cache. A non-terminating completion on each server reached
absolute position 32,768, triggering TorchInductor's `index out of bounds: 0 <= index < 32768`
device assertion and killing all four engines. Only 236/500 MATH instances returned outputs and no
AIME instance did; the emitted aggregate scores must not be used.

The corrected rerun is
[Beaker 01KYC0TNBX48WTTZT8HK1XBDDW](https://beaker.org/ex/01KYC0TNBX48WTTZT8HK1XBDDW).
It keeps vLLM at the checkpoint's native 32,768-token context and caps both MATH-500 and AIME 2025
responses at 20,480 tokens, matching the RL rollout cap and leaving ample room for these short
prompts. It makes no RoPE, checkpoint, parser, or olmo-eval code changes.

## Five-checkpoint sanity-evaluation sweep

All three 100-step runs save checkpoints at steps 20, 40, 60, 80, and 100 (Megatron iterations
19, 39, 59, 79, and 99). Each distributed checkpoint was exported to an immutable HF directory
under `slime-runs/qwen3-30b-a3b-base-dapo/hf-checkpoints/`. The 15 corresponding single-replicate
evaluations are grouped under
[adaptive-compute-slime-qwen3-base-eval-sweep-20260725](https://beaker.org/orgs/ai2/workspaces/holmes-testing/groups/01KYD3BM9NR3CZ48SKN0PF84FR)
and recorded in `notes/beaker_jobs.jsonl` with phase `slime-checkpoint-eval-sweep`.

Every evaluation exactly reuses the corrected step-20 recipe above: MATH-500 plus AIME 2025
pass@32, a native 32,768-token total context, a 20,480-token response cap for both tasks, four
TP=1 vLLM engines, and no RoPE override. The inference routing policy matches training: native
normalized K=8, native normalized K=6, or K=4 with K=8-reference-scaled weights. Jobs run in
`ai2/holmes-testing` on `ai2/jupiter` at urgent priority. This is a trajectory sanity check rather
than a three-replicate final comparison.

## Remaining gates

1. Follow the 100-step K=8 pilot through its first dynamically sampled batch, optimizer step, W&B
   metrics, and step-20 checkpoint.
2. Inspect sampled completions and reward extra-info rather than relying only on aggregate reward.
3. Add a held-out math evaluation (for example AIME 2024/2025) before promoting a longer run.
4. After the fixed-K baseline is stable, design the matched Megatron/SGLang lower-K or adaptive-K
   intervention; do not change rollout K without changing learner K and revalidating replay.

## Step-100 to step-500 continuations

All three completed 100-step training conditions were resumed in place from Megatron iteration 99
(user-facing step 100) on 2026-07-26. Before launch, each checkpoint root was verified to contain
`common.pt`, all 16 distributed-checkpoint shards, and
`rollout/global_dataset_state_dict_99.pt`, so the continuation restores model, optimizer, RNG, and
rollout-dataset state rather than starting a new trajectory. The total rollout limit is 500 and the
save interval is 50, yielding new checkpoints at user-facing steps 150, 200, 250, 300, 350, 400,
450, and 500. All other training, batching, generation, learning-rate, and routing settings are
unchanged from the corresponding 100-step runs.

The continuations use one 8xB200 node each in `ai2/linear-rnns` on `ai2/titan`, at urgent priority,
with a 96-hour timeout:

- Native normalized K=8: [Beaker 01KYEA06CVVM70JS3X2RM44M9F](https://beaker.org/ex/01KYEA06CVVM70JS3X2RM44M9F)
- Native normalized K=6: [Beaker 01KYEA09YJGK893DSZ8JDGZHGT](https://beaker.org/ex/01KYEA09YJGK893DSZ8JDGZHGT)
- K=4 with K=8-reference-scaled weights: [Beaker 01KYEA0E3HBKZSK6QVB27YQ0QV](https://beaker.org/ex/01KYEA0E3HBKZSK6QVB27YQ0QV)

The first continuation attempts reached and read iteration 99 correctly, then Megatron rejected the
new 500-step horizon because the checkpoint recorded the old 100-step optimizer-scheduler bound
(`12,800` samples versus the new `64,000`). The replacement jobs above use Megatron's explicit
optimizer-scheduler override: this keeps the new horizon while still restoring the checkpoint's
`num_steps`, optimizer moments, model, RNG, and dataset state. This does not change the constant
`1e-6` learning rate or constant `0.1` weight decay. The failed preflight-scale attempts remain in
the Beaker tracker for auditability.

The continuations write back to their original checkpoint roots, preserving the existing step
20/40/60/80/100 checkpoints and adding the new 50-step cadence. Intermediate held-out evaluations
can therefore be launched tomorrow without interrupting training, using immutable exports of the
desired checkpoint iterations.

## Continuation checkpoint evaluations

On 2026-07-26, every continuation checkpoint available at launch time was exported to an immutable
HF directory and queued for one held-out evaluation replicate. This covers K=8 normalized at steps
150/200/250 and K=6 normalized plus K=4 reference-scaled at steps 150/200/250/300, for 11 jobs
total. They are grouped under
[adaptive-compute-slime-qwen3-base-continuation-evals-20260726](https://beaker.org/orgs/ai2/workspaces/holmes-testing/groups/01KYFRB7MS516CK9J80S6GXN3Q)
and individually recorded in `notes/beaker_jobs.jsonl`.

The jobs exactly reuse the corrected checkpoint-evaluation recipe: MATH-500 plus AIME 2025
pass@32, a native 32,768-token total context, a 20,480-token response cap, four TP=1 vLLM engines,
and the routing policy used during training. They run in `ai2/holmes-testing` on `ai2/jupiter` at
urgent priority. No K=8 step-300 evaluation was launched because that checkpoint did not yet exist.

All 11 evaluations succeeded and were collected on 2026-07-26. Each contains all 500 MATH-500
examples and all 30 AIME problems with 32 samples per problem, with an empty metric-error list.
Continuation results are:

| Training policy | Step | MATH-500 | AIME pass@1 | AIME pass@8 |
|:--|--:|--:|--:|--:|
| K=8 normalized | 150 | 79.8% | 12.40% | 23.16% |
| K=8 normalized | 200 | 81.2% | 15.83% | 26.63% |
| K=8 normalized | 250 | **83.0%** | **16.67%** | 29.77% |
| K=6 normalized | 150 | 81.4% | 12.40% | 30.20% |
| K=6 normalized | 200 | 80.8% | 11.56% | 27.54% |
| K=6 normalized | 250 | 80.4% | 13.13% | 29.05% |
| K=6 normalized | 300 | **81.6%** | **13.96%** | **31.81%** |
| K=4, K=8-reference scaled | 150 | 78.0% | 10.21% | 24.92% |
| K=4, K=8-reference scaled | 200 | 79.8% | 10.83% | **28.39%** |
| K=4, K=8-reference scaled | 250 | 78.6% | 10.42% | 26.94% |
| K=4, K=8-reference scaled | 300 | **80.2%** | **11.15%** | 26.40% |

These remain one evaluation per checkpoint, not three-replicate estimates. The strongest trend is
K=8's continued improvement through step 250: relative to step 100, MATH-500 rises from 80.0% to
83.0% and AIME pass@1 from 12.19% to 16.67%. K=6 and K=4 also improve over their early checkpoints,
but are noisier and flatter after roughly step 150. The AIME pass@8 estimates fluctuate more than
MATH-500 and pass@1 and should not be read as monotonic training curves.

The machine-readable results and updated plots are:

- `notes/slime_qwen3_base_dapo_checkpoint_evals.csv`;
- `notes/plots/slime_qwen3_base_dapo_rl_math500_by_step.png`;
- `notes/plots/slime_qwen3_base_dapo_rl_aime_by_step.png`; and
- `notes/plots/slime_qwen3_base_dapo_rl_aime_pass_at_8_by_step.png`.

## K=6 reference-scaled training and step-0 baselines

On 2026-07-26, a fourth matched RL arm was launched from the untouched Qwen3-30B-A3B-Base
checkpoint: K=6 with weights scaled by the native top-eight denominator. This retains the top six
expert IDs and their native relative weights, but their sum is the mass those six experts would
have received inside native K=8 rather than one. The SGLang rollout and Megatron learner/replay
paths use the same policy. The generic routing tests were explicitly parameterized at K=6 before
launch and all seven cases passed.

The run uses the same DAPO data, one 8xB200 Titan node, 16 retained prompts x 8 samples, global
batch 128, 64-prompt dynamic-sampling candidate pool, 20,480-token response cap, constant LR
`1e-6`, and all other settings as the existing arms. It runs directly to 500 updates and saves
every 50 steps:

- K=6, K=8-reference-scaled: [Beaker 01KYGAW9SY51BFRACV3PP4053D](https://beaker.org/ex/01KYGAW9SY51BFRACV3PP4053D)
- checkpoint root: `/weka/oe-adapt-default/jacobm/olmoe3/adaptive-compute/slime-runs/qwen3-30b-a3b-base-dapo/checkpoints/slime-qwen3-base-dapo-k6-reference-k8-500step-8gpu-v1-20260726`

The first full update passed end to end. The initial dynamically sampled batch retained 128
responses with mean raw reward `-0.640625`, mean response length 717 tokens, and no truncations.
Reference-log-prob replay and all three training microbatches completed; optimizer step 0 had
finite gradient norm 2.173, learner/rollout mean log-prob absolute difference 0.01222, and constant
LR `1e-6`. Updated weights then began the six-part synchronization back into SGLang. No OOM, NaN,
routing-shape mismatch, or worker failure appeared.

Step-0 evaluations were also launched for all four training policies and all four succeeded. Each
evaluates the same untouched local HF checkpoint with the corrected checkpoint-evaluation recipe:
MATH-500 chat plus
AIME 2025 pass@32, native 32,768-token context, 20,480-token output cap, and four TP=1 vLLM
engines. These are single trajectory baselines, not three-replicate estimates:

- K=8 normalized: [Beaker 01KYGB0HBMJ4F3SGEMKNJ7JT9W](https://beaker.org/ex/01KYGB0HBMJ4F3SGEMKNJ7JT9W)
- K=6 normalized: [Beaker 01KYGB0RSVVKWCCE11VHFY32QC](https://beaker.org/ex/01KYGB0RSVVKWCCE11VHFY32QC)
- K=6, K=8-reference-scaled: [Beaker 01KYGB11109BVZYFT235T0GJ2Y](https://beaker.org/ex/01KYGB11109BVZYFT235T0GJ2Y)
- K=4, K=8-reference-scaled: [Beaker 01KYGB185WR27QC1AK3J0EMSHC](https://beaker.org/ex/01KYGB185WR27QC1AK3J0EMSHC)

The base evaluation does use the same Qwen3 thinking chat template as training and the exported
checkpoints. Slime and olmo-eval both call the checkpoint tokenizer's
`apply_chat_template(..., add_generation_prompt=True)`. The untouched base and exported step-20
tokenizer files are byte-identical (`tokenizer_config.json` SHA-256
`3c04ed3ca964ea2f6b2b5faf0dc4d31aec1cb1e8b4bcf63f402d295046b422b5`), including the identical
4,116-character chat template. Thus step 0 and later checkpoints differ in learned weights and
routing policy, not tokenizer or chat wrapping.

The collected step-0 single-run baselines are:

| Inference routing | MATH-500 | AIME Pass@1 | AIME Pass@8 |
|---|---:|---:|---:|
| K=8 normalized | 75.0% | 10.63% | 26.76% |
| K=6 normalized | 67.4% | 5.94% | 22.31% |
| K=6, K=8-reference-scaled | 70.0% | 8.23% | 25.63% |
| K=4, K=8-reference-scaled | 67.0% | 7.40% | 21.70% |

## Cross-routing checkpoint evaluation

To disentangle training-time routing from inference-time routing, two matched counterfactual
curves were queued on 2026-07-26:

- checkpoints trained with normalized K=8, evaluated with normalized K=6; and
- checkpoints trained with normalized K=6, evaluated with normalized K=8.

The complete matrix contains 19 evaluations: both directions at steps
20/40/60/80/100/150/200/250, K=8-trained/K=6-evaluated at step 300, and
K=6-trained/K=8-evaluated at steps 300/350. They use exactly the same MATH-500 + AIME 2025
pass@32 recipe, context/output caps, and four TP=1 engines as the native-routing checkpoint
evaluations. They are tracked in [the cross-routing Beaker group](https://beaker.org/orgs/ai2/workspaces/holmes-testing/groups/01KYGB9X18QP6ZD4QHBRAVHHR6).

Step zero is not rerun redundantly: the K=8-trained/K=6-eval curve reuses the untouched base K=6
evaluation, while the K=6-trained/K=8-eval curve reuses the untouched base K=8 evaluation. Before
collecting results, the plotter was extended so color denotes the training arm and line style
denotes the inference intervention. Native-routing lines are solid; cross-routing lines are dashed
with hollow markers but retain their training arm's color.

Four newly completed native checkpoints required catch-up evaluation: K=8 step 300, K=6 step
350, and K=4-reference-scaled steps 350/400. Their immutable HF exports completed as one-GPU jobs
in `ai2/linear-rnns` on `ai2/titan`, leaving the 32-GPU `ai2/holmes-testing` allocation for the
actual four-GPU evaluations:

- K=8 step 300 export: [Beaker 01KYGBH63YYZ41P8PW0SGDCBN2](https://beaker.org/ex/01KYGBH63YYZ41P8PW0SGDCBN2)
- K=6 step 350 export: [Beaker 01KYGBHASWZW4XC3J4Q1ZFXGX1](https://beaker.org/ex/01KYGBHASWZW4XC3J4Q1ZFXGX1)
- K=4 reference-scaled steps 350/400 export: [Beaker 01KYGBHZ2EVG118SJK6BZS1JQW](https://beaker.org/ex/01KYGBHZ2EVG118SJK6BZS1JQW)

All four native catch-ups and both corresponding cross-routing evaluations are now queued:

- K=8 native step 300: [Beaker 01KYGC465HW07DYX00VACQQE1F](https://beaker.org/ex/01KYGC465HW07DYX00VACQQE1F)
- K=8-trained step 300, evaluated at K=6: [Beaker 01KYGC4CSN8KVRV4EAY38NJZ0V](https://beaker.org/ex/01KYGC4CSN8KVRV4EAY38NJZ0V)
- K=6 native step 350: [Beaker 01KYGC6XSN2KJGRPQTS6N3EYY5](https://beaker.org/ex/01KYGC6XSN2KJGRPQTS6N3EYY5)
- K=6-trained step 350, evaluated at K=8: [Beaker 01KYGC73W7FC0QHASCY2EEA4V2](https://beaker.org/ex/01KYGC73W7FC0QHASCY2EEA4V2)
- K=4 reference-scaled native step 350: [Beaker 01KYGC7P5DWRZX0M1CCQGDT5KT](https://beaker.org/ex/01KYGC7P5DWRZX0M1CCQGDT5KT)
- K=4 reference-scaled native step 400: [Beaker 01KYGCC0YK0K3FRYYTZ70N9PZX](https://beaker.org/ex/01KYGCC0YK0K3FRYYTZ70N9PZX)

All 19 cross-routing evaluations and the six catch-ups above succeeded and were collected by
2026-07-27. On MATH-500, the K=8-trained model's K=6 inference penalty shrinks from 7.6 points at
step 0 to 0.4 at step 100 and remains within -2.4 to +1.4 points through step 300. For the
K=6-trained model, K=8 inference helps MATH-500 by 2.6--5.0 points at several early checkpoints,
but changes the score by at most 1.2 points at steps 100/150/200/300/350; step 250 is a +3.0-point
exception. AIME Pass@8 is much noisier and has no stable inference-K direction. These are
single-evaluation checkpoint comparisons, not variance estimates.

The K=4-reference-scaled native curve reaches 81.4% MATH-500 at step 350, drops to 73.2% at step
400, and partially recovers to 76.0% at step 450. AIME Pass@1 is 13.02%, 11.35%, and 13.65%,
respectively. The step-400 evaluation was complete and configuration-matched, but showed a real
long/repetitive-output tail: MATH mean length rose from 917 to 2,223 tokens, max-length outputs
from 6 to 32, and invalid/missing answers from 2 to 19. Step 450 improves these to 1,817 tokens,
19 max-length outputs, and 15 invalid/missing answers, but does not return to step-350 behavior.
Training logs show the same qualitative drift in response length, entropy, and reference-policy
divergence without numerical or export errors. This is late-training instability/partial
mode-and-length collapse rather than an eval failure or total capability collapse.

The export and plot code supports the new K=6-reference-scaled arm. Its step-50 and step-100 exports completed on
2026-07-27 together with K=6 normalized step 400 and K=4 reference-scaled step 450. Five matching
evaluations were then queued: those four native checkpoints plus the K=6 step-400 checkpoint
evaluated at K=8. All are tracked in `notes/beaker_jobs.jsonl`.

Later on 2026-07-27, eight additional checkpoints became available and were exported in four
successful urgent one-GPU Titan jobs: K=8 step 350, K=6 normalized steps 450/500, K=4
reference-scaled step 500, and K=6 reference-scaled steps 150/200/250/300. Eleven evaluations
were queued in the same [cross-routing Beaker group](https://beaker.org/orgs/ai2/workspaces/holmes-testing/groups/01KYGB9X18QP6ZD4QHBRAVHHR6):
native and cross-routed evaluations for the K=8 and K=6-normalized checkpoints, and native
evaluations for the reference-scaled checkpoints. These reuse the corrected 32,768-token context,
20,480-token output cap, MATH-500 plus AIME 2025 pass@32 tasks, and four TP=1 engines. The exact
11 experiment IDs and commands are recorded in `notes/beaker_jobs.jsonl`.

All 11 evaluations succeeded and were collected on 2026-07-27. Each result passed the completeness
checks (500 MATH-500 instances, 30 AIME problems with 32 generations each, and no metric errors),
bringing `notes/slime_qwen3_base_dapo_checkpoint_evals.csv` to 71 rows. The three checkpoint plots
were regenerated through K=8 step 350, K=6 normalized step 500, K=4 reference-scaled step 500,
and K=6 reference-scaled step 300.

The K=8 run subsequently wrote step 400. Its one-GPU HF export
[succeeded](https://beaker.org/ex/01KYJED1HFJYT928JER7SM3MD3), and both the
[native-K=8](https://beaker.org/ex/01KYJER2PV95MEETA2HWX1F3S3) and
[counterfactual-K=6](https://beaker.org/ex/01KYJER928YCQB3MBBYG1H0XZP) evaluations were queued with
the same corrected recipe. Both succeeded and were collected on 2026-07-27, extending both blue
curves to step 400 and bringing `notes/slime_qwen3_base_dapo_checkpoint_evals.csv` to 73 rows.
Native K=8 scores 83.0% MATH-500, 16.15% AIME Pass@1, and 27.21% Pass@8; counterfactual K=6
scores 84.8%, 15.73%, and 28.84%. The K=6-reference-scaled training arm subsequently wrote steps
350/400/450/500 and completed successfully. All four were exported in
[Beaker 01KYKFWV03TPCCYWN12RMBPMVX](https://beaker.org/ex/01KYKFWV03TPCCYWN12RMBPMVX), and their
four native-policy evaluations were launched on 2026-07-28. The experiment IDs are recorded in
`notes/beaker_jobs.jsonl` and will extend the K=6-reference curve through step 500 after collection.

## Observed compute-cost accounting

`scripts/adaptive_experts/plot_slime_rl_compute_costs.py` produces an all-observed compute-cost
figure and `notes/slime_qwen3_base_dapo_compute_costs.csv`. Training points use the elapsed time
between finalized distributed checkpoints on the 8xB200 jobs, expressed as GPU-hours per 50 RL
updates; the allocation restart gap is excluded. Eval points use the recorded four-H100 experiment
duration, shown both as total GPU-hours for MATH-500 + AIME and as GPU-seconds per million recorded
completion tokens. The current source table has 38 training intervals and 56 eval measurements.

These are end-to-end costs, so training captures both the K-dependent kernels and behavioral
changes such as response length and dynamic-sampling difficulty. The K=8 arm becomes much more
expensive after step 250, while early intervals across arms are similar. Token-normalized eval
efficiency is noisy and does not establish a clean K ordering. The figure labels reference-scaled
olmo-eval points as K=8 zero-mask compute; in contrast, reference-scaled SGLang rollout and
Megatron training actually dispatch only their target K.

For planning MATH-500 replication, the 71 collected checkpoint jobs report task-level MATH-500
generation times from 6.5 to 31.0 minutes, with a 12.3-minute median and 13.6-minute mean on four
H100 engines. Model and harness startup adds roughly 2--3 minutes. After the four final K=6
reference-scaled points, there are 71 nonzero checkpoint/routing conditions (75 distinct
conditions including four step-zero baselines). Two additional MATH-only replicates therefore
require 142 jobs, or 150 with the baselines. At the 32-GPU Holmes cap, eight four-GPU jobs can run
concurrently; ideal fully allocated wall time is approximately five hours, with six to eight hours
a safer operational allowance for queueing and stragglers.

The full replication sweep was launched on 2026-07-28. It includes two new MATH-500-only runs for
all 75 distinct checkpoint/routing conditions, including the four true step-zero routing
baselines, for 150 jobs total. Synthetic cross-routing step-zero aliases are not rerun. The jobs
retain the existing stochastic MATH recipe (temperature 0.6, top-p 0.95, top-k 20, one sample),
32,768-token native context, 20,480-token generation cap, four TP=1 H100 engines, and the routing
policy used by each original checkpoint evaluation. The new runs are tracked as replicates 2 and
3; the MATH task embedded in the original MATH+AIME evaluation is replicate 1.

All jobs are urgent on `ai2/jupiter` in `ai2/holmes-testing` and are grouped in
[Beaker group 01KYKJ2J1AAE8V0MGHG7HGK9GN](https://beaker.org/orgs/ai2/workspaces/holmes-testing/groups/01KYKJ2J1AAE8V0MGHG7HGK9GN).
The post-launch audit found 150/150 unique expected tags and IDs, no extras, and the exact family
counts of 24 K=8-normalized, 22 K=8-trained/K=6-eval, 28 K=6-normalized, 26
K=6-trained/K=8-eval, 28 K=4-reference-scaled, and 22 K=6-reference-scaled jobs. Exact IDs and
commands are in `notes/beaker_jobs.jsonl` under phase `slime-checkpoint-math500-replicate`.

All 150 jobs succeeded and were collected on 2026-07-28, along with the four original
K=6-reference-scaled step-350/400/450/500 MATH+AIME evaluations. The collector now requires all
three complete 500-example MATH results for each of the 75 real checkpoint/routing conditions
before emitting a row; no partial averages are allowed. The two synthetic cross-routing step-zero
aliases copy the applicable three-run routing-policy aggregate. The resulting CSV has 77 plotted
rows, and every row reports the mean, sample standard deviation, min, max, the three individual
scores, and all three experiment IDs. Across real conditions, the median MATH sample SD is 0.92
percentage points, the mean is 0.97 points, and the maximum is 2.34 points.

The regenerated MATH-500 plot uses the three-run mean with ±1 SD whiskers. The averaged curves
make the convergence pattern clearer: K=8-trained checkpoints evaluated at K=6 are within 0.53
points of native K=8 at steps 80--300 except for step 350 (-1.20), and are +0.87 at step 400;
K=6-trained checkpoints evaluated at K=8 are within 0.87 points of native K=6 from steps 100--400
except for +1.53 at step 450, and are +0.20 at step 500. K=4 reference-scaled still shows the
late instability after averaging, falling from 81.33% at step 350 to 73.00% at step 400 before
recovering to 77.80%/78.47% at steps 450/500. K=6 reference-scaled is flatter, at 80.20% at step
350 and 78.93% at step 500. AIME remains based on the original pass@32 evaluation per checkpoint.

The K=8-trained step-450 checkpoint was exported on 2026-07-28 and evaluated under both its native
normalized K=8 policy and the existing normalized-K=6 counterfactual. For each routing condition,
one combined MATH-500 + AIME 2025 pass@32 experiment and two MATH-500-only replicates were queued,
giving three MATH measurements and one AIME pass@32 measurement per condition. The combined jobs
are [native K=8](https://beaker.org/ex/01KYMT3JFT4BFRQ1AV1JNJSB7Y) and
[evaluated at K=6](https://beaker.org/ex/01KYMT3RN1ZA1G5123QWJ4122T); all four replicate experiment
IDs are recorded in `notes/beaker_jobs.jsonl`. All jobs use the same corrected 32,768-token native
context, 20,480-token output cap, stochastic sampling recipe, four TP=1 engines, urgent priority,
`ai2/holmes-testing`, and `ai2/jupiter` as the earlier checkpoint curves.

## Step-350 inference-time plateau sweep

The matched step-350 checkpoints from the normalized-K=8, K=8-reference-scaled-K=6, and
K=8-reference-scaled-K=4 training arms are being evaluated across inference K=2--12. Each model
is swept under both normalized and K=8-reference-scaled inference routing with three evaluations
per point. Normalized and reference-scaled K=8 are mathematically identical, so that point is run
once and reused, leaving 189 unique jobs for 198 logical model/routing/K/replicate points.

Each job runs the current MATH-500, GPQA Diamond, IFBench, and HumanEval suite. The initial
matrix explicitly enabled the Qwen3 server-side reasoning parser; the collection audit found that
this returned empty answer text for these RL checkpoints and invalidated every completed result.
The original jobs were urgent, four-H100, unallocated experiments in `ai2/holmes-testing` on
`ai2/jupiter`, grouped in
[Beaker 01KYN5JCZTQB4XZ1J1PYZ8X9GE](https://beaker.org/orgs/ai2/workspaces/holmes-testing/groups/01KYN5JCZTQB4XZ1J1PYZ8X9GE).
The exact 189/189 launch audit passed with unique experiment IDs and tags, and the first allocated
job successfully initialized its four vLLM engines and started inference.

On 2026-07-29, 61 completed bundles from that group were downloaded. All were structurally
complete but had zero metrics because the reasoning parser returned empty `text` despite positive
generated-token counts. They are invalid and excluded; the remaining 128 jobs were canceled.
A one-GPU no-parser smoke produced non-empty text for all six underlying tasks, validating the
fix. The full matrix was relaunched with unique `v2-no-parser` tags in
[corrected Beaker group 01KYQ8RJ0NPKZQN6X6W9KY126G](https://beaker.org/orgs/ai2/workspaces/holmes-testing/groups/01KYQ8RJ0NPKZQN6X6W9KY126G).
Its launch audit has exactly 189 unique expected experiments, split 63/63/63 across the three
training checkpoints. The first corrected status snapshot has eight running, 181 queued, zero
succeeded, and zero failed jobs.

The scope was subsequently narrowed to MATH-500 only. The 189 four-suite jobs above were
canceled, and no results from them will enter the plateau analysis. The replacement uses the
same MATH recipe as the existing RL checkpoint plots: `math500:chat`, temperature 0.6, top-p
0.95, top-k 20, one sample, 20,480 generated tokens, 32,768-token context, four TP=1 engines,
the same source snapshot, and no server-side reasoning parser.

The analysis will reuse the existing three-replicate aggregates for K=8-trained/normalized-K=8,
K=8-trained/normalized-K=6, K=6-reference-trained/reference-K=6, and
K=4-reference-trained/reference-K=4. Thus 12 redundant experiments are omitted. The MATH-only
group has 177 nonduplicate new jobs, tracked under `slime-rl-step350-plateau-math500`, in
[Beaker group 01KYQA30P3M92HFVVJ0QAHAGRY](https://beaker.org/orgs/ai2/workspaces/holmes-testing/groups/01KYQA30P3M92HFVVJ0QAHAGRY).
