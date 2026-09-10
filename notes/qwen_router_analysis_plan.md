# Qwen3 MoE router-distribution analysis plan

## Objective

Test whether Qwen3-30B-A3B remains robust when reducing active experts because router mass and/or
actual expert contributions are concentrated in the highest-ranked experts.

The first pass measures router distributions under the checkpoint-default K=8. A later causal pass
can measure the hidden-state and next-token effects of recombining the same expert outputs at other
values of K.

## Model and data

- Model: `Qwen/Qwen3-30B-A3B`, hybrid thinking, default K=8.
- Tasks: MATH-500, GPQA Diamond, and IFEval OOD.
- Sample size: 20 prompts per task, 60 prompts total.
- Initial stratification: 10 correct and 10 incorrect prompts per task from the validated K=8
  `variance-r2-20260710` result, experiment `01KX6KPE77ANFQHN4D4MP1VSEZ`.
- Selection is deterministic with a recorded seed. If a task lacks ten examples in a stratum, use
  every available example and fill the remainder from the other stratum.
- The instrumented generation is stochastic, so score its new outputs and report both the original
  selection stratum and the actual new correctness stratum.

The existing olmo-eval predictions cannot be teacher-forced as exact generation trajectories:
Qwen's reasoning parser retained final content but discarded the full reasoning-token prefix. A
fresh generation is required to capture valid hidden states and router logits for reasoning tokens.

## Generation configuration

Match the post-training evaluation configuration:

- Chat template with thinking enabled.
- Temperature 0.6, top-p 0.95, top-k 20.
- Maximum 32,768 new tokens, dynamically capped per prompt as
  `min(32768, configured_context_limit - prompt_tokens)` so no request is rejected for exceeding
  the total context window.
- One sample per prompt with a deterministic per-example seed.
- Default eight active experts; do not change K in this profiling run.
- Preserve raw token IDs and split phases into prompt, reasoning (before `</think>`), and final
  answer (after `</think>`).

Use a custom incremental decode loop with KV caching and `output_router_logits=True`. Do not use
`generate(..., output_router_logits=True)` in a way that retains every router tensor until the end;
compute and flush summaries online.

## Router measurements

For every token and MoE layer, compute the full 128-expert softmax in float32 and aggregate:

- Sorted probabilities through at least rank 16.
- Cumulative raw mass C1, C2, C4, C8, and C16.
- Selected-weight renormalization multipliers `1 / Ck`.
- Router entropy and effective expert count `exp(entropy)`.
- Logit/probability margins at ranks 1/2, 4/5, and 8/9.
- Top-16 expert IDs, selection frequencies, co-occurrence, and adjacent-token route churn.
- Separate aggregates by task, layer, prompt/reasoning/final phase, and correctness stratum.

Avoid storing all `[tokens, layers, 128]` logits. Store per-example/layer/phase sufficient
statistics plus a fixed-size uniform token sample containing top-16 IDs and probabilities. This
should keep result storage below roughly 1 GB.

## Analysis outputs

1. Rank-probability curves with percentile bands.
2. Layer-by-K heatmaps of cumulative mass Ck.
3. CDFs of C2, C4, C8, and the corresponding renormalization multipliers.
4. Layer/task expert-frequency and load-balance heatmaps.
5. Prompt versus reasoning versus final-token comparisons.
6. Correct versus incorrect comparisons with bootstrap confidence intervals.
7. Route-churn and expert-overlap measurements across adjacent tokens.
8. Correlations between router concentration and the existing K-sweep sensitivity.

The hypothesis predicts that robust regions have high C2/C4, low effective expert count, and small
rank-5-through-8 weights. A flat router distribution would instead imply large renormalization when
K is reduced.

## Follow-up causal analysis

Router probability alone may not equal functional importance because expert output norms and
directions differ. After inspecting the distribution results, run a smaller second pass that:

- Captures individual weighted expert contributions `alpha_e * E_e(h)` for selected layers/tokens.
- Measures contribution norms and cosine alignment with the total MoE output.
- Recombines the same hidden state and expert outputs at K=1/2/4/8/16.
- Measures MoE-output cosine/L2 change and next-token KL divergence.
- Separates expert removal from the renormalization effect.

Start with a few representative layers and two prompts per dataset rather than instrumenting all
60 trajectories.

## GPU plan

Qwen3-30B-A3B has 30.5B total BF16 parameters (about 61 GB of weight storage), 48 layers, 128
experts, and eight active experts. The minimum expected serving shape is one 80 GB H100 per
Transformers replica.

1. Smoke test: one H100, one representative long prompt. Record peak allocated/reserved memory,
   prefill throughput, decode throughput, and output equivalence with instrumentation disabled.
2. Full job: four H100s, four independent single-GPU replicas, 15 prompts per GPU. Do not use
   tensor parallelism unless the smoke test shows that one replica cannot stay below about 76 GB.
3. Fallback: if BF16 Transformers OOMs or has inadequate headroom, use TP=2 per replica and eight
   H100s total. Do not switch to a quantized checkpoint because that would change the model under
   analysis.

Four single-GPU replicas are recommended rather than required: one H100 should fit, but four should
reduce wall time by approximately four while keeping the implementation simple.

## Validation gates

- The selected top-eight IDs reconstructed from router logits must match the model's routed IDs.
- Instrumented and uninstrumented generation must match for a fixed seed over a short smoke sample.
- Router-stat token counts must equal processed token counts times the number of MoE layers.
- No prompt may silently disappear; failures and truncations remain in denominators and metadata.
- Save exact prompt IDs, seeds, model revision, Transformers revision, and launch configuration.

## Launch environment

- Cluster: `ai2/jupiter`
- Workspace: `ai2/olmo-instruct`
- Priority: urgent
- Suggested run tags: `qwen3-router-smoke-20260712` and `qwen3-router-profile-20260712`
- Record every Beaker experiment in `notes/beaker_jobs.jsonl` before considering the launch complete.

## Launch status (2026-07-12)

- Deterministic manifest: `notes/router_profile/qwen3_k8_manifest.jsonl` (60 prompts; 20 per
  task, each split into 10 baseline-correct and 10 baseline-incorrect examples).
- Initial smoke `01KXBS802245A093T1W4QQ6X2G` found a Transformers cached-decode issue in
  Qwen's auxiliary-load-balancing calculation when a full attention mask is supplied alongside
  one-token router logits. Batch size is one with no padding, so the repaired decode omits that
  unnecessary mask.
- Passing smoke: `01KXBSJ3CXCWT0WC5JN7R0FX2R`. Peak memory was 57.67 GiB allocated and
  59.17 GiB reserved. All 48 router layers matched Qwen's selected top-eight IDs with zero
  mismatches. The 69 prompt tokens and 128 generated tokens were recorded at every layer.
- Full K=8 profile: `01KXBSST8ZP8VBZ469GWB1CWCR`, four H100 replicas via `torchrun`, 15
  prompts per rank, full 32,768-token dynamic generation cap. It recovered 53/60 complete
  per-example records, then failed because the first-finished rank exceeded the default 30-minute
  distributed barrier timeout while the remaining ranks processed long MATH generations.
- MATH recovery (2026-07-13): `01KXD4JDGY5ZC96XHB5E3VKKDC`, containing exactly the seven
  missing MATH prompts. The process-group timeout is extended to 10 hours so rank imbalance cannot
  terminate completed work. Its manifest is
  `notes/router_profile/qwen3_k8_missing_math_manifest.jsonl`.

## K=8 selected-mixture shares (2026-07-15)

To measure how much of Qwen's actual K=8 mixture would remain at a smaller K, normalize the raw
top-k router mass by the raw top-eight mass independently for every sampled token and layer:
`selected_share(k) = Ck / C8`. This is distinct from `Ck`, which measures mass against the full
128-expert softmax. Qwen renormalizes its selected eight experts, so `Ck / C8` is the relevant
gate-weight share before removing experts ranked `k+1` through 8.

| Retained K | Cumulative share of K=8 mixture | Rank K's individual share | Removed share | Renormalization of retained weights |
|---:|---:|---:|---:|---:|
| 1 | 21.65% | 21.65% | 78.35% | 4.620x |
| 2 | 38.30% | 16.66% | 61.70% | 2.611x |
| 3 | 52.25% | 13.94% | 47.75% | 1.914x |
| 4 | 64.23% | 11.98% | 35.77% | 1.557x |
| 5 | 74.68% | 10.46% | 25.32% | 1.339x |
| 6 | 83.96% | 9.27% | 16.04% | 1.191x |
| 7 | 92.32% | 8.36% | 7.68% | 1.083x |
| 8 | 100.00% | 7.68% | 0.00% | 1.000x |

The estimate uses the saved uniform token-layer reservoirs for 22 prompts with available samples
(5 GPQA, 5 IFEval, and 12 MATH), first averaging within each prompt and then equally across the
three tasks. Three original workers lost their reservoirs when the distributed timeout terminated
the job, so an exact all-60-prompt estimate would require replaying the saved token trajectories.
The result is nevertheless stable across tasks: the top-four cumulative share ranges only from
64.00% on MATH to 64.62% on GPQA. The main conclusion is that experts ranked 5--8 collectively
carry about 35.8% of the K=8 gate mixture; they are not a negligible tail by router weight alone.

The layer-resolved [plot](plots/qwen3_k8_top8_share_by_layer.png) and
[CSV](router_profile/qwen3_k8_top8_share_by_layer.csv) apply the same prompt- and task-balanced
normalization separately at each of the 48 MoE layers. The within-top-eight distribution is much
flatter across depth than the raw top-eight mass against all 128 experts. The top-four share ranges
from 61.69% (layer 24) to 67.32% (layer 0) and has essentially no linear relationship with depth
(`r=-0.006`); rank eight alone ranges from 6.53% to 8.45%. Thus, the late-layer increase in raw C8
appears to move the selected eight collectively away from the other 120 experts, rather than making
the K=8 mixture itself substantially more top-heavy.

## Primary references

- Qwen model card: <https://huggingface.co/Qwen/Qwen3-30B-A3B>
- Qwen3 MoE implementation: <https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py>
- vLLM routed-expert IDs: <https://docs.vllm.ai/en/v0.21.0/training/routed_experts_replay/>
