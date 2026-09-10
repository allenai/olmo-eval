# Qwen routing-policy sweep: implementation and proposed design

Status: implementation is complete for K=8 interventions and K=12 reference scaling. The seven
smoke conditions passed, and the approved 40-evaluation sweep was launched on 2026-07-17. See
`qwen_routing_policy_sweep_launch.md` for the complete launch record.

## Common experimental setup

- Model: Qwen3-30B-A3B hybrid-thinking checkpoint (`qwen3-30b-a3b`).
- Native router selection: top eight expert IDs are selected exactly as usual.
- Evaluation suite: `adaptive_experts:hybrid_pilot` (MATH-500, GPQA Diamond, IFEval OOD).
- Maximum generation length: 32,768 tokens.
- Serving: four independent TP=1 vLLM engines per evaluation.
- Repetitions: three complete evaluations per condition.
- Planned Beaker placement: `ai2/jupiter`, workspace `ai2/holmes-testing`, urgent priority.
- Every submitted job will be recorded in `notes/beaker_jobs.jsonl`.

The K=8 interventions operate after vLLM has selected and normalized Qwen's native top-eight
weights. The K=12 condition analogously starts from normalized top-twelve routing. Every policy
leaves selected expert IDs unchanged. Except for the reference-scaled conditions in experiment 4,
transformed weights are normalized back to the native row sum.

The intervention code is split into a pure Torch policy module and a small vLLM integration hook:

- `src/olmo_eval/compat/qwen_expert_weight_policies.py`: policy definitions, validation, and
  tensor transforms.
- `src/olmo_eval/compat/vllm_qwen_expert_weights.py`: installs the transform after vLLM routing.
- `src/sitecustomize.py`: opt-in startup hook for the isolated vLLM environment.
- `scripts/adaptive_experts/launch.sh`: validated environment-to-Beaker configuration.

Important implementation limitation: adaptive and truncated policies operate inside the router's
fixed-width selected tensor. This gives the intended model outputs, but the fused MoE kernel still
dispatches every selected slot, including zero-weight slots. The first sweep therefore measures
quality at a counterfactual expert budget, not wall-clock speedup. If the policy is promising, the
next systems step is variable-K dispatch that actually avoids unused expert calls.

## Experiment 1: router-weight temperature / power sweep

For each token and layer, transform native selected weights `g_i` as

`g'_i = g_i^p / sum_j(g_j^p)`.

- `p < 1` flattens the correct expert/weight assignment.
- `p > 1` sharpens it.
- `p = 0` is exactly uniform weighting.
- `p = 1` is an exact identity transform.

Recommended first matrix:

| Exponent `p` | Purpose | New 3x evaluations? |
|---:|:--|:--|
| 0 | Uniform endpoint already completed | No; reuse as a plotted anchor |
| 0.5 | Strong flattening | Yes |
| 0.75 | Mild flattening | Yes |
| 1.0 | Patched identity/backend control | Yes, once only |
| 1.5 | Moderate sharpening | Yes |
| 2.0 | Strong sharpening | Yes |

The single patched `p=1` control is a sanity check against native K=8. It uses the same vLLM hook
and modular Triton MoE path as the interventions, so a close score rules out a gross backend or
hook problem without spending three full repetitions on another identity condition. Optional
denser points `p=0.25` and `p=1.25` remain deferred.

## Experiment 2: fixed global rank profile

Keep each token's native top-eight expert IDs and their rank order, but replace token-specific
weights with the global mean rank profile measured in the K=8 router analysis:

```text
[0.21562765, 0.16642320, 0.13938729, 0.12015191,
 0.10476987, 0.09296770, 0.08376831, 0.07690407]
```

This is one condition, run three times. It preserves the average preference for higher-ranked
experts while removing token-by-token confidence calibration. Uniform weighting is the existing
flat-profile control. Task-specific profiles are not recommended because they introduce avoidable
evaluation leakage, and the measured rank profile varies little by task. A separate profile per
layer is the natural lower-priority extension (experiment 5), not part of this first sweep.

The profile was computed from the saved native K=8 router samples. At each sampled token and layer,
the top-eight weights were divided by their top-eight sum. Samples were averaged first within each
prompt, prompts within each task, and then equally across GPQA Diamond, IFEval OOD, and MATH-500.
Finally, the 48 layer profiles were averaged equally. The available reservoirs contain 22 prompts
(5 GPQA, 5 IFEval, and 12 MATH); the task-balanced top-four share was very stable across tasks.

## Experiment 3: cumulative-mass adaptive K

At every token and layer, keep the smallest native rank prefix whose normalized top-eight
cumulative mass reaches threshold `tau`, then renormalize the retained weights. Use `min_k=1` and
`max_k=8` initially.

Recommended thresholds:

| Threshold `tau` | Approximate K from the global mean profile |
|---:|---:|
| 0.50 | 3 |
| 0.60 | 4 |
| 0.70 | 5 |
| 0.80 | 6 |
| 0.90 | 7 |

Each threshold gets three complete evaluations. These thresholds span the useful fixed-K range
without concentrating several conditions around the same typical K. The actual K varies by token
and layer; before interpreting compute-quality tradeoffs, report its distribution using the saved
router trajectories and, if we proceed to real variable-K dispatch, collect the live K histogram
from that implementation.

The first sweep should use renormalization for all adaptive conditions. Mixing the renormalization
question into this matrix would double its size; experiment 4 isolates that issue more cleanly.

## Experiment 4: dropping expert mass versus renormalizing survivors

Use native K=8 as the weight-scale reference and compare:

1. `renormalize=true`: normalize the selected K experts to sum to one, as Qwen normally does.
2. `renormalize=false`: preserve the K=8 denominator. Below K=8 this removes mass without
   amplifying survivors; above K=8 it adds expert mass without shrinking the original top eight.

Recommended paired matrix:

| Target K | Normalized baseline | New unnormalized repetitions |
|---:|:--|---:|
| 4 | Reuse completed native 3x | 3 |
| 6 | Reuse completed native 3x | 3 |
| 12 | Reuse completed native 3x | 3 |

For K=4 and K=6, the non-renormalized condition keeps their original shares of the normalized K=8
mixture, so its weight sum is below one. For K=12, the mirror-image non-renormalized condition
selects the native top twelve but divides them by the top-eight mass; the original top-eight
weights retain their K=8 scale and ranks 9--12 add mass, so the total is above one. The
renormalized K=12 condition is native top-twelve normalization.

This makes K=12 a genuine “other side of K=8” control. K=2 is excluded because its earlier
generation behavior was broken. K=4 remains central because the local activation analysis found
that retaining and renormalizing the top four made the MoE update 1.377 times the K=8 update norm
on average.

The existing native K=4, K=6, and K=12 post-training sweeps each have three complete evaluations
with this same suite and serving configuration. Reuse them as the normalized side and launch only
the three unnormalized counterparts. The patched `p=1` sanity run checks that the modified backend
is close enough to the native K=8 baseline for this reuse to be credible.

## Proposed launch size and staging

The recommended launch matrix contains 14 conditions:

- Temperature: four conditions run 3x, plus one K=8 identity sanity run.
- Fixed rank profile: 1 condition.
- Adaptive mass: 5 conditions.
- K=8-reference scaling: unnormalized K=4, K=6, and K=12.

That is 40 full-suite Beaker evaluations. Each evaluation uses four independent one-GPU engines as
in the previous Qwen sweep.

Before the full sweep, run one-example MATH-500 smokes covering both extremes of each code path:

- Temperature `p=0.5` and `p=2.0`.
- Fixed global rank profile.
- Adaptive mass `tau=0.50` and `tau=0.90`.
- Unnormalized K=4 and K=12 reference scaling.

All seven conditions passed. The first fixed-profile attempt exposed a CUDA graph-capture issue
when constructing the constant profile tensor; it was fixed and its replacement smoke passed.
After the smoke gate, every approved condition was launched three times, with the `p=1` identity
condition remaining the one-run exception.

## Lower-priority experiment 5

Defer layer-specific policies for now. The implemented fixed profile is deliberately global and
simple. A later extension can load one profile or K schedule per layer, informed by the saved
48-layer router/contribution plots, after the global interventions establish whether this direction
has enough signal.

## Implementation validation completed

- 28 policy unit tests pass.
- The compatibility-policy tests plus the existing Beaker-launcher tests pass: 127 tests total.
- Ruff passes on all changed Python files.
- `bash -n` passes for the launcher.
- Dry-run Beaker specs pass for temperature, rank-profile, adaptive-mass, and truncation modes, and
  contain the expected vLLM environment variables.
- Seven smoke conditions passed before the full sweep was submitted.
