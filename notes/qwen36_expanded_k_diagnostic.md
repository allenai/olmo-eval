# Qwen3.6 above-native-K routing diagnostic

Last updated: 2026-08-16

## Question

The corrected reference-scaled Qwen3.6 TBLite sweep has not degraded through
K=16 and has a strong partial K=32 result. Test whether this is because ranks
9–32 carry little router mass, because their expert-output vectors are weak or
redundant, or because reference scaling preserves the native top-eight update.

## Design

Generate continuations once with the untouched native K=8 Transformers model,
then replay identical prompt and continuation tokens. At every MoE layer and
incoming hidden state, compare:

1. native K=8;
2. router-K=32/keep-8 as a numerical equivalence control;
3. router-K=32/keep-32 scaled by the raw top-eight mass, matching the current
   reference-scaled vLLM intervention;
4. router-K=32 conventionally normalized over all 32 selected experts.

Record raw all-256-softmax mass through ranks 8/16/32, the total post-scaling
weight, routed-update and tail norms, vector alignment/projection, the shared
expert contribution, and relative changes to the complete MoE update. A short
end-to-end teacher-forced probe measures how each policy compounds through all
40 layers.

The full prompt set is the existing stratified manifest with 20 examples each
from MATH-500, GPQA Diamond, and IFEval OOD. Fresh continuations use Qwen's
temperature 1.0, top-p 0.95, top-k 20 recipe, capped at 64 tokens because the
goal is router/contribution measurement rather than task scoring.

## Jobs

- Smoke: [Beaker 01M05V3XE73EZYMP9GQZWCEJRW](https://beaker.org/ex/01M05V3XE73EZYMP9GQZWCEJRW),
  one B200 in `ai2/OLMo-3-moe-experiments`, urgent.
- H100 fallback smoke: [Beaker 01M05VK93NF5R3J1J72TD055VK](https://beaker.org/ex/01M05VK93NF5R3J1J72TD055VK),
  one H100 eligible on Jupiter/Ceres, launched because the Titan node executor
  left the first smoke scheduled behind three earlier jobs.
- Revised-control smoke: [Beaker 01M05VX2NN2HVW7PYZMWMGNH4P](https://beaker.org/ex/01M05VX2NN2HVW7PYZMWMGNH4P),
  succeeded; exact native expert replay had zero error. Widening the BF16
  top-k call from 8 to 32 changed at least one of its first eight IDs in some
  token-layer events, so this real tie-breaking effect is retained as a
  separate control.
- Full 60-prompt job: [Beaker 01M05W0SA6Q8RKAP96N94S1T6G](https://beaker.org/ex/01M05W0SA6Q8RKAP96N94S1T6G),
  four H100s eligible on Jupiter/Ceres, urgent.

The full job was launched after the revised smoke verified exact native expert
replay, counterfactual output reconstruction, and the saved result schema.

## Results

The full job succeeded and all 60 prompts were collected: 20 each from
MATH-500, GPQA Diamond, and IFEval OOD. Every trajectory contains its complete
prompt plus 64 freshly generated native-K=8 tokens. Exact replay of the
captured native expert IDs and weights reconstructed the native routed branch
with zero error on all four workers.

Token-weighted means across all 40 layers and both prompt/generated tokens:

| Measure | Result |
|---|---:|
| Raw all-256 probability in top 8 | 19.38% |
| Raw all-256 probability in top 16 | 27.55% |
| Raw all-256 probability in top 32 | 39.30% |
| Raw probability in ranks 9–32 | 19.93% |
| Reference-scaled K=32 total routed weight | 2.205 |
| Top-eight share under normalized K=32 | 47.09% |
| Tail (ranks 9–32) / native routed norm | 45.65% |
| Tail versus K32-top-eight cosine | 0.073 |
| Reference-scaled local MoE delta vs native | 32.61% relative L2 |
| Normalized local MoE delta vs native | 41.43% relative L2 |
| Reference-scaled/native routed norm ratio | 1.121 |
| Normalized/native routed norm ratio | 0.528 |
| Reference-scaled/native complete-MoE norm ratio | 1.073 |
| Normalized/native complete-MoE norm ratio | 0.781 |

The extra experts are therefore **not negligible in aggregate router mass**.
Each low-ranked expert is small, but ranks 9–32 together carry slightly more
raw mass than ranks 1–8. Reference scaling preserves the native top-eight
amplitude and adds this tail, producing a routed update 12.1% larger and a
complete routed-plus-shared MoE update 7.3% larger in mean norm. In contrast,
normalizing all 32 selected weights assigns only 47.1% of their mass to the
first eight; it cuts the routed update to 52.8% and the complete MoE update to
78.1% of the native mean norm.

The tail's mean norm is substantial (45.7% of the native routed norm), but it
is nearly orthogonal to the top-eight vector (cosine 0.073) and projects only
weakly onto the native routed/complete update. This makes the observed
reference-scaled robustness plausible: the intervention retains the trained
native mixture instead of replacing half of it, while residual, shared-expert,
and later normalization pathways can absorb a relatively unaligned added
component. Reference scaling is locally less disruptive than normalization at
nearly every layer (32.6% versus 41.4% mean relative L2), with no material
hybrid-attention split: full-attention and DeltaNet layers are very similar.

Widening PyTorch's BF16 `topk` call from 8 to 32 changes at least one member of
the first eight on 19.0% of token-layer events because tied logits can be
resolved differently. Keeping only the first eight returned by the K=32 call
causes a 2.81% mean local change, while replaying the exact native IDs and
weights has zero error. This is a real widened-router effect rather than an
instrumentation failure, but the vLLM fused router may have a different exact
tie rate.

The short end-to-end probe compounds each policy through all 40 layers on a
fixed token prefix. Native replay is exact; the K32-top8-only control preserves
the same top token; and both K=32 policies preserve the same top token on this
single probe while materially moving the rest of the distribution. This is a
mechanistic diagnostic, not a task score.

## Interpretation and next check

The results strongly support the hypothesis that reference scaling is what
allows above-native K to remain robust, but for a more precise reason than
"new experts have essentially zero weight": reference scaling preserves the
trained top-eight update. A normalized K=32 policy would halve the routed
branch and change the shared/routed balance substantially, so a larger task
degradation is plausible. It is not yet proven by this analysis; the decisive
benchmark check is a matched normalized TBLite run at K=16 and K=32.

Saved artifacts:

- `notes/qwen36_expanded_k_diagnostic/analysis/qwen36_expanded_k_diagnostic.png`
- `notes/qwen36_expanded_k_diagnostic/analysis/qwen36_expanded_k_router_mass_curve.png`
- `notes/qwen36_expanded_k_diagnostic/analysis/summary.md`
- `notes/qwen36_expanded_k_diagnostic/full/` (raw collected result bundle)
