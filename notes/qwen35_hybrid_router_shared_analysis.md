# Qwen3.5 hybrid-attention router and shared-expert analysis

Last updated: 2026-07-26

## Question

At Qwen3.5-35B-A3B's native K=8, test whether router concentration changes between its
DeltaNet (`linear_attention`) layers and full-attention layers. Measure the always-on shared
expert separately from the eight routed experts so its functional contribution is not mistaken
for router probability mass.

The model has 40 MoE layers. Layers 4, 8, ..., 40 (one-based) use full attention; the other 30 use
DeltaNet. Every layer has 256 routed experts, selects eight, renormalizes those eight weights to
approximately one in BF16, and adds a separately sigmoid-gated shared-expert output.

## Prompt set and generation

- Checkpoint: `Qwen/Qwen3.5-35B-A3B`, revision
  `59d61f3ce65a6d9863b86d2e96597125219dc754` in the passing smoke.
- Data: 20 prompts each from MATH-500, GPQA Diamond, and IFEval OOD; within each task, ten were
  baseline-correct and ten baseline-incorrect under the completed default-K=8 evaluation.
- Manifest: `notes/router_profile/qwen35_k8_manifest.jsonl`.
- Fresh full-trajectory generation with thinking enabled, temperature 0.6, top-p 0.95, top-k 20,
  and a 32,768-token dynamic generation cap.
- Statistics are retained separately for prompt, hidden reasoning, and returned final-answer
  tokens, with per-example summaries as well as token-weighted aggregates.

The fresh generation is necessary because the olmo-eval prediction files preserve parsed output,
not every hidden reasoning token required to reconstruct the inference trajectory.

## Measurements

For the routed branch at every token and layer, the profiler records all-256 softmax concentration,
the eight selected and renormalized router weights, cumulative within-top-eight shares, entropy,
effective expert count, and the rank-8/rank-9 boundary gap.

For the shared branch it records three complementary notions of contribution:

1. The mean sigmoid shared gate, which is the explicit learned scalar multiplier but does not
   account for the shared expert's activation magnitude.
2. `||shared|| / (||routed|| + ||shared||)`, which compares branch magnitudes after the shared gate.
3. `shared dot total / ||total||^2`, the shared branch's signed projection share of the actual
   combined MoE update. The routed and shared projection shares sum to one; this captures vector
   alignment and cancellation that a norm ratio misses.

The raw and gated shared norms, routed norm, total norm, routed/shared cosine, and coherence ratio
are also saved. These measures directly address the possibility that a small gate can still yield
a large functional update because the underlying expert activation is large.

## Validation and launch

The first smoke exposed BF16 ties at the K=8 boundary: CUDA can break an exact tie differently when
asked for top-8 versus top-9. Validation was corrected to replay the model's exact top-8 call;
top-9 is used only to measure the boundary gap.

The corrected one-B200 smoke passed:

- [Beaker 01KYFVSXMFTT9B7KJGDY6CS7D4](https://beaker.org/ex/01KYFVSXMFTT9B7KJGDY6CS7D4)
- selected-expert ID mismatches: 0;
- instrumented versus ordinary next-token logits: exactly identical;
- maximum routed-plus-shared reconstruction relative L2: 0.729%;
- peak allocation/reservation: 65.47/65.51 GiB;
- one 73-token prompt plus 16 generated tokens completed in 1.92 seconds after loading.

The full 60-prompt job ran on four independent B200 replicas in
`ai2/OLMo-3-moe-experiments`, on `ai2/titan`, at urgent priority:

- [Beaker 01KYFVX9N8X1R3XTJ87YQS96T4](https://beaker.org/ex/01KYFVX9N8X1R3XTJ87YQS96T4)

The layer-resolved plot and comparison tables will be written to
`notes/qwen35_router_shared/` by
`scripts/adaptive_experts/analyze_qwen35_hybrid_router_shared.py`. Gold vertical bands in the plot
identify full-attention layers; unshaded layers are DeltaNet layers.

## Status

The smoke values are only a schema and magnitude sanity check from one truncated MATH trajectory;
they are not reported as findings. Conclusions will use the complete stratified prompt set and
will compare both token-weighted and prompt-balanced views before attributing differences to layer
type.

The job completed successfully on 2026-07-27 and was collected on 2026-07-28. All 60 planned
trajectories are present: 20 each from MATH-500, GPQA Diamond, and IFEval OOD. Several deliberately
uncapped reasoning trajectories reached the full 32,768-token generation limit; these remain valid
profile data and explain the worker imbalance. Prompt/task-balanced and token-weighted summaries
for both all tokens and generated tokens are saved under `notes/qwen35_router_shared/`.

For generated tokens in the prompt/task-balanced view, the selected K=8 rank distribution is
nearly identical between the 30 DeltaNet layers and 10 full-attention layers: top-4 cumulative
share is 61.74% versus 61.53%, and raw top-8 router mass is 12.00% versus 11.84%. The shared branch
is somewhat stronger in full-attention layers: its branch-norm fraction is 51.21% versus 47.73%,
and its output projection share is 50.84% versus 45.70%. Reconstruction relative-L2 remains below
0.24% in both layer types.
