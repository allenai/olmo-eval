# Qwen3 K=8 expert-contribution experiment

## Question

Test whether router weight understates the functional importance of lower-ranked selected experts.
Qwen normalizes the top-eight router weights but does not normalize each expert output before
combination, so an expert with a small gate weight could still contribute a large vector.

For selected expert rank `i`, define the unweighted output `e_i = E_i(h)`, normalized gate weight
`g_i`, weighted contribution `c_i = g_i e_i`, and full routed update `y_8 = sum_i c_i`.

## Fixed-trajectory design

- Model: `Qwen/Qwen3-30B-A3B`, checkpoint-default K=8.
- Data: the exact 60 saved trajectories from the completed router profile: 20 each from GPQA
  Diamond, IFEval OOD, and MATH-500, balanced by the original baseline-correctness stratum.
- Replay the saved token IDs with causal KV caching rather than sampling new answers. This keeps
  token trajectories fixed and isolates expert contribution from generation drift.
- Process prompt, reasoning, and final-answer phases separately.
- Instrument every one of the 48 MoE layers, computing sufficient statistics online rather than
  saving `[token, layer, expert, hidden]` activation tensors.
- Compute all vector reductions in float32 while leaving the model forward in BF16.

## Measurements

Per selected gate rank 1--8:

- normalized gate weight `g_i`;
- unweighted expert-output norm `||e_i||`;
- weighted contribution norm `||c_i||` and its share of `sum_j ||c_j||`;
- signed projection `dot(c_i, y_8) / ||y_8||^2`;
- rank of `||c_i||` among the eight selected experts;
- leave-one-out cosine and relative L2 after removing rank `i` and renormalizing the survivors.

For every counterfactual retained K=1--8, form the exact local top-k recombination
`y_k = sum_{i<=k} c_i / sum_{i<=k} g_i` and measure:

- cumulative gate and contribution-norm shares;
- cosine between `y_k` and `y_8`;
- `||y_k - y_8|| / ||y_8||`;
- `||y_k|| / ||y_8||`.

Also measure the full MoE update norm, its norm relative to the decoder residual, the coherence
ratio `||sum_i c_i|| / sum_i ||c_i||`, the number of bottom-four/top-four contribution-norm
inversions, and top-four overlap between gate ranking and contribution-norm ranking. The audited
Transformers Qwen3 sparse block has no separate shared-expert branch.

## Validation gates

1. Instrumented and ordinary model logits match on a short fixed prefix within BF16 tolerance.
2. Selected gate weights sum to one and selected IDs remain in descending gate order.
3. The instrumented weighted contributions reproduce the ordinary MoE output.
4. Every replay prompt is accounted for; phase/layer token counts match expected processed tokens.
5. No full activation vectors are retained after an online reduction.

## Execution plan

1. Unit-test the vector metric calculations on synthetic expert outputs, including identical,
   orthogonal, canceling, and activation-rank-reversal cases.
2. Run a one-prompt smoke on one H100 and record output equivalence, memory, and throughput.
3. Run four independent one-H100 replicas on urgent `ai2/jupiter`, 15 trajectories per replica,
   with a long distributed timeout for rank imbalance.
4. Merge per-example sufficient statistics, bootstrap over prompts, and report overall, task,
   phase, and layer results.
5. Produce rank-wise gate-versus-contribution plots and counterfactual K-by-layer plots.

## Decision criteria

The activation-size hypothesis is supported if lower gate ranks frequently move upward under
`||g_i E_i(h)||`, or if experts 5--8 have leave-one-out effects disproportionate to their gate
weights. Large contribution norms but small leave-one-out effects instead support redundancy or
cancellation. Small contribution norms and small marginal effects support simple router-rank
truncation.

## Execution record

- Smoke r1 replaced the expert forward and correctly failed the logit-equivalence gate.
- Smoke r2 preserved the original forward, but exposed that raw absolute reconstruction error is
  a misleading BF16 validation criterion for large expert vectors.
- [Smoke r3](https://beaker.org/ex/01KXMQ1B2Z4HDHDWR54YW9WF3Y) passed: instrumented logits were
  exactly equal to ordinary logits, maximum contribution-reconstruction error was 0.36% relative
  L2, and peak allocated memory was 57.7 GiB on one H100.
- [Full 60-response run](https://beaker.org/ex/01KXMSXEM2K66D7TM89PQ8XQD3): four H100s, urgent
  priority on `ai2/ceres` in `ai2/OLMo-3-moe-experiments`, chunk size 128. A one-response smoke
  test first verified exact logit equivalence and sub-0.43% relative reconstruction error under
  the cached Ceres PyTorch 2.7.1 / CUDA 12.8 image. The full run replays
  326,567 completion tokens plus 9,529 prompt tokens. Identical unscheduled copies in Jupiter were
  canceled after the Ceres replacement was confirmed and scheduled; an earlier Ceres copy using
  the uncached runtime was also canceled after the cached-image replacement was scheduled.

## Results

The full run completed all 60 responses: 20 each from GPQA Diamond, IFEval OOD, and MATH-500.
It covered 9,529 prompt tokens and 326,567 generated tokens (296,715 reasoning and 29,852 final),
or 15,675,216 generated-token-by-layer observations. All four workers had bit-identical ordinary
and instrumented logits with matching top-1 tokens. The worst observed K=8 reconstruction error
was 0.47% relative L2. Reported intervals below are 95% task-stratified bootstrap intervals over
the 60 responses, with token-layer weighting within each resample.

### Activation magnitude reinforces the gate ordering on average

| Router rank | Gate mass | Raw expert-output norm vs. rank 1 | Contribution-norm share |
|---:|---:|---:|---:|
| 1 | 21.12% | 100.0% | 25.39% |
| 2 | 16.36% | 93.3% | 18.20% |
| 3 | 13.83% | 84.8% | 14.22% |
| 4 | 12.00% | 77.2% | 11.42% |
| 5 | 10.59% | 69.6% | 9.43% |
| 6 | 9.48% | 64.5% | 8.02% |
| 7 | 8.64% | 60.8% | 7.02% |
| 8 | 7.98% | 58.1% | 6.31% |

Rank 8's unweighted activation norm was 58.1% of rank 1's (95% CI 57.5--58.8%). Thus the lower
router weights are not generally offset by larger expert activations. The top four contain 63.31%
of the normalized K=8 gate mass (62.28--64.05%) but 69.23% of the summed contribution norms
(68.09--70.02%). Conversely, experts 5--8 contain 30.77% of contribution norm (29.98--31.91%)
and only 19.65% of signed projection onto the final K=8 mixture (18.57--21.22%). The coherence
ratio is 47.12%, showing substantial cancellation among selected expert vectors.

There is still meaningful local reordering. At least one bottom-four expert exceeds at least one
top-four expert in contribution norm on 56.88% of token-layer observations (53.54--62.11%), but
only 9.84% of all bottom-four/top-four pairs are inverted (8.59--11.87%). The gate-ranked and
contribution-ranked top-four sets overlap by 83.56% on average (81.41--84.91%). This is compatible
with frequent boundary swaps, not a systematic reversal of the router ranking.

### Local K=4 recombination

Renormalizing only the top four selected experts, as Qwen does when changing K, yields a local MoE
update with cosine 0.9188 to the K=8 update (0.9120--0.9233). Its norm is 1.377 times the K=8 norm
(1.373--1.381), and its relative L2 difference is 0.596 (0.584--0.614). The high cosine alongside
the larger L2 difference means that dropping experts 5--8 mostly preserves direction but amplifies
the update after the surviving gate weights are renormalized. The K=8 MoE update itself averages
15.23% of the incoming residual norm (14.36--15.98%).

This is a local counterfactual on fixed K=8 hidden states and token trajectories. It does not
replace a full K=4 forward pass, where altered hidden states compound across layers and can change
later routing and generated tokens. It therefore explains why activation size is unlikely to be a
hidden source of bottom-expert importance, but it does not by itself predict an evaluation score.

### Task, phase, and layer consistency

- Top-four contribution-norm share is similar on GPQA (69.86%), IFEval OOD (69.45%), and
  MATH-500 (68.74%). Rank-8 activation norm relative to rank 1 is 59.1%, 59.5%, and 57.2%,
  respectively.
- Reasoning and final-answer tokens give nearly identical top-four contribution shares (69.17%
  and 69.79%). Rank-8 relative activation norm is 58.5% during reasoning and 54.9% during final
  answers.
- Across the 48 MoE layers, the top-four contribution share never falls below 64.63% and reaches
  75.95%. K=4/K=8 cosine ranges from 0.887 to 0.980. Layers 22--24 are the least top-heavy and
  have the weakest K=4 agreement; the final layer is the most aligned. Within every layer,
  rank-8 mean activation norm remains below rank 1 (layerwise ratio 44.4--78.3%).

The activation-size hypothesis is therefore not supported as a population-level explanation for
why K=8 might matter. Lower-ranked experts sometimes move across the top-four boundary, but their
smaller raw activations make their aggregate weighted contribution less important than gate mass
alone would suggest.

Artifacts:

- [Rank-wise plot](plots/qwen3_k8_expert_contribution_by_rank.png)
- [Layer-wise plot](plots/qwen3_k8_expert_contribution_by_layer.png)
- [Machine-readable summary](expert_contribution/qwen3_k8_expert_contribution_summary.json)
- [Per-layer table](expert_contribution/qwen3_k8_expert_contribution_by_layer.csv)
