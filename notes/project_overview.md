# Adaptive expert compute: overall plan and results

Last updated: 2026-07-22 22:49 UTC

## Objective

Determine how far mixture-of-experts models can reduce active expert computation while preserving
capability, why some models and tasks are more robust than others, and whether inference or
training can be made natively adaptive.

The project currently separates four questions:

1. How does evaluation quality change when fixed active K is varied at inference time?
2. Which parts of routing matter: selected expert identities, their relative weights, total
   retained mass, renormalization, layer, token, and task?
3. Can an adaptive policy choose fewer experts while retaining quality?
4. If inference-only changes are insufficient, can post-training or pretraining make one
   checkpoint robust across compute budgets?

Detailed records remain in the linked experiment notes. This document is the living high-level
index of what has and has not been done.

## Evaluation and reporting conventions

- The original post-training suite was MATH-500, GPQA Diamond, and IFEval OOD. The current
  evaluation suite is MATH-500, GPQA Diamond, the three-task IFBench macro with a 32,768-token
  generation cap, and standard HumanEval.
- In this document, **historical three-eval suite** means MATH-500 + GPQA Diamond + standalone
  IFEval OOD. **Current four-eval suite** means MATH-500 + GPQA Diamond + IFBench-32k macro +
  standard HumanEval. A fixed-K curve is not called complete for the current suite unless all four
  measurements are available.
- Chat-model generations generally allow up to 32,768 new tokens. The earlier diagnostic
  Dolci-think checkpoint required a 30,000-token cap within its native 32,768-token context; the
  corrected terminal-EOS-v2 checkpoint advertises the longer context and uses the standard 32k cap.
- Accuracy uses the full dataset denominator. Truncated, missing, or unparseable answers remain
  incorrect rather than being removed.
- Stochastic full evaluations are normally repeated three times and reported as mean ± sample
  standard deviation. Explicit exceptions are identified below.
- Invalid infrastructure results, incorrect answer parsers, and incomplete-output serving failures
  are excluded from summary curves.
- AIME 2026 produces 32 samples per problem in one evaluation. Its pass@1 through pass@32 values
  are not equivalent to the between-run variance from three complete evaluations.
- Every launched Beaker experiment is retained in the append-only
  [job ledger](beaker_jobs.jsonl).

## Plotting plan

`IFBench macro (32k)` is now the primary instruction-following series for Qwen and GPT-OSS.
IFBench macro is the unweighted mean of IFEval OOD and the two multi-turn IFBench variants, so
showing it together with standalone IFEval OOD in the same high-level plot would overrepresent
that task family. The current backfills populate this adopted metric across K.

Standalone IFEval OOD is a historical/backward-comparison metric and remains only in detailed
tables or secondary analyses where its constituent behavior is useful. The earlier 2,048-token
IFBench protocol must remain labeled separately and must not be mixed with the 32k refresh. GLM
and Nemotron plots will not be retrofitted with IFBench until those models have matching
evaluations.

## Status at a glance

| Experimental area | Current state |
|:--|:--|
| Qwen3 Base fixed-K OLMoBase sweep | Complete and plotted |
| Qwen3 hybrid-thinking fixed-K sweep | Historical suite complete; current four-eval normalized and reference-scaled curves complete and plotted for K=3--16 |
| GPT-OSS fixed-K sweep | Historical and current-suite normalized/reference-scaled K=1--8 curves complete and plotted |
| GLM-4.5-Air fixed-K sweep | Historical suite complete for K={1,2,3,4,5,7,8}; current suite not planned yet |
| Nemotron 3 Super normalized sweep | Historical suite complete for K={11,13,15,17,19,21,22}; current suite not run |
| Qwen3 AIME 2026 | Complete and plotted for all fixed-K conditions |
| GPT-OSS and GLM AIME 2026 | Complete for retained conditions; GPT-OSS K=1 is invalid |
| Dolci-think SFT checkpoints | Earlier checkpoints were trained incorrectly; corrected terminal-EOS-v2 normalized/reference-scaled K=3--12 sweep is complete and plotted (60/60) |
| Qwen3.5 hybrid fixed-K sweep | Complete and plotted: normalized/reference-scaled K=4--8, three replicates, all four current-suite metrics |
| Qwen3.5 hybrid router/shared profile | Complete at native K=8: all 60 trajectories collected with prompt-balanced and token-weighted layer plots |
| Qwen router-mass analysis | Complete at K=8, including layer profiles |
| Qwen activation/contribution analysis | Complete on 60 fixed trajectories |
| Qwen expert-weight perturbations | Complete: native, shuffled, and uniform, three runs each |
| Qwen routing-policy interventions | Complete: temperature, fixed profile, adaptive mass, and reference scaling |
| IFBench + HumanEval capability follow-up | Original protocol and 15/15-run 32k IFBench refresh complete |
| Wider Qwen/GPT adaptive and reference-scaled sweeps | Complete and collected: all 87 Qwen jobs and all 45 planned GPT-OSS jobs |
| Nemotron router/contribution analysis | Complete: full 60-prompt profile, one-GPU aggregate, Qwen comparison, and layer plots collected |
| Qwen reference-preserving adaptive thresholds | Complete/collected/plotted: tau={0.5,0.6,0.7,0.8}, n=2 at tau=0.5 and n=3 otherwise, with eager realized-K telemetry |
| True variable-K serving speedup | Not implemented |
| Adaptive SFT/RL/pretraining | Fixed-K DAPO baselines trained at K=8, K=6, and K=4 reference-scaled; continuations to step 500 are running and held-out evaluations are collected through steps 250/300 |

## 1. Qwen3-30B-A3B-Base fixed-K evaluation

### What we have

- Consolidated OLMoBase evaluations at K={1,2,3,4,5,6,7,8,9,10,11,12,16}.
- Three measurements of the stochastic OLMoBase math suite at every K; deterministic suites were
  run once.
- Both aggregate and individual constituent-evaluation plots.
- The individual-task view shows that the aggregate behavior is real rather than an averaging
  artifact. Median constituent-to-aggregate correlations are approximately 0.94--0.99.

The main shape is not a simple monotonic benefit from more experts. Most score-based suites improve
rapidly through K=4--5, are strongest around K=5--11, and then collapse at K=12/16. In contrast,
several bits-per-byte metrics continue to improve or remain flat at larger K. The math replicates
also show a suite-scope/random-stream effect at large K, so their high variance should not be
interpreted as pure independent run variance.

### What remains

- No additional base-model repetitions are currently planned.
- If this result becomes central, rerun the stochastic math suite with an identical isolated
  launch shape at every K to remove the full-suite versus math-only random-stream confound.
- The base checkpoint has not been included in router-weight or expert-contribution analysis; the
  current mechanistic work intentionally focuses on chat/reasoning models.

Detailed results: [adaptive expert sweep results](adaptive_experts_results.md).

## 2. Qwen3-30B-A3B hybrid-thinking fixed-K evaluation

### What we have

- The historical three-eval suite at K=1--16 and K=32. These completed runs do not include
  HumanEval or the full IFBench macro.
- Three complete runs at most K values; K=13--15 have two runs.
- Generation-length, final-answer, and token-cap diagnostics.

The curve has a sharp low-K transition. Macro score is 0.00 at K=1, 0.25 at K=2, 36.88 at K=3,
59.74 at K=4, and then roughly 63--67 from K=5 through K=16. K=32 degrades to 48.62. MATH-500 is
nearly saturated by K=4, while GPQA and especially instruction following retain more K
sensitivity. The failure modes at K=1/2 and K=32 include very long or truncated reasoning, so the
quality curve is also a generation-stability curve.

This is the original evidence that the off-the-shelf model does not require all eight default
experts for many capabilities: K=5--7 broadly preserves the K=8 baseline, though the exact
threshold is task-dependent.

A prompt-matched MATH-500 response analysis additionally covers the full normalized K=1--16 and
K=32 sweep. Its primary length comparison aggregates repetitions within each of the same 500 prompt
IDs at every K, without conditioning on correctness. A fixed 477-prompt K=8-solvable cohort and a
416-prompt cohort correct in every K=4--16 run give nearly identical length curves, ruling out the
concern that different easy successful prompts create the trend. K=4 uses a median 1.29 times as
many total generated tokens as K=8; K=5, K=6, and K=7 smoothly converge toward native length at
1.14x, 1.09x, and 1.03x. K=3 uses 2.08x on the all-prompt comparison and produces an empty returned
response on 29.1% of samples. K=32 is bimodal: its all-prompt median is 0.85x K=8, but 19.1% of
generations hit the 32k cap and 18.8% expose no returned answer. Thus reduced per-token expert
compute can be partly offset by longer reasoning, especially below the quality transition, while
excessive K introduces a distinct tail-stability failure.

Decomposing the full completion count on the 416 common K=4--16 successes shows that the length
curve is almost entirely hidden reasoning rather than the parser-returned worked solution. The
returned solution stays at roughly 600--615 Qwen tokens across the range. From K=4 to K=8, mean
full length falls by 1,129 tokens: inferred hidden reasoning falls by 1,139 while returned text grows
by nine. From K=8 to K=16, inferred reasoning accounts for 99.0% of the additional 458-token drop.

Repeating the decomposition on K=8-reference-scaled runs produces a larger 434-prompt
common-correct cohort and removes the normalized curve's reasoning-length gradient. Accuracy stays
between 94.33% and 95.47%, while mean full completions remain within 3,983--4,222 tokens across
K=4--16. On the stricter 406 prompts shared by both routing cohorts, normalized K=4 averages 5,131
full tokens, reference-scaled K=4 averages 3,816, and native K=8 averages 4,024; returned solutions
remain nearly identical. The reduced-K reasoning-length penalty is therefore largely a
renormalization effect rather than an unavoidable cost of activating fewer experts.

Matched GPQA and IFBench analyses show the same mechanism generalizes. GPQA's normalized K=4
matched median generation is 1.586 times K=8, while reference-scaled K=4 is 1.007 times K=8 and
recovers 4.71 score points. IFBench's normalized K=4 ratio is 1.308, while reference scaling lowers
it to 1.119 and recovers 6.14 macro points. Direct cohorts shared across both policies confirm both
effects. GPQA also shows that normalized low K can lengthen the returned answer, not only hidden
reasoning. On IFBench, reference scaling mitigates but does not fully remove the K=4 length penalty;
K=6--7 is effectively back in the native capability and typical-length regime.

### What remains

- A third replicate was never launched for K=13--15.
- The current four-eval normalized and K=8-reference-scaled curves are complete at K=3--16.
  The normalized four-eval mean is 70.20 at K=4, 74.48 at K=5, 76.71 at K=6, 76.98 at K=7,
  and 76.46 at native K=8. Reference scaling raises K=4 to 74.03 and K=3 from 37.87 to 67.97.
- The historical fixed-K plot retains standalone IFEval OOD for backward comparison. The adopted
  current-suite plot uses IFBench-32k and HumanEval.
- Prompt-level qualitative analysis of IFBench examples that consistently change across the K=8
  boundary is planned but not executed. The matched GPQA and IFBench response/token analyses are
  complete.

Plots:
[historical core post-training](plots/posttraining_expert_sweep.png),
[current suite normalized versus reference-scaled](plots/qwen3_hybrid_current_suite_expert_sweep.png), and
[AIME 2026](plots/qwen3_hybrid_aime2026_expert_sweep.png).
Response analysis: [MATH-500 coherence across K](qwen_math_response_coherence.md).
Additional response analysis:
[GPQA and IFBench normalized versus reference-scaled](qwen_gpqa_ifbench_response_analysis.md).

## 3. GPT-OSS-120B fixed-K evaluation

### What we have

- Three valid historical-suite runs at every K from 1 through 8.
- K=3/5/6/7 use TP=2 eager/NCCL serving; K=1/2/4/8 use TP=1.
- Failed or empty-output serving diagnostics are excluded.

GPT-OSS is exceptionally robust around its default K=4. Macro is 31.00 at K=1, jumps to 75.57 at
K=2, and stays approximately 74--77 for K=2--8. Thus half of the default active experts preserves
the measured baseline, and increasing K above four does not help this suite.

The adopted current suite confirms this with complete normalized and reference-scaled K=1--8
curves. Normalized four-eval mean is 26.82 at K=1, jumps to 83.10 at K=2, and stays within
82.11--83.95 through K=8. Reference-scaled mean is 37.22 at K=1, 81.41 at K=2, and 82.83--84.52
at K=3--8. Reference scaling is modestly higher at K=5--8 but not K=2--3, with pointwise
differences generally comparable to evaluation variability. The robust K=2 result, rather than
either scaling rule, is the main finding.

### What remains

- Current-suite normalized and reference-scaled K=1--8 are both complete at three replicates per
  K.
- Reference-scaled K=8 has its planned three valid runs. The original experiment's in-place
  restart and a separately submitted replacement both succeeded; the redundant fourth result is
  validated but excluded from the summary curve.
- Router mass, rank profiles, expert contributions, and adaptive thresholds have not yet been
  measured for GPT-OSS.
- No additional GPT-OSS fixed-K repetitions are currently needed; both policies preserve the
  plateau from K=2 upward.

Plots: [historical post-training](plots/gptoss_120b_posttraining_expert_sweep.png) and
[current suite](plots/gptoss_120b_current_suite_expert_sweep.png).

## 4. GLM-4.5-Air fixed-K evaluation

### What we have

- Three historical-suite runs at K={1,2,3,4,5,7,8}.
- Every run uses two TP=4 vLLM engines.

GLM shows another sharp transition but at a different point. Macro is 0.17 at K=1, 31.04 at K=2,
57.58 at K=3, 63.42 at K=4, and approximately 64--65 at K=5/7/8. Half of the default K=8 is
therefore close to full performance on the historical suite; this has not been confirmed on
IFBench-32k or HumanEval.

### What remains

- K=6 is absent.
- GLM is intentionally excluded from the current IFBench/HumanEval backfill to control GPU cost.
- No reference-scaled, adaptive-mass, router-profile, or contribution analysis has been run.

Plot: [GLM post-training](plots/glm45_air_posttraining_expert_sweep.png).

## 5. Nemotron 3 Super LatentMoE evaluation

### What we have

- Three normalized runs at K={11,13,15,17,19,21,22}; checkpoint default K=22.
- Three raw, unnormalized ×5 runs at K=11 and K=22.
- A validated vLLM configuration using TP=2 expert parallelism and the non-FlashInfer FP8 MoE
  backend.

The normalized macro curve rises smoothly from 74.67 at K=11 to 79.38 at K=22. K=17 reaches
78.64, so roughly full performance is preserved around 17 of 22 routed experts rather than near
half of the default. Almost all separation comes from GPQA: K=11 trails K=22 by 12.12 points on
GPQA, while MATH-500 is flat and IFEval OOD is within 1.55 points.

Raw ×5 routing at K=11 and K=22 causes near-total generation failure. This is a valid negative
result, not an answer-extraction problem, but it does not isolate a useful reference-scale
condition because `5 * sum(top-K sigmoid scores)` can be much larger than the normalized routed
scale.

### What remains

The K=22 router and contribution profiler passed its reconstruction smoke. The deployed FP8
checkpoint's unified ModelOpt format is not supported faithfully by Transformers 5.7, so the
mechanistic analysis uses NVIDIA's corresponding BF16 checkpoint with the same K=22, 512-expert,
shared-expert LatentMoE architecture. The four balanced 15-prompt production shards are complete,
downloaded, and validated; their aggregate router/contribution comparison with Qwen remains to be
computed.

Nemotron differs from the standard Qwen analysis in several important ways:

- router scores are sigmoid rather than a full softmax competition;
- selected routed weights may be normalized and are then multiplied by a fixed scaling factor of
  five;
- a shared expert remains active;
- the LatentMoE architecture may change how selected routed contributions interact with the
  shared path.

The collected analysis has two stages:

1. **Router/rank profile:** begin from the native K=22 execution and measure rank-wise normalized
   shares within the selected top 22, raw top-K sigmoid sums, margins, effective selected K,
   expert frequency/load balance, and layer/task/phase differences. Report cumulative shares for
   K={1,2,4,8,11,13,15,17,19,21,22}.
2. **Functional contribution:** on fixed trajectories, measure routed expert activation norms,
   weighted contribution norms and directions, local recombinations at the evaluated K values,
   renormalized versus scale-preserving removal, and the routed mixture relative to both the
   shared-expert update and the complete routed-plus-shared MoE update.

The key question is whether K≈17 corresponds to a cumulative-mass threshold, whether ranks 18--22
have individually small but collectively important contributions, or whether the shared and
latent routed paths create a different form of dependence than Qwen.

The aggregate post-processing job succeeded and is tracked as
[01KY61MX7B4T9WCZ4W41MT2WRA](https://beaker.org/ex/01KY61MX7B4T9WCZ4W41MT2WRA). It requests one
H100 only for priority scheduling and does not perform additional model inference.

The selected K=22 weights are diffuse (effective K 20.97). Top 11 contains 60.08% of selected
router mass but 71.25% of contribution-norm mass; top 17 contains 83.04% and 88.25%. Thus expert
activation magnitude strengthens the high-router-rank head rather than rescuing the weak tail.
Reference-preserving removal is locally closer to the native routed+shared update than
renormalization, especially at K=11 (relative L2 0.182 versus 0.379). Router mass alone still does
not explain Nemotron's weaker half-K evaluation robustness: its half-K mass profile is comparable
to Qwen's, despite a larger capability loss.

Detailed results and plot:
[Nemotron routing sweep](nemotron3_super_routing_sweep.md) and
[normalized curve](plots/nemotron3_super_posttraining_expert_sweep.png).
Implementation and smoke details:
[Nemotron router/contribution analysis](nemotron_router_contribution_analysis.md).

Adaptive training roadmap: [elastic-K and DAPO plan](adaptive_training_and_rl_plan.md).

## 6. AIME 2026 hard-reasoning sweeps

### What we have

- Qwen: all K={1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,32} complete.
- GPT-OSS: K=2--8 complete; K=1 is invalid because Harmony decoding repeatedly returned HTTP 500
  and no model outputs.
- GLM: K={2,4,8} complete.

Qwen again shows a mid-K plateau. K=1/2 score zero, K=3 reaches 27.60 pass@1 and 53.33 pass@32,
K=4 reaches 62.71/80.00, and K=5--12 generally reaches 86.67--90.00 pass@32. K=32 falls to
18.54 pass@1 and 53.33 pass@32.

GPT-OSS is nearly invariant from K=2--8: pass@1 is about 79--83 and pass@32 is 96.67 at every K.
GLM rises strongly from K=2 to K=4 and more modestly to K=8.

### What remains

- No AIME runs use adaptive-mass or reference-scaled policies yet.
- GLM lacks the intermediate K values.
- Because AIME already samples 32 times per problem, follow-ups should retain all pass@k values
  rather than treating three independent jobs as the only uncertainty summary.

Plots:
[Qwen](plots/qwen3_hybrid_aime2026_expert_sweep.png),
[GPT-OSS](plots/gptoss_120b_aime2026_expert_sweep.png), and
[GLM](plots/glm45_air_aime2026_expert_sweep.png).

## 7. User-trained Qwen/Dolci-think checkpoints

### What we have

The previously evaluated 100k SFT checkpoint is:

`/weka/oe-adapt-default/jacobm/olmoe3/post-training/checkpoints/qwen3-30b-a3b-dolci-think-olmo-core-sft-100k-20260710-174355-hf`

One OLMo3-parser run was completed at K={2,4,6,8}. Macro scores were 7.96, 46.02, 52.58, and
51.46. We subsequently learned that this checkpoint and the related full-SFT checkpoint were
trained incorrectly. These numbers are retained only as pipeline diagnostics and must not be used
to conclude that correct post-training makes Qwen less robust.

Several invalid attempts were useful operationally:

- advertising a 40,960-token context for a native 32,768-token checkpoint caused engine failures;
- the Qwen3 reasoning parser could hide an entire OLMo-thinking response when `</think>` was
  absent;
- the correct configuration uses the native context, a 30,000-token output cap, and
  `reasoning_parser=olmo3`.

The corrected checkpoint now under evaluation is:

`/weka/oe-adapt-default/jacobm/olmoe3/post-training/checkpoints/qwen3-30b-a3b-dolci-think-olmo-core-sft-full-terminal-eos-v2-lr4e-5-20260715-071349-hf`

Its normalized K=8 and reference-scaled K=3 smokes passed with the OLMo3 parser. A 60-job
normalized/reference-scaled K=3--12 sweep on the current four-eval suite is now complete. All 60
jobs succeeded; every bundle is downloaded and fully validated, giving three replicates at every
K/mode point.

This checkpoint is much more K-sensitive than off-the-shelf Qwen. MATH is nearly saturated by
K=4--5 and IFBench by K=6--8, while GPQA and especially HumanEval continue to improve toward K=8.
Normalized four-eval mean peaks at 56.40 at K=8 and drifts to 54.50 at K=12. Reference scaling
peaks at 57.99 at K=9, stays strongest at K=8--10, and falls to 55.95 at K=12. Reference scaling
helps at K=3 and K=8--12 but hurts at K=4--7, so its benefit does not transfer uniformly after this
SFT. Extra experts above eight do not consistently help.

### What remains

- The corrected terminal-EOS-v2 K=3--12 evaluation sweep is complete; no reruns are currently
  required.
- Investigate why HumanEval is unusually K-sensitive and why reference scaling changes sign across
  K after SFT before selecting this checkpoint for adaptive training.
- The old plot is retained as a diagnostic artifact but is excluded from project conclusions:
  [incorrect-checkpoint diagnostic](plots/qwen3_dolci_think_sft100k_posttraining_expert_sweep.png).
- No correctly trained checkpoint has yet been optimized explicitly for adaptive K.

Detailed launch and final results:
[corrected terminal-EOS-v2 sweep](corrected_dolci_terminal_eos_v2_sweep.md) and
[current-suite plot](plots/qwen3_dolci_terminal_eos_v2_current_suite_expert_sweep.png).

## 8. Qwen router-mass profile

### What we have

The native K=8 profile used 20 prompts per core dataset, stratified by baseline correctness. A
partial distributed run plus a MATH recovery produced the saved trajectories; the within-top-eight
reservoir analysis uses 22 prompts with available uniform token samples.

Within the normalized K=8 selected mixture:

| Retained prefix | Share of top-eight mixture | Survivor renormalization |
|--:|--:|--:|
| K=1 | 21.65% | 4.620× |
| K=2 | 38.30% | 2.611× |
| K=3 | 52.25% | 1.914× |
| K=4 | 64.23% | 1.557× |
| K=5 | 74.68% | 1.339× |
| K=6 | 83.96% | 1.191× |
| K=7 | 92.32% | 1.083× |
| K=8 | 100.00% | 1.000× |

The top-four share is fairly flat across the 48 MoE layers, ranging from 61.69% to 67.32%.
Experts 5--8 therefore carry about 35.8% of the selected gate mixture; they are not a negligible
router-weight tail even though evaluations can be robust after dropping some of them.

### What remains

- An exact all-60-prompt router-reservoir aggregate would require replaying trajectories lost at
  the distributed timeout.
- Router mass alone does not establish marginal expert value; the contribution experiment below
  addresses that question for Qwen.
- The equivalent Nemotron 60-prompt profile is collected and validated; its aggregate analysis
  and direct comparison with Qwen remain to be completed. A matched GPT-OSS profile is planned so
  Qwen softmax K=8, GPT-OSS K=4, and Nemotron sigmoid LatentMoE K=22 can be compared using the
  same prompts and output schema. GLM remains unprofiled.

Detailed note and plot:
[router analysis](qwen_router_analysis_plan.md) and
[layer profile](plots/qwen3_k8_top8_share_by_layer.png).

## 9. Qwen expert activation and functional contribution

### What we have

Sixty fixed K=8 trajectories—20 each from MATH-500, GPQA Diamond, and IFEval OOD—were replayed
through all 48 MoE layers. Instrumented and ordinary logits matched exactly; contribution
reconstruction stayed below 0.47% relative L2.

Activation size does not rescue the lower-ranked experts on average:

- rank-8 raw expert-output norm is 58.1% of rank 1;
- the top four contain 63.31% of gate mass but 69.23% of summed contribution norm;
- the bottom four contain 30.77% of contribution norm but only 19.65% of signed projection onto
  the final K=8 update;
- contribution vectors have substantial cancellation: the coherence ratio is 47.12%;
- local K=4 renormalization has cosine 0.9188 to K=8 but 1.377× its norm.

Boundary swaps still occur, but not as a systematic inversion: only 9.84% of bottom-four/top-four
pairs invert their contribution-norm order.

### What remains

- The current results are local counterfactuals on K=8 hidden states, not full K=4 trajectories.
- Next-token KL and full regeneration from selected high-effect layers/tokens have not been run.
- Random selected-expert subsets and individual-rank ablations remain useful controls.

Detailed note and plots:
[contribution experiment](qwen_expert_contribution_experiment.md),
[rank view](plots/qwen3_k8_expert_contribution_by_rank.png), and
[layer view](plots/qwen3_k8_expert_contribution_by_layer.png).

## 10. Qwen expert-weight perturbations

### What we have

The selected top-eight expert IDs were held fixed while weights were native, shuffled among those
experts, or made uniform. Each condition has three complete historical-suite runs.

| Condition | Macro | Outputs hitting 32,768-token cap |
|:--|--:|--:|
| Native weights | 66.66 ± 0.81 | 1.67% ± 0.29 |
| Shuffled weights | 0.00 ± 0.00 | 99.90% ± 0.10 |
| Uniform weights | 29.36 ± 0.94 | 55.04% ± 0.75 |

The relative assignment of weights to selected experts is therefore essential. Shuffling is
catastrophic even though the expert set, weight multiset, and total mass are unchanged. Uniform
weights retain some capability but remain severely degraded.

### What remains

- No equivalent perturbation has been run on GPT-OSS, GLM, or Nemotron.
- This experiment does not identify whether sensitivity comes from a small set of layers or token
  phases.

Detailed note: [expert-weight perturbations](qwen_expert_weight_perturbation.md).

## 11. Qwen routing-policy interventions

### What we have

Forty complete historical-suite evaluations cover four policy families:

- router-weight power/temperature at p={0.5,0.75,1.0,1.5,2.0};
- a fixed global mean profile by router rank;
- cumulative-mass routing at tau={0.50,0.60,0.70,0.80,0.90};
- K=8-reference scaling at K={4,6,12}.

Key macro results:

| Policy | Macro |
|:--|--:|
| Identity p=1 | 66.01 |
| Mild p=0.5/0.75/1.5 | 64.75 / 65.35 / 65.51 |
| Strong sharpening p=2 | 50.16 ± 0.88 |
| Fixed global rank profile | 62.50 ± 0.30 |
| Adaptive tau=0.50/0.60/0.70/0.80/0.90 | 25.43 / 52.88 / 62.74 / 64.27 / 65.86 |
| K=8-reference scaled K=4/6/12 | 64.14 / 64.78 / 64.89 |

The cumulative-mass curve is smooth. Roughly 60--70% retained selected mass is where the tested
quality curve approaches the full baseline, but this is an empirical Qwen threshold rather than
a universal constant.

The expanded current four-eval sweep is now complete at three runs per threshold and includes
realized-K telemetry:

| tau | Realized mean K | Current four-eval macro |
|---:|---:|---:|
| 0.50 | 4.125 ± 0.009 | 30.82 ± 2.27 |
| 0.60 | 4.791 ± 0.015 | 63.17 ± 1.66 |
| 0.70 | 5.395 ± 0.015 | 74.27 ± 0.15 |
| 0.80 | 6.260 ± 0.008 | 76.89 ± 0.31 |
| 0.90 | 7.196 ± 0.002 | 76.12 ± 0.59 |

This richer suite sharpens the transition: tau=0.50 is not viable despite averaging more than four
experts, tau=0.60 is intermediate, and tau=0.70 is close to the plateau. Tau=0.80 and tau=0.90
are statistically similar on the aggregate, with tau=0.80 using about 0.94 fewer experts per
token-layer.

K=4 reference scaling substantially outperforms native renormalized K=4 on the core macro
(64.14 versus 59.74). K=6 and K=12 are similar with or without scale preservation. This indicates
that forcing survivor mass back to one can be harmful at low K; total routed mass does not itself
need to equal one for full quality.

### What remains

- The full Qwen reference-scaled curve is complete at K={3,5,6,7,9,10,11,12,13,14,15,16}.
  The normalized Qwen IFBench/HumanEval backfill is also complete through K=16.
- GPT-OSS normalized and reference-scaled K=1--8 are complete. Both the restarted
  reference-scaled K=8 experiment and its separately submitted replacement succeeded; the
  redundant fourth result is excluded from summaries.
- The current adaptive hook masks fixed-width slots and does not yet reduce actual expert
  dispatch, so no speedup claim is warranted.
- Equivalent adaptive policies have not been implemented for Nemotron's sigmoid/shared-expert
  LatentMoE.

Detailed results: [routing-policy sweep](qwen_routing_policy_sweep_results.md).

## 12. IFBench and coding capability follow-up

### What we have

The first five-condition Qwen follow-up evaluated:

- native K=8;
- normalized K=4;
- K=4 preserving native K=8 mass;
- adaptive tau=0.80;
- adaptive tau=0.90.

All conditions have three complete standard-HumanEval runs and three complete IFBench runs under
the original 2,048-token IFBench cap. HumanEval pass@1 is 93.29 at native K=8, 82.32 at normalized
K=4, 86.18 with K=4 reference scaling, 92.68 at tau=0.80, and 94.11 at tau=0.90.

The original IFBench cap was binding for a substantial fraction of outputs, so those numbers are
retained only as a separately labeled protocol. All 15 runs in the 32k IFBench refresh are now
complete:

| Condition | n | IFBench macro (32k) |
|:--|--:|--:|
| Native K=8 | 3 | 55.96 ± 0.49 |
| Normalized K=4 | 3 | 48.15 ± 0.35 |
| K=4 preserving native K=8 mass | 3 | 54.29 ± 0.41 |
| Adaptive tau=0.80 | 3 | 54.89 ± 0.39 |
| Adaptive tau=0.90 | 3 | 56.33 ± 0.53 |

Reference-scaled K=4 and adaptive tau=0.80 are much closer to native K=8 than normalized K=4.
This strengthens the scale/renormalization result on instruction following.

HumanEval+ was smoke-tested but rejected without changing olmo-eval because a hidden-test payload
exceeded the sandbox transport's per-argument limit and would have been counted incorrectly.
Standard HumanEval works and is retained.

### What remains

- The 32k IFBench capability-policy refresh is complete and locally collected.
- The Qwen normalized and reference-scaled current-suite curves are complete and plotted.
- GPT-OSS current-suite normalized and reference-scaled curves are complete and plotted.
- Standalone IFEval OOD remains only in detailed historical views.
- No matching IFBench/HumanEval backfill is planned for GLM in the current queue.

Detailed note: [capability-policy follow-up](qwen_capability_policy_sweep.md).

## 13. Current expanded production queue

### What we have launched

The active matrix contains 132 valid production jobs in `ai2/holmes-testing`, all urgent on
`ai2/jupiter`:

- Qwen adaptive mass: 15 jobs, tau={0.5,0.6,0.7,0.8,0.9}, three runs each, with realized-K
  recording;
- Qwen K=8-reference-scaled curve: 36 jobs at missing K={3,5,6,7,9,10,11,12,13,14,15,16};
- Qwen normalized IFBench/HumanEval backfill: 36 jobs at the same missing K values;
- GPT-OSS K=4-reference-scaled curve: 21 jobs at K={1,2,3,5,6,7,8};
- GPT-OSS normalized IFBench/HumanEval backfill: 24 jobs at K=1--8.

All three smoke paths passed. Qwen telemetry was validated across all 48 layers. GPT-OSS TP=1 and
TP=2 routing hooks generated and scored cleanly. TP=1 retains 64 sequences while reserving GPU
headroom with memory utilization 0.94.

At the 2026-07-22 22:49 UTC check, all 132 original jobs have succeeded and are downloaded and
validated. This includes the complete 87-job Qwen matrix, all 21 planned GPT-OSS reference-scaled
jobs, and all 24 normalized GPT-OSS jobs. Forty-five superseded GPT-OSS attempts are terminal and
canceled. The separately submitted K=8 replacement also succeeded after the original experiment's
successful in-place restart; it is validated but excluded as a redundant fourth replicate:
[01KXZ1GH969HK261P949Q9HKZ1](https://beaker.org/ex/01KXZ1GH969HK261P949Q9HKZ1).

### What remains

- The production queue and collection pass are complete.
- No action is needed on GPT-OSS K=8; the successful in-place restart has been collected.
- Compare adaptive thresholds at equal realized average K rather than only nominal tau.

Detailed launch audit: [expanded routing sweeps](expanded_routing_sweeps_launch.md).

## 14. Efficiency and actual adaptive execution

### What we have

- Quality experiments can modify active K, selected weights, retained mass, and adaptive masks.
- Realized-K telemetry is implemented and passed its Qwen smoke test.
- Current jobs report serving configuration and GPU metrics.

### What remains

- The adaptive policy still presents fixed-width routed slots to the expert kernel. It tests
  quality but does not establish lower FLOPs, higher throughput, or lower latency.
- Implement true variable-K dispatch only after the quality policies are selected.
- Benchmark tokens/s, time to first token, decode latency, memory, and expert calls at matched
  quality and matched request distributions.
- Consider request-level or layer-level policies first if token-level dynamic dispatch causes
  poor batching or kernel utilization.

## 15. Adaptive training

### What we have

- Three fixed-policy Qwen3 base DAPO baselines have been trained through at least 250--300 steps:
  native normalized K=8, native normalized K=6, and K=4 with K=8-reference-scaled weights. Their
  matched continuations to step 500 are running.
- Single-replicate MATH-500 and AIME 2025 checkpoint evaluations are collected at steps
  20/40/60/80/100 for all three policies, through step 250 for K=8, and through step 300 for K=6
  and K=4. K=8 reaches 83.0% MATH-500 and 16.67% AIME pass@1 at step 250.
- SGLang and Megatron implement and validate matching K=4 reference-scaled routing, including
  rollout/replay consistency and two full smoke-test optimizer updates.
- No model has yet been trained with an adaptive or sampled K policy.
- The incorrectly trained SFT checkpoints are excluded from training conclusions. Correct
  replacements exist but have not yet been evaluated for K robustness.
- Inference experiments have identified useful candidate supervision signals: retained mass,
  router margins, entropy, layer, and contribution disagreement.

### What remains

The current training plan, after the evaluation/mechanistic work is stable:

1. **Elastic SFT:** randomly sample K during training, initially with frozen experts and
   router/LoRA adaptation. Distill low-K logits or hidden states from native-K execution.
2. **Selected-expert dropout:** randomly remove routed experts while preserving clear controls for
   renormalized and scale-preserving mixtures.
3. **Layer/prompt/token-adaptive SFT:** train a small policy or budget conditioner to choose K,
   with an explicit expert-call penalty.
4. **Compute-aware RL:** optimize task reward minus expert cost after the action space and serving
   implementation are stable.
5. **Variable-K pretraining:** eventually sample budgets across batches, tokens, or layers, perhaps
   conditioned on a compute-budget token.

Expert pruning or merging is a separate direction: reducing active K lowers active computation but
does not remove expert parameters from memory.

## Prioritized next steps

1. Collect the running Qwen reference-preserving cumulative-mass sweep and compare each threshold
   at its measured realized K against fixed-K and renormalized-adaptive curves.
2. Run the same prompt-level router/contribution schema for GPT-OSS at native K=4.
3. Scope the first Qwen/DAPO elasticity pilot: fixed K=8, fixed reference-scaled K=4, and elastic
   reference-scaled K sampled from {4,6,8}, followed by matched cross-K evaluation.
4. Compare model families at matched fractions of default K, matched retained selected mass, and
   matched realized expert calls.
5. Analyze IFBench and HumanEval prompts that reliably cross the quality boundary, especially the
   corrected Dolci checkpoint's unusually sharp HumanEval dependence, while separating behavioral
   failures from cap hits and parser failures.
6. Select one promising inference policy and implement true variable-K dispatch for a
   quality-versus-throughput benchmark.
7. Begin elastic SFT only after the above establishes which policies and signals should be taught.

## Conclusions so far

1. **Off-the-shelf MoEs often use more active experts than are necessary for the measured
   capabilities.** GPT-OSS retains its baseline at K=2 versus default K=4; Qwen and GLM preserve
   much of their core performance around K=5--6 and K=4--5 versus default K=8.
2. **Robustness is not universal.** Nemotron appears to need roughly K=17 of 22 for full
   historical-suite aggregate performance. The incorrectly trained Qwen checkpoints are excluded
   from this comparison.
3. **Sensitivity is capability- and training-specific.** MATH often saturates early, while GPQA,
   instruction following, and coding can expose larger expert-count differences. Nemotron's
   normalized curve is almost entirely separated by GPQA; the corrected Dolci checkpoint's slope
   is driven especially by HumanEval.
4. **Correct relative expert weights matter enormously.** Shuffling weights among the same Qwen
   expert IDs collapses generation; uniform weights also cause severe degradation.
5. **Total selected mass need not equal one, but scale preservation is not universally better.**
   Off-the-shelf Qwen K=4 benefits strongly from reference scaling, while GPT-OSS is robust under
   either rule and the corrected Dolci checkpoint changes which rule wins across K. Renormalizing a
   small retained set can over-amplify the MoE update, but SFT can alter the useful calibration.
6. **Lower-ranked Qwen experts are not simply large-activation specialists.** Their raw
   activations decline with router rank. The top four contain more contribution norm than gate
   mass predicts, although cancellation and local boundary swaps remain important.
7. **Router mass is a promising adaptive signal, not yet an efficiency result.** Qwen quality
   approaches baseline around tau=0.8--0.9, but realized K and real variable-dispatch benchmarks
   are required before claiming compute savings.
8. **Architecture matters.** Nemotron's sigmoid routing, ×5 scale, shared expert, and LatentMoE
   structure require their own mass and contribution analysis rather than assuming Qwen's
   thresholds transfer.
