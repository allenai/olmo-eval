# Qwen3 GPQA and IFBench response analysis across active-expert counts

Date: 2026-07-21

## Scope and method

This extends the MATH-500 token-decomposition analysis to GPQA Diamond and the adopted
IFBench-32k macro. Each figure contains:

1. performance over every prompt, with the mean and replicate min--max range;
2. mean full-generation tokens split into parser-returned response tokens and inferred hidden
   reasoning plus boundary tokens; and
3. prompt-matched median token ratios relative to native K=8.

The hidden component is reconstructed as the server's full completion-token count minus the Qwen
token count of the parser-returned response. It therefore includes hidden reasoning plus a small
number of delimiter, finalization, EOS, or other boundary tokens.

The token panels condition on prompts correct in every available replicate at every K from 4
through 16. This prevents changes in the difficulty of the successful subset from masquerading as
length changes. Normalized GPQA has two replicates at K={13,14,15} and three elsewhere; every other
condition has three replicates at every K.

For IFBench, performance is the unweighted mean of the three task-level prompt-loose accuracies,
matching the adopted project metric. Token summaries likewise average the three tasks equally, so
the 1,774-prompt MT WildChat task does not dominate the 300-prompt IFEval OOD task.

In this note, **reference-scaled** means retaining the selected top-K expert weights at their K=8
reference scale rather than renormalizing the retained weights to sum to one. It is the
"unnormalized" condition in the current discussion. Native K=8 is shared by both definitions.

## GPQA Diamond

The normalized common-correct cohort contains 45 of 198 questions; the reference-scaled cohort
contains 57. Performance and prompt-matched median full-generation ratios are:

| K | Normalized score | Reference-scaled score | Normalized token ratio vs K=8 | Reference token ratio vs K=8 |
|---:|---:|---:|---:|---:|
| 4 | 56.06 | 60.77 | 1.586 | 1.007 |
| 5 | 60.27 | 59.76 | 1.275 | 1.056 |
| 6 | 62.46 | 61.45 | 1.111 | 1.033 |
| 7 | 62.46 | 63.47 | 1.064 | 1.025 |
| 8 | 61.28 | 61.28 | 1.000 | 1.000 |
| 9 | 62.46 | 63.80 | 0.960 | 1.016 |
| 10 | 62.46 | 64.65 | 0.964 | 0.966 |
| 11 | 63.30 | 62.96 | 0.867 | 0.915 |
| 12 | 60.94 | 62.79 | 0.892 | 0.933 |
| 13 | 60.10 | 66.33 | 0.881 | 0.924 |
| 14 | 61.62 | 62.96 | 0.927 | 0.911 |
| 15 | 63.38 | 61.95 | 0.892 | 0.917 |
| 16 | 58.25 | 62.12 | 0.882 | 0.946 |

The low-K contrast is large. Normalized K=4 takes 1.59 times K=8's matched median generation
length and loses 5.22 score points relative to K=8. Reference-scaled K=4 is essentially length
neutral at 1.01 times K=8 and scores within 0.51 points of K=8.

The result survives a stricter direct comparison on the 35 questions shared by both routing
cohorts:

| Condition | Full tokens | Returned response | Inferred reasoning + boundaries |
|:--|---:|---:|---:|
| Normalized K=4 | 4,513 | 712 | 3,800 |
| Reference-scaled K=4 | 3,056 | 548 | 2,508 |
| Native K=8 | 3,127 | 591 | 2,536 |

Unlike MATH-500, normalized low-K GPQA lengthens both hidden reasoning and the returned answer.
Reference scaling removes both effects: its K=4 hidden reasoning is nearly identical to K=8 and
its returned response is slightly shorter.

## IFBench-32k

The normalized common-success cohort contains 1,242 prompts: 24 IFEval OOD, 937 MT WildChat, and
281 MT OOD WildChat. The reference-scaled cohort contains 1,493 prompts: 35, 1,114, and 344 from
those tasks, respectively.

| K | Normalized macro | Reference-scaled macro | Normalized token ratio vs K=8 | Reference token ratio vs K=8 |
|---:|---:|---:|---:|---:|
| 4 | 48.15 | 54.29 | 1.308 | 1.119 |
| 5 | 53.97 | 55.46 | 1.104 | 1.082 |
| 6 | 56.23 | 56.64 | 1.049 | 1.075 |
| 7 | 56.22 | 56.23 | 1.017 | 1.024 |
| 8 | 55.96 | 55.96 | 1.000 | 1.000 |
| 9 | 56.78 | 56.15 | 1.040 | 1.006 |
| 10 | 56.85 | 56.80 | 1.045 | 1.017 |
| 11 | 56.65 | 56.77 | 1.016 | 1.019 |
| 12 | 56.30 | 56.80 | 1.034 | 1.006 |
| 13 | 55.81 | 57.28 | 1.045 | 0.999 |
| 14 | 55.22 | 57.64 | 1.020 | 1.003 |
| 15 | 55.21 | 56.39 | 1.052 | 1.014 |
| 16 | 55.35 | 56.49 | 1.059 | 0.963 |

Reference scaling at K=4 recovers 6.14 macro points and reduces the matched median length penalty
from 31% to 12%. On the 1,157 prompts shared by both routing cohorts, with tasks still weighted
equally, the direct comparison is:

| Condition | Full tokens | Returned response | Inferred reasoning + boundaries |
|:--|---:|---:|---:|
| Normalized K=4 | 1,280 | 351 | 930 |
| Reference-scaled K=4 | 1,047 | 318 | 729 |
| Native K=8 | 865 | 289 | 577 |

Reference scaling therefore mitigates rather than completely removes IFBench's K=4 generation
penalty. Both hidden reasoning and returned text remain somewhat longer than K=8. By K=6--7,
performance and typical response length are effectively in the native regime under either
routing condition.

The mean-token bars have several visible spikes that are not reflected in the median-ratio panel.
They come primarily from rare, extremely long parser-returned responses in the small IFEval OOD
common-success subset, especially reference-scaled K=5 and K={14,15}, and normalized K=16. The
right-hand prompt-matched median panel is the better description of typical behavior; the middle
panel usefully preserves the tail-cost signal.

## Conclusions

1. The normalized low-K generation penalty generalizes beyond MATH-500. It is particularly large
   at K=4 on GPQA and IFBench.
2. Preserving K=8 weight scale almost completely removes that penalty on GPQA and substantially
   reduces it on IFBench.
3. Capability moves in the same direction: reference-scaled K=4 is much closer to K=8 on both
   GPQA and IFBench than normalized K=4.
4. The mechanism is not purely hidden-chain length on every task. GPQA and IFBench also show some
   low-K change in returned response length, although hidden reasoning remains the larger component.
5. K=5--7 remains the strongest reduced-compute region: capability is generally at the K=8 level
   and typical generation lengths are close to native.

## Artifacts

- Normalized GPQA plot:
  `notes/plots/qwen3_gpqa_common_correct_token_decomposition_normalized.png` and `.svg`
- Reference-scaled GPQA plot:
  `notes/plots/qwen3_gpqa_common_correct_token_decomposition_reference_scaled.png` and `.svg`
- Normalized IFBench plot:
  `notes/plots/qwen3_ifbench_common_success_token_decomposition_normalized.png` and `.svg`
- Reference-scaled IFBench plot:
  `notes/plots/qwen3_ifbench_common_success_token_decomposition_reference_scaled.png` and `.svg`
- Machine-readable reports: `notes/qwen_gpqa_token_decomposition_{normalized,reference_scaled}.json`
  and `notes/qwen_ifbench_token_decomposition_{normalized,reference_scaled}.json`
- Analysis and plotting scripts: `scripts/adaptive_experts/analyze_qwen_task_token_decomposition.py`
  and `scripts/adaptive_experts/plot_qwen_task_token_decomposition.py`
