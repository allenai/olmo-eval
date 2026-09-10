# Prompt-level MATH-500 transitions across expert counts

This analysis matches the same 500 prompts across every K and uses all three
replicates of the step-350 **reference-scaled** plateau sweep. A prompt is called
correct at a K when at least two of three generations are correct. The individual
replicates are independent samples, so the analysis treats 0/3 through 3/3 as an
empirical success rate rather than pairing replicate 1 across K.

## Summary

| Training run | Always correct | Always wrong | Clean K threshold | Non-monotonic | Any replicate disagreement |
|---|---:|---:|---:|---:|---:|
| trained K=8 normalized | 220 (44.0%) | 38 (7.6%) | 146 (29.2%) | 96 (19.2%) | 358 (71.6%) |
| trained K=6 reference-scaled | 224 (44.8%) | 48 (9.6%) | 120 (24.0%) | 108 (21.6%) | 378 (75.6%) |
| trained K=4 reference-scaled | 252 (50.4%) | 49 (9.8%) | 109 (21.8%) | 90 (18.0%) | 335 (67.0%) |

`Clean K threshold` means a majority-correct sequence that changes from wrong to
correct at most once over K=2…12. `Non-monotonic` means a majority-correct prompt
later becomes majority-wrong at a larger K at least once.

## The policy-relevant K=4/6/8 slice

Patterns are majority correctness at K=4, K=6, and K=8 respectively.

| Training run | 000 | 001 | 011 | 111 | Non-monotonic patterns | Corr. 4↔6 | Corr. 6↔8 | Corr. 4↔8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| trained K=8 normalized | 53 | 10 | 20 | 390 | 27 | 0.817 | 0.835 | 0.797 |
| trained K=6 reference-scaled | 63 | 10 | 24 | 368 | 35 | 0.803 | 0.775 | 0.735 |
| trained K=4 reference-scaled | 62 | 8 | 13 | 385 | 32 | 0.814 | 0.812 | 0.795 |

Mean scores from the same prediction files:

| Training run | K=2 | K=3 | K=4 | K=5 | K=6 | K=7 | K=8 | K=9 | K=10 | K=11 | K=12 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| trained K=8 normalized | 0.478 | 0.737 | 0.809 | 0.832 | 0.829 | 0.832 | 0.841 | 0.831 | 0.846 | 0.836 | 0.821 |
| trained K=6 reference-scaled | 0.489 | 0.735 | 0.785 | 0.811 | 0.802 | 0.790 | 0.794 | 0.805 | 0.807 | 0.816 | 0.815 |
| trained K=4 reference-scaled | 0.542 | 0.766 | 0.813 | 0.826 | 0.825 | 0.816 | 0.827 | 0.819 | 0.812 | 0.815 | 0.819 |

## Interpretation

There is a real but small monotonic signal in the policy-relevant K=4/6/8 range.
Across the three checkpoints, 73.6--78.0% of prompts are majority-correct at every
one of K=4, K=6, and K=8, while 10.6--12.6% are wrong at all three. Only 4.2--6.8%
show a clean transition that actually needs K=6 or K=8. Another 5.4--7.0% have a
non-monotonic majority pattern. Prompt difficulty is nevertheless fairly stable:
the K=4/K=8 empirical-success correlations are 0.735--0.797.

Looking directly at the three-sample empirical probabilities rather than majority
labels makes the limited separation especially clear. K=4 and K=8 have identical
success rates on 69--79% of prompts. K=8 is better on 11.6--16.0%, while K=4 is
better on 7.8--15.0%; the net average K=8 advantage is only 0.9--3.1 percentage
points, depending on the trained checkpoint.

Generation variance is material. Even within only K=4/6/8, 27.8--38.6% of prompts
have at least one K where the three generations disagree. Over the full K=2--12
sweep that rises to 67.0--75.6%. A hard label such as ‘this prompt needs K=6’ from
one generation would therefore be noisy and highly imbalanced. If we pursue a prompt
selector, it should use multi-sample success probabilities/confidence bounds and first
report an oracle quality-versus-average-K curve. Direct cost-aware RL remains especially
well motivated because it need not pretend every prompt has one deterministic minimum K.

![Matched-prompt heatmaps](/weka/oe-adapt-default/jacobm/olmoe3/adaptive-compute/olmo-eval/notes/plots/slime_qwen3_base_dapo_prompt_k_heatmaps.png)

Per-prompt probabilities: `/weka/oe-adapt-default/jacobm/olmoe3/adaptive-compute/olmo-eval/notes/slime_qwen3_base_dapo_prompt_k_transitions.csv`
