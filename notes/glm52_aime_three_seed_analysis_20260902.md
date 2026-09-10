# GLM-5.2 AIME 2026 three-seed analysis

Status timestamp: 2026-09-02 UTC.

## Scope and run health

This analysis compares the completed GLM-5.2 native `K=8` baseline with the
native-reference-truncated `K=4` intervention. Each policy has seeds 0, 1, and
2; every run contains all 30 AIME 2026 problems and 32 generations per problem
(960 generations per seed, 2,880 per policy). The four replicate jobs for
seeds 1 and 2 all exited with code 0 and produced complete metrics and
prediction artifacts. No rerun is needed.

The native-reference-truncated policy retains the native top four experts but
divides their sigmoid scores by the sum of the native top-eight scores. It is
the scale-preserving reduced-`K` analogue, not renormalized top-four routing.

| Policy | Seed | Correct samples | Pass@1 | Pass@32 | Problems with no success |
|---|---:|---:|---:|---:|---|
| Native `K=8` | 0 | 807/960 | 84.06% | 96.67% | 15 |
| Native `K=8` | 1 | 793/960 | 82.60% | 96.67% | 15 |
| Native `K=8` | 2 | 782/960 | 81.46% | 96.67% | 15 |
| Reference-truncated `K=4` | 0 | 742/960 | 77.29% | 93.33% | 15, 30 |
| Reference-truncated `K=4` | 1 | 714/960 | 74.38% | 93.33% | 15, 30 |
| Reference-truncated `K=4` | 2 | 751/960 | 78.23% | 96.67% | 15 |

## Aggregate scores

Values below are the mean across three seeds, with the sample standard
deviation across seeds in parentheses.

| Policy | Pass@1 | Pass@4 | Pass@8 | Pass@16 | Pass@32 |
|---|---:|---:|---:|---:|---:|
| Native `K=8` | 82.71% (1.31) | 94.38% (0.37) | 95.63% (0.33) | 96.46% (0.16) | 96.67% (0.00) |
| Reference-truncated `K=4` | 76.63% (2.01) | 91.60% (0.65) | 93.21% (0.66) | 93.88% (0.97) | 94.44% (1.92) |

The paired seed-wise pass@1 differences (`K=4 - K=8`) are -6.77, -8.23, and
-3.23 percentage points, for a mean of **-6.08 points** and a seed SD of 2.57
points. The direction is consistent across all three seeds. Pooling all 96
samples per problem does not improve coverage beyond 29/30 for either policy:
problem 15 receives no correct generation under either policy, while the
reference policy's problem-30 miss in seeds 0 and 1 is recovered in seed 2.

A paired bootstrap over the 30 problem clusters gives a 95% interval of
[-12.85, +0.63] points for the pass@1 difference. This interval is wide because
the intervention is strongly problem-dependent, not because the seed-level
direction is unstable. Treat the three-seed point estimate as compelling
evidence of a mean loss on this set, but not as a precise population estimate
for arbitrary hard-math problems.

## Problem heterogeneity

Aggregating 96 generations per problem, reference-truncated `K=4` is worse on
20 problems, tied on one, and better on nine. The largest changes are:

| Problem | Native correct | Reference correct | Difference |
|---:|---:|---:|---:|
| 29 | 80/96 | 25/96 | -57.29 points |
| 10 | 67/96 | 28/96 | -40.62 points |
| 6 | 94/96 | 62/96 | -33.33 points |
| 17 | 93/96 | 69/96 | -25.00 points |
| 3 | 44/96 | 83/96 | +40.62 points |
| 20 | 69/96 | 90/96 | +21.88 points |
| 12 | 72/96 | 91/96 | +19.79 points |

The losses therefore are not a uniform quality shift. Reduced `K` changes
which reasoning paths are reliable: it is much worse on several combinatorial,
geometric, and algebraic items while materially better on a smaller set.

## Output validity and length

| Policy | Empty visible output | No extracted answer | Hit 163,840-token cap | Gross repetition loop |
|---|---:|---:|---:|---:|
| Native `K=8` | 0/2,880 | 224/2,880 | 0/2,880 | 0/2,880 |
| Reference-truncated `K=4` | 9/2,880 | 143/2,880 | 2/2,880 | 5/2,880 |

“Gross repetition loop” is a deliberately conservative automatic flag:
four-gram repetition at least 0.90 and zlib compression ratio at most 0.10.
Manual inspection confirms that the five flagged outputs are genuine long
loops. This is a rare event (0.17%), but it and the nine empty visible outputs
show a real tail-reliability regression at `K=4`. The smaller no-answer count
at `K=4` does not imply better parsing: many native outputs state a correct
answer without the exact boxing format expected by Minerva.

| Policy | Total tokens, mean | Total tokens, median | P90 | P95 | Max | Visible characters, median |
|---|---:|---:|---:|---:|---:|---:|
| Native `K=8` | 19,694 | 17,818 | 37,777 | 44,031 | 64,095 | 2,824 |
| Reference-truncated `K=4` | 19,122 | 17,641 | 30,774 | 35,395 | 163,840 | 2,481 |

Typical `K=4` generations are modestly shorter: median total generation length
is 1.0% lower, mean length is 2.9% lower, P95 is 19.6% lower, and median visible
text is 12.1% shorter. The maximum reverses because two rare `K=4` failures hit
the full output cap. The stored predictions retain total generated-token counts
but not hidden `reasoning_content`, so the three-seed analysis does not claim a
fresh exact split between hidden reasoning and visible tokens.

The like-for-like subset with nonempty visible text in both policies contains
2,871 paired seed/problem/sample positions. Accuracy is 82.76% for native
`K=8` and 76.87% for reference `K=4`, a 5.89-point difference. Restricting
further to the 2,534 positions where both outputs have an extracted answer
gives 89.50% versus 79.48%. Thus the observed degradation is not driven by the
nine empty outputs.

## Qualitative output audit

There is no broad fluency or coherence collapse. Most `K=4` outputs are
well-formed mathematical solutions. Manual review of discordant and outlier
samples shows three recurring failure modes:

1. **Plausible but wrong derivations.** On problem 29, a `K=4` sample claims
   that concatenations of `(1, 2, 1)` are the only valid sequences and returns
   1 rather than 157. On problem 10, another follows an incorrect rotation and
   area derivation to 41 rather than 156. These are substantive reasoning
   errors with polished prose.
2. **Correct content with extractor-hostile formatting.** Some scored failures
   explicitly derive the correct result but omit `\boxed{}` or let the extractor
   select a nearby variable. Examples include problem 6 ending with 441 while
   extracting `x`, and problem 17 ending with 243 while extracting
   `\sqrt{N}`. A conservative final-answer-text heuristic finds this pattern in
   roughly 20% of scored errors under both policies, so it affects absolute
   scores but does not explain the `K=4` gap.
3. **Rare tail degeneration.** The nine empty outputs, two cap hits, and five
   gross loops occur only under reference-truncated `K=4`. They are too rare to
   dominate pass@1, but should be tracked as a reliability metric in future
   reduced-`K` experiments.

Overall, native-reference truncation is substantially better than the earlier
normalized-`K=4` seed-0 result, but it does not preserve native GLM-5.2 quality:
the three-seed estimate is a roughly six-point pass@1 loss, about a two-point
mean pass@32 loss, shorter typical generations, and a small increase in severe
tail failures.

## Reproduction and artifacts

Collected artifacts are under `results/glm52_aime_replicates/`. Recompute the
analysis from the prediction JSONL files with:

```bash
python scripts/adaptive_experts/analyze_glm52_aime_replicates.py
```

The original experiment IDs and result-dataset IDs remain recorded in
[`holmes_cuda13_large_model_eval_20260826.md`](holmes_cuda13_large_model_eval_20260826.md).
