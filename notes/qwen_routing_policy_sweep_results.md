# Qwen routing-policy sweep results

Last collected: 2026-07-17T16:56:48Z

This is the live result record for the 40-evaluation sweep launched in
`qwen_routing_policy_sweep_launch.md`. Downloaded bundles are stored under
`results/adaptive_experts/<beaker-experiment-id>/`; per-run machine-readable values are in
`routing_policy/qwen3_runs.csv`.

Scores use each task's complete dataset as the denominator. MATH-500 has 500 examples, GPQA
Diamond has 198, and IFEval OOD has 300. Values are percentages. Condition summaries with at
least two completed runs report mean ± sample standard deviation in percentage points; a lone
run is displayed without an uncertainty estimate.

## Collection status

| State | Runs |
|:--|--:|
| Complete, downloaded, and validated | 40 |
| Running or queued | 0 |
| Total | 40 |

All 40 successful bundles are local. Each has all 998 expected prediction rows (500 MATH, 198
GPQA, and 300 IFEval), an empty task-error list, and all four expected summary metrics. In total,
39,920 predictions were validated.

The third K=12 reference-scaled experiment,
[01KXQ4CYSACXYQV1PZE2Z4ABRJ](https://beaker.org/ex/01KXQ4CYSACXYQV1PZE2Z4ABRJ),
had an initial job canceled during startup because its node was cordoned after an unrecoverable
GPU Xid 31. Beaker automatically retried the experiment on a healthy node; that second job
completed successfully with all 998 predictions and no task errors. The successful retry is the
replicate-3 result below.

## Condition summaries

Values are mean ± sample standard deviation in percentage points. “Cap” is the percentage of all
998 outputs that reached 32,768 generated tokens. “Empty” is the percentage with no parsed final
content. These diagnostic outputs remain in the evaluation denominators.

| Condition | Valid runs | MATH-500 ↑ | GPQA Diamond ↑ | IFEval OOD ↑ | Macro ↑ | Cap | Empty |
|:--|--:|--:|--:|--:|--:|--:|--:|
| Temperature `p=0.5` | 3/3 | 95.33 ± 0.42 | 59.60 ± 0.51 | 39.33 ± 0.33 | 64.75 ± 0.28 | 2.14 ± 0.31 | 1.54 ± 0.15 |
| Temperature `p=0.75` | 3/3 | 94.87 ± 0.23 | 61.28 ± 2.28 | 39.89 ± 0.51 | 65.35 ± 0.72 | 2.10 ± 0.27 | 1.34 ± 0.31 |
| Temperature `p=1.0` identity sanity | 1/1 | 95.40 | 63.64 | 39.00 | 66.01 | 1.70 | 0.80 |
| Temperature `p=1.5` | 3/3 | 94.40 ± 0.35 | 64.48 ± 1.46 | 37.67 ± 1.00 | 65.51 ± 0.26 | 1.44 ± 0.38 | 1.00 ± 0.27 |
| Temperature `p=2.0` | 3/3 | 65.40 ± 1.59 | 57.74 ± 1.91 | 27.33 ± 0.88 | 50.16 ± 0.88 | 8.18 ± 0.61 | 25.25 ± 0.96 |
| Fixed global rank profile | 3/3 | 93.73 ± 0.50 | 59.43 ± 1.05 | 34.33 ± 0.33 | 62.50 ± 0.30 | 9.22 ± 0.44 | 6.68 ± 0.25 |
| Adaptive mass `tau=0.50` | 3/3 | 37.93 ± 2.08 | 35.02 ± 1.62 | 3.33 ± 0.33 | 25.43 ± 1.12 | 24.48 ± 0.95 | 59.15 ± 0.90 |
| Adaptive mass `tau=0.60` | 3/3 | 84.20 ± 1.59 | 53.87 ± 0.58 | 20.56 ± 2.84 | 52.88 ± 1.55 | 7.15 ± 0.47 | 18.60 ± 1.05 |
| Adaptive mass `tau=0.70` | 3/3 | 95.27 ± 0.46 | 60.94 ± 3.36 | 32.00 ± 1.45 | 62.74 ± 1.54 | 2.54 ± 0.35 | 3.81 ± 0.56 |
| Adaptive mass `tau=0.80` | 3/3 | 95.53 ± 0.42 | 61.95 ± 1.62 | 35.33 ± 0.67 | 64.27 ± 0.66 | 1.70 ± 0.36 | 1.07 ± 0.15 |
| Adaptive mass `tau=0.90` | 3/3 | 95.33 ± 0.31 | 62.12 ± 0.51 | 40.11 ± 3.50 | 65.86 ± 1.27 | 1.74 ± 0.35 | 1.14 ± 0.21 |
| K=4, K=8-reference scaled | 3/3 | 94.87 ± 0.42 | 60.77 ± 1.05 | 36.78 ± 1.90 | 64.14 ± 0.64 | 2.87 ± 0.15 | 1.30 ± 0.10 |
| K=6, K=8-reference scaled | 3/3 | 95.13 ± 0.46 | 60.77 ± 1.27 | 38.44 ± 1.26 | 64.78 ± 0.25 | 1.67 ± 0.47 | 0.73 ± 0.25 |
| K=12, K=8-reference scaled | 3/3 | 94.73 ± 0.64 | 59.60 ± 3.54 | 40.33 ± 0.58 | 64.89 ± 1.16 | 1.60 ± 0.66 | 0.90 ± 0.46 |

## Initial takeaways

- The `p=1` patched identity result is close to the earlier native K=8 controls, supporting
  comparability between the patched and native backends.
- Mild flattening (`p=0.5`, `p=0.75`) and sharpening (`p=1.5`) preserve most performance. Strong
  sharpening (`p=2`) reproducibly loses about 16 macro points versus the identity control and
  produces many empty parsed responses.
- Replacing token-specific weights with the global mean rank profile retains substantial
  capability but loses about 3.5 macro points versus the identity control. This suggests that
  token-level confidence calibration contributes beyond the average rank shape.
- Cumulative-mass routing shows a smooth quality curve. `tau=0.80` reaches 64.27 macro and
  `tau=0.90` reaches 65.86, while lower thresholds fail progressively.
- Preserving K=8 weight scale at K=4 scores 64.14 macro, versus 59.74 for the earlier normalized
  native K=4 evaluation. K=6 is similar on both scales (64.78 reference-scaled versus 65.21
  normalized), as is adding K=12 mass above the K=8 scale (64.89 versus 65.36 normalized). This
  supports the hypothesis that survivor renormalization can itself be harmful at low K,
  especially K=4.

The full per-run metrics and diagnostics are in `routing_policy/qwen3_runs.csv`. The launch record
`qwen_routing_policy_sweep_launch.md` contains every Beaker link and the exact policy semantics.

## Recommended next evaluations

No jobs in this section have been launched.

1. **Broaden the strongest Qwen comparisons.** Evaluate native/identity K=8, normalized K=4,
   K=8-reference-scaled K=4, adaptive `tau=0.80`, and adaptive `tau=0.90` on the existing
   `ifbench` multi-turn instruction-following suite and
   `humaneval_plus:chat:pass_at_1:qwen3_thinking`. Run each condition three times. This directly
   tests the two promising compute policies on the capability classes least represented by
   MATH-500/GPQA/IFEval OOD.
2. **Use AIME 2026 as a cheap hard-reasoning discriminator.** Run adaptive `tau=0.80`,
   adaptive `tau=0.90`, and reference-scaled K=4 on `aime_2026:pass_at_32`. One evaluation per
   condition is sufficient because each evaluation already produces 32 samples per problem.
   Existing native K=4 and K=8 AIME runs provide comparison points.
3. **Test whether the mechanism generalizes across MoEs.** First measure GPT-OSS-120B and
   GLM-4.5-Air router-mass profiles so thresholds correspond to meaningful average K values. Then
   run GPT-OSS reference-scaled K=2 plus one adaptive threshold near average K=2–3, and GLM
   reference-scaled K=4 plus thresholds near average K=4 and K=6. Use the existing three-task
   suite and three repetitions after one smoke per code path.
4. **Instrument realized K before making efficiency claims.** The current adaptive implementation
   zeroes unused fixed-width slots but still dispatches them. Record live K by layer, task, prompt
   versus generation, and reasoning versus final-answer phase; then implement true variable-K
   dispatch and evaluate quality together with tokens/s, latency, and memory.

Additional temperature points and more fixed-global-profile evaluations are lower priority:
they clarify router calibration but do not reduce expert computation. MMLU also remains deferred
because this repository still lacks the desired chat-formatted task; GPQA already supplies a
knowledge-heavy signal.
